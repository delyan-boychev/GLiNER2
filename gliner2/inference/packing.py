"""Correctness-first inference sequence packing.

The module owns planning, block-mask construction, encoder capability checks,
overflow fallback, and exact inverse mapping.  It intentionally knows nothing
about GLiNER2 task heads: callers receive one exact-length hidden-state tensor
per original request and continue through the existing extraction pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from gliner2.inference.encoder_adapters import (
    SequencePackingUnsupportedError,
    get_encoder_packing_adapter,
)

logger = logging.getLogger(__name__)


class PackingOverflowError(ValueError):
    """Raised when an encoded sequence exceeds the configured packing limit."""


@dataclass(frozen=True)
class PackingConfig:
    """Opt-in inference sequence-packing configuration.

    Packing is disabled by default.  ``max_packed_length`` applies only to the
    packing container; it never truncates tokenized records.
    """

    enabled: bool = False
    max_packed_length: int = 512
    max_segments_per_stream: int = 32
    strategy: str = "best_fit"
    attention_backend: str = "dense_block"
    length_buckets: Tuple[int, ...] = (128, 256, 384, 512)
    overflow: str = "fallback"
    min_fill_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.max_packed_length <= 0:
            raise ValueError("max_packed_length must be > 0")
        if self.max_segments_per_stream <= 0:
            raise ValueError("max_segments_per_stream must be > 0")
        if self.strategy != "best_fit":
            raise ValueError("only strategy='best_fit' is currently supported")
        if self.attention_backend != "dense_block":
            raise ValueError("only attention_backend='dense_block' is currently supported")
        if self.overflow not in {"fallback", "error"}:
            raise ValueError("overflow must be 'fallback' or 'error'")
        buckets = tuple(self.length_buckets)
        if any(bucket <= 0 for bucket in buckets):
            raise ValueError("length_buckets must contain only positive lengths")
        if any(left >= right for left, right in zip(buckets, buckets[1:])):
            raise ValueError("length_buckets must be strictly increasing")
        if not 0.0 <= self.min_fill_ratio <= 1.0:
            raise ValueError("min_fill_ratio must be in [0, 1]")


@dataclass(frozen=True)
class PackedSegment:
    """Inverse-map entry for one request inside a packed stream."""

    request_index: int
    stream_index: int
    offset: int
    length: int


@dataclass
class PackedEncoderBatch:
    """Tensor inputs and inverse map for a dense block-masked encoder call."""

    input_ids: torch.Tensor
    token_attention_mask: torch.Tensor
    pair_attention_mask: torch.Tensor
    position_ids: Optional[torch.Tensor]
    segment_ids: torch.Tensor
    segments: List[PackedSegment]
    original_count: int


@dataclass(frozen=True)
class PackingStats:
    """Per-model-batch diagnostics for rollout and benchmark reporting."""

    activated: bool
    reason: str
    original_count: int
    packed_count: int = 0
    fallback_count: int = 0
    stream_count: int = 0
    packed_length: int = 0
    fill_ratio: float = 0.0
    baseline_cost: float = 0.0
    packed_cost: float = 0.0


@dataclass
class _Stream:
    used: int
    segments: List[Tuple[int, int]]


def supports_sequence_packing(encoder_config) -> bool:
    """Return whether a config describes a currently supported safe encoder.

    This conservative check deliberately accepts only Transformers'
    DeBERTa-v2/v3 family and rejects the optional cross-token convolution.
    Runtime adapter selection performs stricter class and version validation.
    """

    if encoder_config is None:
        return False
    model_type = str(getattr(encoder_config, "model_type", "")).replace("_", "-").lower()
    if model_type != "deberta-v2" and type(encoder_config).__name__ != "DebertaV2Config":
        return False
    if int(getattr(encoder_config, "conv_kernel_size", 0) or 0) > 0:
        return False
    return True


def _sequence_lengths(attention_mask: torch.Tensor) -> List[int]:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    active = attention_mask.bool()
    lengths = active.sum(dim=1).tolist()
    width = attention_mask.shape[1]
    positions = torch.arange(width, device=attention_mask.device).unsqueeze(0)
    expected = positions < torch.tensor(lengths, device=attention_mask.device).unsqueeze(1)
    if not torch.equal(active, expected):
        raise ValueError("packing requires prefix-contiguous attention masks")
    return [int(length) for length in lengths]


def _best_fit_plan(
    request_lengths: Sequence[Tuple[int, int]],
    config: PackingConfig,
) -> Tuple[List[PackedSegment], List[int]]:
    """Stable best-fit decreasing with deterministic tie breaking."""

    ordered = sorted(request_lengths, key=lambda item: (-item[1], item[0]))
    streams: List[_Stream] = []

    for request_index, length in ordered:
        if length < 0:
            raise ValueError("encoded lengths must be non-negative")
        if length > config.max_packed_length:
            raise PackingOverflowError(
                f"request {request_index} has encoded length {length}, exceeding "
                f"max_packed_length={config.max_packed_length}; packing never truncates"
            )

        candidates = []
        for stream_index, stream in enumerate(streams):
            if len(stream.segments) >= config.max_segments_per_stream:
                continue
            remaining_after = config.max_packed_length - stream.used - length
            if remaining_after >= 0:
                candidates.append((remaining_after, stream_index))

        if candidates:
            _, stream_index = min(candidates)
            stream = streams[stream_index]
        else:
            stream_index = len(streams)
            stream = _Stream(used=0, segments=[])
            streams.append(stream)

        stream.segments.append((request_index, length))
        stream.used += length

    segments: List[PackedSegment] = []
    for stream_index, stream in enumerate(streams):
        offset = 0
        for request_index, length in stream.segments:
            segments.append(PackedSegment(request_index, stream_index, offset, length))
            offset += length
        if offset != stream.used:  # pragma: no cover - defensive invariant
            raise AssertionError("packing planner offset mismatch")
    return segments, [stream.used for stream in streams]


def plan_packing(
    lengths: Sequence[int],
    config: PackingConfig,
    request_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[PackedSegment], List[int]]:
    """Plan packing without allocating the quadratic pair mask."""

    if request_indices is None:
        request_indices = list(range(len(lengths)))
    if len(request_indices) != len(lengths):
        raise ValueError("request_indices and lengths must have equal size")
    if len(set(int(index) for index in request_indices)) != len(request_indices):
        raise ValueError("request_indices must be unique")
    return _best_fit_plan(
        [(int(index), int(length)) for index, length in zip(request_indices, lengths)],
        config,
    )


def _bucket_length(used_length: int, config: PackingConfig) -> int:
    if used_length == 0:
        return 0
    for bucket in config.length_buckets:
        if used_length <= bucket <= config.max_packed_length:
            return bucket
    return config.max_packed_length


def pack_requests(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: PackingConfig,
    *,
    request_indices: Optional[Sequence[int]] = None,
    original_count: Optional[int] = None,
    pad_token_id: int = 0,
) -> PackedEncoderBatch:
    """Pack already-valid encoder records without truncation or chunking.

    Every supplied record must fit.  High-level inference separates overflow
    records before calling this function and sends them through the ordinary
    encoder path when ``overflow='fallback'``.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    lengths = _sequence_lengths(attention_mask)
    if request_indices is None:
        request_indices = list(range(len(lengths)))
    request_indices = [int(index) for index in request_indices]
    if len(request_indices) != len(lengths):
        raise ValueError("request_indices and input batch must have equal size")
    if original_count is None:
        original_count = len(lengths)
    if original_count < len(lengths):
        raise ValueError("original_count cannot be smaller than the packed record count")
    if any(index < 0 or index >= original_count for index in request_indices):
        raise ValueError("request index is outside original_count")

    segments, stream_lengths = plan_packing(lengths, config, request_indices)
    stream_count = len(stream_lengths)
    packed_length = _bucket_length(max(stream_lengths, default=0), config)
    device = input_ids.device

    packed_ids = torch.full(
        (stream_count, packed_length),
        int(pad_token_id),
        dtype=input_ids.dtype,
        device=device,
    )
    token_mask = torch.zeros((stream_count, packed_length), dtype=torch.bool, device=device)
    position_ids = torch.zeros((stream_count, packed_length), dtype=torch.long, device=device)
    segment_ids = torch.full(
        (stream_count, packed_length), -1, dtype=torch.long, device=device
    )

    source_rows = {request_index: row for row, request_index in enumerate(request_indices)}
    for segment in segments:
        row = source_rows[segment.request_index]
        stop = segment.offset + segment.length
        if stop > packed_length:  # pragma: no cover - planner/bucket invariant
            raise AssertionError("planned segment exceeds packed tensor")
        packed_ids[segment.stream_index, segment.offset:stop] = input_ids[
            row, :segment.length
        ]
        token_mask[segment.stream_index, segment.offset:stop] = True
        position_ids[segment.stream_index, segment.offset:stop] = torch.arange(
            segment.length, device=device
        )
        segment_ids[segment.stream_index, segment.offset:stop] = segment.request_index

    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    pair_mask = (
        same_segment
        & token_mask[:, :, None]
        & token_mask[:, None, :]
    )

    return PackedEncoderBatch(
        input_ids=packed_ids,
        token_attention_mask=token_mask,
        pair_attention_mask=pair_mask,
        position_ids=position_ids,
        segment_ids=segment_ids,
        segments=segments,
        original_count=int(original_count),
    )


def unpack_hidden_states(
    packed_hidden: torch.Tensor,
    packed: PackedEncoderBatch,
    *,
    expected_input_ids: Optional[torch.Tensor] = None,
    expected_request_indices: Optional[Sequence[int]] = None,
) -> Dict[int, torch.Tensor]:
    """Unpack exact segment slices and assert the complete inverse map."""

    if packed_hidden.ndim != 3:
        raise ValueError("packed_hidden must have shape [streams, sequence, hidden]")
    if packed_hidden.shape[:2] != packed.input_ids.shape:
        raise ValueError("packed_hidden stream/sequence shape does not match packed input")

    assigned = torch.zeros_like(packed.token_attention_mask)
    unpacked: Dict[int, torch.Tensor] = {}
    for segment in packed.segments:
        stop = segment.offset + segment.length
        if segment.stream_index >= packed_hidden.shape[0] or stop > packed_hidden.shape[1]:
            raise AssertionError("packed segment is outside encoder output bounds")
        region = assigned[segment.stream_index, segment.offset:stop]
        if bool(region.any()):
            raise AssertionError("a packed token was assigned to multiple requests")
        region.fill_(True)
        if segment.request_index in unpacked:
            raise AssertionError(f"request {segment.request_index} appears more than once")

        hidden = packed_hidden[segment.stream_index, segment.offset:stop]
        if hidden.shape[0] != segment.length:
            raise AssertionError("unpacked hidden-state length mismatch")
        if expected_input_ids is not None:
            expected = expected_input_ids[segment.request_index, :segment.length]
            actual = packed.input_ids[segment.stream_index, segment.offset:stop]
            if not torch.equal(actual, expected):
                raise AssertionError("packed token IDs do not match the original request")
        unpacked[segment.request_index] = hidden

    if not torch.equal(assigned, packed.token_attention_mask):
        raise AssertionError("active packed tokens are not covered exactly once")
    if expected_request_indices is None:
        expected_request_indices = range(packed.original_count)
    expected_set = {int(index) for index in expected_request_indices}
    if set(unpacked) != expected_set:
        missing = sorted(expected_set - set(unpacked))
        extra = sorted(set(unpacked) - expected_set)
        raise AssertionError(f"packing inverse map mismatch: missing={missing}, extra={extra}")
    return unpacked


def estimate_baseline_cost(
    lengths: Sequence[int], alpha: float, beta: float
) -> float:
    """Estimate ordinary padded encoder work using ``alpha*B*M^2 + beta*B*M``."""

    if not lengths:
        return 0.0
    maximum = max(int(length) for length in lengths)
    batch = len(lengths)
    return float(alpha * batch * maximum * maximum + beta * batch * maximum)


def estimate_packed_cost(
    stream_lengths: Sequence[int],
    token_count: int,
    alpha: float,
    beta: float,
) -> float:
    """Estimate ideal packed work using ``alpha*sum(S^2) + beta*sum(L)``."""

    return float(
        alpha * sum(int(length) ** 2 for length in stream_lengths)
        + beta * int(token_count)
    )


def _execution_cost(
    batch: int, width: int, alpha: float, beta: float
) -> float:
    return float(alpha * batch * width * width + beta * batch * width)


def _normal_encode(encoder, input_ids, attention_mask) -> torch.Tensor:
    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    return outputs.last_hidden_state


def _cost_weights(encoder) -> Tuple[float, float]:
    """Analytic transformer work weights used until backend profiles override them."""

    config = getattr(encoder, "config", None)
    if config is None and hasattr(encoder, "_orig_mod"):
        config = getattr(encoder._orig_mod, "config", None)
    hidden = max(1, int(getattr(config, "hidden_size", 768)))
    # Approximate attention score/value work and projection+FFN token work.
    return float(2 * hidden), float(12 * hidden * hidden)


_WARNED_UNSUPPORTED = set()


def _warn_unsupported_once(reason: str) -> None:
    if reason not in _WARNED_UNSUPPORTED:
        _WARNED_UNSUPPORTED.add(reason)
        logger.warning("sequence packing disabled for this batch: %s", reason)


def encode_batch_with_packing(
    encoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: PackingConfig,
    *,
    pad_token_id: int = 0,
    return_padded_fallback: bool = False,
) -> Tuple[Union[List[torch.Tensor], torch.Tensor], PackingStats]:
    """Encode a model batch and return exact-length states in request order.

    The function is inference-only and does not mutate inputs.  Unsupported
    encoders, uneconomic shapes, and configured overflow records use the normal
    public encoder path.  A sequence is never sliced or padded and presented as
    though the padding had been encoded content.  Runtime callers may set
    ``return_padded_fallback`` to reuse the existing padded extraction path when
    packing is not activated; activated packing always returns exact segments.
    """

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be matching 2D tensors")
    lengths = _sequence_lengths(attention_mask)
    count = len(lengths)
    if count == 0:
        return [], PackingStats(False, "empty_batch", 0)

    def normal(reason: str, *, warn: bool = False):
        if warn:
            _warn_unsupported_once(reason)
        hidden = _normal_encode(encoder, input_ids, attention_mask)
        if return_padded_fallback:
            return hidden, PackingStats(
                False, reason, count, fallback_count=count
            )
        exact = [hidden[index, :length] for index, length in enumerate(lengths)]
        return exact, PackingStats(False, reason, count, fallback_count=count)

    if not config.enabled:
        return normal("disabled")
    if input_ids.device.type == "mps":
        encoder_parameter = next(encoder.parameters(), None)
        encoder_dtype = (
            encoder_parameter.dtype if encoder_parameter is not None else torch.float32
        )
        if encoder_dtype in {torch.float16, torch.bfloat16}:
            return normal("mps_reduced_precision_failed_parity", warn=True)
    encoder_config = getattr(encoder, "config", None)
    if encoder_config is None and hasattr(encoder, "_orig_mod"):
        encoder_config = getattr(encoder._orig_mod, "config", None)
    if not supports_sequence_packing(encoder_config):
        return normal("unsupported_encoder_config", warn=True)
    try:
        adapter = get_encoder_packing_adapter(encoder)
    except SequencePackingUnsupportedError as error:
        return normal(str(error), warn=True)

    eligible = [
        index
        for index, length in enumerate(lengths)
        if length <= config.max_packed_length
    ]
    overflow = [
        index
        for index, length in enumerate(lengths)
        if length > config.max_packed_length
    ]
    if overflow and config.overflow == "error":
        details = ", ".join(f"{index}:{lengths[index]}" for index in overflow)
        raise PackingOverflowError(
            f"encoded requests exceed max_packed_length={config.max_packed_length}: {details}"
        )
    if len(eligible) < 2:
        return normal("fewer_than_two_packable_requests")

    eligible_lengths = [lengths[index] for index in eligible]
    segments, stream_lengths = plan_packing(eligible_lengths, config, eligible)
    segment_counts = [0] * len(stream_lengths)
    for segment in segments:
        segment_counts[segment.stream_index] += 1
    if max(segment_counts, default=0) < 2:
        return normal("no_stream_combines_requests")

    packed_length = _bucket_length(max(stream_lengths), config)
    fill_ratio = sum(eligible_lengths) / (len(stream_lengths) * packed_length)
    if fill_ratio < config.min_fill_ratio:
        return normal("fill_ratio_below_minimum")

    alpha, beta = _cost_weights(encoder)
    baseline_cost = _execution_cost(count, max(lengths), alpha, beta)
    packed_cost = _execution_cost(
        len(stream_lengths), packed_length, alpha, beta
    )
    if overflow:
        overflow_width = max(lengths[index] for index in overflow)
        packed_cost += _execution_cost(len(overflow), overflow_width, alpha, beta)
    if packed_cost > 0.9 * baseline_cost:
        return normal("estimated_cost_not_lower")

    subset_ids = input_ids[eligible]
    subset_mask = attention_mask[eligible]
    packed = pack_requests(
        subset_ids,
        subset_mask,
        config,
        request_indices=eligible,
        original_count=count,
        pad_token_id=pad_token_id,
    )
    packed_hidden = adapter.encode(
        packed.input_ids,
        packed.token_attention_mask,
        packed.pair_attention_mask,
        packed.position_ids,
    )
    unpacked = unpack_hidden_states(
        packed_hidden,
        packed,
        expected_input_ids=input_ids,
        expected_request_indices=eligible,
    )

    if overflow:
        overflow_width = max(lengths[index] for index in overflow)
        fallback_hidden = _normal_encode(
            encoder,
            input_ids[overflow, :overflow_width],
            attention_mask[overflow, :overflow_width],
        )
        for row, request_index in enumerate(overflow):
            unpacked[request_index] = fallback_hidden[row, :lengths[request_index]]

    if set(unpacked) != set(range(count)):
        raise AssertionError("not every request was restored after packed/fallback encoding")
    exact = [unpacked[index] for index in range(count)]
    for index, hidden in enumerate(exact):
        if hidden.shape[0] != lengths[index]:
            raise AssertionError("restored request length differs from encoded length")

    return exact, PackingStats(
        activated=True,
        reason="packed",
        original_count=count,
        packed_count=len(eligible),
        fallback_count=len(overflow),
        stream_count=len(stream_lengths),
        packed_length=packed_length,
        fill_ratio=float(fill_ratio),
        baseline_cost=baseline_cost,
        packed_cost=packed_cost,
    )


__all__ = [
    "PackingConfig",
    "PackedSegment",
    "PackedEncoderBatch",
    "PackingStats",
    "PackingOverflowError",
    "supports_sequence_packing",
    "plan_packing",
    "pack_requests",
    "unpack_hidden_states",
    "estimate_baseline_cost",
    "estimate_packed_cost",
    "encode_batch_with_packing",
]
