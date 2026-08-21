"""CUDA Triton implementation of inference-only DeBERTa attention.

The positional projections and active relative-position pruning are shared with
``InferenceDisentangledSelfAttention``.  A Triton kernel then fuses content
scores, positional-score lookup, masking, online softmax, and multiplication by
V, avoiding materialized ``[B, H, L, L]`` score and probability tensors.

This module remains importable on machines without Triton.  Instantiating the
class is allowed there, but its forward method reports that CUDA/Triton is
required.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

from .optimized import InferenceDisentangledSelfAttention
from .original import DebertaAttentionConfig, make_log_bucket_position

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is intentionally optional on CPU and macOS.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _deberta_attention_forward_kernel(
        query,
        key,
        value,
        c2p,
        p2c,
        delta_to_local_slot,
        attention_mask,
        output,
        NUM_HEADS: tl.constexpr,
        SEQUENCE_LENGTH: tl.constexpr,
        ACTIVE_SLOTS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SCORE_SCALE_LOG2: tl.constexpr,
        HAS_C2P: tl.constexpr,
        HAS_P2C: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // NUM_HEADS

        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dimension_offsets = tl.arange(0, HEAD_DIM)
        query_in_bounds = query_offsets < SEQUENCE_LENGTH

        query_base = query + batch_head * SEQUENCE_LENGTH * HEAD_DIM
        key_base = key + batch_head * SEQUENCE_LENGTH * HEAD_DIM
        value_base = value + batch_head * SEQUENCE_LENGTH * HEAD_DIM
        output_base = output + batch_head * SEQUENCE_LENGTH * HEAD_DIM

        query_values = tl.load(
            query_base
            + query_offsets[:, None] * HEAD_DIM
            + dimension_offsets[None, :],
            mask=query_in_bounds[:, None],
            other=0.0,
        )
        query_is_kept = tl.load(
            attention_mask + batch * SEQUENCE_LENGTH + query_offsets,
            mask=query_in_bounds,
            other=0,
        ).to(tl.int1)

        row_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for key_start in tl.range(0, SEQUENCE_LENGTH, BLOCK_N):
            key_offsets = key_start + tl.arange(0, BLOCK_N)
            key_in_bounds = key_offsets < SEQUENCE_LENGTH
            key_is_kept = tl.load(
                attention_mask + batch * SEQUENCE_LENGTH + key_offsets,
                mask=key_in_bounds,
                other=0,
            ).to(tl.int1)

            key_values = tl.load(
                key_base
                + key_offsets[:, None] * HEAD_DIM
                + dimension_offsets[None, :],
                mask=key_in_bounds[:, None],
                other=0.0,
            )
            if IS_FP32:
                scores = tl.dot(
                    query_values,
                    tl.trans(key_values),
                    input_precision="ieee",
                )
            else:
                scores = tl.dot(query_values, tl.trans(key_values))

            pair_in_bounds = query_in_bounds[:, None] & key_in_bounds[None, :]
            delta_index = (
                query_offsets[:, None]
                - key_offsets[None, :]
                + SEQUENCE_LENGTH
                - 1
            )
            local_slot = tl.load(
                delta_to_local_slot + delta_index,
                mask=pair_in_bounds,
                other=0,
            ).to(tl.int64)

            if HAS_C2P:
                c2p_base = c2p + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS
                c2p_values = tl.load(
                    c2p_base
                    + query_offsets[:, None] * ACTIVE_SLOTS
                    + local_slot,
                    mask=pair_in_bounds,
                    other=0.0,
                )
                scores += c2p_values

            if HAS_P2C:
                p2c_base = p2c + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS
                p2c_values = tl.load(
                    p2c_base
                    + key_offsets[None, :] * ACTIVE_SLOTS
                    + local_slot,
                    mask=pair_in_bounds,
                    other=0.0,
                )
                scores += p2c_values

            scores *= SCORE_SCALE_LOG2
            attended = (
                query_is_kept[:, None]
                & key_is_kept[None, :]
                & pair_in_bounds
            )
            scores = tl.where(attended, scores, -float("inf"))

            # HF masks every score of a padded query with the same finite value;
            # its softmax is consequently uniform across the complete key row.
            padded_query_row = query_in_bounds[:, None] & ~query_is_kept[:, None]
            scores = tl.where(padded_query_row & key_in_bounds[None, :], 0.0, scores)

            # Give out-of-range rows a harmless value so the final partial tile
            # does not create NaNs. Those rows are never stored.
            scores = tl.where(
                ~query_in_bounds[:, None] & (key_offsets[None, :] == 0),
                0.0,
                scores,
            )

            new_row_max = tl.maximum(row_max, tl.max(scores, axis=1))
            correction = tl.math.exp2(row_max - new_row_max)
            probabilities = tl.math.exp2(scores - new_row_max[:, None])
            new_row_sum = row_sum * correction + tl.sum(probabilities, axis=1)

            value_values = tl.load(
                value_base
                + key_offsets[:, None] * HEAD_DIM
                + dimension_offsets[None, :],
                mask=key_in_bounds[:, None],
                other=0.0,
            )
            accumulator *= correction[:, None]
            if IS_FP32:
                accumulator += tl.dot(
                    probabilities,
                    value_values,
                    input_precision="ieee",
                )
            elif IS_BF16:
                accumulator += tl.dot(probabilities.to(tl.bfloat16), value_values)
            else:
                accumulator += tl.dot(probabilities.to(tl.float16), value_values)

            row_max = new_row_max
            row_sum = new_row_sum

        accumulator /= row_sum[:, None]
        tl.store(
            output_base
            + query_offsets[:, None] * HEAD_DIM
            + dimension_offsets[None, :],
            accumulator,
            mask=query_in_bounds[:, None],
        )


def _base_2d_mask(attention_mask: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Recover the per-token mask without creating a 4D mask."""

    if attention_mask.dim() == 2:
        mask = attention_mask
    elif attention_mask.dim() == 3:
        mask = attention_mask.bool().any(dim=-2)
    elif attention_mask.dim() == 4:
        if attention_mask.size(1) != 1:
            raise ValueError("the Triton path requires a head-broadcastable attention mask")
        mask = attention_mask[:, 0].bool().any(dim=-2)
    else:
        raise ValueError(
            "attention_mask must have shape [B, L], [B, L, L], or [B, 1, L, L]"
        )

    if mask.size(-1) != sequence_length:
        raise ValueError(
            f"attention mask length {mask.size(-1)} does not match input length "
            f"{sequence_length}"
        )
    return mask.bool().contiguous()


class TritonPreparedPositionPlan(NamedTuple):
    """Immutable positional inputs for one Triton sequence length."""

    sequence_length: int
    active_slots: torch.Tensor
    delta_to_local: torch.Tensor
    pos_key: torch.Tensor | None
    pos_query: torch.Tensor | None


class TritonInferenceDisentangledSelfAttention(InferenceDisentangledSelfAttention):
    """Forward-only CUDA Triton DeBERTa-v2/v3 self-attention.

    The first implementation specializes the fused kernel for the 64-element
    attention heads used by the official Microsoft DeBERTa-v3 checkpoints.
    C2P/P2C projection GEMMs remain PyTorch matmuls so cuBLAS can select their
    implementation; the score/softmax/PV portion is the custom Triton kernel.
    """

    def __init__(self, config: DebertaAttentionConfig | Any) -> None:
        super().__init__(config)
        self._triton_position_plan_cache: dict[
            tuple[int, str], TritonPreparedPositionPlan
        ] = {}

    def clear_inference_cache(self) -> None:
        super().clear_inference_cache()
        if hasattr(self, "_triton_position_plan_cache"):
            self._triton_position_plan_cache.clear()

    @torch.no_grad()
    def prepare_shape(
        self,
        sequence_length: int,
        device: torch.device | str | None = None,
    ) -> TritonPreparedPositionPlan:
        """Prepare the relative LUT and pruned Pq/Pk for one CUDA length."""

        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.training:
            raise RuntimeError("prepare_shape() requires module.eval()")

        resident_device = self._plan_device()
        resolved_device = torch.device(device) if device is not None else resident_device
        if resolved_device.type == resident_device.type and resolved_device.index is None:
            resolved_device = resident_device
        if resolved_device.type != "cuda":
            raise ValueError("Triton shape plans must be prepared on CUDA")
        cache_key = sequence_length, str(resolved_device)
        cached = self._triton_position_plan_cache.get(cache_key)
        if cached is not None:
            return cached

        deltas = torch.arange(
            -(sequence_length - 1),
            sequence_length,
            dtype=torch.long,
            device="cpu",
        )
        if self.position_buckets > 0:
            deltas = make_log_bucket_position(
                deltas,
                self.position_buckets,
                self.max_relative_positions,
            )
        global_slots = torch.clamp(
            deltas + self.pos_ebd_size,
            0,
            self.pos_ebd_size * 2 - 1,
        )
        active_slots = torch.unique(global_slots, sorted=True)
        delta_to_local = torch.searchsorted(active_slots, global_slots).to(torch.int32)

        active_slots = active_slots.to(device=resolved_device)

        pos_key = None
        if self.relative_attention and "c2p" in self.pos_att_type:
            if self._cached_pos_key is None:
                raise RuntimeError("call prepare_for_inference() before prepare_shape()")
            pos_key = self._cached_pos_key.index_select(1, active_slots).contiguous()

        pos_query = None
        if self.relative_attention and "p2c" in self.pos_att_type:
            if self._cached_pos_query is None:
                raise RuntimeError("call prepare_for_inference() before prepare_shape()")
            pos_query = self._cached_pos_query.index_select(1, active_slots).contiguous()

        plan = TritonPreparedPositionPlan(
            sequence_length=sequence_length,
            active_slots=active_slots,
            delta_to_local=delta_to_local.to(device=resolved_device),
            pos_key=pos_key,
            pos_query=pos_query,
        )
        self._triton_position_plan_cache[cache_key] = plan
        return plan

    def _validate_triton_call(self, hidden_states: torch.Tensor) -> None:
        if triton is None:
            raise RuntimeError("Triton is not installed; this backend requires CUDA and Triton")
        if hidden_states.device.type != "cuda":
            raise RuntimeError("TritonInferenceDisentangledSelfAttention requires CUDA")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "the Triton path supports torch.float16, torch.bfloat16, and torch.float32"
            )
        if self.attention_head_size != 64:
            raise ValueError(
                "the initial Triton kernel is specialized for DeBERTa-v3 head_dim=64"
            )
        self._validate_inference_call()

    def forward_prepared(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        plan: TritonPreparedPositionPlan,
    ) -> tuple[torch.Tensor, None]:
        """Pure forward using a plan prepared completely outside the hot path."""

        self._validate_triton_call(hidden_states)

        batch_size, sequence_length = hidden_states.shape[:2]
        if sequence_length != plan.sequence_length:
            raise ValueError(
                f"prepared length {plan.sequence_length} does not match input length "
                f"{sequence_length}"
            )
        query_layer = self.transpose_for_scores(
            self.query_proj(hidden_states), self.num_attention_heads
        ).view(
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        )
        key_layer = self.transpose_for_scores(
            self.key_proj(hidden_states), self.num_attention_heads
        ).view(
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        )
        value_layer = self.transpose_for_scores(
            self.value_proj(hidden_states), self.num_attention_heads
        ).view(
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        )

        has_c2p = self.relative_attention and "c2p" in self.pos_att_type
        has_p2c = self.relative_attention and "p2c" in self.pos_att_type
        active_slot_count = plan.active_slots.numel()

        if has_c2p:
            if plan.pos_key is None:
                raise ValueError("prepared Triton plan has no content-to-position keys")
            c2p = torch.matmul(query_layer, plan.pos_key.transpose(-1, -2)).contiguous()
        else:
            c2p = query_layer

        if has_p2c:
            if plan.pos_query is None:
                raise ValueError("prepared Triton plan has no position-to-content queries")
            p2c = torch.matmul(key_layer, plan.pos_query.transpose(-1, -2)).contiguous()
        else:
            p2c = key_layer

        base_mask = _base_2d_mask(attention_mask, sequence_length)
        output = torch.empty_like(query_layer)
        scale_factor = 1 + int(has_c2p) + int(has_p2c)
        score_scale_log2 = self._scale(scale_factor) ** -1 * 1.4426950408889634

        if sequence_length <= 32:
            block_m, block_n, num_warps, num_stages = 16, 32, 4, 2
        elif sequence_length <= 128:
            block_m, block_n, num_warps, num_stages = 32, 64, 4, 3
        else:
            block_m, block_n, num_warps, num_stages = 64, 64, 8, 3

        grid = (triton.cdiv(sequence_length, block_m), batch_size * self.num_attention_heads)
        _deberta_attention_forward_kernel[grid](
            query_layer,
            key_layer,
            value_layer,
            c2p,
            p2c,
            plan.delta_to_local,
            base_mask,
            output,
            NUM_HEADS=self.num_attention_heads,
            SEQUENCE_LENGTH=sequence_length,
            ACTIVE_SLOTS=active_slot_count,
            HEAD_DIM=self.attention_head_size,
            SCORE_SCALE_LOG2=score_scale_log2,
            HAS_C2P=has_c2p,
            HAS_P2C=has_p2c,
            IS_BF16=hidden_states.dtype == torch.bfloat16,
            IS_FP32=hidden_states.dtype == torch.float32,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        context_layer = (
            output.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, sequence_length, self.all_head_size)
        )
        return context_layer, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Convenience wrapper with lazy preparation outside the pure hot path."""

        self._validate_triton_call(hidden_states)
        if output_attentions:
            raise ValueError("output_attentions=True is not supported by the Triton path")
        if query_states is not None:
            raise ValueError("the Triton path currently supports self-attention only")
        if relative_pos is not None:
            raise ValueError("custom relative_pos tensors are not supported by the Triton path")

        has_c2p = self.relative_attention and "c2p" in self.pos_att_type
        has_p2c = self.relative_attention and "p2c" in self.pos_att_type
        needs_key = has_c2p and self._cached_pos_key is None
        needs_query = has_p2c and self._cached_pos_query is None
        if needs_key or needs_query:
            if rel_embeddings is None:
                raise ValueError(
                    "rel_embeddings is required until prepare_for_inference() has populated the cache"
                )
            self.prepare_for_inference(rel_embeddings)

        plan = self.prepare_shape(hidden_states.size(1), hidden_states.device)
        return self.forward_prepared(hidden_states, attention_mask, plan)


__all__ = [
    "TritonInferenceDisentangledSelfAttention",
    "TritonPreparedPositionPlan",
]
