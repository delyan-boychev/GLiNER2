"""Version-gated DeBERTa-v2/v3 sequence-packing adapter.

Hugging Face's DeBERTa-v2 encoder supports a 3D pair mask, but the enclosing
``DebertaV2Model`` also passes ``attention_mask`` to its embedding layer.  That
embedding path expects a token mask and cannot consume the pair mask.  This
adapter mirrors the relevant official model-forward steps while routing the
two masks to the components that own their semantics.

DeBERTa-v3 checkpoints use ``DebertaV2Model`` in Transformers, hence the name.
"""

from __future__ import annotations

import re
from typing import Tuple

import torch
import transformers

from .base import EncoderPackingAdapter, SequencePackingUnsupportedError


_SUPPORTED_TRANSFORMERS_RANGES = (
    ((4, 40), (5, 0)),
    # Transformers 5.15 retains the same DebertaV2Model/Encoder mask routing;
    # this range is separately pinned so unreviewed 5.x changes still fail shut.
    ((5, 15), (5, 16)),
)


def _major_minor(version: str) -> Tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise SequencePackingUnsupportedError(
            f"cannot parse Transformers version {version!r}"
        )
    return int(match.group(1)), int(match.group(2))


def _unwrap_compiled(module):
    """Return the original module behind ``torch.compile``, when present."""
    return getattr(module, "_orig_mod", module)


class DebertaV2PackingAdapter(EncoderPackingAdapter):
    """Pack-aware adapter for the official Transformers DeBERTa-v2 model."""

    def __init__(self, encoder) -> None:
        version = _major_minor(transformers.__version__)
        if not any(lower <= version < upper
                   for lower, upper in _SUPPORTED_TRANSFORMERS_RANGES):
            supported = ", ".join(
                f">={lower[0]}.{lower[1]},<{upper[0]}.{upper[1]}"
                for lower, upper in _SUPPORTED_TRANSFORMERS_RANGES
            )
            raise SequencePackingUnsupportedError(
                "DeBERTa packing is version-gated to Transformers "
                f"{supported}; found {transformers.__version__}"
            )

        model = _unwrap_compiled(encoder)
        try:
            from transformers.models.deberta_v2.modeling_deberta_v2 import (
                DebertaV2Model,
            )
        except ImportError as error:  # pragma: no cover - version gate normally catches this
            raise SequencePackingUnsupportedError(
                "Transformers does not provide DebertaV2Model"
            ) from error

        if not isinstance(model, DebertaV2Model):
            raise SequencePackingUnsupportedError(
                "packing currently supports only the official Transformers "
                f"DebertaV2Model, found {type(model).__module__}.{type(model).__name__}"
            )
        if getattr(model, "z_steps", 0) > 1:
            raise SequencePackingUnsupportedError(
                "DeBERTa z_steps > 1 has not been parity-validated for packing"
            )
        if getattr(model.encoder, "conv", None) is not None:
            raise SequencePackingUnsupportedError(
                "DeBERTa convolution is not segment-aware"
            )
        self.model = model
        self._compiled_encoder = encoder if model is not encoder else None
        self._compiled_call = None

    def encode(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pair_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("packed input_ids must have shape [streams, sequence]")
        if token_attention_mask.shape != input_ids.shape:
            raise ValueError("token_attention_mask must match input_ids")
        if pair_attention_mask.shape != (
            input_ids.shape[0], input_ids.shape[1], input_ids.shape[1]
        ):
            raise ValueError("pair_attention_mask must have shape [streams, sequence, sequence]")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match input_ids")

        if self._compiled_encoder is not None:
            if self._compiled_call is None:
                compiler_config = self._compiled_encoder.get_compiler_config()
                mode = (
                    "max-autotune-no-cudagraphs"
                    if compiler_config.get("max_autotune", False)
                    else "default"
                )
                dynamic = bool(
                    getattr(self._compiled_encoder.dynamo_ctx, "_dynamic", True)
                )
                self._compiled_call = torch.compile(
                    self._encode_eager, dynamic=dynamic, mode=mode
                )
            return self._compiled_call(
                input_ids,
                token_attention_mask,
                pair_attention_mask,
                position_ids,
            )
        return self._encode_eager(
            input_ids,
            token_attention_mask,
            pair_attention_mask,
            position_ids,
        )

    def _encode_eager(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pair_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Mirrors DebertaV2Model.forward in the supported Transformers range.
        # The only intentional split is mask routing: embeddings receive the
        # active-token mask while the encoder receives the block-diagonal mask.
        token_type_ids = torch.zeros_like(input_ids)
        embedding_output = self.model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=token_attention_mask,
            inputs_embeds=None,
        )
        encoder_outputs = self.model.encoder(
            embedding_output,
            pair_attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        return encoder_outputs.hidden_states[-1]
