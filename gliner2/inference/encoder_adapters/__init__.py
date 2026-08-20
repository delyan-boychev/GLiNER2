"""Safe encoder-adapter selection for sequence packing."""

from __future__ import annotations

from .base import EncoderPackingAdapter, SequencePackingUnsupportedError
from .deberta_v2 import DebertaV2PackingAdapter


def get_encoder_packing_adapter(encoder) -> EncoderPackingAdapter:
    """Return a validated adapter or raise instead of guessing semantics."""
    cached = getattr(encoder, "_gliner2_packing_adapter", None)
    compiled = hasattr(encoder, "_orig_mod")
    if cached is not None and (
        (compiled and getattr(cached, "_compiled_encoder", None) is encoder)
        or (not compiled and getattr(cached, "model", None) is encoder)
    ):
        return cached
    config = getattr(encoder, "config", None)
    if config is None and hasattr(encoder, "_orig_mod"):
        config = getattr(encoder._orig_mod, "config", None)
    model_type = str(getattr(config, "model_type", "")).replace("_", "-").lower()
    if model_type == "deberta-v2" or type(config).__name__ == "DebertaV2Config":
        adapter = DebertaV2PackingAdapter(encoder)
        # The adapter is not an nn.Module, so this cache does not affect state
        # dicts.  It preserves a separately compiled packed graph across calls.
        object.__setattr__(encoder, "_gliner2_packing_adapter", adapter)
        return adapter
    raise SequencePackingUnsupportedError(
        "packing currently supports only DeBERTa-v2/v3 encoders; "
        f"found model_type={model_type or '<unknown>'!r}"
    )


__all__ = [
    "EncoderPackingAdapter",
    "SequencePackingUnsupportedError",
    "DebertaV2PackingAdapter",
    "get_encoder_packing_adapter",
]
