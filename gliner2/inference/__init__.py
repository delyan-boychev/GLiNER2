"""Inference utilities and opt-in runtime configuration.

Imports stay lazy so the schema/API-only top-level package remains usable when
the optional local-model dependencies are not installed.
"""

_PACKING_EXPORTS = {
    "PackingConfig",
    "PackedSegment",
    "PackedEncoderBatch",
    "PackingStats",
    "PackingOverflowError",
}


def __getattr__(name):
    if name not in _PACKING_EXPORTS:
        raise AttributeError(f"module 'gliner2.inference' has no attribute {name!r}")
    from . import packing
    value = getattr(packing, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(list(globals()) + list(_PACKING_EXPORTS)))


__all__ = sorted(_PACKING_EXPORTS)
