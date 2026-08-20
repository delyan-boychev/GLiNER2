"""Deterministic encoder-backend selection for local inference.

FlashDeBERTa is deliberately selected before model construction.  Moving a
model to CUDA later does not change its encoder implementation.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from transformers import PretrainedConfig


SUPPORTED_ENCODER_BACKENDS = ("auto", "transformers", "flashdeberta")
SUPPORTED_FLASHDEBERTA_VERSIONS = frozenset({"0.0.7"})
MIN_FLASHDEBERTA_CAPABILITY = (8, 0)
FLASH_DTYPES = frozenset({torch.float16, torch.bfloat16})

_DTYPE_ALIASES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
_FALSE_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})


class EncoderBackendError(RuntimeError):
    """Raised when an explicitly requested encoder backend is unsupported."""


@dataclass(frozen=True)
class EncoderBackendCapabilities:
    """Runtime facts consumed by the pure backend resolver.

    Tests can supply this object directly, so backend policy is testable on a
    machine without CUDA, Triton, or FlashDeBERTa.
    """

    cuda_available: bool
    cuda_capability: Optional[Tuple[int, int]]
    flashdeberta_version: Optional[str]
    flashdeberta_importable: bool
    python_version: Tuple[int, int]


@dataclass(frozen=True)
class EncoderBackendResolution:
    """Resolved backend plus a stable, user-readable explanation."""

    requested: str
    backend: str
    reason: str
    effective_dtype: Optional[torch.dtype]
    compatibility_override: bool = False


def resolve_load_dtype(
    dtype: Optional[Union[str, torch.dtype]],
    *,
    quantize: bool = False,
) -> Optional[torch.dtype]:
    """Normalize the supported inference dtypes.

    ``quantize=True`` remains the historical spelling for FP16.  Supplying a
    conflicting explicit dtype is rejected instead of applying two casts.
    """

    if isinstance(dtype, str):
        normalized = _DTYPE_ALIASES.get(dtype.strip().lower())
    else:
        normalized = dtype

    if dtype is not None and normalized not in FLASH_DTYPES:
        choices = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(
            f"dtype must be torch.float16, torch.bfloat16, or one of: {choices}"
        )
    if quantize and normalized not in (None, torch.float16):
        raise ValueError("quantize=True is the FP16 path and conflicts with dtype=bf16")
    return torch.float16 if quantize else normalized


def _destination_device(map_location) -> torch.device:
    if map_location is None:
        return torch.device("cpu")
    try:
        return torch.device(map_location)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            "map_location must identify one destination device when selecting "
            "an encoder backend"
        ) from exc


def detect_encoder_backend_capabilities(map_location) -> EncoderBackendCapabilities:
    """Probe optional-package and CUDA facts without importing FlashDeBERTa."""

    destination = _destination_device(map_location)
    cuda_available = torch.cuda.is_available()
    capability = None
    if destination.type == "cuda" and cuda_available:
        capability = torch.cuda.get_device_capability(destination)

    try:
        version = importlib.metadata.version("flashdeberta")
    except importlib.metadata.PackageNotFoundError:
        version = None
    try:
        importable = importlib.util.find_spec("flashdeberta") is not None
    except (ImportError, AttributeError, ValueError):
        importable = False

    return EncoderBackendCapabilities(
        cuda_available=cuda_available,
        cuda_capability=capability,
        flashdeberta_version=version,
        flashdeberta_importable=importable,
        python_version=(sys.version_info.major, sys.version_info.minor),
    )


def _is_deberta_v2(config: PretrainedConfig) -> bool:
    # DeBERTa-v3 checkpoints intentionally use the DeBERTa-v2 Transformers
    # architecture/configuration.
    return (
        getattr(config, "model_type", None) == "deberta-v2"
        or config.__class__.__name__ == "DebertaV2Config"
    )


def _unsupported_reason(
    config: PretrainedConfig,
    destination: torch.device,
    dtype: Optional[torch.dtype],
    capabilities: EncoderBackendCapabilities,
) -> Optional[str]:
    if not _is_deberta_v2(config):
        model_type = getattr(config, "model_type", config.__class__.__name__)
        return f"encoder type {model_type!r} is not DeBERTa-v2/v3"
    if destination.type != "cuda":
        return f"destination device is {destination.type!r}, not CUDA"
    if dtype not in FLASH_DTYPES:
        return "effective dtype is not FP16 or BF16"
    if capabilities.python_version < (3, 10):
        return "flashdeberta 0.0.7 requires Python 3.10+"
    if not capabilities.cuda_available:
        return "CUDA is not available"
    if capabilities.cuda_capability is None:
        return "CUDA compute capability could not be determined"
    if capabilities.cuda_capability < MIN_FLASHDEBERTA_CAPABILITY:
        major, minor = capabilities.cuda_capability
        return (
            f"CUDA compute capability {major}.{minor} is below the validated "
            "minimum of 8.0"
        )
    if capabilities.flashdeberta_version is None or not capabilities.flashdeberta_importable:
        return "flashdeberta is not installed"
    if capabilities.flashdeberta_version not in SUPPORTED_FLASHDEBERTA_VERSIONS:
        versions = ", ".join(sorted(SUPPORTED_FLASHDEBERTA_VERSIONS))
        return (
            f"flashdeberta {capabilities.flashdeberta_version} is not validated; "
            f"supported version: {versions}"
        )
    return None


def resolve_encoder_backend(
    requested: str,
    *,
    encoder_config: PretrainedConfig,
    map_location=None,
    effective_dtype: Optional[torch.dtype] = None,
    capabilities: Optional[EncoderBackendCapabilities] = None,
    environ=None,
) -> EncoderBackendResolution:
    """Resolve ``auto``/``transformers``/``flashdeberta`` deterministically."""

    if not isinstance(requested, str):
        raise ValueError(
            f"encoder_backend must be one of {SUPPORTED_ENCODER_BACKENDS}, "
            f"got {type(requested).__name__}"
        )
    requested = requested.strip().lower()
    if requested not in SUPPORTED_ENCODER_BACKENDS:
        raise ValueError(
            f"encoder_backend must be one of {SUPPORTED_ENCODER_BACKENDS}, "
            f"got {requested!r}"
        )

    env = os.environ if environ is None else environ
    env_value = str(env.get("USE_FLASHDEBERTA", "")).strip().lower()
    compatibility_override = env_value not in _FALSE_ENV_VALUES
    if compatibility_override:
        warnings.warn(
            "USE_FLASHDEBERTA is deprecated; pass "
            "encoder_backend='flashdeberta' to from_pretrained() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if requested == "auto":
            requested = "flashdeberta"

    if requested == "transformers":
        reason = "Transformers backend explicitly requested"
        if compatibility_override:
            reason += "; deprecated USE_FLASHDEBERTA override ignored"
        return EncoderBackendResolution(
            requested=requested,
            backend="transformers",
            reason=reason,
            effective_dtype=effective_dtype,
            compatibility_override=compatibility_override,
        )

    destination = _destination_device(map_location)
    caps = capabilities or detect_encoder_backend_capabilities(destination)
    unsupported = _unsupported_reason(
        encoder_config, destination, effective_dtype, caps
    )
    if unsupported is not None:
        if requested == "flashdeberta":
            raise EncoderBackendError(
                f"FlashDeBERTa was explicitly requested but is unsupported: {unsupported}. "
                "Use encoder_backend='transformers' or correct the load-time "
                "device/dtype/dependency configuration."
            )
        return EncoderBackendResolution(
            requested=requested,
            backend="transformers",
            reason=f"automatic FlashDeBERTa selection skipped: {unsupported}",
            effective_dtype=effective_dtype,
            compatibility_override=compatibility_override,
        )

    major, minor = caps.cuda_capability or (0, 0)
    return EncoderBackendResolution(
        requested=requested,
        backend="flashdeberta",
        reason=(
            "automatic requirements satisfied: DeBERTa-v2/v3, "
            f"CUDA capability {major}.{minor}, {effective_dtype}, and "
            f"flashdeberta {caps.flashdeberta_version}"
        ),
        effective_dtype=effective_dtype,
        compatibility_override=compatibility_override,
    )


__all__ = [
    "EncoderBackendCapabilities",
    "EncoderBackendError",
    "EncoderBackendResolution",
    "MIN_FLASHDEBERTA_CAPABILITY",
    "SUPPORTED_ENCODER_BACKENDS",
    "SUPPORTED_FLASHDEBERTA_VERSIONS",
    "detect_encoder_backend_capabilities",
    "resolve_encoder_backend",
    "resolve_load_dtype",
]
