"""Dependency-free policy and routing tests for encoder backends."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import BertConfig, DebertaV2Config

from gliner2.models.base import BaseExtractorModel
from gliner2.models.encoder_backend import (
    EncoderBackendCapabilities,
    EncoderBackendError,
    resolve_encoder_backend,
    resolve_load_dtype,
)
from gliner2.models.flashdeberta import build_flashdeberta_encoder


def _caps(**overrides):
    values = dict(
        cuda_available=True,
        cuda_capability=(8, 0),
        flashdeberta_version="0.0.7",
        flashdeberta_importable=True,
        python_version=(3, 10),
    )
    values.update(overrides)
    return EncoderBackendCapabilities(**values)


def _resolve(config=None, **kwargs):
    return resolve_encoder_backend(
        kwargs.pop("requested", "auto"),
        encoder_config=config or DebertaV2Config(),
        map_location=kwargs.pop("map_location", "cuda"),
        effective_dtype=kwargs.pop("effective_dtype", torch.float16),
        capabilities=kwargs.pop("capabilities", _caps()),
        environ=kwargs.pop("environ", {}),
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_auto_selects_flashdeberta_for_validated_configuration(dtype):
    result = _resolve(effective_dtype=dtype)
    assert result.backend == "flashdeberta"
    assert "requirements satisfied" in result.reason


@pytest.mark.parametrize(
    ("config", "map_location", "dtype", "caps", "reason"),
    [
        (BertConfig(), "cuda", torch.float16, _caps(), "not DeBERTa"),
        (DebertaV2Config(), "cpu", torch.float16, _caps(), "not CUDA"),
        (DebertaV2Config(), "mps", torch.float16, _caps(), "not CUDA"),
        (DebertaV2Config(), "cuda", None, _caps(), "not FP16 or BF16"),
        (
            DebertaV2Config(), "cuda", torch.float16,
            _caps(cuda_capability=(7, 5)), "below the validated minimum",
        ),
        (
            DebertaV2Config(), "cuda", torch.float16,
            _caps(cuda_available=False), "CUDA is not available",
        ),
        (
            DebertaV2Config(), "cuda", torch.float16,
            _caps(python_version=(3, 9)), "requires Python 3.10+",
        ),
        (
            DebertaV2Config(), "cuda", torch.float16,
            _caps(flashdeberta_version=None, flashdeberta_importable=False),
            "not installed",
        ),
        (
            DebertaV2Config(), "cuda", torch.float16,
            _caps(flashdeberta_version="0.0.8"), "not validated",
        ),
    ],
)
def test_auto_falls_back_with_recorded_reason(
    config, map_location, dtype, caps, reason
):
    result = _resolve(
        config, map_location=map_location, effective_dtype=dtype, capabilities=caps
    )
    assert result.backend == "transformers"
    assert reason in result.reason


def test_explicit_flashdeberta_never_silently_falls_back():
    with pytest.raises(EncoderBackendError, match="destination device.*not CUDA"):
        _resolve(requested="flashdeberta", map_location="cpu")


def test_explicit_transformers_ignores_flash_requirements():
    result = _resolve(
        BertConfig(), requested="transformers", map_location="cpu",
        effective_dtype=None, capabilities=_caps(cuda_available=False),
    )
    assert result.backend == "transformers"
    assert "explicitly requested" in result.reason


def test_deprecated_environment_override_is_explicit_and_checked():
    with pytest.warns(DeprecationWarning, match="USE_FLASHDEBERTA"):
        with pytest.raises(EncoderBackendError, match="not CUDA"):
            _resolve(map_location="cpu", environ={"USE_FLASHDEBERTA": "1"})


def test_false_environment_value_does_not_override_auto():
    result = _resolve(map_location="cpu", environ={"USE_FLASHDEBERTA": "0"})
    assert result.backend == "transformers"
    assert not result.compatibility_override


def test_explicit_transformers_wins_over_deprecated_environment_override():
    with pytest.warns(DeprecationWarning):
        result = _resolve(
            requested="transformers", environ={"USE_FLASHDEBERTA": "1"}
        )
    assert result.backend == "transformers"
    assert "override ignored" in result.reason


def test_dtype_normalization_preserves_quantize_as_fp16():
    assert resolve_load_dtype(None, quantize=True) is torch.float16
    assert resolve_load_dtype("fp16") is torch.float16
    assert resolve_load_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(ValueError, match="conflicts"):
        resolve_load_dtype("bf16", quantize=True)
    with pytest.raises(ValueError, match="dtype must"):
        resolve_load_dtype(torch.float32)


def test_compile_routing_skips_only_flash_encoder(monkeypatch):
    calls = []

    def fake_compile(module, *, dynamic):
        calls.append((module, dynamic))
        return ("compiled", module)

    monkeypatch.setattr(torch, "compile", fake_compile)
    ordinary = SimpleNamespace(encoder_backend="transformers", encoder="ordinary")
    assert BaseExtractorModel._compile_encoder(ordinary, dynamic=True)
    assert ordinary.encoder == ("compiled", "ordinary")

    flash = SimpleNamespace(encoder_backend="flashdeberta", encoder="flash")
    assert not BaseExtractorModel._compile_encoder(flash, dynamic=True)
    assert flash.encoder == "flash"
    assert calls == [("ordinary", True)]


def test_span_compile_still_compiles_task_heads_for_flash(monkeypatch):
    from gliner2.models.span.model import SpanExtractorModel

    calls = []

    def fake_compile(module, *, dynamic):
        calls.append((module, dynamic))
        return f"compiled:{module}"

    monkeypatch.setattr(torch, "compile", fake_compile)
    fake = SimpleNamespace(
        encoder_backend="flashdeberta",
        encoder="flash-encoder",
        _compute_span_rep_core="span-head",
        count_embed="count-head",
    )
    fake._compile_encoder = types.MethodType(BaseExtractorModel._compile_encoder, fake)

    assert SpanExtractorModel.compile(fake) is fake
    assert fake.encoder == "flash-encoder"
    assert fake._compute_span_rep_core == "compiled:span-head"
    assert fake.count_embed == "compiled:count-head"
    assert calls == [("span-head", True), ("count-head", True)]


class _FakeFlashModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.z_steps = 0

    def forward(self, *args, **kwargs):  # replaced by the 0.0.7 adapter
        raise AssertionError("unadapted fake forward called")


def _fake_flash_module():
    return types.SimpleNamespace(FlashDebertaV2Model=_FakeFlashModel)


def test_fake_flash_backend_instantiation_is_dependency_free(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashdeberta", _fake_flash_module())
    monkeypatch.setattr(
        "gliner2.models.base.importlib.metadata.version", lambda name: "0.0.7"
    )
    encoder = BaseExtractorModel._load_encoder(
        "unused", DebertaV2Config(), encoder_backend="flashdeberta"
    )
    assert isinstance(encoder, _FakeFlashModel)


def test_flash_adapter_rejects_training_gradients_and_attentions(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashdeberta", _fake_flash_module())
    encoder = build_flashdeberta_encoder(DebertaV2Config(), "0.0.7")

    with pytest.raises(RuntimeError, match="inference-only"):
        encoder(input_ids=torch.ones(1, 2, dtype=torch.long))

    encoder.eval()
    with pytest.raises(RuntimeError, match="gradient-enabled"):
        encoder(input_ids=torch.ones(1, 2, dtype=torch.long))

    with torch.no_grad(), pytest.raises(NotImplementedError, match="output_attentions"):
        encoder(
            input_ids=torch.ones(1, 2, dtype=torch.long),
            output_attentions=True,
        )
