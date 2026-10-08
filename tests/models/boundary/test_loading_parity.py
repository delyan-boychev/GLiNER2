"""Offline lifecycle parity tests for boundary extractors."""

from __future__ import annotations

import logging
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from gliner2.auto import AutoExtractor
from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import (
    build_tiny_boundary_model,
    save_tiny_boundary_checkpoint,
)


ENTITY_SAMPLE = (
    "apple released iphone .",
    {"entities": {"company": ["apple"], "product": ["iphone"]}},
)


def test_tiny_boundary_quantized_load_is_fp16(tmp_path):
    save_tiny_boundary_checkpoint(tmp_path)

    model = AutoExtractor.from_pretrained(
        str(tmp_path), map_location="cpu", quantize=True
    )

    assert model.architecture == "boundary"
    assert {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    } == {torch.float16}


def test_boundary_quantize_method_is_explicit():
    model = build_tiny_boundary_model()

    assert model.quantize(method="bf16") is model
    assert {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    } == {torch.bfloat16}
    with pytest.raises(ValueError, match="quantize method"):
        model.quantize(method="int4")


def test_tiny_boundary_load_quantizes_before_compile(tmp_path, monkeypatch):
    save_tiny_boundary_checkpoint(tmp_path)
    from gliner2.inference.engine import BoundaryExtractor

    events = []
    original_quantize = BoundaryExtractor.quantize

    def tracked_quantize(self, method="fp16"):
        events.append("quantize")
        return original_quantize(self, method=method)

    def fake_compile(self, dynamic=True):
        assert next(self.parameters()).dtype == torch.float16
        events.append(("compile", dynamic))
        return self

    monkeypatch.setattr(BoundaryExtractor, "quantize", tracked_quantize)
    monkeypatch.setattr(BoundaryExtractor, "compile", fake_compile)

    model = AutoExtractor.from_pretrained(
        str(tmp_path), quantize=True, compile=True
    )

    assert model.architecture == "boundary"
    assert events == ["quantize", ("compile", True)]


def test_tiny_boundary_compile_wraps_tensor_subgraphs(monkeypatch):
    model = build_tiny_boundary_model()
    calls = []

    def fake_torch_compile(module, *, dynamic):
        calls.append((module, dynamic))
        return module

    monkeypatch.setattr(torch, "compile", fake_torch_compile)

    assert model.compile(dynamic=False) is model
    assert len(calls) == 5
    assert all(dynamic is False for _, dynamic in calls)


def test_boundary_load_extends_short_checkpoint_embeddings(tmp_path):
    model = build_tiny_boundary_model()
    model.save_pretrained(str(tmp_path))
    weights_path = tmp_path / "model.safetensors"
    state = load_file(str(weights_path))
    key = "encoder.embeddings.word_embeddings.weight"
    original = state[key].clone()
    state[key] = original[:-2].clone()
    save_file(state, str(weights_path))

    reloaded = AutoExtractor.from_pretrained(str(tmp_path))
    loaded = reloaded.state_dict()[key]

    assert loaded.shape == original.shape
    assert torch.equal(loaded[:-2], original[:-2])


def test_boundary_direct_loader_rejects_unknown_option(tmp_path):
    save_tiny_boundary_checkpoint(tmp_path)
    from gliner2.inference.engine import BoundaryExtractor

    with pytest.raises(TypeError, match="does not accept.*threshold"):
        BoundaryExtractor.from_pretrained(str(tmp_path), threshold=0.5)


def test_boundary_push_to_hub_uploads_complete_checkpoint(monkeypatch):
    import huggingface_hub

    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create", kwargs))

        def upload_folder(self, **kwargs):
            folder = kwargs["folder_path"]
            assert os.path.isfile(os.path.join(folder, "config.json"))
            assert os.path.isfile(os.path.join(folder, "model.safetensors"))
            calls.append(("upload", {"repo_id": kwargs["repo_id"]}))
            return "uploaded"

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    result = build_tiny_boundary_model().push_to_hub("org/boundary", private=False)

    assert result == "uploaded"
    assert calls == [
        (
            "create",
            {"repo_id": "org/boundary", "private": False, "exist_ok": True},
        ),
        ("upload", {"repo_id": "org/boundary"}),
    ]


def test_boundary_apply_lora_wraps_task_head():
    pytest.importorskip("peft")
    model = build_tiny_boundary_model()

    wrapped = model.apply_lora(
        r=2, alpha=4, dropout=0.0, targets=["classification_head"]
    )

    assert wrapped.peft_config
    assert any("lora_" in name for name, _ in wrapped.named_parameters())


def _training_batch(model):
    collator = ExtractorCollator(
        model.processor,
        is_training=True,
        architecture="boundary",
        max_gold_per_query=model.boundary_head.settings.max_gold_per_query,
        build_targets=True,
    )
    batch = collator([ENTITY_SAMPLE])
    # Exercise the record auxiliary branch even though this entity-only sample
    # has no record instances.
    batch.targets.records = [[]]
    return batch


def test_dropped_auxiliary_losses_are_logged(monkeypatch, caplog):
    model = build_tiny_boundary_model()
    model.train()
    batch = _training_batch(model)
    monkeypatch.setattr(
        model, "_record_loss", lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("bad record target")
        )
    )
    monkeypatch.setattr(
        model, "_relation_loss", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("bad relation target")
        )
    )

    with caplog.at_level(logging.WARNING):
        output = model(batch)

    assert torch.isfinite(output.total_loss)
    assert "Dropped record auxiliary loss" in caplog.text
    assert "Dropped relation auxiliary loss" in caplog.text


def test_fatal_auxiliary_device_error_propagates(monkeypatch):
    model = build_tiny_boundary_model()
    model.train()
    batch = _training_batch(model)
    monkeypatch.setattr(
        model, "_record_loss", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("CUDA error: device-side assert triggered")
        )
    )

    with pytest.raises(RuntimeError, match="device-side assert"):
        model(batch)
