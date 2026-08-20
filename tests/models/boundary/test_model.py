"""BoundaryExtractorModel integration: construction, dispatch, encode, save/reload."""

from __future__ import annotations

import torch

from gliner2 import ExtractorConfig
from gliner2.auto import AutoExtractor
from gliner2.models.outputs import CandidateTensorBatch
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def test_boundary_model_builds_with_expected_modules():
    model = build_tiny_boundary_model()
    assert model.architecture == "boundary"
    assert set(model.task_module_names()) == {
        "classifier", "boundary_head", "record_decoder", "relation_scorer"
    }
    # Boundary head submodules exist; no span-specific span_rep/count modules.
    assert hasattr(model.boundary_head, "boundary_proposer")
    assert hasattr(model.boundary_head, "pair_scorer")
    assert not hasattr(model, "span_rep")


def test_auto_extractor_dispatches_boundary_from_config():
    from gliner2.configuration import architecture_from_config
    from gliner2.inference.engine import BoundaryExtractor

    config = ExtractorConfig(architecture="boundary", model_name="tiny-bert-fixture")
    cls = AutoExtractor._resolve_class(architecture_from_config(config))
    assert cls is BoundaryExtractor


def test_boundary_head_from_config_produces_valid_candidates():
    # The head dims are wired from config; exercise it on synthetic states
    # (the tiny offline tokenizer lacks schema markers for a full encode()).
    model = build_tiny_boundary_model()
    torch.manual_seed(0)
    B, L, Q, H = 2, 12, 2, model.hidden_size
    token_states = torch.randn(B, L, H)
    text_mask = torch.ones(B, L, dtype=torch.bool)
    query_states = torch.randn(B, Q, H)
    query_mask = torch.ones(B, Q, dtype=torch.bool)

    out = model.boundary_head(token_states, text_mask, query_states, query_mask)
    assert isinstance(out.candidates, CandidateTensorBatch)
    out.candidates.validate(text_mask.sum(dim=1).long())


def test_boundary_save_reload_preserves_architecture_and_weights(tmp_path):
    model = build_tiny_boundary_model()
    save_dir = tmp_path / "boundary_ckpt"
    model.save_pretrained(str(save_dir))

    # Architecture stamped in config for AutoExtractor dispatch.
    reloaded_config = ExtractorConfig.from_pretrained(str(save_dir))
    assert reloaded_config.architecture == "boundary"

    reloaded = AutoExtractor.from_pretrained(str(save_dir))
    assert reloaded.architecture == "boundary"

    a = dict(model.state_dict())
    b = dict(reloaded.state_dict())
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), f"weight mismatch for {k}"


def test_boundary_public_entity_inference_uses_standard_span_keys():
    model = build_tiny_boundary_model()
    result = model.extract_entities(
        "Apple released iPhone.",
        ["company", "product"],
        threshold=0.0,
        include_spans=True,
    )

    assert set(result["entities"]) == {"company", "product"}
    for items in result["entities"].values():
        assert items
        for item in items:
            assert {"text", "start", "end"} <= set(item)
            assert "char_start" not in item and "char_end" not in item
            assert item["text"] == "Apple released iPhone."[item["start"] : item["end"]]
