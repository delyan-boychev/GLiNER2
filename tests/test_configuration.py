"""Tests for architecture-aware ``ExtractorConfig`` (PR 2).

Covers validation, serialization round-trips, legacy fallback (missing
architecture and top-level ``max_width``), and config migration.
"""

from __future__ import annotations

import json

import pytest

from gliner2.configuration import (
    ExtractorConfig,
    SpanHeadSettings,
    BoundaryHeadSettings,
    normalize_architecture,
    validate_span_head,
    validate_boundary_head,
    migrate_config_dict,
    architecture_from_config,
)


# ---------------------------------------------------------------------------
# normalize_architecture
# ---------------------------------------------------------------------------

def test_normalize_architecture_defaults_and_casing():
    assert normalize_architecture(None) == "span"
    assert normalize_architecture("SPAN") == "span"
    assert normalize_architecture("  Boundary ") == "boundary"


def test_normalize_architecture_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_architecture("triangle")


# ---------------------------------------------------------------------------
# Head validation
# ---------------------------------------------------------------------------

def test_validate_span_head_defaults():
    result = validate_span_head({})
    assert result["max_width"] == SpanHeadSettings().max_width
    assert result["span_mode"] == SpanHeadSettings().span_mode


@pytest.mark.parametrize("bad", [{"max_width": 0}, {"max_width": -3}, {"dropout": 1.0}])
def test_validate_span_head_rejects_invalid(bad):
    with pytest.raises(ValueError):
        validate_span_head(bad)


def test_validate_boundary_head_defaults():
    result = validate_boundary_head({})
    defaults = BoundaryHeadSettings()
    assert result["candidate_budget"] == defaults.candidate_budget
    assert result["boundary_refinement_layers"] == defaults.boundary_refinement_layers
    assert result["boundary_ffn_multiplier"] == defaults.boundary_ffn_multiplier


@pytest.mark.parametrize(
    "bad",
    [
        {"boundary_refinement_layers": -1},
        {"boundary_ffn_multiplier": 0},
        {"boundary_ffn_multiplier": -0.5},
    ],
)
def test_validate_boundary_head_rejects_invalid_refinement(bad):
    with pytest.raises(ValueError):
        validate_boundary_head(bad)


def test_validate_boundary_head_budget_ordering():
    with pytest.raises(ValueError):
        validate_boundary_head({"candidate_budget": 200, "training_candidate_budget": 100})


def test_validate_boundary_head_loss_defaults():
    result = validate_boundary_head({})
    defaults = BoundaryHeadSettings()
    assert result["boundary_negative_weight"] == defaults.boundary_negative_weight
    assert result["boundary_marginal_loss"] == defaults.boundary_marginal_loss
    assert result["classification_loss_weight"] == defaults.classification_loss_weight


@pytest.mark.parametrize(
    "bad",
    [
        {"boundary_negative_weight": 0.0},
        {"boundary_negative_weight": -0.1},
        {"boundary_negative_weight": 1.5},
        {"boundary_marginal_loss": "softmax"},
        {"classification_loss_weight": -1.0},
    ],
)
def test_validate_boundary_head_rejects_invalid_loss_config(bad):
    with pytest.raises(ValueError):
        validate_boundary_head(bad)


def test_validate_boundary_head_accepts_valid_loss_config():
    result = validate_boundary_head({
        "boundary_negative_weight": 0.25,
        "boundary_marginal_loss": "asymmetric_focal",
        "classification_loss_weight": 2.0,
    })
    assert result["boundary_negative_weight"] == 0.25
    assert result["boundary_marginal_loss"] == "asymmetric_focal"
    assert result["classification_loss_weight"] == 2.0


# ---------------------------------------------------------------------------
# Legacy fallback
# ---------------------------------------------------------------------------

def test_missing_architecture_resolves_to_span():
    cfg = ExtractorConfig()
    assert cfg.architecture == "span"
    assert architecture_from_config(cfg) == "span"


def test_legacy_top_level_max_width_migrates_into_span_head():
    cfg = ExtractorConfig(max_width=11)
    assert cfg.span_head["max_width"] == 11
    # Legacy attribute preserved for the span implementation.
    assert cfg.max_width == 11


def test_migrate_config_dict_is_idempotent_and_fills_span():
    raw = {"model_name": "x", "max_width": 5}
    once = migrate_config_dict(raw)
    twice = migrate_config_dict(once)
    assert once["architecture"] == "span"
    assert once["span_head"]["max_width"] == 5
    assert twice == once


def test_migrate_config_dict_boundary_path():
    raw = {"architecture": "boundary", "boundary_head": {"candidate_budget": 64,
                                                          "training_candidate_budget": 64}}
    migrated = migrate_config_dict(raw)
    assert migrated["architecture"] == "boundary"
    assert migrated["boundary_head"]["candidate_budget"] == 64


# ---------------------------------------------------------------------------
# Boundary config
# ---------------------------------------------------------------------------

def test_boundary_config_ignores_max_width_with_warning():
    with pytest.warns(UserWarning):
        cfg = ExtractorConfig(architecture="boundary", max_width=8)
    assert cfg.architecture == "boundary"
    assert not hasattr(cfg, "max_width") or getattr(cfg, "max_width", None) is None


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

def test_span_config_serialization_round_trip(tmp_path):
    cfg = ExtractorConfig(model_name="tiny", max_width=7)
    cfg.save_pretrained(str(tmp_path))
    loaded = ExtractorConfig.from_pretrained(str(tmp_path))
    assert loaded.architecture == "span"
    assert loaded.span_head["max_width"] == 7
    assert loaded.max_width == 7


def test_boundary_config_serialization_round_trip(tmp_path):
    cfg = ExtractorConfig(
        architecture="boundary",
        boundary_head={
            "candidate_budget": 96,
            "training_candidate_budget": 128,
            "boundary_refinement_layers": 3,
            "boundary_ffn_multiplier": 1.5,
        },
    )
    cfg.save_pretrained(str(tmp_path))
    loaded = ExtractorConfig.from_pretrained(str(tmp_path))
    assert loaded.architecture == "boundary"
    assert loaded.boundary_head["candidate_budget"] == 96
    assert loaded.boundary_head["boundary_refinement_layers"] == 3
    assert loaded.boundary_head["boundary_ffn_multiplier"] == 1.5


def test_legacy_config_json_without_architecture_loads_as_span(tmp_path):
    """A config.json produced before architecture existed must load as span."""
    legacy = {
        "model_type": "extractor",
        "model_name": "legacy-bert",
        "max_width": 9,
        "counting_layer": "count_lstm",
        "token_pooling": "first",
    }
    (tmp_path / "config.json").write_text(json.dumps(legacy))
    loaded = ExtractorConfig.from_pretrained(str(tmp_path))
    assert loaded.architecture == "span"
    assert loaded.max_width == 9


def test_legacy_boundary_without_version_migrates_on_real_load(tmp_path):
    legacy = {
        "model_type": "extractor",
        "model_name": "legacy-boundary-encoder",
        "architecture": "boundary",
        "boundary_head": {},
    }
    (tmp_path / "config.json").write_text(json.dumps(legacy))

    loaded = ExtractorConfig.from_pretrained(str(tmp_path))

    assert loaded.config_version == ExtractorConfig.current_config_version
    assert loaded.boundary_head["enable_records"] is False
    assert loaded.boundary_head["enable_relations"] is False

    loaded.save_pretrained(str(tmp_path))
    reloaded = ExtractorConfig.from_pretrained(str(tmp_path))
    assert reloaded.to_dict() == loaded.to_dict()


def test_legacy_boundary_explicit_structured_heads_survive_migration(tmp_path):
    legacy = {
        "model_type": "extractor",
        "model_name": "legacy-boundary-encoder",
        "architecture": "boundary",
        "boundary_head": {
            "enable_records": True,
            "enable_relations": True,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(legacy))

    loaded = ExtractorConfig.from_pretrained(str(tmp_path))

    assert loaded.boundary_head["enable_records"] is True
    assert loaded.boundary_head["enable_relations"] is True
