"""Decode-level parity checks for metadata-driven boundary features."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gliner2 import AttributeGroup, RegexValidator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def _core(model, query_count: int, text_length: int):
    hidden = model.hidden_size
    return {
        "text_states": torch.zeros(1, text_length, hidden),
        "text_mask": torch.ones(1, text_length, dtype=torch.bool),
        "query_states": torch.zeros(1, query_count, hidden),
        "query_mask": torch.ones(1, query_count, dtype=torch.bool),
    }


def test_boundary_entities_honor_scalar_metadata_and_attribute_groups(monkeypatch):
    model = build_tiny_boundary_model()
    specs = [
        {"task_type": "entities", "field_name": "person"},
        {"task_type": "entities", "field_name": "sentiment: positive"},
        {"task_type": "entities", "field_name": "sentiment: negative"},
    ]
    core = _core(model, len(specs), 2)

    def explicit(*args, **kwargs):
        # Attribute queries are selected in schema order; one retained span.
        return torch.tensor([[[3.0], [-1.0]]])

    monkeypatch.setattr(model.boundary_head, "score_explicit_spans", explicit)
    group = AttributeGroup(
        ["positive", "negative"],
        applies_to=["person"],
        qualify_labels=True,
    )
    metadata = {
        "entity_order": ["person"],
        "entity_metadata": {
            "person": {
                "dtype": "str",
                "threshold": 0.8,
                "validators": [RegexValidator(r"^Alice$")],
            }
        },
        "entity_attribute_groups": {"sentiment": group},
        "entity_attribute_prompt_labels": {
            "positive": "sentiment: positive",
            "negative": "sentiment: negative",
        },
        "entity_attribute_labels": {
            "sentiment: positive", "sentiment: negative"
        },
    }
    result = model._decode_entities(
        0,
        core,
        specs,
        [[(0.9, 0, 1)], [], []],
        metadata,
        None,
        "nested",
        0,
        [0, 6],
        [5, 10],
        "Alice Acme",
        2,
        True,
        True,
    )

    assert list(result) == ["person"]
    assert result["person"]["text"] == "Alice"
    assert result["person"]["sentiment"]["label"] == "positive"
    assert result["person"]["start"] == 0
    assert result["person"]["end"] == 5

    thresholds = model._query_thresholds(
        [[specs[0]]], [metadata], 0.5, torch.device("cpu")
    )
    assert thresholds.item() == pytest.approx(0.8)

    schema = model.create_schema().entities(
        "person", validators=[RegexValidator(r"^Alice$")]
    )
    assert len(schema._entity_metadata["person"]["validators"]) == 1


def test_boundary_legacy_structure_honors_choices_validators_and_dtype(
    monkeypatch,
):
    model = build_tiny_boundary_model()
    specs = [
        {
            "task_type": "json_structures",
            "task_name": "product",
            "field_name": "tier",
        },
        {
            "task_type": "json_structures",
            "task_name": "product",
            "field_name": "name",
        },
    ]
    core = _core(model, len(specs), 3)
    batch = SimpleNamespace(
        text_tokens=[["basic", "premium", "Product"]]
    )

    def explicit(*args, **kwargs):
        # "premium" wins the scalar enum field.
        return torch.tensor([[[-2.0, 2.0]]])

    monkeypatch.setattr(model.boundary_head, "score_explicit_spans", explicit)
    metadata = {
        "field_orders": {"product": ["tier", "name"]},
        "field_metadata": {
            "product.tier": {
                "dtype": "str",
                "threshold": 0.5,
                "choices": ["basic", "premium"],
                "validators": [],
            },
            "product.name": {
                "dtype": "str",
                "threshold": 0.5,
                "choices": None,
                "validators": [RegexValidator(r"^Product$")],
            },
        },
    }
    result = model._decode_legacy_structures(
        batch,
        0,
        core,
        specs,
        [[], [(0.9, 2, 3)]],
        metadata,
        set(),
        "nested",
        2,
        [0],
        [7],
        "Product",
        1,
        0.5,
        True,
        True,
    )

    instance = result["product"][0]
    assert instance["tier"]["text"] == "premium"
    assert instance["name"] == {
        "text": "Product",
        "confidence": 0.9,
        "start": 0,
        "end": 7,
    }
