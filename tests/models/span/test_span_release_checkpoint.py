"""Release-gating checks for the published span checkpoint."""
from __future__ import annotations

import gc

import pytest
import torch

from gliner2 import AutoExtractor


CHECKPOINT = "fastino/gliner2-base-v1"
TEXT = "Apple CEO Tim Cook announced iPhone 15 in Cupertino."
ENTITY_TYPES = ["company", "person", "product", "location"]

pytestmark = [pytest.mark.slow, pytest.mark.quality]


@pytest.fixture(scope="module")
def model():
    loaded = AutoExtractor.from_pretrained(CHECKPOINT, map_location="cpu")
    loaded.eval()
    return loaded


def _entity_texts(result):
    return {
        label: {
            item["text"] if isinstance(item, dict) else item
            for item in values
        }
        for label, values in result["entities"].items()
    }


def test_span_checkpoint_dispatch_and_quality_floor(model):
    assert model.config.architecture == "span"
    result = model.extract_entities(
        TEXT,
        ENTITY_TYPES,
        include_confidence=True,
        include_spans=True,
    )
    texts = _entity_texts(result)
    assert "Apple" in texts["company"]
    assert "Tim Cook" in texts["person"]
    assert "iPhone 15" in texts["product"]
    assert "Cupertino" in texts["location"]
    for entities in result["entities"].values():
        for entity in entities:
            assert TEXT[entity["start"]:entity["end"]] == entity["text"]

    classification = model.classify_text(
        TEXT,
        {"topic": {"labels": ["technology", "sports", "politics"]}},
    )
    assert classification["topic"] == "technology"


def test_span_checkpoint_batch_and_long_document_parity(model):
    texts = [TEXT, "Google CEO Sundar Pichai introduced Gemini in California."]
    singles = [
        model.extract_entities(text, ENTITY_TYPES, include_spans=True)
        for text in texts
    ]
    batched = model.batch_extract_entities(
        texts,
        ENTITY_TYPES,
        batch_size=2,
        include_spans=True,
    )
    assert batched == singles

    long_text = ("Background context without named products. " * 80) + TEXT
    result = model.extract_entities_long(
        long_text,
        ENTITY_TYPES,
        chunk_size=96,
        chunk_overlap=24,
        include_spans=True,
    )
    entities = [item for values in result["entities"].values() for item in values]
    assert entities
    for entity in entities:
        assert long_text[entity["start"]:entity["end"]] == entity["text"]


@pytest.mark.parametrize(
    "policy",
    ["allow", "nested", "flat", "disallow", "longest"],
)
def test_span_checkpoint_accepts_explicit_overlap_policies(model, policy):
    result = model.extract_entities(TEXT, ENTITY_TYPES, overlap_policy=policy)
    assert isinstance(result["entities"], dict)


def test_span_checkpoint_local_roundtrip(model, tmp_path):
    save_dir = tmp_path / "span"
    model.save_pretrained(str(save_dir))
    reloaded = AutoExtractor.from_pretrained(str(save_dir), map_location="cpu")
    reloaded.eval()
    assert reloaded.config.architecture == "span"
    assert reloaded.extract_entities(TEXT, ENTITY_TYPES) == model.extract_entities(
        TEXT, ENTITY_TYPES
    )


def test_span_checkpoint_quantized_load():
    quantized = AutoExtractor.from_pretrained(
        CHECKPOINT,
        map_location="cpu",
        quantize=True,
    )
    floating = [
        parameter
        for parameter in quantized.parameters()
        if parameter.is_floating_point()
    ]
    assert floating
    assert {parameter.dtype for parameter in floating} == {torch.float16}
    del quantized
    gc.collect()
