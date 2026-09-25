"""Boundary-specific processor/collator integration tests."""

from __future__ import annotations

import pytest

from gliner2.processor import SamplingConfig, SchemaTransformer
from gliner2.training import ExtractorCollator


def _processor(tokenizer, *, shuffle_entities=False):
    return SchemaTransformer(
        tokenizer=tokenizer,
        sampling_config=SamplingConfig(
            shuffle_entities=shuffle_entities,
            synthetic_entity_label_prob=0.0,
        ),
    )


def test_boundary_training_collation_aligns_reordered_entity_targets(
    tiny_tokenizer, monkeypatch
):
    processor = _processor(tiny_tokenizer, shuffle_entities=True)
    monkeypatch.setattr("random.shuffle", lambda values: values.reverse())
    collator = ExtractorCollator(
        processor, is_training=True, architecture="boundary", max_gold_per_query=4
    )

    batch = collator([
        (
            "John works at Apple.",
            {"entities": {"person": ["John"], "company": ["Apple"]}},
        )
    ])

    layout = batch.query_layouts[0]
    assert [query.role_name for query in layout.queries] == ["company", "person"]
    assert batch.targets is not None
    pairs = {
        layout.query(query_id).role_name: (int(start), int(end))
        for query_id, (start, end) in enumerate(batch.targets.mention_pairs[0, :, 0])
    }
    assert pairs == {"company": (3, 4), "person": (0, 1)}


def test_boundary_inference_collation_has_layout_without_targets(tiny_tokenizer):
    processor = _processor(tiny_tokenizer)
    batch = ExtractorCollator(
        processor, is_training=False, architecture="boundary"
    )([("Apple released iPhone.", {"entities": {"company": "", "product": ""}})])

    assert [q.role_name for q in batch.query_layouts[0].queries] == ["company", "product"]
    assert batch.targets is None


def test_pinned_relation_head_collapses_the_gold_cross_product(tiny_tokenizer):
    processor = SchemaTransformer(
        tokenizer=tiny_tokenizer,
        sampling_config=SamplingConfig(
            remove_relations_prob=0.0, swap_head_tail_prob=0.0
        ),
    )
    collator = ExtractorCollator(
        processor, is_training=True, architecture="boundary", max_gold_per_query=4
    )
    text = "Pushkin street runs past the Pushkin monument."

    def gold_pairs(head):
        batch = collator([
            (text, {"relations": [{"near": {"head": head, "tail": "monument"}}]})
        ])
        pairs, mask = batch.targets.edge_targets[:2]
        return pairs[mask].tolist()

    # a repeated head surface supervises both cross-products
    assert gold_pairs("Pushkin") == [[0, 1, 6, 7], [5, 6, 6, 7]]
    assert gold_pairs({"text": "Pushkin", "start": 29, "end": 36}) == [[5, 6, 6, 7]]


def test_boundary_training_rejects_missing_entity_annotation(tiny_tokenizer):
    processor = _processor(tiny_tokenizer)
    collator = ExtractorCollator(processor, is_training=True, architecture="boundary")

    with pytest.raises(ValueError, match="was not found"):
        collator([("Apple released iPhone.", {"entities": {"company": ["Google"]}})])
