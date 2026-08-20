"""Parity tests for synchronization-collapsed span decoding."""

from __future__ import annotations

import pytest
import torch

from gliner2.inference.runtime import ExtractorRuntimeMixin


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_linear_span_logits_shape_and_values(dtype):
    span_rep = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 4.0]]], dtype=dtype
    )
    struct_proj = torch.tensor(
        [[[5.0, 6.0], [7.0, 8.0]]], dtype=dtype
    )
    expected = torch.tensor(
        [[[[17.0], [39.0]], [[23.0], [53.0]]]], dtype=dtype
    )
    runtime = _FakeRuntime()
    actual = runtime._compute_span_logits(span_rep, struct_proj)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


class _FakeBatch:
    def __init__(self):
        self.input_ids = torch.zeros(1, 2, dtype=torch.long)
        self.schema_tokens_list = [[
            ["[P]", "prompt", "entities", "[E]", "person"],
            ["[P]", "prompt", "sentiment", "[C]", "positive", "[C]", "negative"],
            ["[P]", "prompt", "works_for", "[R]", "head", "[R]", "tail"],
            ["[P]", "prompt", "product", "[E]", "name", "[E]", "price"],
        ]]
        self.task_types = [[
            "entities", "classifications", "relations", "json_structures"
        ]]
        self.text_tokens = [["Alice", "Apple"]]
        self.original_texts = ["Alice Apple"]
        self.start_mappings = [[0, 6]]
        self.end_mappings = [[5, 11]]
        self.original_schemas = [{
            "entities": {"person": ""},
            "classifications": [{
                "task": "sentiment",
                "labels": ["positive", "negative"],
                "multi_label": False,
                "cls_threshold": 0.5,
            }],
            "relations": [{"works_for": {"head": "", "tail": ""}}],
            "json_structures": [{"product": {"name": "", "price": ""}}],
        }]

    def __len__(self):
        return 1


class _FakeRuntime(ExtractorRuntimeMixin):
    def count_pred(self, embedding):
        logits = embedding.new_full((embedding.shape[0], 20), -10.0)
        logits[:, 1] = 10.0
        return logits

    def count_embed(self, field_embeddings, predicted_count):
        return field_embeddings.unsqueeze(0).expand(
            predicted_count, -1, -1
        )

    def classifier(self, embeddings):
        return embeddings.sum(dim=-1, keepdim=True)


def test_collapsed_decode_matches_existing_mixed_task_decoder():
    runtime = _FakeRuntime()
    batch = _FakeBatch()
    header = torch.zeros(4)
    schema_embs = [[
        [header, torch.tensor([2.0, 0.0, 0.0, 0.0])],
        [
            header,
            torch.tensor([2.0, 0.0, 0.0, 0.0]),
            torch.tensor([-2.0, 0.0, 0.0, 0.0]),
        ],
        [
            header,
            torch.tensor([2.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 2.0, 0.0, 0.0]),
        ],
        [
            header,
            torch.tensor([2.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 2.0, 0.0, 0.0]),
        ],
    ]]
    token_embs = [torch.zeros(2, 4)]
    span_info = [{
        "span_rep": torch.tensor([
            [[2.0, 0.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0, 0.0]],
        ])
    }]
    metadata = [{
        "field_metadata": {},
        "entity_metadata": {},
        "relation_metadata": {},
        "field_orders": {
            "works_for": ["head", "tail"],
            "product": ["name", "price"],
        },
        "entity_order": ["person"],
        "relation_order": ["works_for"],
        "classification_tasks": ["sentiment"],
        "entity_attribute_groups": {},
    }]

    expected = runtime._extract_sample(
        token_embs=token_embs[0],
        schema_embs=schema_embs[0],
        schema_tokens_list=batch.schema_tokens_list[0],
        task_types=batch.task_types[0],
        text_tokens=batch.text_tokens[0],
        original_text=batch.original_texts[0],
        schema=batch.original_schemas[0],
        start_mapping=batch.start_mappings[0],
        end_mapping=batch.end_mappings[0],
        threshold=0.6,
        metadata=metadata[0],
        include_confidence=True,
        include_spans=True,
        span_info=span_info[0],
    )
    actual = runtime._extract_from_batch_sync_collapsed(
        batch=batch,
        all_schema_embs=schema_embs,
        all_span_info=span_info,
        threshold=0.6,
        metadata_list=metadata,
        include_confidence=True,
        include_spans=True,
    )

    assert actual == [expected]
    assert list(actual[0]) == ["entities", "sentiment", "works_for", "product"]
