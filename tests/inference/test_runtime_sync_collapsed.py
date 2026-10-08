"""Parity tests for synchronization-collapsed span decoding."""

from __future__ import annotations

import torch

from gliner2.inference.runtime import ExtractorRuntimeMixin


class _FakeBatch:
    def __init__(self):
        self.input_ids = torch.zeros(1, 2, dtype=torch.long)
        self.schema_tokens_list = [[
            ["[P]", "prompt", "entities", "[E]", "person"],
            [
                "[P]", "prompt", "sentiment",
                "[C]", "positive", "[C]", "negative",
            ],
            [
                "[P]", "prompt", "works_for",
                "[R]", "head", "[R]", "tail",
            ],
            [
                "[P]", "prompt", "product",
                "[E]", "name", "[E]", "price",
            ],
        ]]
        self.task_types = [[
            "entities", "classifications", "relations", "json_structures",
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
    def __init__(self):
        self.count_shapes = []

    def count_pred(self, embedding):
        self.count_shapes.append(tuple(embedding.shape))
        logits = embedding.new_full((embedding.shape[0], 20), -10.0)
        logits[:, 1] = 10.0
        return logits

    def count_embed(self, field_embeddings, predicted_count):
        return field_embeddings.unsqueeze(0).expand(predicted_count, -1, -1)

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
    runtime.count_shapes.clear()

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
    assert runtime.count_shapes
    assert all(shape[0] == 1 for shape in runtime.count_shapes)


def test_collapsed_decode_is_enabled_by_default():
    assert ExtractorRuntimeMixin.sync_collapsed_decode is True


def _collapsed_and_eager(model, texts, schema, threshold=0.05):
    from gliner2.training.trainer import ExtractorCollator

    schema_dicts, metadata = model._build_schema_dicts_and_metadata(
        [schema for _ in texts]
    )
    collator = ExtractorCollator(
        model.processor, is_training=False, architecture=model.architecture
    )
    batch = collator(list(zip(texts, schema_dicts)))

    with torch.inference_mode():
        token_embs, schema_embs = model.processor.extract_embeddings_from_batch(
            model.encode_tokens(batch), batch.input_ids, batch
        )
        span_info = model.compute_span_rep_batched(token_embs)
        collapsed = model._extract_from_batch_sync_collapsed(
            batch=batch,
            all_schema_embs=schema_embs,
            all_span_info=span_info,
            threshold=threshold,
            metadata_list=metadata,
            include_confidence=True,
            include_spans=True,
        )
        eager = [
            model._extract_sample(
                token_embs=token_embs[i],
                schema_embs=schema_embs[i],
                schema_tokens_list=batch.schema_tokens_list[i],
                task_types=batch.task_types[i],
                text_tokens=batch.text_tokens[i],
                original_text=batch.original_texts[i],
                schema=batch.original_schemas[i],
                start_mapping=batch.start_mappings[i],
                end_mapping=batch.end_mappings[i],
                threshold=threshold,
                metadata=metadata[i],
                include_confidence=True,
                include_spans=True,
                span_info=span_info[i],
            )
            for i in range(len(batch))
        ]
    return collapsed, eager


TEXTS = [
    "Tim Cook works for Apple in Cupertino on 4 May 2026.",
    "The Aurora Phone camera is excellent but its battery is disappointing.",
    "Vertex Audio shipped Vertex Buds in Berlin for 249 euros.",
]


def test_collapsed_decode_matches_eager_for_multiple_samples(tiny_span_model):
    schema = (
        tiny_span_model.create_schema()
        .entities(["person", "organization", "location"])
        .classification("sentiment", ["positive", "negative"])
        .structure("employment")
        .field("person")
        .field("organization")
    )

    collapsed, eager = _collapsed_and_eager(tiny_span_model.eval(), TEXTS, schema)

    assert len(collapsed) == len(TEXTS)
    assert collapsed == eager


def test_collapsed_decode_matches_eager_with_entity_attributes(tiny_span_model):
    from gliner2.inference.schema import AttributeGroup

    schema = (
        tiny_span_model.create_schema()
        .entities(["product", "organization"])
        .entity_attributes(
            {
                "sentiment": AttributeGroup(
                    ["positive", "negative", "neutral"],
                    applies_to=["product"],
                    qualify_labels=True,
                ),
                "aspect": AttributeGroup(
                    ["camera", "battery", "audio"],
                    multi_label=True,
                    threshold=0.35,
                    applies_to=["product"],
                ),
            }
        )
    )

    collapsed, eager = _collapsed_and_eager(tiny_span_model.eval(), TEXTS, schema)

    assert collapsed == eager
