"""Public long-document wrapper and merge contracts."""

from __future__ import annotations

import pytest

from gliner2.inference.runtime import ExtractorRuntimeMixin
from gliner2.inference.schema import Schema


class _WrapperProbe(ExtractorRuntimeMixin):
    def __init__(self):
        self.call = None

    def create_schema(self):
        return Schema()

    def extract_long(self, text, schema, **kwargs):
        self.call = ("single", text, schema, kwargs)
        return {"ok": True}

    def batch_extract_long(self, texts, schemas, **kwargs):
        self.call = ("batch", texts, schemas, kwargs)
        return [{"ok": text} for text in texts]


@pytest.mark.parametrize(
    ("method", "argument", "schema_key"),
    [
        ("extract_entities_long", ["person"], "entities"),
        ("classify_text_long", {"sentiment": ["positive"]}, "classifications"),
        ("extract_json_long", {"item": ["name::str"]}, "json_structures"),
        ("extract_relations_long", ["works_at"], "relations"),
    ],
)
def test_single_long_convenience_wrappers_route_all_options(
    method, argument, schema_key
):
    runtime = _WrapperProbe()
    result = getattr(runtime, method)(
        "document",
        argument,
        chunk_size=12,
        chunk_overlap=3,
        batch_size=2,
        num_workers=1,
        include_confidence=True,
        include_spans=True,
        overlap_policy="allow",
    )

    assert result == {"ok": True}
    _, text, schema, kwargs = runtime.call
    assert text == "document"
    assert schema_key in schema.build()
    assert kwargs["overlap_policy"] == "allow"
    assert kwargs["chunk_size"] == 12
    assert kwargs["chunk_overlap"] == 3


@pytest.mark.parametrize(
    ("method", "argument", "schema_key"),
    [
        ("batch_extract_entities_long", ["person"], "entities"),
        ("batch_classify_text_long", {"sentiment": ["positive"]}, "classifications"),
        ("batch_extract_json_long", {"item": ["name::str"]}, "json_structures"),
        ("batch_extract_relations_long", ["works_at"], "relations"),
    ],
)
def test_batch_long_convenience_wrappers_preserve_document_order(
    method, argument, schema_key
):
    runtime = _WrapperProbe()
    result = getattr(runtime, method)(
        ["first", "second"],
        argument,
        batch_size=2,
        overlap_policy="flat",
    )

    assert result == [{"ok": "first"}, {"ok": "second"}]
    _, texts, schema, kwargs = runtime.call
    assert texts == ["first", "second"]
    assert schema_key in schema.build()
    assert kwargs["overlap_policy"] == "flat"


class _ChunkRuntime(ExtractorRuntimeMixin):
    architecture = "span"

    def batch_extract(self, texts, schemas, **kwargs):
        results = []
        for text in texts:
            start = text.find("target")
            spans = []
            if start >= 0:
                spans.append(
                    {
                        "text": "target",
                        "confidence": 0.9,
                        "start": start,
                        "end": start + len("target"),
                    }
                )
            results.append({"entities": {"item": spans}})
        return results


def test_runtime_long_merge_remaps_offsets_and_deduplicates_overlap():
    text = "aa bb target cc dd"
    runtime = _ChunkRuntime()

    result = runtime.batch_extract_long(
        [text],
        {"entities": ["item"]},
        chunk_size=3,
        chunk_overlap=1,
        include_confidence=True,
        include_spans=True,
        overlap_policy="allow",
    )[0]

    assert result == {
        "entities": {
            "item": [
                {
                    "text": "target",
                    "confidence": 0.9,
                    "start": text.index("target"),
                    "end": text.index("target") + len("target"),
                }
            ]
        }
    }


def test_batch_extract_long_uses_processor_word_splitter(monkeypatch):
    from types import SimpleNamespace

    from gliner2.inference.chunking import TextChunk

    sentinel = object()
    captured = {}

    def fake_split(text, chunk_size=384, chunk_overlap=64, word_splitter=None):
        captured["word_splitter"] = word_splitter
        return [
            TextChunk(
                text=text,
                start_char=0,
                end_char=len(text),
                start_word=0,
                end_word=1,
            )
        ]

    monkeypatch.setattr(
        "gliner2.inference.runtime.split_text_into_chunks", fake_split
    )

    class Runtime(ExtractorRuntimeMixin):
        def __init__(self):
            self.processor = SimpleNamespace(word_splitter=sentinel)

        def batch_extract(self, texts, schemas, **kwargs):
            return [{} for _ in texts]

        @staticmethod
        def _scalar_entity_labels(schema):
            return set()

        def _resolved_overlap_policy(self, policy):
            return policy or "disallow"

    Runtime().batch_extract_long(
        ["hello"],
        {"entities": ["item"]},
        chunk_size=4,
        chunk_overlap=1,
    )
    assert captured["word_splitter"] is sentinel
