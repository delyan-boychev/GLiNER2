"""Transport-level regression tests for GLiNER2API's public wrappers."""
from __future__ import annotations

import json
import warnings

import pytest

from gliner2.api_client import (
    AuthenticationError,
    GLiNER2API,
    GLiNER2APIError,
    ServerError,
    ValidationError,
)


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.content = json.dumps(body).encode() if body is not None else b""

    def json(self):
        if self._body is None:
            raise ValueError("empty response")
        return self._body


class _Session:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code
        self.calls = []
        self.headers = {}

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "payload": json, "timeout": timeout})
        body = self.body(json, len(self.calls)) if callable(self.body) else self.body
        if isinstance(body, tuple):
            body, status_code = body
        else:
            status_code = self.status_code
        return _Response(body, status_code)

    def close(self):
        pass


def _client(body={"result": {"canned": True}}, status_code=200):
    client = GLiNER2API(api_key="test-key", api_base_url="https://api.example.test")
    client.session = _Session(body, status_code)
    return client


def _payload(client):
    assert len(client.session.calls) == 1
    return client.session.calls[0]["payload"]


def test_entity_single_and_batch_preserve_reply_shape_asymmetry():
    single = _client({"result": {"person": ["Ada"]}})
    assert single.extract_entities("Ada", {"person": "human"}) == {
        "entities": {"person": ["Ada"]}
    }
    assert _payload(single)["schema"] == ["person"]

    batch = _client({"result": {"person": ["Ada"]}})
    assert batch.batch_extract_entities(["Ada"], ["person"]) == [{"person": ["Ada"]}]
    assert _payload(batch)["task"] == "extract_entities"


def test_classification_preserves_single_and_batch_wire_contracts():
    single = _client({"result": {"classification": "positive"}})
    assert single.classify_text("great", {"sentiment": ["positive", "negative"]}) == {
        "sentiment": "positive"
    }
    assert _payload(single)["task"] == "classify_text"
    assert _payload(single)["schema"] == {"categories": ["positive", "negative"]}

    multi = _client({"result": {"sentiment": "positive", "topic": "tech"}})
    multi.classify_text("great", {"sentiment": ["positive"], "topic": ["tech"]})
    assert _payload(multi)["task"] == "schema"

    batch = _client({"result": {"sentiment": "positive"}})
    assert batch.batch_classify_text(["great"], {"sentiment": ["positive"]}) == [
        {"sentiment": "positive"}
    ]
    assert _payload(batch)["task"] == "schema"


def test_all_batch_wrappers_return_one_result_per_input():
    json_client = _client({"result": {"invoice": {}}})
    assert json_client.batch_extract_json(["x"], {"invoice": ["total"]}) == [{"invoice": {}}]

    relation_client = _client({"result": {"relation_extraction": {}}})
    assert relation_client.batch_extract_relations(["x"], ["works_at"]) == [
        {"relation_extraction": {}}
    ]

    extract_client = _client({"result": {"only": "a dict"}})
    assert extract_client.batch_extract(["x"], {"entities": ["person"]}) == [
        {"only": "a dict"}
    ]


def test_generic_extract_preserves_validation_short_circuit_and_fanout_warning():
    client = _client()
    assert client.batch_extract([], {"entities": ["person"]}) == []
    assert client.session.calls == []

    with pytest.raises(ValueError, match="at least one extraction task"):
        client.extract("x", {"unknown": True})

    fanout = _client({"result": {"ok": True}})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert fanout.batch_extract(["a", "b"], [{"entities": ["a"]}, {"entities": ["b"]}]) == [
            {"ok": True},
            {"ok": True},
        ]
    assert len(fanout.session.calls) == 2
    assert [warning.category for warning in captured] == [UserWarning]


def test_generic_extract_keeps_schema_api_wire_metadata():
    client = _client({"result": {}})
    schema = client.create_schema().entities(["person"], dtype="str", threshold=0.4)
    schema = schema.classification("sentiment", ["positive"], cls_threshold=0.8)
    client.extract("x", schema)
    assert _payload(client)["schema"] == {
        "entities": ["person"],
        "entity_dtype": "str",
        "entity_threshold": 0.4,
        "classifications": {
            "sentiment": {
                "labels": ["positive"],
                "multi_label": False,
                "cls_threshold": 0.8,
            }
        },
    }


@pytest.mark.parametrize(
    ("status_code", "body", "exception"),
    [
        (401, {"detail": "bad key"}, AuthenticationError),
        (400, {"detail": "bad request"}, ValidationError),
        (422, None, ValidationError),
        (500, {"detail": "server"}, ServerError),
        (418, {"detail": "teapot"}, GLiNER2APIError),
    ],
)
def test_http_error_mapping_preserves_exception_data(status_code, body, exception):
    client = _client(body, status_code)
    with pytest.raises(exception) as caught:
        client.extract_entities("x", ["person"])
    assert caught.value.status_code == status_code
    assert caught.value.response_data == body


def test_empty_success_response_is_an_api_error():
    client = _client(None)
    with pytest.raises(GLiNER2APIError, match="Empty response body from API") as caught:
        client.extract_entities("x", ["person"])
    assert caught.value.status_code == 200


def test_batch_size_partitions_requests_and_preserves_order():
    def respond(payload, _):
        return {
            "result": [
                {"value": text}
                for text in payload["text"]
            ]
        }

    client = _client(respond)
    texts = ["a", "b", "c", "d", "e"]
    result = client.batch_extract_json(
        texts,
        {"item": ["value"]},
        batch_size=2,
    )

    assert result == [{"value": text} for text in texts]
    payloads = [call["payload"] for call in client.session.calls]
    assert [payload["text"] for payload in payloads] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]
    assert all("batch_size" not in payload for payload in payloads)


def test_partition_failure_exposes_completed_results_and_failed_range():
    def respond(payload, call_number):
        if call_number == 2:
            return ({"detail": "partition failed"}, 500)
        return {
            "result": [
                {"value": text}
                for text in payload["text"]
            ]
        }

    client = _client(respond)
    with pytest.raises(ServerError, match="partition failed") as caught:
        client.batch_extract_json(
            ["a", "b", "c", "d", "e"],
            {"item": ["value"]},
            batch_size=2,
        )

    assert caught.value.partial_results == [{"value": "a"}, {"value": "b"}]
    assert caught.value.failed_range == (2, 4)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(batch_size):
    client = _client()
    with pytest.raises(ValueError, match="greater than 0"):
        client.batch_extract_json(
            ["a"], {"item": ["value"]}, batch_size=batch_size
        )
    assert client.session.calls == []


def test_entity_long_chunks_locally_and_remaps_global_offsets():
    def respond(payload, _):
        results = []
        for chunk_text in payload["text"]:
            start = chunk_text.find("target")
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
            results.append({"item": spans})
        return {"result": results}

    text = "aa bb target cc dd"
    client = _client(respond)
    result = client.extract_entities_long(
        text,
        ["item"],
        chunk_size=3,
        chunk_overlap=1,
        batch_size=1,
        include_confidence=True,
        include_spans=True,
        overlap_policy="flat",
    )

    start = text.index("target")
    assert result == {
        "entities": {
            "item": [
                {
                    "text": "target",
                    "confidence": 0.9,
                    "start": start,
                    "end": start + len("target"),
                }
            ]
        }
    }
    payloads = [call["payload"] for call in client.session.calls]
    assert all(payload["task"] == "extract_entities" for payload in payloads)
    assert all(
        set(payload) == {
            "task",
            "text",
            "schema",
            "threshold",
            "include_confidence",
            "include_spans",
            "format_results",
        }
        for payload in payloads
    )


def test_generic_long_batch_fans_out_per_document_schemas_in_order():
    def respond(payload, _):
        label = payload["schema"]["entities"][0]
        surface = payload["text"].split()[0]
        return {
            "result": {
                "entities": {
                    label: [
                        {
                            "text": surface,
                            "confidence": 0.8,
                            "start": 0,
                            "end": len(surface),
                        }
                    ]
                }
            }
        }

    client = _client(respond)
    texts = ["Alice here", "Acme there"]
    schemas = [
        {"entities": ["person"]},
        {"entities": ["company"]},
    ]
    with pytest.warns(UserWarning, match="Multi-schema batch"):
        result = client.batch_extract_long(
            texts,
            schemas,
            batch_size=2,
            chunk_size=10,
            chunk_overlap=0,
            include_spans=True,
        )

    assert result == [
        {
            "entities": {
                "person": [{"text": "Alice", "start": 0, "end": 5}]
            }
        },
        {
            "entities": {
                "company": [{"text": "Acme", "start": 0, "end": 4}]
            }
        },
    ]
    assert [
        call["payload"]["schema"] for call in client.session.calls
    ] == schemas


def test_json_classification_and_relation_long_wrappers_merge_offline():
    json_client = _client(
        {
            "result": [
                {
                    "item": [
                        {
                            "name": {
                                "text": "Alice",
                                "confidence": 0.8,
                                "start": 0,
                                "end": 5,
                            }
                        }
                    ]
                }
            ]
        }
    )
    assert json_client.extract_json_long(
        "Alice",
        {"item": ["name"]},
        chunk_size=10,
        chunk_overlap=0,
    ) == {"item": [{"name": "Alice"}]}
    assert _payload(json_client)["task"] == "extract_json"

    classification_client = _client(
        {
            "result": [
                {
                    "sentiment": {
                        "label": "positive",
                        "confidence": 0.9,
                    }
                }
            ]
        }
    )
    assert classification_client.classify_text_long(
        "great",
        {"sentiment": ["positive", "negative"]},
        chunk_size=10,
        chunk_overlap=0,
    ) == {"sentiment": "positive"}
    assert _payload(classification_client)["task"] == "schema"

    relation = {
        "head": {
            "text": "Alice",
            "confidence": 0.7,
            "start": 0,
            "end": 5,
        },
        "tail": {
            "text": "Acme",
            "confidence": 0.7,
            "start": 9,
            "end": 13,
        },
    }
    relation_client = _client(
        {
            "result": [
                {"relation_extraction": {"works_at": [relation]}}
            ]
        }
    )
    assert relation_client.extract_relations_long(
        "Alice at Acme",
        ["works_at"],
        chunk_size=10,
        chunk_overlap=0,
        include_spans=True,
    ) == {
        "relation_extraction": {
            "works_at": [
                {
                    "head": {"text": "Alice", "start": 0, "end": 5},
                    "tail": {"text": "Acme", "start": 9, "end": 13},
                }
            ]
        }
    }
    assert _payload(relation_client)["task"] == "schema"


@pytest.mark.parametrize(
    "method_name",
    [
        "extract_entities_long",
        "batch_extract_entities_long",
        "classify_text_long",
        "batch_classify_text_long",
        "extract_json_long",
        "batch_extract_json_long",
        "extract_relations_long",
        "batch_extract_relations_long",
        "extract_long",
        "batch_extract_long",
    ],
)
def test_api_exposes_complete_long_document_surface(method_name):
    assert hasattr(GLiNER2API, method_name)
