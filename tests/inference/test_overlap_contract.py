"""Shared explicit overlap semantics across decoders and chunk merging."""

from __future__ import annotations

import pytest

from gliner2.inference.candidate_decoder import apply_overlap_policy
from gliner2.inference.chunking import TextChunk, merge_chunk_results
from gliner2.inference.overlap import normalize_overlap_policy
from gliner2.models.boundary.engine import _resolve_spans
from gliner2.models.candidates import ScoredSpanCandidate


SCORED = [
    (0.9, 0, 5),
    (0.6, 1, 4),
    (0.7, 3, 8),
    (0.5, 8, 10),
]

EXPECTED = {
    "allow": [(0, 5), (3, 8), (1, 4), (8, 10)],
    "nested": [(0, 5), (1, 4), (8, 10)],
    "disallow": [(0, 5), (8, 10)],
    "longest": [(0, 5), (3, 8), (8, 10)],
}


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("allow", "allow"),
        ("all", "allow"),
        ("nested", "nested"),
        ("flat", "disallow"),
        ("disallow", "disallow"),
        ("longest", "longest"),
    ],
)
def test_overlap_aliases_are_normalized(alias, canonical):
    assert normalize_overlap_policy(alias) == canonical


@pytest.mark.parametrize("policy", ["allow", "nested", "disallow", "longest"])
def test_span_and_boundary_decoders_share_explicit_semantics(policy):
    candidates = [
        ScoredSpanCandidate(0, start, end, 0.0, score)
        for score, start, end in SCORED
    ]
    span_result = apply_overlap_policy(candidates, policy)
    boundary_result = _resolve_spans(SCORED, policy)

    assert [(item.start, item.end) for item in span_result] == EXPECTED[policy]
    assert [(start, end) for _, start, end in boundary_result] == EXPECTED[policy]


@pytest.mark.parametrize("policy", ["allow", "nested", "disallow", "longest"])
def test_chunk_merge_uses_the_same_explicit_policy(policy):
    text = "abcdefghij"
    chunks = [
        TextChunk(text, 0, len(text), 0, 10),
        TextChunk(text, 0, len(text), 0, 10),
    ]
    rows = [
        {
            "text": text[start:end],
            "confidence": score,
            "start": start,
            "end": end,
        }
        for score, start, end in SCORED
    ]
    results = [
        {"entities": {"item": rows[:2]}},
        {"entities": {"item": rows[2:]}},
    ]

    merged = merge_chunk_results(
        text,
        chunks,
        results,
        include_confidence=True,
        include_spans=True,
        overlap_policy=policy,
    )
    actual = [
        (item["start"], item["end"])
        for item in merged["entities"]["item"]
    ]
    # Chunk output is document-ordered, while decoder output is score-ranked.
    assert actual == sorted(EXPECTED[policy])
