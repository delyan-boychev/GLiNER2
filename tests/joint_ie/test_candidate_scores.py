"""Sparse CandidateScoreSet + span-lattice bridge + optimizer integration."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from gliner2.joint_ie.candidate_scores import (
    CandidateScoreSet,
    MentionScore,
    ScoredRelationEdge,
    candidate_score_set_to_problem,
    score_lattice_to_candidate_score_set,
)
from gliner2.joint_ie.constraints import TypedEndpoints, UniqueRelationSlot
from gliner2.joint_ie.optimizers import BeamOptimizer, GreedyOptimizer


def _score_set():
    return CandidateScoreSet(
        text="Alice works at Acme",
        mentions=(
            MentionScore(0, "person", 0, 1, 4.0, 0.98),
            MentionScore(1, "org", 3, 4, 4.0, 0.98),
            MentionScore(0, "person", 2, 3, -4.0, 0.02),  # below threshold, dropped
        ),
    )


def test_candidate_score_set_to_problem_thresholds_and_builds_edges():
    css = _score_set()
    edges = [ScoredRelationEdge("works_for", ("person", 0, 1), ("org", 3, 4), 4.0, 0.98)]
    problem = candidate_score_set_to_problem(css, edges)
    assert len(problem.nodes) == 2  # low-prob mention filtered
    assert len(problem.edges) == 1
    assert problem.edges[0].head == ("person", 0, 1)


def test_edge_referencing_dropped_mention_is_pruned():
    css = _score_set()
    bad = [ScoredRelationEdge("works_for", ("person", 2, 3), ("org", 3, 4), 4.0, 0.98)]
    problem = candidate_score_set_to_problem(css, bad)
    assert len(problem.edges) == 0


def test_typed_endpoints_and_optimizers_select_relation():
    css = _score_set()
    edges = [ScoredRelationEdge("works_for", ("person", 0, 1), ("org", 3, 4), 4.0, 0.98)]
    constraints = [TypedEndpoints("works_for", ("person",), ("org",))]
    problem = candidate_score_set_to_problem(css, edges, constraints=constraints)

    for optimizer in (GreedyOptimizer(), BeamOptimizer(beam_width=8)):
        solution = optimizer.optimize(problem)
        assert {e.relation_type for e in solution.edges} == {"works_for"}
        assert len(solution.nodes) == 2


def test_sparse_boundary_edges_receive_distinct_relation_slots():
    css = CandidateScoreSet(
        text="A X B Y",
        mentions=(
            MentionScore(0, "person", 0, 1, 4.0, 0.98),
            MentionScore(1, "org", 1, 2, 4.0, 0.98),
            MentionScore(0, "person", 2, 3, 4.0, 0.98),
            MentionScore(1, "org", 3, 4, 4.0, 0.98),
        ),
        edges=(
            ScoredRelationEdge(
                "works_for", ("person", 0, 1), ("org", 1, 2), 4.0, 0.98
            ),
            ScoredRelationEdge(
                "works_for", ("person", 2, 3), ("org", 3, 4), 4.0, 0.98
            ),
        ),
    )
    problem = candidate_score_set_to_problem(
        css,
        constraints=(UniqueRelationSlot("works_for", "slot"),),
    )
    solution = GreedyOptimizer().optimize(problem)

    assert len(solution.edges) == 2
    assert len({edge.slot for edge in solution.edges}) == 2


def test_score_lattice_to_candidate_score_set_maps_halfopen_spans():
    # A minimal ScoreLattice-shaped object: one entity task, L=2, W=2.
    role_logits = torch.full((1, 2, 2, 2), -5.0)   # [count, types, L, W]
    role_logits[0, 0, 0, 0] = 5.0                   # type "a", start 0 width 0 -> [0,1)
    role_probs = torch.sigmoid(role_logits)
    hyp = SimpleNamespace(role_logits=role_logits, role_probabilities=role_probs)
    task = SimpleNamespace(
        task_type="entities", roles=("a", "b"), count_hypotheses=[hyp]
    )
    span_starts = torch.tensor([[0, 0], [1, 1]])
    span_ends = torch.tensor([[0, 1], [1, 1]])       # inclusive ends
    valid = torch.tensor([[True, True], [True, False]])
    lattice = SimpleNamespace(
        text="x y", span_starts=span_starts, span_ends=span_ends,
        valid_span_mask=valid, tasks=[task],
    )

    css = score_lattice_to_candidate_score_set(lattice)
    high = [m for m in css.mentions if m.probability > 0.5]
    assert len(high) == 1
    m = high[0]
    assert (m.entity_type, m.start, m.end) == ("a", 0, 1)  # inclusive 0 -> half-open [0,1)
