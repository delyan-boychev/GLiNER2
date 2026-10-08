"""Internal Hungarian matcher: exhaustive-permutation parity + rectangular."""

from __future__ import annotations

import itertools

import torch

from gliner2.training.matching import (
    build_record_matching_cost,
    linear_sum_assignment,
    match_record_instances,
)


def _brute_force_min(cost: torch.Tensor) -> float:
    r, c = cost.shape
    n = min(r, c)
    best = float("inf")
    for rows in itertools.permutations(range(r), n):
        for cols in itertools.permutations(range(c), n):
            total = sum(float(cost[rows[k], cols[k]]) for k in range(n))
            best = min(best, total)
    return best


def test_hungarian_matches_bruteforce_square_up_to_6():
    torch.manual_seed(0)
    for n in range(1, 7):
        for _ in range(5):
            cost = torch.randint(0, 20, (n, n)).to(torch.float64)
            row, col = linear_sum_assignment(cost)
            got = float(cost[row, col].sum())
            assert abs(got - _brute_force_min(cost)) < 1e-9, (n, cost)
            assert torch.equal(row, torch.arange(n))
            assert sorted(col.tolist()) == list(range(n))


def test_hungarian_rectangular_both_orientations():
    torch.manual_seed(1)
    for r, c in [(2, 5), (5, 2), (3, 4), (4, 3)]:
        cost = torch.randint(0, 30, (r, c)).to(torch.float64)
        row, col = linear_sum_assignment(cost)
        assert len(row) == min(r, c)
        got = float(cost[row, col].sum())
        assert abs(got - _brute_force_min(cost)) < 1e-9, (r, c, cost)
        # rows ascending, columns distinct
        assert row.tolist() == sorted(row.tolist())
        assert len(set(col.tolist())) == len(col)


def test_hungarian_empty():
    row, col = linear_sum_assignment(torch.zeros(0, 3))
    assert len(row) == 0 and len(col) == 0


def test_hungarian_accepts_mps_costs():
    if not torch.backends.mps.is_available():
        return
    cost = torch.tensor(
        [[3.0, 1.0], [1.0, 3.0]],
        device="mps",
    )

    row, col = linear_sum_assignment(cost)

    assert row.tolist() == [0, 1]
    assert col.tolist() == [1, 0]


def test_record_matching_is_permutation_invariant():
    torch.manual_seed(2)
    inst_queries, fields, cands = 5, 2, 4
    object_logits = torch.randn(inst_queries)
    field_pointer_logits = torch.randn(inst_queries, fields, cands)
    gold = [[0, 1], [2, 3], [1, 0]]

    res = match_record_instances(object_logits, field_pointer_logits, gold)
    shuffled = [gold[i] for i in (2, 0, 1)]
    res2 = match_record_instances(object_logits, field_pointer_logits, shuffled)
    assert abs(res.cost - res2.cost) < 1e-9
