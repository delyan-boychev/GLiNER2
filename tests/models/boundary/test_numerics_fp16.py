"""Long-sequence half-precision numerical guards."""

import torch

from gliner2.models.boundary.heads import BoundaryQueryHead
from gliner2.models.boundary.pool import PooledCandidates, SharedPoolScorer
from gliner2.models.boundary.relations import RelationPairBatch, SparseRelationScorer
from gliner2.models.boundary.scoring import interval_prefix_score


def test_inside_prefix_half_model_accumulates_in_fp32():
    torch.manual_seed(3)
    length = 4096
    head = BoundaryQueryHead(8, 8, query_dim=8, dropout=0.0).half()
    boundary_states = torch.randn(1, length + 1, 8).half()
    text_states = torch.randn(1, length, 8).half()
    query_states = torch.randn(1, 1, 8).half()
    boundary_mask = torch.ones(1, length + 1, dtype=torch.bool)
    text_mask = torch.ones(1, length, dtype=torch.bool)
    query_mask = torch.ones(1, 1, dtype=torch.bool)
    output = head(
        boundary_states,
        boundary_mask,
        text_states,
        text_mask,
        query_states,
        query_mask,
    )
    assert output.inside_prefix.dtype == torch.float32
    assert torch.isfinite(output.inside_prefix).all()
    starts = torch.tensor([[[17, 1024, 2048]]])
    ends = torch.tensor([[[4011, 2048, 4096]]])
    actual = interval_prefix_score(
        output.inside_prefix, starts, ends, output.inside_prefix_mean
    )
    expected = torch.stack(
        [
            output.inside_logits[0, 0, 17:4011].float().sum(),
            output.inside_logits[0, 0, 1024:2048].float().sum(),
            output.inside_logits[0, 0, 2048:4096].float().sum(),
        ]
    ).view(1, 1, 3)
    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-4)


def test_shared_pool_scorer_accepts_fp32_integer_derived_features_in_fp16():
    scorer = SharedPoolScorer(
        boundary_dim=8,
        query_dim=8,
        pair_dim=8,
        dropout=0.0,
        candidate_attention_layers=0,
        candidate_attention_heads=1,
        query_attention_layers=0,
        enable_span_content=False,
        content_dim=4,
        content_soft_max_pool=False,
        text_hidden_size=8,
    ).half().eval()
    pooled = PooledCandidates(
        indices=torch.tensor([[[0, 2], [1, 4]]]),
        mask=torch.ones(1, 2, dtype=torch.bool),
        # Proposal scores may be accumulated in FP32 for numerical stability.
        proposal_logits=torch.randn(1, 2),
        gold_mask=None,
    )

    scores, candidates = scorer(
        boundary_states=torch.randn(1, 5, 8).half(),
        query_states=torch.randn(1, 2, 8).half(),
        query_mask=torch.ones(1, 2, dtype=torch.bool),
        pooled=pooled,
        start_logits=torch.randn(1, 2, 5).half(),
        end_logits=torch.randn(1, 2, 5).half(),
        inside_prefix=None,
        text_lengths=torch.tensor([4]),
        text_states=torch.randn(1, 4, 8).half(),
        text_mask=torch.ones(1, 4, dtype=torch.bool),
    )

    assert scores.dtype == torch.float16
    assert candidates.dtype == torch.float16


def test_relation_scorer_builds_positional_features_in_fp16():
    scorer = SparseRelationScorer(8).half().eval()
    pairs = RelationPairBatch(
        batch_index=torch.tensor([0]),
        relation_index=torch.tensor([0]),
        head_start=torch.tensor([0]),
        head_end=torch.tensor([2]),
        tail_start=torch.tensor([3]),
        tail_end=torch.tensor([5]),
        head_prob=torch.ones(1),
        tail_prob=torch.ones(1),
    )

    logits = scorer(
        torch.randn(1, 6, 8).half(),
        torch.randn(1, 1, 8).half(),
        None,
        pairs,
    )

    assert logits.dtype == torch.float16
    assert torch.isfinite(logits).all()
