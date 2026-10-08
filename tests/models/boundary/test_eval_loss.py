"""Boundary eval loss integrity (Finding 1 / Phase 1.1).

An extraction-only eval set must produce a finite, non-zero, supervised eval
loss (not crash and not silently collapse to classification-only). The eval
collator builds gold targets while the model runs in eval mode; gold injection
into proposals stays gated on ``model.training`` so eval remains unbiased.
"""

from __future__ import annotations

import torch

from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


ENTITIES_ONLY = ("apple released iphone .", {"entities": {"company": ["apple"], "product": ["iphone"]}})


def _eval_collator(model, build_targets):
    return ExtractorCollator(
        model.processor,
        is_training=False,
        architecture="boundary",
        max_gold_per_query=model.boundary_head.settings.max_gold_per_query,
        build_targets=build_targets,
    )


def test_entities_only_eval_loss_is_finite_and_nonzero():
    model = build_tiny_boundary_model()
    model.eval()

    batch = _eval_collator(model, build_targets=True)([ENTITIES_ONLY])
    assert batch.targets is not None  # eval collator builds supervision

    with torch.no_grad():
        out = model(batch)

    assert out.total_loss is not None
    loss = float(out.total_loss)
    assert torch.isfinite(out.total_loss)
    assert loss > 0.0


def test_eval_loss_moves_with_boundary_weights():
    model = build_tiny_boundary_model()
    model.eval()
    collator = _eval_collator(model, build_targets=True)
    batch = collator([ENTITIES_ONLY])

    with torch.no_grad():
        base = float(model(batch).total_loss)

    # Perturb the boundary head; a supervised eval loss must respond.
    with torch.no_grad():
        for p in model.boundary_head.parameters():
            p.add_(torch.randn_like(p) * 0.5)
        perturbed = float(model(batch).total_loss)

    assert abs(perturbed - base) > 1e-6


def test_plain_inference_collation_builds_no_targets():
    # Guards the decoupling: default inference still yields no supervision, so
    # ordinary extraction is unaffected by the eval-target change.
    model = build_tiny_boundary_model()
    batch = _eval_collator(model, build_targets=None)([ENTITIES_ONLY])
    assert batch.targets is None


# A malformed record (any shape ExtractorProcessor._transform_record rejects;
# here, relations in the flat/legacy shape `_process_relations` cannot parse)
# only ever crashed under `error_policy="fallback"` (the eval collator's
# default) *and* `build_targets=True` (the eval-loss decoupling above): the
# transform failure is swallowed and substituted with
# SchemaTransformer._create_fallback_record, whose `structure_labels` used to
# encode a fabricated (0, 0) mention as a bare tuple instead of a list of
# tuples, and omitted `text_word_first_positions` (leaving text_length at 0).
# Plain inference (build_targets=False/None) never touched either field, so
# this was silently inert until an eval loss started requesting real targets.
MALFORMED_RELATIONS = (
    "the kiox 300 pairs with the kiox 400c on this generation .",
    {
        "relations": [
            {"name": "same_generation_family", "head": "kiox 300", "tail": "kiox 400c"}
        ]
    },
)


def test_fallback_record_survives_eval_loss_build_targets():
    model = build_tiny_boundary_model()
    model.eval()

    # error_policy left at its "fallback" default (what the real Trainer eval
    # dataloader uses) so the malformed record is substituted, not raised.
    collator = ExtractorCollator(
        model.processor,
        is_training=False,
        architecture="boundary",
        max_gold_per_query=model.boundary_head.settings.max_gold_per_query,
        build_targets=True,
    )
    batch = collator([ENTITIES_ONLY, MALFORMED_RELATIONS])
    assert batch.targets is not None

    with torch.no_grad():
        out = model(batch)
    assert out.total_loss is not None
    assert torch.isfinite(out.total_loss)


def test_fallback_record_contributes_zero_gold_targets():
    # Pins the fix at the level it actually operates on (targets), not just
    # "the forward pass didn't crash": a fabricated-but-validly-shaped
    # mention (e.g. structure_labels=[[1, [[[(0, 0)]]]]], a plausible partial
    # re-implementation) would also survive the loss-based assertions above
    # without this check, since it happens to be loss-invisible today.
    model = build_tiny_boundary_model()
    model.eval()
    batch = _eval_collator(model, build_targets=True)([ENTITIES_ONLY, MALFORMED_RELATIONS])
    fallback_idx = 1
    assert int(batch.targets.mention_mask[fallback_idx].sum()) == 0


def test_fallback_record_has_correct_text_length():
    # Pins the second, independent half of the fix: text_word_first_positions
    # must actually be populated (previously silently defaulted to [], giving
    # text_length=0 for every fallback record). Regressing this alone leaves
    # the loss-based assertions above green, since a count=0 target never
    # exercises text_length at all.
    model = build_tiny_boundary_model()
    model.eval()
    batch = _eval_collator(model, build_targets=True)([ENTITIES_ONLY, MALFORMED_RELATIONS])
    fallback_idx = 1
    assert batch.text_word_counts[fallback_idx] == len(batch.text_tokens[fallback_idx]) == 1
