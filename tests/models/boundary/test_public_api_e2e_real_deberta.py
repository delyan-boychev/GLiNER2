"""End-to-end boundary lifecycle through the *public* API on a real encoder.

Unlike ``test_end_to_end_real_deberta`` (which drives the model internals),
this test exercises the wired public surface:

* ``BoundaryExtractor`` construction from an ``ExtractorConfig``.
* Training via ``ExtractorTrainer`` on ``InputExample`` (entities +
  classification combined) — the same trainer the span architecture uses.
* Inference via ``extract_entities`` / ``classify_text``.
* Save via the trainer checkpoint, load via ``AutoExtractor.from_pretrained``
  (architecture-aware dispatch), and re-inference on the reloaded model.

It is marked ``slow`` because it initializes and trains a real DeBERTa-v3
encoder. Inference quality is release-gated separately against the published
``fastino/gliner2.5-multi-v1`` checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from gliner2 import AutoExtractor, ExtractorConfig
from gliner2.inference.engine import BoundaryExtractor
from gliner2.training.data import Classification, InputExample
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

MODEL_NAME = "microsoft/deberta-v3-xsmall"
ENTITY_TYPES = ["company", "product"]
TOPIC_LABELS = ["technology", "sports"]

EXAMPLES = [
    InputExample(
        text="Apple released the iPhone in California during the winter season",
        entities={"company": ["Apple"], "product": ["iPhone"]},
        classifications=[
            Classification(task="topic", labels=TOPIC_LABELS, true_label="technology")
        ],
    ),
    InputExample(
        text="Sony launched the Walkman in Tokyo during a rainy afternoon",
        entities={"company": ["Sony"], "product": ["Walkman"]},
        classifications=[
            Classification(task="topic", labels=TOPIC_LABELS, true_label="technology")
        ],
    ),
    InputExample(
        text="The Lakers defeated the Celtics at the downtown arena last night",
        entities={"company": ["Lakers"], "product": ["Celtics"]},
        classifications=[
            Classification(task="topic", labels=TOPIC_LABELS, true_label="sports")
        ],
    ),
    InputExample(
        text="The Yankees beat the Dodgers inside the packed stadium this weekend",
        entities={"company": ["Yankees"], "product": ["Dodgers"]},
        classifications=[
            Classification(task="topic", labels=TOPIC_LABELS, true_label="sports")
        ],
    ),
]


def _build_model() -> BoundaryExtractor:
    cfg = ExtractorConfig(
        model_name=MODEL_NAME,
        architecture="boundary",
        boundary_head=dict(
            boundary_dim=96, pair_dim=96, start_top_k=32, end_top_k=32,
            ends_per_start=16, starts_per_end=16, candidate_budget=96,
            training_candidate_budget=128, max_gold_per_query=16,
            end_block_size=32, dropout=0.0,
        ),
        token_pooling="first",
    )
    model = BoundaryExtractor(cfg)
    model.boundary_head.loss_weights["pair"] = 4.0
    return model


def _entity_recall(model, examples) -> float:
    hits = total = 0
    for ex in examples:
        result = model.extract_entities(ex.text, ENTITY_TYPES, include_spans=True)
        entities = result.get("entities", {})
        for label, expected in ex.entities.items():
            predicted = [
                e["text"] if isinstance(e, dict) else e
                for e in entities.get(label, [])
            ]
            for surface in expected:
                total += 1
                hits += surface in predicted
    return hits / total if total else 0.0


def _cls_accuracy(model, examples) -> float:
    hits = 0
    for ex in examples:
        result = model.classify_text(ex.text, {"topic": {"labels": TOPIC_LABELS}})
        pred = result.get("topic")
        if isinstance(pred, dict):
            pred = pred.get("labels", [None])
            pred = pred[0] if pred else None
        elif isinstance(pred, list):
            pred = pred[0] if pred else None
        hits += pred == ex.classifications[0].true_label
    return hits / len(examples)


@pytest.mark.slow
def test_boundary_public_api_lifecycle_real_deberta(tmp_path):
    torch.manual_seed(0)
    model = _build_model()

    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        # This is a lifecycle contract, not an overfit/quality test. A single
        # optimizer step proves the public trainer path without introducing a
        # seed-sensitive convergence gate.
        max_steps=1,
        batch_size=4,
        gradient_accumulation_steps=1,
        encoder_lr=2e-5,
        task_lr=5e-3,
        warmup_ratio=0.0,
        scheduler_type="constant",
        logging_steps=1,
        eval_strategy="no",
        fp16=False,
        bf16=False,
    )
    trainer = ExtractorTrainer(model, config)
    result = trainer.train(train_data=EXAMPLES)
    assert torch.isfinite(torch.tensor(result["train_metrics_history"][-1]["loss"]))

    # ---- Inference through the public API (in-memory) ------------------------
    entity_result = model.extract_entities(
        EXAMPLES[0].text, ENTITY_TYPES, include_spans=True
    )
    classification_result = model.classify_text(
        EXAMPLES[0].text, {"topic": {"labels": TOPIC_LABELS}}
    )
    assert isinstance(entity_result.get("entities"), dict)
    assert "topic" in classification_result

    # ---- Save (trainer checkpoint) -> load (architecture-aware) -------------
    save_dir = tmp_path / "run" / "final"
    reloaded = AutoExtractor.from_pretrained(str(save_dir))
    assert type(reloaded).__name__ == "BoundaryExtractor"
    assert reloaded.config.architecture == "boundary"

    reloaded_entities = reloaded.extract_entities(
        EXAMPLES[0].text, ENTITY_TYPES, include_spans=True
    )
    reloaded_classification = reloaded.classify_text(
        EXAMPLES[0].text, {"topic": {"labels": TOPIC_LABELS}}
    )
    assert isinstance(reloaded_entities.get("entities"), dict)
    assert "topic" in reloaded_classification
