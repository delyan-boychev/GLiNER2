from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gliner2.training.data import InputExample
from gliner2.training.trainer import ExtractorCollator
from gliner2.processor import SamplingConfig, SchemaTransformer
from recipes.boundary_pretrain import train as recipe
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer


def _write_jsonl(path: Path, prefix: str, count: int) -> None:
    rows = [
        {
            "input": f"{prefix}-{index} appears here.",
            "output": {"entities": {"item": [f"{prefix}-{index}"]}},
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_config(path: Path, train_path: str) -> None:
    config = {
        "seed": 13,
        "data": {
            "train": [{"name": "train", "path": train_path, "weight": 1.0}],
            "mixing": {"epoch_size": 5},
        },
        "model": {
            "model_name": "unused-test-backbone",
            "architecture": "boundary",
            "boundary_head": {
                "boundary_dim": 64,
                "pair_dim": 64,
                "candidate_pool": "shared",
            },
        },
        "training": {
            "output_dir": "./output",
            "max_steps": 1,
            "batch_size": 1,
            "eval_batch_size": 1,
            "fp16": False,
            "bf16": False,
            "num_workers": 0,
            "fused_optimizer": False,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_yaml_loading_and_relative_path_resolution(tmp_path: Path) -> None:
    config_path = tmp_path / "nested" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text("data:\n  train: []\n", encoding="utf-8")

    assert recipe.load_yaml_config(config_path)["data"]["train"] == []
    assert recipe.resolve_path(config_path, "../data/train.jsonl") == (
        tmp_path / "data" / "train.jsonl"
    ).resolve()


def test_empty_source_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    empty_path = tmp_path / "empty.jsonl"
    config_path.write_text("{}\n", encoding="utf-8")
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        recipe.load_sources(
            config_path,
            [{"name": "empty", "path": "empty.jsonl", "weight": 1.0}],
            seed=1,
        )


def test_invalid_source_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    invalid_path = tmp_path / "invalid.jsonl"
    config_path.write_text("{}\n", encoding="utf-8")
    invalid_path.write_text(
        json.dumps(
            {
                "input": "Alice is present.",
                "output": {"entities": {"person": ["Bob"]}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="not found"):
        recipe.load_sources(
            config_path,
            [{"name": "invalid", "path": "invalid.jsonl"}],
            seed=1,
        )


def test_recipe_skips_malformed_and_invalid_examples(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    data_path = tmp_path / "mixed.jsonl"
    config_path.write_text("{}\n", encoding="utf-8")
    data_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "input": "Alice is present.",
                        "output": {"entities": {"person": ["Alice"]}},
                    }
                ),
                "{not valid json}",
                json.dumps(
                    {
                        "input": "Alice is present.",
                        "output": {"entities": {"person": ["Bob"]}},
                    }
                ),
                json.dumps(
                    {
                        "input": "Alice has a memory.",
                        "output": {
                            "json_structures": [
                                {"memory": {"owner": ["Alice", {"bad": "value"}]}}
                            ]
                        },
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    source = recipe.load_sources(
        config_path,
        [{"name": "mixed", "path": "mixed.jsonl"}],
        seed=1,
        skip_invalid_examples=True,
    )[0]

    assert len(source.examples) == 1
    assert source.skipped_examples == 3


def test_weighted_mixing_is_exact_and_deterministic(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    _write_jsonl(tmp_path / "a.jsonl", "a", 3)
    _write_jsonl(tmp_path / "b.jsonl", "b", 3)
    sources = recipe.load_sources(
        config_path,
        [
            {"name": "a", "path": "a.jsonl", "weight": 3.0},
            {"name": "b", "path": "b.jsonl", "weight": 1.0},
        ],
        seed=5,
    )

    first, first_counts = recipe.mix_sources(sources, seed=9, epoch_size=8)
    second, second_counts = recipe.mix_sources(sources, seed=9, epoch_size=8)

    assert first_counts == second_counts == {"a": 6, "b": 2}
    assert [example.text for example in first.examples] == [
        example.text for example in second.examples
    ]


def test_config_conversion_resolves_output_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    model_config, training_config = recipe.build_configs(
        {
            "model": {
                "model_name": "unused",
                "architecture": "boundary",
                "boundary_head": {"boundary_dim": 64, "pair_dim": 64},
            },
            "training": {
                "output_dir": "relative-output",
                "fp16": False,
                "bf16": False,
                "fused_optimizer": False,
            },
        },
        config_path,
    )

    assert model_config.architecture == "boundary"
    assert model_config.boundary_head["boundary_dim"] == 64
    assert training_config.output_dir == str(
        (tmp_path / "relative-output").resolve()
    )


def test_dry_run_validates_without_allocating_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "train.jsonl"
    config_path = tmp_path / "config.yaml"
    _write_jsonl(data_path, "sample", 2)
    _write_config(config_path, "train.jsonl")

    def fail_if_allocated(*args, **kwargs):
        raise AssertionError("dry-run allocated a model")

    monkeypatch.setattr(recipe, "BoundaryExtractor", fail_if_allocated)
    result = recipe.run(config_path, dry_run=True)

    assert result["dry_run"] is True
    assert result["manifest"]["training_examples"] == 5
    assert not (tmp_path / "output").exists()


def test_legacy_json_structure_uses_natural_first_field_anchor() -> None:
    example = InputExample.from_dict(
        {
            "input": "Alice stores an episodic memory.",
            "output": {
                "json_structures": [
                    {
                        "memory": {
                            "owner": "Alice",
                            "mode": "episodic",
                            "content": "memory",
                        }
                    }
                ]
            },
        }
    )

    assert example.structures[0]._fields["mode"] == "episodic"
    assert example.to_dict()["output"]["record_metadata"] == {
        "memory": {"mode": "natural", "anchor": "owner"}
    }


def test_tolerant_boundary_collator_ignores_missing_entity_surface() -> None:
    processor = SchemaTransformer(
        tokenizer=build_tiny_tokenizer(),
        sampling_config=SamplingConfig(
            remove_entity_prob=0.0,
            shuffle_entities=False,
            synthetic_entity_label_prob=0.0,
        ),
    )
    collator = ExtractorCollator(
        processor,
        architecture="boundary",
        allow_invalid_samples=True,
    )

    batch = collator([(
        "Alice is present",
        {"entities": {"astronomical_object": ["not in this text"]}},
    )])

    assert batch.targets is not None
