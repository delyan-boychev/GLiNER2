"""Dependency-free checks for the CUDA encoder profiler's analysis helpers."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "profile_encoder_cuda.py"
SPEC = importlib.util.spec_from_file_location("profile_encoder_cuda", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_case_word_lengths_expose_padding_profiles():
    uniform = MODULE.CaseSpec(5, 100, 4, "uniform")
    mixed = MODULE.CaseSpec(5, 100, 4, "mixed")
    extreme = MODULE.CaseSpec(5, 100, 4, "extreme")

    assert MODULE.case_word_lengths(uniform) == [100] * 5
    assert MODULE.case_word_lengths(mixed) == [100, 75, 50, 25, 100]
    assert MODULE.case_word_lengths(extreme) == [100, 12, 12, 12, 100]


def test_csv_and_percentile_validation():
    assert MODULE.parse_positive_csv("1, 8,32") == [1, 8, 32]
    assert MODULE.percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_positive_csv("1,0")
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_choice_csv("eager,unknown", ("eager", "compile"))


def test_default_sweep_is_the_full_encoder_evaluation():
    _parser, args = MODULE.parse_args([])

    assert args.batch_sizes == [1, 4, 16]
    assert args.text_lengths == [32, 128, 320]
    assert args.schema_sizes == [4, 16, 32]
    assert args.padding_profiles == ["uniform", "mixed"]
    assert args.execution_modes == ["eager", "compile"]
    assert args.warmup == 5
    assert args.iterations == 20


def _row(case_id, mode, length, latency, padding_waste=0.0):
    return {
        "case_id": case_id,
        "mode": mode,
        "spec": {
            "batch_size": 4,
            "text_words": length,
            "schema_size": 8,
            "padding_profile": "uniform",
        },
        "input": {
            "encoded_length": length,
            "attention_padding_waste_ratio": padding_waste,
        },
        "cuda": {"median_ms": latency},
        "eager_parity": {"allclose": True} if mode == "compile" else None,
    }


def test_mode_comparison_and_length_scaling():
    rows = [
        _row("short", "eager", 64, 2.0),
        _row("short", "compile", 64, 1.0),
        _row("long", "eager", 128, 8.0),
        _row("long", "compile", 128, 4.0),
    ]

    comparisons = MODULE.compare_modes(rows)
    assert [row["compile_speedup"] for row in comparisons] == [2.0, 2.0]

    scaling = MODULE.scaling_analysis(rows)
    eager = next(row for row in scaling if row["mode"] == "eager")
    assert eager["pairs"][0]["latency_scaling_exponent"] == pytest.approx(2.0)


def test_recommendations_are_driven_by_measured_rows():
    rows = [
        _row("padded", "eager", 128, 8.0, padding_waste=0.50),
        _row("padded", "compile", 128, 4.0, padding_waste=0.50),
    ]
    comparisons = MODULE.compare_modes(rows)
    notes = MODULE.recommendations(rows, comparisons, trace=None)

    assert any("Length-bucket" in note for note in notes)
    assert any("torch.compile" in note and "2.000x" in note for note in notes)
