#!/usr/bin/env python3
"""Small CUDA smoke test for the GLiNER2 encoder with torch.compile.

Run from the repository root:

    python try_encoder_compile.py

The script loads the checkpoint on CPU, keeps only its Transformers encoder,
moves that encoder to CUDA, and compares four execution paths:

* ``legacy``: ordinary eager PyTorch;
* ``compiled-default``: the repository's current ``torch.compile`` call; and
* ``reduce-overhead``: compilation with CUDA graphs where supported;
* ``max-autotune-buckets``: max autotuning specialized to each supplied shape.

Every shape prints its first-call time, steady-state latency, and pairwise
max/mean absolute differences for all paths. There is deliberately no
numerical threshold or pass/fail decision. At the end it prints detailed and
aggregate tables and writes a checkpointed JSON report. A runtime failure is
reported without hiding the exception or stopping the remaining cases.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import statistics
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


# These are deliberately plain constants so the default experiment is easy to
# edit directly on the GPU machine.
DEFAULT_MODEL = "fastino/gliner2-base-v1"
DEFAULT_BATCH_SIZES = (1, 4, 8, 16, 32)
DEFAULT_LENGTHS = (16, 32, 64, 128, 256, 512)
DEFAULT_SHAPES = tuple(
    (batch, length)
    for batch in DEFAULT_BATCH_SIZES
    for length in DEFAULT_LENGTHS
)
DEFAULT_WARMUP = 2
DEFAULT_RUNS = 10
DEFAULT_STABILITY_RUNS = 2
PADDING_PROFILES = ("none", "mixed", "extreme", "random")
PADDING_FACTORS = {
    "mixed": (1.0, 0.75, 0.5, 0.25),
    "extreme": (1.0, 0.125, 0.125, 0.125),
}
EXECUTION_PATHS = (
    # name, torch.compile mode, dynamic
    ("legacy", None, None),
    ("compiled-default", "default", True),
    ("reduce-overhead", "reduce-overhead", True),
    ("max-autotune-buckets", "max-autotune", False),
)
PATH_NAMES = tuple(name for name, _, _ in EXECUTION_PATHS)
NUMERICAL_COMPARISONS = tuple(
    (PATH_NAMES[left], PATH_NAMES[right])
    for left in range(1, len(PATH_NAMES))
    for right in range(left)
)
STABILITY_PATH_NAMES = ("legacy", "reduce-overhead", "max-autotune-buckets")
STABILITY_NUMERICAL_COMPARISONS = (
    ("reduce-overhead", "legacy"),
    ("max-autotune-buckets", "legacy"),
    ("max-autotune-buckets", "reduce-overhead"),
)
DEFAULT_OUTPUT = Path("encoder_compile_results.json")


def parse_shapes(value: str) -> list[tuple[int, int]]:
    """Parse shapes such as ``1x64,4x256,8x476``."""
    shapes = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        try:
            batch, length = (int(part.strip()) for part in item.split("x", 1))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"invalid shape {raw_item!r}; expected BxL, for example 8x476"
            ) from exc
        if batch <= 0 or length <= 0:
            raise argparse.ArgumentTypeError("batch and length must be positive")
        shapes.append((batch, length))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return shapes


def parse_padding_profiles(value: str) -> list[str]:
    profiles = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in profiles if item not in PADDING_PROFILES]
    if not profiles or invalid:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated values from {PADDING_PROFILES}; got {invalid}"
        )
    return profiles


def shape_string(shapes: Iterable[tuple[int, int]]) -> str:
    return ",".join(f"{batch}x{length}" for batch, length in shapes)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(samples_ms: list[float]) -> dict[str, Any]:
    return {
        "median_ms": statistics.median(samples_ms),
        "p90_ms": percentile(samples_ms, 0.90),
        "p95_ms": percentile(samples_ms, 0.95),
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def hidden_state(output: Any) -> torch.Tensor:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"unsupported encoder output: {type(output).__name__}")


def make_inputs(
    batch: int,
    length: int,
    vocab_size: int,
    pad_token_id: int,
    padding: str,
    device: torch.device,
    variant: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    # A shape-specific seed makes a failed case exactly reproducible while not
    # depending on which cases ran before it.
    generator = torch.Generator(device=device)
    profile_index = PADDING_PROFILES.index(padding)
    seed = 17_000 + batch * 1_000 + length + profile_index * 100_000 + variant
    generator.manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch, length),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    attention_mask = torch.ones((batch, length), dtype=torch.long, device=device)

    if padding in PADDING_FACTORS:
        # Includes a meaningful padded tail even for B=1, and different valid
        # lengths for larger batches. This exercises DeBERTa's mask path while
        # retaining the requested dense BxL tensor shape.
        fractions = PADDING_FACTORS[padding]
        offset = 1 if batch == 1 else 0
        valid_lengths = [
            max(
                1,
                min(length, round(length * fractions[(index + offset) % len(fractions)])),
            )
            for index in range(batch)
        ]
        for row, valid_length in enumerate(valid_lengths):
            attention_mask[row, valid_length:] = 0
            input_ids[row, valid_length:] = pad_token_id
    elif padding == "random":
        randomizer = random.Random(seed)
        valid_lengths = [
            randomizer.randint(min(2, length), length) for _ in range(batch)
        ]
        for row, valid_length in enumerate(valid_lengths):
            attention_mask[row, valid_length:] = 0
            input_ids[row, valid_length:] = pad_token_id
    else:
        valid_lengths = [length] * batch

    return input_ids, attention_mask, valid_lengths


def call_encoder(encoder, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return hidden_state(encoder(input_ids=input_ids, attention_mask=attention_mask))


def timed_call(encoder, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize(input_ids.device)
    started = time.perf_counter()
    output = call_encoder(encoder, input_ids, attention_mask)
    torch.cuda.synchronize(input_ids.device)
    return output, (time.perf_counter() - started) * 1_000.0


def median_latency(
    encoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    warmup: int,
    runs: int,
    label: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    output = None
    for index in range(warmup):
        print(f"  {label} warmup {index + 1}/{warmup}...", end="", flush=True)
        output, elapsed_ms = timed_call(encoder, input_ids, attention_mask)
        print(f" {elapsed_ms:.3f} ms", flush=True)

    samples = []
    for index in range(runs):
        print(f"  {label} run {index + 1}/{runs}...", end="", flush=True)
        output, elapsed_ms = timed_call(encoder, input_ids, attention_mask)
        samples.append(elapsed_ms)
        print(f" {elapsed_ms:.3f} ms", flush=True)

    assert output is not None
    return latency_summary(samples), output


def configure_recompile_limits(shape_count: int) -> dict[str, Any]:
    """Allow the static bucket path to retain one specialization per shape."""
    required = max(8, shape_count + 8)
    settings = {}
    for name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit"):
        if not hasattr(torch._dynamo.config, name):
            continue
        previous = int(getattr(torch._dynamo.config, name))
        requested = max(previous, required)
        if name == "accumulated_recompile_limit":
            requested = max(previous, required * 2)
        try:
            setattr(torch._dynamo.config, name, requested)
            settings[name] = {
                "previous": previous,
                "effective": int(getattr(torch._dynamo.config, name)),
            }
        except Exception as exc:
            settings[name] = {
                "previous": previous,
                "effective": previous,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return settings


def geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def stability_order(
    shapes: list[tuple[int, int]],
    cycle_index: int,
) -> list[tuple[int, int]]:
    """Return a deterministic order that repeatedly crosses bucket sizes."""
    ordered = list(shapes)
    if cycle_index % 4 == 0:
        zigzag = []
        left = 0
        right = len(ordered) - 1
        while left <= right:
            zigzag.append(ordered[left])
            left += 1
            if left <= right:
                zigzag.append(ordered[right])
                right -= 1
        return zigzag
    if cycle_index % 4 == 1:
        return list(reversed(ordered))
    random.Random(91_000 + cycle_index).shuffle(ordered)
    return ordered


def compare_outputs(
    outputs: dict[str, torch.Tensor],
    indent: str = "  ",
    comparison_pairs: tuple[tuple[str, str], ...] = NUMERICAL_COMPARISONS,
) -> list[dict[str, Any]]:
    comparisons = []
    for left_name, right_name in comparison_pairs:
        if left_name not in outputs or right_name not in outputs:
            comparison = {
                "left": left_name,
                "right": right_name,
                "status": "unavailable",
            }
            comparisons.append(comparison)
            print(
                f"{indent}{left_name} vs {right_name}: unavailable because a path failed",
                flush=True,
            )
            continue
        difference = None
        try:
            difference = (
                outputs[left_name].float() - outputs[right_name].float()
            ).abs()
            max_error = float(difference.max().item())
            mean_error = float(difference.mean().item())
            sum_error = float(difference.sum().item())
            comparison = {
                "left": left_name,
                "right": right_name,
                "status": "ok",
                "max_abs_error": max_error,
                "mean_abs_error": mean_error,
                "sum_abs_error": sum_error,
                "numel": difference.numel(),
            }
            comparisons.append(comparison)
            print(
                f"{indent}{left_name} vs {right_name}: "
                f"max_abs={max_error:.6g} | mean_abs={mean_error:.6g}",
                flush=True,
            )
        except Exception as exc:
            comparison = {
                "left": left_name,
                "right": right_name,
                "status": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            comparisons.append(comparison)
            print(
                f"{indent}ERROR comparing {left_name} vs {right_name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            if difference is not None:
                del difference
    return comparisons


def summarize_numerical(
    records: list[dict[str, Any]],
    comparison_pairs: tuple[tuple[str, str], ...] = NUMERICAL_COMPARISONS,
) -> dict[str, Any]:
    comparison_summary = {}
    for left_name, right_name in comparison_pairs:
        key = f"{left_name}_vs_{right_name}"
        rows = [
            comparison
            for record in records
            for comparison in record.get("numerical_comparisons", [])
            if comparison["left"] == left_name
            and comparison["right"] == right_name
            and comparison["status"] == "ok"
        ]
        total_values = sum(row["numel"] for row in rows)
        total_error = sum(row["sum_abs_error"] for row in rows)
        comparison_summary[key] = {
            "left": left_name,
            "right": right_name,
            "completed_cases": len(rows),
            "unavailable_cases": sum(
                1
                for record in records
                for comparison in record.get("numerical_comparisons", [])
                if comparison["left"] == left_name
                and comparison["right"] == right_name
                and comparison["status"] != "ok"
            ),
            "maximum_abs_error": (
                max(row["max_abs_error"] for row in rows) if rows else None
            ),
            "weighted_mean_abs_error": (
                total_error / total_values if total_values else None
            ),
            "compared_values": total_values,
        }
    return comparison_summary


def summarize_reduce_vs_buckets(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        record
        for record in records
        if record.get("paths", {}).get("reduce-overhead", {}).get("status") == "ok"
        and record.get("paths", {}).get("max-autotune-buckets", {}).get("status")
        == "ok"
    ]
    speedups = [
        record["paths"]["reduce-overhead"]["latency"]["median_ms"]
        / record["paths"]["max-autotune-buckets"]["latency"]["median_ms"]
        for record in usable
    ]
    comparisons = [
        comparison
        for record in records
        for comparison in record.get("numerical_comparisons", [])
        if comparison["left"] == "max-autotune-buckets"
        and comparison["right"] == "reduce-overhead"
        and comparison["status"] == "ok"
    ]
    compared_values = sum(row["numel"] for row in comparisons)
    total_error = sum(row["sum_abs_error"] for row in comparisons)
    return {
        "total_steps": len(records),
        "comparable_steps": len(usable),
        "unavailable_steps": len(records) - len(usable),
        "bucket_wins": sum(value > 1.0 for value in speedups),
        "reduce_overhead_wins": sum(value < 1.0 for value in speedups),
        "ties": sum(value == 1.0 for value in speedups),
        "geometric_mean_bucket_speedup_vs_reduce_overhead": geometric_mean(speedups),
        "median_bucket_speedup_vs_reduce_overhead": (
            statistics.median(speedups) if speedups else None
        ),
        "minimum_bucket_speedup_vs_reduce_overhead": min(speedups) if speedups else None,
        "maximum_bucket_speedup_vs_reduce_overhead": max(speedups) if speedups else None,
        "maximum_abs_error_bucket_vs_reduce_overhead": (
            max(row["max_abs_error"] for row in comparisons)
            if comparisons
            else None
        ),
        "weighted_mean_abs_error_bucket_vs_reduce_overhead": (
            total_error / compared_values if compared_values else None
        ),
        "compared_values": compared_values,
    }


def build_summary(
    cases: list[dict[str, Any]],
    stability_replays: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stability_replays = stability_replays or []
    path_summary = {}
    fastest_counts = {name: 0 for name in PATH_NAMES}
    for case in cases:
        completed = [
            (name, result)
            for name, result in case.get("paths", {}).items()
            if result.get("status") == "ok"
        ]
        if completed:
            fastest_name, _ = min(
                completed,
                key=lambda item: item[1]["latency"]["median_ms"],
            )
            fastest_counts[fastest_name] += 1

    for path_name in PATH_NAMES:
        rows = [
            case["paths"][path_name]
            for case in cases
            if path_name in case.get("paths", {})
            and case["paths"][path_name].get("status") == "ok"
        ]
        failures = [
            case["paths"][path_name]
            for case in cases
            if path_name in case.get("paths", {})
            and case["paths"][path_name].get("status") == "error"
        ]
        medians = [row["latency"]["median_ms"] for row in rows]
        speedups = [
            row["speedup_vs_legacy"]
            for row in rows
            if row.get("speedup_vs_legacy") is not None
        ]
        total_seconds = sum(
            row["latency"]["median_ms"] * row["iterations"] / 1_000.0
            for row in rows
        )
        total_documents = sum(row["batch_size"] * row["iterations"] for row in rows)
        total_padded_tokens = sum(
            row["batch_size"]
            * row["sequence_length"]
            * row["iterations"]
            for row in rows
        )
        total_valid_tokens = sum(
            row["valid_tokens"] * row["iterations"] for row in rows
        )
        path_summary[path_name] = {
            "completed_cases": len(rows),
            "runtime_failures": len(failures),
            "fastest_case_count": fastest_counts[path_name],
            "median_of_case_medians_ms": statistics.median(medians) if medians else None,
            "p95_of_case_medians_ms": percentile(medians, 0.95) if medians else None,
            "geometric_mean_speedup_vs_legacy": geometric_mean(speedups),
            "minimum_speedup_vs_legacy": min(speedups) if speedups else None,
            "maximum_speedup_vs_legacy": max(speedups) if speedups else None,
            "wins_vs_legacy": sum(value > 1.0 for value in speedups),
            "aggregate_documents_per_second": (
                total_documents / total_seconds if total_seconds else None
            ),
            "aggregate_padded_tokens_per_second": (
                total_padded_tokens / total_seconds if total_seconds else None
            ),
            "aggregate_valid_tokens_per_second": (
                total_valid_tokens / total_seconds if total_seconds else None
            ),
            "total_first_call_ms": sum(row["first_call_ms"] for row in rows),
        }

    stability_path_summary = {}
    for path_name in STABILITY_PATH_NAMES:
        rows = [
            replay["paths"][path_name]
            for replay in stability_replays
            if path_name in replay.get("paths", {})
            and replay["paths"][path_name].get("status") == "ok"
        ]
        failures = [
            replay["paths"][path_name]
            for replay in stability_replays
            if path_name in replay.get("paths", {})
            and replay["paths"][path_name].get("status") == "error"
        ]
        medians = [row["latency"]["median_ms"] for row in rows]
        speedups = [
            row["speedup_vs_legacy"]
            for row in rows
            if row.get("speedup_vs_legacy") is not None
        ]
        stability_path_summary[path_name] = {
            "completed_steps": len(rows),
            "runtime_failures": len(failures),
            "median_replay_ms": statistics.median(medians) if medians else None,
            "p95_replay_ms": percentile(medians, 0.95) if medians else None,
            "geometric_mean_speedup_vs_legacy": geometric_mean(speedups),
            "minimum_speedup_vs_legacy": min(speedups) if speedups else None,
            "maximum_speedup_vs_legacy": max(speedups) if speedups else None,
        }

    return {
        "total_recorded_shapes": len(cases),
        "completed_shapes": sum(case.get("status") == "completed" for case in cases),
        "skipped_shapes": sum(case.get("status") == "skipped" for case in cases),
        "input_failures": sum(case.get("status") == "error" for case in cases),
        "runtime_failures": sum(
            result.get("status") == "error"
            for case in cases
            for result in case.get("paths", {}).values()
        ),
        "paths": path_summary,
        "numerical_comparisons": summarize_numerical(cases),
        "stability": {
            "total_steps": len(stability_replays),
            "completed_steps": sum(
                replay.get("status") == "completed" for replay in stability_replays
            ),
            "failed_steps": sum(
                replay.get("status") == "error" for replay in stability_replays
            ),
            "paths": stability_path_summary,
            "numerical_comparisons": summarize_numerical(
                stability_replays,
                STABILITY_NUMERICAL_COMPARISONS,
            ),
            "reduce_overhead_vs_buckets": summarize_reduce_vs_buckets(
                stability_replays
            ),
            "by_padding_profile": {
                profile: summarize_reduce_vs_buckets([
                    replay
                    for replay in stability_replays
                    if replay.get("padding_profile") == profile
                ])
                for profile in PADDING_PROFILES
                if any(
                    replay.get("padding_profile") == profile
                    for replay in stability_replays
                )
            },
        },
    }


def write_report(
    output: Path,
    metadata: dict[str, Any],
    cases: list[dict[str, Any]],
    status: str,
    stability_replays: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stability_replays = stability_replays or []
    report = {
        "status": status,
        "updated_at": utc_now(),
        "metadata": metadata,
        "cases": cases,
        "stability_replays": stability_replays,
        "summary": build_summary(cases, stability_replays),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    return report


def run_stability_phase(
    execution_paths,
    args: argparse.Namespace,
    device: torch.device,
    vocab_size: int,
    pad_token_id: int,
    max_positions: int,
    metadata: dict[str, Any],
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if args.stability_cycles == 0:
        return []

    selected_paths = {
        name: (path_encoder, compile_mode, dynamic, setup_error)
        for name, path_encoder, compile_mode, dynamic, setup_error in execution_paths
        if name in STABILITY_PATH_NAMES
    }
    valid_shapes = [
        shape
        for shape in args.shapes
        if not max_positions or shape[1] <= max_positions
    ]
    stability_replays = []
    total_steps = args.stability_cycles * len(valid_shapes)
    previous_shape = None
    step_number = 0
    print(
        "\nPOST-PRECOMPILE STABILITY REPLAY\n"
        "  comparing legacy, reduce-overhead, and max-autotune-buckets\n"
        "  all static buckets have already been compiled by the main sweep",
        flush=True,
    )

    with torch.inference_mode():
        for cycle_index in range(args.stability_cycles):
            padding_profile = args.stability_padding_profiles[
                cycle_index % len(args.stability_padding_profiles)
            ]
            order = stability_order(valid_shapes, cycle_index)
            print(
                f"\nSTABILITY CYCLE {cycle_index + 1}/{args.stability_cycles} "
                f"padding={padding_profile}",
                flush=True,
            )
            for cycle_step, (batch, length) in enumerate(order, start=1):
                step_number += 1
                shape_name = f"b{batch}_l{length}"
                print(
                    f"\n[stability {step_number}/{total_steps}] {shape_name} "
                    f"after={previous_shape or 'main-sweep'} padding={padding_profile}",
                    flush=True,
                )
                replay = {
                    "replay_id": f"cycle{cycle_index + 1}_step{cycle_step}_{shape_name}",
                    "cycle": cycle_index + 1,
                    "cycle_step": cycle_step,
                    "batch_size": batch,
                    "sequence_length": length,
                    "previous_shape": previous_shape,
                    "padding_profile": padding_profile,
                    "status": "running",
                    "paths": {},
                    "numerical_comparisons": [],
                }
                try:
                    input_ids, attention_mask, valid_lengths = make_inputs(
                        batch,
                        length,
                        vocab_size,
                        pad_token_id,
                        padding_profile,
                        device,
                        variant=cycle_index * 10_000 + cycle_step,
                    )
                except Exception as exc:
                    replay["status"] = "error"
                    replay["input_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    stability_replays.append(replay)
                    write_report(
                        args.output,
                        metadata,
                        cases,
                        "running",
                        stability_replays,
                    )
                    print(f"  INPUT ERROR: {type(exc).__name__}: {exc}", flush=True)
                    previous_shape = shape_name
                    continue

                valid_tokens = int(sum(valid_lengths))
                replay["valid_lengths"] = valid_lengths
                replay["valid_tokens"] = valid_tokens
                replay["padding_ratio"] = 1.0 - valid_tokens / (batch * length)
                print(f"  valid_lengths={valid_lengths}", flush=True)

                outputs = {}
                legacy_ms = None
                for path_name in STABILITY_PATH_NAMES:
                    path_encoder, compile_mode, dynamic, setup_error = selected_paths[
                        path_name
                    ]
                    result = {
                        "status": "running",
                        "compile_mode": compile_mode or "eager",
                        "dynamic": dynamic,
                    }
                    replay["paths"][path_name] = result
                    if setup_error is not None or path_encoder is None:
                        result["status"] = "error"
                        result["error"] = setup_error or {
                            "type": "RuntimeError",
                            "message": "execution path was not created",
                        }
                        print(f"  ERROR {path_name}: path setup failed", flush=True)
                        continue
                    try:
                        latency, path_output = median_latency(
                            path_encoder,
                            input_ids,
                            attention_mask,
                            warmup=0,
                            runs=args.stability_runs,
                            label=f"stability {path_name}",
                        )
                        path_output = path_output.detach().clone()
                        outputs[path_name] = path_output
                        del path_output
                        path_ms = latency["median_ms"]
                        if path_name == "legacy":
                            legacy_ms = path_ms
                        speedup = (
                            legacy_ms / path_ms if legacy_ms is not None else None
                        )
                        result.update({
                            "status": "ok",
                            "latency": latency,
                            "runs": args.stability_runs,
                            "speedup_vs_legacy": speedup,
                            "documents_per_second": batch * 1_000.0 / path_ms,
                            "valid_tokens_per_second": valid_tokens * 1_000.0 / path_ms,
                        })
                        speedup_text = (
                            f" vs_legacy={speedup:.3f}x"
                            if speedup is not None
                            else ""
                        )
                        print(
                            f"  REPLAY {path_name}: median={path_ms:.3f} ms"
                            f"{speedup_text}",
                            flush=True,
                        )
                    except Exception as exc:
                        result["status"] = "error"
                        result["error"] = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                        print(
                            f"  ERROR {path_name}: {type(exc).__name__}: {exc}",
                            flush=True,
                        )

                print("  numerical differences:", flush=True)
                replay["numerical_comparisons"] = compare_outputs(
                    outputs,
                    indent="    ",
                    comparison_pairs=STABILITY_NUMERICAL_COMPARISONS,
                )
                del outputs
                del input_ids, attention_mask
                replay["status"] = "completed"
                stability_replays.append(replay)
                write_report(
                    args.output,
                    metadata,
                    cases,
                    "running",
                    stability_replays,
                )
                previous_shape = shape_name

    return stability_replays


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def print_final_tables(cases: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("\nFINAL PER-SHAPE RESULTS", flush=True)
    detailed_rows = []
    for case in cases:
        shape = case["case_id"]
        if case.get("status") == "skipped":
            detailed_rows.append([
                shape, "-", "SKIP", "-", "-", "-", "-", "-", "-", "-",
            ])
            continue
        for path_name in PATH_NAMES:
            result = case.get("paths", {}).get(path_name)
            if not result or result.get("status") != "ok":
                detailed_rows.append([
                    shape, path_name, "ERROR", "-", "-", "-", "-", "-", "-", "-",
                ])
                continue
            latency = result["latency"]
            if path_name == "legacy":
                max_abs_legacy = 0.0
                mean_abs_legacy = 0.0
            else:
                comparison = next(
                    (
                        item
                        for item in case.get("numerical_comparisons", [])
                        if item["left"] == path_name
                        and item["right"] == "legacy"
                        and item["status"] == "ok"
                    ),
                    None,
                )
                max_abs_legacy = (
                    comparison["max_abs_error"] if comparison is not None else None
                )
                mean_abs_legacy = (
                    comparison["mean_abs_error"] if comparison is not None else None
                )
            detailed_rows.append([
                shape,
                path_name,
                "OK",
                format_number(latency["median_ms"]),
                format_number(latency["p95_ms"]),
                format_number(result["documents_per_second"], 1),
                format_number(result["valid_tokens_per_second"], 0),
                format_number(result.get("speedup_vs_legacy"), 3),
                format_number(max_abs_legacy, 7),
                format_number(mean_abs_legacy, 9),
            ])
    print_table(
        [
            "shape", "path", "status", "median_ms", "p95_ms", "docs/s",
            "valid_tok/s", "x_legacy", "max_abs_legacy", "mean_abs_legacy",
        ],
        detailed_rows,
    )

    print("\nFINAL PATH SUMMARY", flush=True)
    path_rows = []
    for path_name in PATH_NAMES:
        row = summary["paths"][path_name]
        path_rows.append([
            path_name,
            str(row["completed_cases"]),
            str(row["runtime_failures"]),
            str(row["fastest_case_count"]),
            format_number(row["median_of_case_medians_ms"]),
            format_number(row["geometric_mean_speedup_vs_legacy"]),
            format_number(row["aggregate_documents_per_second"], 1),
            format_number(row["aggregate_valid_tokens_per_second"], 0),
        ])
    print_table(
        ["path", "cases", "errors", "fastest", "median_ms", "geo_x_legacy", "docs/s", "valid_tok/s"],
        path_rows,
    )

    print("\nFINAL NUMERICAL SUMMARY", flush=True)
    numerical_rows = []
    for comparison in summary["numerical_comparisons"].values():
        numerical_rows.append([
            f"{comparison['left']} vs {comparison['right']}",
            str(comparison["completed_cases"]),
            str(comparison["unavailable_cases"]),
            format_number(comparison["maximum_abs_error"], 7),
            format_number(comparison["weighted_mean_abs_error"], 9),
        ])
    print_table(
        ["comparison", "cases", "missing", "max_abs", "weighted_mean_abs"],
        numerical_rows,
    )

    stability = summary.get("stability", {})
    if stability.get("total_steps", 0):
        print("\nPOST-PRECOMPILE STABILITY SUMMARY", flush=True)
        stability_rows = []
        for path_name in STABILITY_PATH_NAMES:
            row = stability["paths"][path_name]
            stability_rows.append([
                path_name,
                str(row["completed_steps"]),
                str(row["runtime_failures"]),
                format_number(row["median_replay_ms"]),
                format_number(row["p95_replay_ms"]),
                format_number(row["geometric_mean_speedup_vs_legacy"]),
                format_number(row["minimum_speedup_vs_legacy"]),
            ])
        print_table(
            [
                "path", "steps", "errors", "median_ms", "p95_ms",
                "geo_x_legacy", "min_x_legacy",
            ],
            stability_rows,
        )

        print("\nPOST-PRECOMPILE NUMERICAL SUMMARY", flush=True)
        replay_numerical_rows = []
        for comparison in stability["numerical_comparisons"].values():
            replay_numerical_rows.append([
                f"{comparison['left']} vs {comparison['right']}",
                str(comparison["completed_cases"]),
                str(comparison["unavailable_cases"]),
                format_number(comparison["maximum_abs_error"], 7),
                format_number(comparison["weighted_mean_abs_error"], 9),
            ])
        print_table(
            ["comparison", "steps", "missing", "max_abs", "weighted_mean_abs"],
            replay_numerical_rows,
        )

        print("\nREDUCE-OVERHEAD VS PRECOMPILED BUCKETS", flush=True)
        decision_rows = []
        decision_groups = [
            ("all", stability["reduce_overhead_vs_buckets"]),
            *stability["by_padding_profile"].items(),
        ]
        for profile, row in decision_groups:
            decision_rows.append([
                profile,
                str(row["comparable_steps"]),
                str(row["unavailable_steps"]),
                str(row["bucket_wins"]),
                str(row["reduce_overhead_wins"]),
                format_number(
                    row["geometric_mean_bucket_speedup_vs_reduce_overhead"]
                ),
                format_number(row["minimum_bucket_speedup_vs_reduce_overhead"]),
                format_number(row["maximum_bucket_speedup_vs_reduce_overhead"]),
                format_number(
                    row["maximum_abs_error_bucket_vs_reduce_overhead"], 7
                ),
                format_number(
                    row["weighted_mean_abs_error_bucket_vs_reduce_overhead"], 9
                ),
            ])
        print_table(
            [
                "padding", "steps", "missing", "bucket_wins", "reduce_wins",
                "geo_bucket_x", "min_bucket_x", "max_bucket_x", "max_abs",
                "mean_abs",
            ],
            decision_rows,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Compare four eager/compiled GLiNER2 encoder paths across CUDA "
            "shapes and save detailed plus aggregate results."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=list(DEFAULT_SHAPES),
        help="comma-separated batch-by-sequence shapes, for example 1x64,8x476",
    )
    parser.add_argument("--padding", choices=PADDING_PROFILES, default="mixed")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--stability-padding-profiles",
        type=parse_padding_profiles,
        default=list(PADDING_PROFILES),
        help="valid-token mask profiles replayed after every bucket is compiled",
    )
    parser.add_argument(
        "--stability-cycles",
        type=int,
        default=len(PADDING_PROFILES),
        help="post-precompile bucket-transition cycles; zero disables the phase",
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=DEFAULT_STABILITY_RUNS,
        help="calls per path after each bucket transition",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="require a single graph so graph breaks become explicit failures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="checkpointed JSON report",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.stability_cycles < 0:
        parser.error("--stability-cycles must be non-negative")
    if args.stability_runs <= 0:
        parser.error("--stability-runs must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; no CUDA device is visible to PyTorch")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("this CUDA device does not support BF16")
    if not hasattr(torch, "compile"):
        raise SystemExit("this PyTorch build does not provide torch.compile")

    # This experiment is specifically for the ordinary Transformers encoder.
    # Prevent the legacy environment override from replacing it during load.
    flash_override = os.environ.pop("USE_FLASHDEBERTA", None)
    if flash_override is not None:
        print("Ignoring USE_FLASHDEBERTA for this Transformers encoder test.", flush=True)

    from gliner2 import GLiNER2

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"Loading {args.model} on CPU...", flush=True)
    model = GLiNER2.from_pretrained(args.model).eval()

    # Keep only the encoder before moving anything to CUDA. The task heads and
    # processor are no longer referenced and can be reclaimed on CPU.
    encoder = model.encoder.eval()
    del model
    gc.collect()
    encoder.to(device=device, dtype=dtype)

    config = encoder.config
    vocab_size = int(config.vocab_size)
    pad_token_id = int(getattr(config, "pad_token_id", 0) or 0)
    max_positions = int(getattr(config, "max_position_embeddings", 0) or 0)
    print(
        f"GPU: {torch.cuda.get_device_name(device)} | torch={torch.__version__} | "
        f"encoder={type(encoder).__name__} | dtype={args.dtype}",
        flush=True,
    )
    print(
        f"Paths: {', '.join(PATH_NAMES)} | fullgraph={args.fullgraph} | "
        f"shapes={shape_string(args.shapes)}",
        flush=True,
    )
    print(
        "max-autotune-buckets treats every supplied BxL shape as a static bucket.",
        flush=True,
    )

    recompile_limits = configure_recompile_limits(len(args.shapes))
    if recompile_limits:
        print(f"Compiler cache limits: {recompile_limits}", flush=True)

    execution_paths = []
    for name, compile_mode, dynamic in EXECUTION_PATHS:
        if compile_mode is None:
            execution_paths.append((name, encoder, compile_mode, dynamic, None))
            continue
        compile_kwargs = {
            "dynamic": dynamic,
            "fullgraph": args.fullgraph,
        }
        # Omitting mode exactly mirrors the repository's existing compiled
        # encoder call. Calling it mode="default" would be equivalent.
        if compile_mode != "default":
            compile_kwargs["mode"] = compile_mode
        try:
            compiled_encoder = torch.compile(encoder, **compile_kwargs)
            setup_error = None
        except Exception as exc:
            compiled_encoder = None
            setup_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"ERROR creating {name}: {type(exc).__name__}: {exc}", flush=True)
        execution_paths.append((
            name,
            compiled_encoder,
            compile_mode,
            dynamic,
            setup_error,
        ))

    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"
    device_properties = torch.cuda.get_device_properties(device)
    metadata = {
        "started_at": utc_now(),
        "model": args.model,
        "dtype": args.dtype,
        "padding": args.padding,
        "warmup": args.warmup,
        "iterations": args.runs,
        "fullgraph": args.fullgraph,
        "stability": {
            "cycles": args.stability_cycles,
            "runs_per_transition": args.stability_runs,
            "padding_profiles": args.stability_padding_profiles,
            "paths": list(STABILITY_PATH_NAMES),
            "numerical_comparisons": [
                {"left": left, "right": right}
                for left, right in STABILITY_NUMERICAL_COMPARISONS
            ],
        },
        "requested_shapes": [
            {"batch_size": batch, "sequence_length": length}
            for batch, length in args.shapes
        ],
        "paths": [
            {
                "name": name,
                "compile_mode": compile_mode or "eager",
                "dynamic": dynamic,
                "bucketed": name == "max-autotune-buckets",
            }
            for name, compile_mode, dynamic in EXECUTION_PATHS
        ],
        "encoder": {
            "class": type(encoder).__name__,
            "parameter_count": sum(parameter.numel() for parameter in encoder.parameters()),
            "vocab_size": vocab_size,
            "max_position_embeddings": max_positions,
            "hidden_size": int(getattr(config, "hidden_size", 0) or 0),
            "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0) or 0),
            "num_attention_heads": int(getattr(config, "num_attention_heads", 0) or 0),
        },
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "gpu_total_memory_bytes": int(device_properties.total_memory),
            "gpu_multiprocessor_count": int(device_properties.multi_processor_count),
            "cpu": platform.processor() or platform.machine(),
            "platform": platform.platform(),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "compiler_cache_limits": recompile_limits,
    }

    cases = []
    write_report(args.output, metadata, cases, "running")
    print(f"Checkpoint JSON: {args.output.resolve()}", flush=True)

    total = len(args.shapes)
    with torch.inference_mode():
        for case_number, (batch, length) in enumerate(args.shapes, start=1):
            case = f"B={batch} L={length}"
            print(f"\n[{case_number}/{total}] {case}", flush=True)
            case_record = {
                "case_id": f"b{batch}_l{length}",
                "batch_size": batch,
                "sequence_length": length,
                "status": "running",
                "paths": {},
                "numerical_comparisons": [],
            }
            if max_positions and length > max_positions:
                print(f"  SKIP: length exceeds max_position_embeddings={max_positions}", flush=True)
                case_record["status"] = "skipped"
                case_record["skip_reason"] = (
                    f"sequence length exceeds max_position_embeddings={max_positions}"
                )
                cases.append(case_record)
                write_report(args.output, metadata, cases, "running")
                continue

            try:
                input_ids, attention_mask, valid_lengths = make_inputs(
                    batch,
                    length,
                    vocab_size,
                    pad_token_id,
                    args.padding,
                    device,
                )
            except Exception as exc:
                case_record["status"] = "error"
                case_record["input_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                print(f"  INPUT ERROR: {type(exc).__name__}: {exc}", flush=True)
                cases.append(case_record)
                write_report(args.output, metadata, cases, "running")
                continue
            valid_tokens = int(sum(valid_lengths))
            case_record["valid_lengths"] = valid_lengths
            case_record["valid_tokens"] = valid_tokens
            case_record["padded_tokens"] = batch * length
            case_record["padding_ratio"] = 1.0 - valid_tokens / (batch * length)
            print(
                f"  input_ids={tuple(input_ids.shape)} mask_valid={valid_lengths}",
                flush=True,
            )

            legacy_ms = None
            outputs = {}
            for path_name, path_encoder, compile_mode, dynamic, setup_error in execution_paths:
                print(f"\n  --- {path_name} ---", flush=True)
                path_result = {
                    "status": "running",
                    "compile_mode": compile_mode or "eager",
                    "dynamic": dynamic,
                    "bucket": (
                        {"batch_size": batch, "sequence_length": length}
                        if path_name == "max-autotune-buckets"
                        else None
                    ),
                }
                case_record["paths"][path_name] = path_result
                if setup_error is not None or path_encoder is None:
                    path_result["status"] = "error"
                    path_result["error"] = setup_error or {
                        "type": "RuntimeError",
                        "message": "execution path was not created",
                    }
                    print(f"  ERROR {path_name}: path setup failed", flush=True)
                    continue
                try:
                    compiling = path_name != "legacy"
                    print(
                        f"  {path_name} first call"
                        + (" (may compile/recompile)" if compiling else "")
                        + "...",
                        end="",
                        flush=True,
                    )
                    first_output, first_ms = timed_call(
                        path_encoder, input_ids, attention_mask
                    )
                    print(f" {first_ms:.3f} ms", flush=True)
                    del first_output

                    latency, path_output = median_latency(
                        path_encoder,
                        input_ids,
                        attention_mask,
                        args.warmup,
                        args.runs,
                        path_name,
                    )

                    # reduce-overhead may return CUDA-graph-managed buffers
                    # that a later invocation overwrites. Clone immediately.
                    path_output = path_output.detach().clone()
                    outputs[path_name] = path_output
                    del path_output
                    path_ms = latency["median_ms"]
                    if path_name == "legacy":
                        legacy_ms = path_ms
                    speedup = legacy_ms / path_ms if legacy_ms is not None else None
                    speedup_text = (
                        f" | vs_legacy={speedup:.2f}x" if speedup is not None else ""
                    )
                    print(
                        f"  RESULT {path_name}: first={first_ms:.3f} ms | "
                        f"median={path_ms:.3f} ms | p95={latency['p95_ms']:.3f} ms"
                        f"{speedup_text}",
                        flush=True,
                    )
                    path_result.update({
                        "status": "ok",
                        "first_call_ms": first_ms,
                        "latency": latency,
                        "iterations": args.runs,
                        "batch_size": batch,
                        "sequence_length": length,
                        "valid_tokens": valid_tokens,
                        "speedup_vs_legacy": speedup,
                        "documents_per_second": batch * 1_000.0 / path_ms,
                        "padded_tokens_per_second": batch * length * 1_000.0 / path_ms,
                        "valid_tokens_per_second": valid_tokens * 1_000.0 / path_ms,
                    })
                except Exception as exc:  # one mode must not hide the others
                    path_result["status"] = "error"
                    path_result["error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    print(
                        f"  ERROR {path_name}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)

            print("\n  --- numerical differences ---", flush=True)
            case_record["numerical_comparisons"] = compare_outputs(outputs)

            del outputs
            del input_ids, attention_mask
            case_record["status"] = "completed"
            cases.append(case_record)
            report = write_report(args.output, metadata, cases, "running")
            print(
                f"  saved {len(cases)}/{total} shapes to {args.output}",
                flush=True,
            )

    stability_replays = run_stability_phase(
        execution_paths,
        args,
        device,
        vocab_size,
        pad_token_id,
        max_positions,
        metadata,
        cases,
    )

    runtime_failures = sum(
        result.get("status") == "error"
        for case_record in cases
        for result in case_record.get("paths", {}).values()
    ) + sum(
        result.get("status") == "error"
        for replay in stability_replays
        for result in replay.get("paths", {}).values()
    )
    input_failures = sum(case_record.get("status") == "error" for case_record in cases)
    input_failures += sum(
        replay.get("status") == "error" for replay in stability_replays
    )
    metadata["completed_at"] = utc_now()
    final_status = (
        "completed_with_errors" if runtime_failures or input_failures else "completed"
    )
    report = write_report(
        args.output,
        metadata,
        cases,
        final_status,
        stability_replays,
    )
    print(
        f"\nResults saved to {args.output.resolve()} | "
        f"runtime_failures={runtime_failures} input_failures={input_failures}",
        flush=True,
    )
    print_final_tables(cases, report["summary"])
    return 1 if runtime_failures or input_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
