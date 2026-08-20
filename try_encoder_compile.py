#!/usr/bin/env python3
"""Small CUDA smoke test for the GLiNER2 encoder with torch.compile.

Run from the repository root:

    python try_encoder_compile.py

The script loads the checkpoint on CPU, keeps only its Transformers encoder,
moves that encoder to CUDA, and compares three execution paths:

* ``legacy``: ordinary eager PyTorch;
* ``compiled-default``: the repository's current ``torch.compile`` call; and
* ``max-autotune-buckets``: max autotuning on a small set of padded shapes.

For every fixed batch size, each measured run uses a different logical batch.
That exact batch is shared by all three paths, and the script prints latency plus
pairwise max/mean absolute differences without imposing a parity threshold.
Compilation/warmup uses separately generated batches. At the end, checkpointed
JSON and tables summarize only paired measurements. Per-bucket groups report
valid tokens/s and documents/s; the overall comparison reports valid tokens/s.

The comparison holds batch size fixed while varying exact token dimensions
between runs. Eager and repository-default compilation see the exact collated
BxL tensor. Max-autotune sees the same logical tokens padded to the nearest
configured L bucket. There are no separately measured static-shape cases.
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
DEFAULT_LENGTHS = (16, 20, 24, 28, 32, 64, 128, 256, 512)
DEFAULT_SHAPES = tuple(
    (batch, length)
    for batch in DEFAULT_BATCH_SIZES
    for length in DEFAULT_LENGTHS
)
DEFAULT_WARMUP = 2
DEFAULT_RUNS = 10
WARMUP_VARIANT_OFFSET = 1_000_000_000
EXECUTION_PATHS = (
    # name, torch.compile mode, dynamic
    ("legacy", None, None),
    ("compiled-default", "default", True),
    ("max-autotune-buckets", "max-autotune", False),
)
PATH_NAMES = tuple(name for name, _, _ in EXECUTION_PATHS)
NUMERICAL_COMPARISONS = tuple(
    (PATH_NAMES[left], PATH_NAMES[right])
    for left in range(1, len(PATH_NAMES))
    for right in range(left)
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
    embedding_vocab_size: int,
    pad_token_id: int,
    device: torch.device,
    variant: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    # The integers select actual learned rows from the checkpoint's embedding
    # table. We never feed synthetic random floating-point embeddings.
    seed = 17_000 + batch * 1_000 + length + variant
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sample_size = embedding_vocab_size - int(0 <= pad_token_id < embedding_vocab_size)
    if sample_size <= 0:
        raise ValueError("encoder embedding table has no non-padding rows")
    input_ids = torch.randint(
        low=0,
        high=sample_size,
        size=(batch, length),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    if 0 <= pad_token_id < embedding_vocab_size:
        input_ids += (input_ids >= pad_token_id).to(input_ids.dtype)
    attention_mask = torch.ones((batch, length), dtype=torch.long, device=device)

    # ``length`` is the longest document in this collated batch. Keep one
    # document at that length and sample the others near it, producing the
    # ordinary within-batch padding seen during inference.
    randomizer = random.Random(seed)
    valid_lengths = [length]
    for _ in range(batch - 1):
        relative_length = 0.30 + 0.70 * randomizer.betavariate(5.0, 2.0)
        valid_lengths.append(max(1, min(length, round(length * relative_length))))
    randomizer.shuffle(valid_lengths)
    for row, valid_length in enumerate(valid_lengths):
        attention_mask[row, valid_length:] = 0
        input_ids[row, valid_length:] = pad_token_id

    return input_ids, attention_mask, valid_lengths


def nearest_length_bucket(actual_length: int, length_buckets: list[int]) -> int:
    """Return the smallest configured max-autotune bucket that fits a length."""
    return next(
        bucket
        for bucket in sorted(length_buckets)
        if actual_length <= bucket
    )


def comparison_schedule(
    shape_grid: list[tuple[int, int]],
    runs_per_batch: int,
) -> list[dict[str, Any]]:
    """Draw a continuous realistic length workload for every fixed batch size."""
    schedule = []
    for batch in sorted({batch for batch, _ in shape_grid}):
        length_boundaries = sorted({
            length
            for shape_batch, length in shape_grid
            if shape_batch == batch
        })
        minimum_length = min(length_boundaries)
        maximum_length = max(length_boundaries)
        # Center the continuous workload on short 1-2 sentence requests while
        # retaining a long tail that can naturally reach the 512-token bucket.
        median_length = min(max(28, minimum_length), maximum_length)
        randomizer = random.Random(91_000 + batch * 1_000)
        seen_lengths = set()
        for run_number in range(1, runs_per_batch + 1):
            for _ in range(100):
                sampled = round(
                    randomizer.lognormvariate(math.log(median_length), 0.90)
                )
                actual_length = max(
                    minimum_length,
                    min(maximum_length, sampled),
                )
                if actual_length not in seen_lengths:
                    break
            seen_lengths.add(actual_length)
            variant = batch * 1_000_000 + run_number
            schedule.append({
                "batch_size": batch,
                "run_number": run_number,
                "actual_sequence_length": actual_length,
                "bucket_sequence_length": nearest_length_bucket(
                    actual_length,
                    length_boundaries,
                ),
                "variant": variant,
            })
    return schedule


def make_actual_and_bucket_inputs(
    batch: int,
    actual_length: int,
    bucket_length: int,
    embedding_vocab_size: int,
    pad_token_id: int,
    device: torch.device,
    variant: int,
) -> dict[str, Any]:
    """Create one logical batch and its max-autotune bucketed representation."""
    actual_ids, actual_mask, valid_lengths = make_inputs(
        batch,
        actual_length,
        embedding_vocab_size,
        pad_token_id,
        device,
        variant=variant,
    )
    if bucket_length < actual_length:
        raise ValueError(
            f"bucket length {bucket_length} is smaller than actual length {actual_length}"
        )
    if bucket_length == actual_length:
        bucket_ids = actual_ids
        bucket_mask = actual_mask
    else:
        bucket_ids = torch.full(
            (batch, bucket_length),
            fill_value=pad_token_id,
            dtype=actual_ids.dtype,
            device=device,
        )
        bucket_mask = torch.zeros(
            (batch, bucket_length),
            dtype=actual_mask.dtype,
            device=device,
        )
        bucket_ids[:, :actual_length] = actual_ids
        bucket_mask[:, :actual_length] = actual_mask
    return {
        "actual_input_ids": actual_ids,
        "actual_attention_mask": actual_mask,
        "bucket_input_ids": bucket_ids,
        "bucket_attention_mask": bucket_mask,
        "valid_lengths": valid_lengths,
    }


def call_encoder(encoder, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return hidden_state(encoder(input_ids=input_ids, attention_mask=attention_mask))


def timed_call(encoder, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize(input_ids.device)
    started = time.perf_counter()
    output = call_encoder(encoder, input_ids, attention_mask)
    torch.cuda.synchronize(input_ids.device)
    return output, (time.perf_counter() - started) * 1_000.0


def configure_recompile_limits(shape_count: int) -> dict[str, Any]:
    """Allow compiled paths enough cached specializations for the workload."""
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


def warmup_execution_paths(
    execution_paths,
    schedule: list[dict[str, Any]],
    embedding_vocab_size: int,
    pad_token_id: int,
    device: torch.device,
    warmup_calls: int,
    max_positions: int,
) -> dict[str, Any]:
    """Compile/warm observed exact and routed shapes on disjoint batches."""
    report = {
        "calls_per_warmup_batch": warmup_calls,
        "variant_offset": WARMUP_VARIANT_OFFSET,
        "uses_measured_batches": False,
        "paths": {
            name: {"calls": 0, "total_ms": 0.0, "errors": []}
            for name in PATH_NAMES
        },
    }
    if warmup_calls == 0:
        print("\nWARMUP DISABLED; compilation may enter measured timings", flush=True)
        return report

    print(
        "\nCOMPILE / WARMUP\n"
        "  independently generated logical batches; exact paths keep actual L "
        "and max-autotune uses routed bucket L; no measured batch is reused",
        flush=True,
    )
    with torch.inference_mode():
        for warm_number, item in enumerate(schedule, start=1):
            batch = item["batch_size"]
            actual_length = item["actual_sequence_length"]
            bucket_length = item["bucket_sequence_length"]
            if max_positions and bucket_length > max_positions:
                continue
            inputs = make_actual_and_bucket_inputs(
                batch,
                actual_length,
                bucket_length,
                embedding_vocab_size,
                pad_token_id,
                device,
                item["variant"] + WARMUP_VARIANT_OFFSET,
            )
            print(
                f"  warm batch {warm_number}/{len(schedule)}: B={batch} "
                f"actual_L={actual_length} bucket_L={bucket_length}",
                flush=True,
            )
            for path_name, path_encoder, _, _, setup_error in execution_paths:
                if setup_error is not None or path_encoder is None:
                    continue
                use_bucket = path_name == "max-autotune-buckets"
                input_ids = inputs[
                    "bucket_input_ids" if use_bucket else "actual_input_ids"
                ]
                attention_mask = inputs[
                    "bucket_attention_mask"
                    if use_bucket
                    else "actual_attention_mask"
                ]
                input_length = bucket_length if use_bucket else actual_length
                for call_number in range(1, warmup_calls + 1):
                    try:
                        output, elapsed_ms = timed_call(
                            path_encoder,
                            input_ids,
                            attention_mask,
                        )
                        del output
                        row = report["paths"][path_name]
                        row["calls"] += 1
                        row["total_ms"] += elapsed_ms
                        print(
                            f"    {path_name} warm {call_number}/{warmup_calls}: "
                            f"{elapsed_ms:.3f} ms",
                            flush=True,
                        )
                    except Exception as exc:
                        report["paths"][path_name]["errors"].append({
                            "batch_size": batch,
                            "actual_sequence_length": actual_length,
                            "input_sequence_length": input_length,
                            "type": type(exc).__name__,
                            "message": str(exc),
                        })
                        print(
                            f"    ERROR {path_name} warmup: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        break
            del input_ids, attention_mask, inputs
    return report


def geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


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


def summarize_paired_distribution(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate fully paired random runs by fixed B and routed L bucket."""
    paired = [
        record
        for record in records
        if all(
            record.get("paths", {}).get(path_name, {}).get("status") == "ok"
            for path_name in PATH_NAMES
        )
    ]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in paired:
        key = (
            record["batch_size"],
            record["bucket_sequence_length"],
        )
        grouped.setdefault(key, []).append(record)

    summaries = []
    for (batch, length_bucket), rows in sorted(grouped.items()):
        path_rows = {}
        for path_name in PATH_NAMES:
            throughputs = [
                row["paths"][path_name]["valid_tokens_per_second"]
                for row in rows
            ]
            document_rates = [
                row["paths"][path_name]["documents_per_second"]
                for row in rows
            ]
            latencies = [
                row["paths"][path_name]["latency"]["median_ms"]
                for row in rows
            ]
            legacy_latencies = [
                row["paths"]["legacy"]["latency"]["median_ms"]
                for row in rows
            ]
            path_rows[path_name] = {
                "mean_valid_tokens_per_second": statistics.mean(throughputs),
                "mean_documents_per_second": statistics.mean(document_rates),
                "mean_latency_ms": statistics.mean(latencies),
                "geometric_mean_speedup_vs_legacy": geometric_mean([
                    legacy_ms / path_ms
                    for legacy_ms, path_ms in zip(legacy_latencies, latencies)
                ]),
            }
        fastest_path = max(
            PATH_NAMES,
            key=lambda name: path_rows[name]["mean_documents_per_second"],
        )
        summaries.append({
            "batch_size": batch,
            "bucket_sequence_length": length_bucket,
            "paired_runs": len(rows),
            "minimum_actual_length": min(
                row["actual_sequence_length"] for row in rows
            ),
            "maximum_actual_length": max(
                row["actual_sequence_length"] for row in rows
            ),
            "mean_actual_length": statistics.mean(
                row["actual_sequence_length"] for row in rows
            ),
            "mean_actual_padding_ratio": statistics.mean(
                row["actual_padding_ratio"] for row in rows
            ),
            "mean_bucket_padding_ratio": statistics.mean(
                row["bucket_padding_ratio"] for row in rows
            ),
            "fastest_path": fastest_path,
            "paths": path_rows,
        })
    return summaries


def build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    path_summary = {}
    fastest_counts = {name: 0 for name in PATH_NAMES}
    fully_paired_cases = [
        case
        for case in cases
        if all(
            case.get("paths", {}).get(path_name, {}).get("status") == "ok"
            for path_name in PATH_NAMES
        )
    ]
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
            case["paths"][path_name]["latency"]["median_ms"]
            * case["paths"][path_name]["iterations"]
            / 1_000.0
            for case in fully_paired_cases
        )
        total_valid_tokens = sum(
            case["paths"][path_name]["valid_tokens"]
            * case["paths"][path_name]["iterations"]
            for case in fully_paired_cases
        )
        path_summary[path_name] = {
            "completed_cases": len(rows),
            "overall_paired_cases": len(fully_paired_cases),
            "runtime_failures": len(failures),
            "fastest_case_count": fastest_counts[path_name],
            "median_of_case_medians_ms": statistics.median(medians) if medians else None,
            "p95_of_case_medians_ms": percentile(medians, 0.95) if medians else None,
            "geometric_mean_speedup_vs_legacy": geometric_mean(speedups),
            "minimum_speedup_vs_legacy": min(speedups) if speedups else None,
            "maximum_speedup_vs_legacy": max(speedups) if speedups else None,
            "wins_vs_legacy": sum(value > 1.0 for value in speedups),
            "aggregate_valid_tokens_per_second": (
                total_valid_tokens / total_seconds if total_seconds else None
            ),
            "total_first_call_ms": sum(row.get("first_call_ms", 0.0) for row in rows),
        }

    return {
        "total_recorded_batches": len(cases),
        "completed_batches": sum(case.get("status") == "completed" for case in cases),
        "skipped_batches": sum(case.get("status") == "skipped" for case in cases),
        "input_failures": sum(case.get("status") == "error" for case in cases),
        "runtime_failures": sum(
            result.get("status") == "error"
            for case in cases
            for result in case.get("paths", {}).values()
        ),
        "paths": path_summary,
        "numerical_comparisons": summarize_numerical(cases),
        "paired_distribution_groups": summarize_paired_distribution(cases),
    }


def write_report(
    output: Path,
    metadata: dict[str, Any],
    cases: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    report = {
        "status": status,
        "updated_at": utc_now(),
        "metadata": metadata,
        "batch_runs": cases,
        "summary": build_summary(cases),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    return report


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
    print("\nFINAL PAIRED BATCH-RUN RESULTS", flush=True)
    detailed_rows = []
    for case in cases:
        shape = case["case_id"]
        if case.get("status") == "skipped":
            detailed_rows.append([
                shape, "-", "SKIP", "-", "-", "-", "-", "-", "-",
            ])
            continue
        for path_name in PATH_NAMES:
            result = case.get("paths", {}).get(path_name)
            if not result or result.get("status") != "ok":
                detailed_rows.append([
                    shape, path_name, "ERROR", "-", "-", "-", "-", "-", "-",
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
                format_number(result["valid_tokens_per_second"], 0),
                format_number(result["documents_per_second"], 1),
                format_number(result.get("speedup_vs_legacy"), 3),
                format_number(max_abs_legacy, 7),
                format_number(mean_abs_legacy, 9),
            ])
    print_table(
        [
            "batch_run", "path", "status", "ms", "valid_tok/s", "docs/s",
            "x_legacy", "max_abs_legacy", "mean_abs_legacy",
        ],
        detailed_rows,
    )

    print("\nOVERALL VALID-TOKEN THROUGHPUT", flush=True)
    path_rows = []
    legacy_overall = summary["paths"]["legacy"][
        "aggregate_valid_tokens_per_second"
    ]
    for path_name in PATH_NAMES:
        row = summary["paths"][path_name]
        overall_tok_s = row["aggregate_valid_tokens_per_second"]
        overall_speedup = (
            overall_tok_s / legacy_overall
            if overall_tok_s is not None and legacy_overall
            else None
        )
        path_rows.append([
            path_name,
            str(row["overall_paired_cases"]),
            str(row["runtime_failures"]),
            format_number(overall_tok_s, 0),
            format_number(overall_speedup),
        ])
    print_table(
        ["path", "runs", "errors", "valid_tok/s", "x_legacy"],
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
        ["comparison", "runs", "missing", "max_abs", "weighted_mean_abs"],
        numerical_rows,
    )

    print("\nPAIRED THROUGHPUT BY BATCH SIZE AND LENGTH BUCKET", flush=True)
    distribution_rows = []
    for group in summary["paired_distribution_groups"]:
        group_name = (
            f"B{group['batch_size']}/bucket{group['bucket_sequence_length']}"
        )
        paths = group["paths"]
        distribution_rows.append([
            group_name,
            str(group["paired_runs"]),
            (
                f"{group['mean_actual_length']:.1f} "
                f"[{group['minimum_actual_length']}-"
                f"{group['maximum_actual_length']}]"
            ),
            f"{group['mean_actual_padding_ratio']:.1%}",
            f"{group['mean_bucket_padding_ratio']:.1%}",
            format_number(paths["legacy"]["mean_valid_tokens_per_second"], 0),
            format_number(
                paths["compiled-default"]["mean_valid_tokens_per_second"], 0
            ),
            format_number(
                paths["max-autotune-buckets"]["mean_valid_tokens_per_second"], 0
            ),
            format_number(paths["legacy"]["mean_documents_per_second"], 1),
            format_number(
                paths["compiled-default"]["mean_documents_per_second"], 1
            ),
            format_number(
                paths["max-autotune-buckets"]["mean_documents_per_second"], 1
            ),
            group["fastest_path"],
        ])
    print_table(
        [
            "group", "n", "actual_L mean[min-max]", "actual_pad", "bucket_pad",
            "legacy tok/s", "default tok/s", "bucket tok/s",
            "legacy docs/s", "default docs/s", "bucket docs/s",
            "fastest",
        ],
        distribution_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Compare three eager/compiled GLiNER2 encoder paths across CUDA "
            "shapes and save detailed plus aggregate results."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=list(DEFAULT_SHAPES),
        help=(
            "comma-separated BxL routing buckets; B values define fixed-batch "
            "random streams and L values route max-autotune inputs/reporting"
        ),
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="different measured batches generated for each fixed batch size",
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

    # Keep only the encoder. Generated token IDs select real learned embedding
    # rows; the benchmark never substitutes synthetic floating-point inputs.
    encoder = model.encoder.eval()
    embedding_vocab_size = int(encoder.get_input_embeddings().num_embeddings)
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
        f"buckets={shape_string(args.shapes)}",
        flush=True,
    )
    print(
        "Measured workload: random changing lengths only; no separately measured "
        "static-shape cases. Only max-autotune is bucket padded.",
        flush=True,
    )
    print(
        "Inputs: independently sampled IDs selecting real checkpoint embedding "
        f"rows | embedding_rows={embedding_vocab_size} | synthetic_embeds=False",
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
        "warmup": args.warmup,
        "runs_per_batch_size": args.runs,
        "fullgraph": args.fullgraph,
        "comparison_schedule": {
            "type": "fixed_batch_random_token_length",
            "random_batch_sizes": False,
            "random_actual_token_lengths": True,
            "runs_per_batch_size": args.runs,
            "mode_order": "rotated_per_measured_batch",
            "separate_static_measurements": False,
            "exact_length_paths": ["legacy", "compiled-default"],
            "padded_length_path": "max-autotune-buckets",
            "grouping": "batch_size_and_routed_length_bucket",
        },
        "input_distribution": {
            "type": "uniform_checkpoint_embedding_rows",
            "synthetic_floating_point_embeddings": False,
            "embedding_rows": embedding_vocab_size,
            "measured_batches_distinct_from_warmup": True,
            "maximum_length_distribution": "truncated_lognormal",
            "maximum_length_target_median_tokens": 28,
            "maximum_length_log_sigma": 0.90,
            "within_batch_length_distribution": "scaled_beta",
            "attention_mask": "realistic_per_example_lengths",
        },
        "routing_buckets": [
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
            "embedding_vocab_size": embedding_vocab_size,
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

    scheduled_cases = comparison_schedule(args.shapes, args.runs)
    metadata["comparison_schedule"]["batch_runs"] = scheduled_cases
    cases = []
    write_report(args.output, metadata, cases, "running")
    print(f"Checkpoint JSON: {args.output.resolve()}", flush=True)
    metadata["warmup_report"] = warmup_execution_paths(
        execution_paths,
        scheduled_cases,
        embedding_vocab_size,
        pad_token_id,
        device,
        args.warmup,
        max_positions,
    )
    write_report(args.output, metadata, cases, "running")

    total = len(scheduled_cases)
    with torch.inference_mode():
        for case_number, scheduled_case in enumerate(scheduled_cases, start=1):
            batch = scheduled_case["batch_size"]
            actual_length = scheduled_case["actual_sequence_length"]
            bucket_length = scheduled_case["bucket_sequence_length"]
            variant = scheduled_case["variant"]
            case = (
                f"B={batch} run={scheduled_case['run_number']}/{args.runs} "
                f"actual_L={actual_length} bucket_L={bucket_length}"
            )
            print(f"\n[{case_number}/{total}] {case}", flush=True)
            case_record = {
                "case_id": (
                    f"b{batch}_run{scheduled_case['run_number']}_"
                    f"l{actual_length}_bucket{bucket_length}"
                ),
                "batch_size": batch,
                "sequence_length": actual_length,
                "actual_sequence_length": actual_length,
                "bucket_sequence_length": bucket_length,
                "run_number": scheduled_case["run_number"],
                "status": "running",
                "paths": {},
                "numerical_comparisons": [],
            }
            if max_positions and bucket_length > max_positions:
                print(
                    "  SKIP: routed bucket length exceeds "
                    f"max_position_embeddings={max_positions}",
                    flush=True,
                )
                case_record["status"] = "skipped"
                case_record["skip_reason"] = (
                    "routed bucket length exceeds "
                    f"max_position_embeddings={max_positions}"
                )
                cases.append(case_record)
                write_report(args.output, metadata, cases, "running")
                continue

            try:
                inputs = make_actual_and_bucket_inputs(
                    batch,
                    actual_length,
                    bucket_length,
                    embedding_vocab_size,
                    pad_token_id,
                    device,
                    variant,
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
            valid_lengths = inputs["valid_lengths"]
            valid_tokens = int(sum(valid_lengths))
            case_record["valid_lengths"] = valid_lengths
            case_record["valid_tokens"] = valid_tokens
            case_record["actual_padded_tokens"] = batch * actual_length
            case_record["bucket_padded_tokens"] = batch * bucket_length
            case_record["actual_padding_ratio"] = (
                1.0 - valid_tokens / (batch * actual_length)
            )
            case_record["bucket_padding_ratio"] = (
                1.0 - valid_tokens / (batch * bucket_length)
            )
            print(
                f"  exact_input=({batch}, {actual_length}) "
                f"bucket_input=({batch}, {bucket_length}) "
                f"valid_lengths={valid_lengths}",
                flush=True,
            )

            outputs = {}
            order_offset = (scheduled_case["run_number"] - 1) % len(execution_paths)
            measured_path_order = (
                execution_paths[order_offset:] + execution_paths[:order_offset]
            )
            print(
                "  mode_order="
                + " -> ".join(path[0] for path in measured_path_order),
                flush=True,
            )
            for path_name, path_encoder, compile_mode, dynamic, setup_error in measured_path_order:
                print(f"\n  --- {path_name} ---", flush=True)
                use_bucket = path_name == "max-autotune-buckets"
                input_ids = inputs[
                    "bucket_input_ids" if use_bucket else "actual_input_ids"
                ]
                attention_mask = inputs[
                    "bucket_attention_mask"
                    if use_bucket
                    else "actual_attention_mask"
                ]
                input_length = bucket_length if use_bucket else actual_length
                path_result = {
                    "status": "running",
                    "compile_mode": compile_mode or "eager",
                    "dynamic": dynamic,
                    "input_batch_size": batch,
                    "input_sequence_length": input_length,
                    "actual_sequence_length": actual_length,
                    "bucket_sequence_length": bucket_length,
                    "bucketed": use_bucket,
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
                    path_output, elapsed_ms = timed_call(
                        path_encoder, input_ids, attention_mask
                    )
                    latency = latency_summary([elapsed_ms])
                    if use_bucket:
                        path_output = path_output[:, :actual_length]
                    # max-autotune may use CUDA-graph-managed buffers that a
                    # later invocation overwrites. Clone immediately.
                    path_output = path_output.detach().clone()
                    outputs[path_name] = path_output
                    del path_output
                    path_ms = latency["median_ms"]
                    print(
                        f"  RUN {scheduled_case['run_number']}/{args.runs} "
                        f"{path_name}: {path_ms:.3f} ms",
                        flush=True,
                    )
                    path_result.update({
                        "status": "ok",
                        "latency": latency,
                        "iterations": 1,
                        "batch_size": batch,
                        "sequence_length": input_length,
                        "actual_sequence_length": actual_length,
                        "bucket_sequence_length": bucket_length,
                        "valid_tokens": valid_tokens,
                        "documents_per_second": batch * 1_000.0 / path_ms,
                        "tokens_per_second": (
                            batch * input_length * 1_000.0 / path_ms
                        ),
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

            legacy_result = case_record["paths"].get("legacy", {})
            legacy_ms = (
                legacy_result.get("latency", {}).get("median_ms")
                if legacy_result.get("status") == "ok"
                else None
            )
            print("  paired speedups:", flush=True)
            for path_name in PATH_NAMES:
                result = case_record["paths"].get(path_name, {})
                path_ms = result.get("latency", {}).get("median_ms")
                speedup = (
                    legacy_ms / path_ms
                    if legacy_ms is not None and path_ms is not None
                    else None
                )
                result["speedup_vs_legacy"] = speedup
                speedup_text = f"{speedup:.3f}x" if speedup is not None else "-"
                print(f"    {path_name}: {speedup_text}", flush=True)

            print("\n  --- numerical differences ---", flush=True)
            case_record["numerical_comparisons"] = compare_outputs(outputs)

            del outputs
            del input_ids, attention_mask, inputs
            case_record["status"] = "completed"
            cases.append(case_record)
            report = write_report(args.output, metadata, cases, "running")
            print(
                f"  saved {len(cases)}/{total} batch runs to {args.output}",
                flush=True,
            )

    runtime_failures = sum(
        result.get("status") == "error"
        for case_record in cases
        for result in case_record.get("paths", {}).values()
    )
    input_failures = sum(case_record.get("status") == "error" for case_record in cases)
    metadata["completed_at"] = utc_now()
    final_status = (
        "completed_with_errors" if runtime_failures or input_failures else "completed"
    )
    report = write_report(
        args.output,
        metadata,
        cases,
        final_status,
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
