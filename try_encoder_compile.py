#!/usr/bin/env python3
"""Small CUDA smoke test for the GLiNER2 encoder with torch.compile.

Run from the repository root:

    python try_encoder_compile.py

The script loads the checkpoint on CPU, keeps only its Transformers encoder,
moves that encoder to CUDA, and compares five execution paths:

* ``legacy``: ordinary eager PyTorch;
* ``compiled-default``: the repository's current ``torch.compile`` call; and
* ``reduce-overhead``: compilation with CUDA graphs where supported;
* ``max-autotune-dynamic``: max autotuning with dynamic shapes; and
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
EXECUTION_PATHS = (
    # name, torch.compile mode, dynamic
    ("legacy", None, None),
    ("compiled-default", "default", True),
    ("reduce-overhead", "reduce-overhead", True),
    ("max-autotune-dynamic", "max-autotune", True),
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
    vocab_size: int,
    pad_token_id: int,
    padding: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    # A shape-specific seed makes a failed case exactly reproducible while not
    # depending on which cases ran before it.
    generator = torch.Generator(device=device)
    generator.manual_seed(17_000 + batch * 1_000 + length)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch, length),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    attention_mask = torch.ones((batch, length), dtype=torch.long, device=device)

    if padding == "mixed":
        # Includes a meaningful padded tail even for B=1, and different valid
        # lengths for larger batches. This exercises DeBERTa's mask path while
        # retaining the requested dense BxL tensor shape.
        fractions = (1.0, 0.75, 0.5, 0.25)
        offset = 1 if batch == 1 else 0
        valid_lengths = [
            max(2, round(length * fractions[(index + offset) % len(fractions)]))
            for index in range(batch)
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


def build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
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

    comparison_summary = {}
    for left_name, right_name in NUMERICAL_COMPARISONS:
        key = f"{left_name}_vs_{right_name}"
        rows = [
            comparison
            for case in cases
            for comparison in case.get("numerical_comparisons", [])
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
                for case in cases
                for comparison in case.get("numerical_comparisons", [])
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
        "numerical_comparisons": comparison_summary,
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
        "cases": cases,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Compare five eager/compiled GLiNER2 encoder paths across CUDA "
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
    parser.add_argument("--padding", choices=("none", "mixed"), default="mixed")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
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
            for left_name, right_name in NUMERICAL_COMPARISONS:
                if left_name not in outputs or right_name not in outputs:
                    comparison = {
                        "left": left_name,
                        "right": right_name,
                        "status": "unavailable",
                    }
                    case_record["numerical_comparisons"].append(comparison)
                    print(
                        f"  {left_name} vs {right_name}: unavailable because a path failed",
                        flush=True,
                    )
                    continue
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
                case_record["numerical_comparisons"].append(comparison)
                print(
                    f"  {left_name} vs {right_name}: "
                    f"max_abs={max_error:.6g} | mean_abs={mean_error:.6g}",
                    flush=True,
                )
                del difference

            del outputs
            del input_ids, attention_mask
            case_record["status"] = "completed"
            cases.append(case_record)
            report = write_report(args.output, metadata, cases, "running")
            print(
                f"  saved {len(cases)}/{total} shapes to {args.output}",
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
    report = write_report(args.output, metadata, cases, final_status)
    print_final_tables(cases, report["summary"])
    print(
        f"\nJSON saved to {args.output.resolve()} | "
        f"runtime_failures={runtime_failures} input_failures={input_failures}",
        flush=True,
    )
    return 1 if runtime_failures or input_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
