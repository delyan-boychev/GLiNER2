#!/usr/bin/env python3
"""Profile the GLiNER2 Transformers encoder on CUDA with real model inputs.

This profiler deliberately stops at ``model.encoder``. Tokenization and schema
construction happen once per case and are excluded from steady-state timings.
The inputs still use GLiNER2's real inference preprocessing, so encoder sequence
length includes both the document and its schema prompts.

The default run:

* sweeps batch size, document length, schema size, and padding distribution;
* compares eager execution with ``torch.compile(dynamic=True)``;
* reports latency percentiles, throughput, peak CUDA memory, and quadratic
  attention work wasted on padding;
* times embeddings and every DeBERTa layer separately in an eager diagnostic;
* writes a Chrome trace containing CPU operators and CUDA kernels for the
  selected case; and
* prints evidence-based optimization candidates instead of changing the model.

Example:

    python benchmarks/profile_encoder_cuda.py \
      --model fastino/gliner2-base-v1 \
      --dtype fp16 \
      --batch-sizes 1,4,16 \
      --text-lengths 32,128,320 \
      --schema-sizes 4,16,32 \
      --padding-profiles uniform,mixed \
      --execution-modes eager,compile \
      --warmup 5 --iterations 20 \
      --trace benchmarks/encoder_cuda_trace.json \
      --output benchmarks/encoder_cuda_profile.json

The trace opens in ``chrome://tracing`` or Perfetto. For an Nsight Systems
capture, use ``--trace-kind nvtx`` so PyTorch's CUPTI profiler does not compete
with Nsight; ranges are named ``gliner2_encoder/...``.
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
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


BASE_DOCUMENTS = (
    "Apple chief executive Tim Cook introduced new hardware at the company's "
    "Cupertino campus while engineers discussed battery life and camera improvements.",
    "Microsoft opened a cloud engineering center in Warsaw after planning the "
    "facility with local universities and public agencies.",
    "Researchers at Stanford University published a study of groundwater levels "
    "using satellite observations and measurements from coastal wells.",
    "The European Central Bank left interest rates unchanged after officials "
    "reviewed inflation, wage growth, and business investment.",
    "A regional hospital expanded its cardiology unit with operating rooms and an "
    "outpatient diagnostics center intended to shorten waiting times.",
    "NVIDIA presented data-center processors during a developer conference in San "
    "Jose, where laboratories described plans for climate and medical research.",
    "A passenger train traveling from Paris to Lyon was delayed after heavy rain "
    "damaged signaling equipment outside Dijon.",
    "Amazon Web Services announced a cloud region in Thailand for banks, retailers, "
    "public agencies, and technology startups.",
)


ENTITY_LABELS = (
    "person", "organization", "location", "date", "product", "event",
    "money", "percentage", "job title", "facility", "law", "language",
    "nationality", "medical condition", "medication", "chemical", "vehicle",
    "artwork", "book", "film", "award", "academic institution",
    "government agency", "sports team", "political party", "technology",
    "scientific concept", "email address", "phone number", "URL", "address",
    "quantity", "duration", "time", "country", "city", "state or province",
    "river", "mountain", "airport", "train station", "company division",
    "research laboratory", "university", "hospital", "currency", "stock symbol",
    "disease", "medical procedure", "gene", "protein", "software",
    "programming language", "hardware", "cloud service", "dataset", "publication",
    "legal case", "contract", "government program", "energy source", "animal",
    "plant", "food",
)


PADDING_FACTORS = {
    "uniform": (1.0,),
    "mixed": (1.0, 0.75, 0.5, 0.25),
    "extreme": (1.0, 0.125, 0.125, 0.125),
}


@dataclass(frozen=True)
class CaseSpec:
    batch_size: int
    text_words: int
    schema_size: int
    padding_profile: str

    @property
    def case_id(self) -> str:
        return (
            f"b{self.batch_size}_w{self.text_words}_q{self.schema_size}_"
            f"{self.padding_profile}"
        )


def parse_positive_csv(value: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def parse_choice_csv(value: str, allowed: Iterable[str]) -> List[str]:
    choices = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed_set = set(allowed)
    invalid = [item for item in choices if item not in allowed_set]
    if not choices or invalid:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated values from {sorted(allowed_set)}; got {invalid}"
        )
    return choices


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_corpus(path: Optional[Path]) -> List[str]:
    if path is None:
        return list(BASE_DOCUMENTS)
    documents = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            value = json.loads(line)
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number} has no non-empty 'text' field")
            documents.append(text.strip())
        else:
            documents.append(line)
    if not documents:
        raise ValueError(f"no documents found in {path}")
    return documents


def resize_document(document: str, target_words: int) -> str:
    words = document.split()
    if not words:
        raise ValueError("cannot resize an empty document")
    repeats = (target_words + len(words) - 1) // len(words)
    return " ".join((words * repeats)[:target_words])


def case_word_lengths(spec: CaseSpec) -> List[int]:
    factors = PADDING_FACTORS[spec.padding_profile]
    return [
        max(4, int(round(spec.text_words * factors[index % len(factors)])))
        for index in range(spec.batch_size)
    ]


def schema_labels(size: int) -> List[str]:
    labels = list(ENTITY_LABELS[:size])
    while len(labels) < size:
        labels.append(f"custom entity type {len(labels) + 1}")
    return labels


def extract_hidden(output: Any) -> torch.Tensor:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"encoder returned unsupported output type {type(output).__name__}")


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def mib(value: int) -> float:
    return value / 2**20


def summarize_samples(samples_ms: Sequence[float]) -> Dict[str, float]:
    return {
        "median_ms": statistics.median(samples_ms),
        "p90_ms": percentile(samples_ms, 0.90),
        "p95_ms": percentile(samples_ms, 0.95),
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
    }


def build_case_inputs(model, corpus: Sequence[str], spec: CaseSpec, device: torch.device):
    word_lengths = case_word_lengths(spec)
    texts = [
        resize_document(corpus[index % len(corpus)], word_lengths[index])
        for index in range(spec.batch_size)
    ]
    schema = model.create_schema().entities(schema_labels(spec.schema_size))
    schemas = [schema] * spec.batch_size
    schema_dicts, _ = model._build_schema_dicts_and_metadata(schemas)
    batch = model.processor.collate_fn_inference(
        list(zip(texts, schema_dicts)),
        architecture=model.architecture,
        error_policy="raise",
    )
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    valid_lengths = attention_mask.sum(dim=1).to(dtype=torch.int64).cpu().tolist()
    encoded_length = int(input_ids.shape[1])
    total_slots = spec.batch_size * encoded_length
    valid_tokens = int(sum(valid_lengths))
    attention_capacity = spec.batch_size * encoded_length * encoded_length
    effective_attention = sum(length * length for length in valid_lengths)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "requested_word_lengths": word_lengths,
        "text_word_counts": list(batch.text_word_counts),
        "valid_lengths": valid_lengths,
        "encoded_length": encoded_length,
        "valid_tokens": valid_tokens,
        "padded_tokens": total_slots,
        "padding_ratio": 1.0 - valid_tokens / total_slots,
        "attention_padding_waste_ratio": 1.0 - effective_attention / attention_capacity,
    }


def timed_forward(encoder, input_ids, attention_mask, device: torch.device):
    synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    hidden = extract_hidden(encoder(input_ids=input_ids, attention_mask=attention_mask))
    end_event.record()
    synchronize(device)
    wall_ms = (time.perf_counter() - wall_start) * 1_000
    cuda_ms = float(start_event.elapsed_time(end_event))
    return hidden, wall_ms, cuda_ms


def measure_case(
    encoder,
    inputs: Mapping[str, Any],
    spec: CaseSpec,
    mode: str,
    warmup: int,
    iterations: int,
    device: torch.device,
    log_every_run: bool,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    hidden, first_wall_ms, first_cuda_ms = timed_forward(
        encoder, input_ids, attention_mask, device
    )
    print(
        f"  [{mode} {spec.case_id}] first-call "
        f"wall={first_wall_ms:.3f} ms cuda={first_cuda_ms:.3f} ms",
        flush=True,
    )
    del hidden

    for index in range(warmup):
        hidden, wall_ms, cuda_ms = timed_forward(
            encoder, input_ids, attention_mask, device
        )
        if log_every_run:
            print(
                f"  [{mode} {spec.case_id}] warmup {index + 1}/{warmup} "
                f"wall={wall_ms:.3f} ms cuda={cuda_ms:.3f} ms",
                flush=True,
            )
        del hidden

    synchronize(device)
    gc.collect()
    base_allocated = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    wall_samples = []
    cuda_samples = []
    final_hidden = None
    for index in range(iterations):
        if final_hidden is not None:
            del final_hidden
        hidden, wall_ms, cuda_ms = timed_forward(
            encoder, input_ids, attention_mask, device
        )
        wall_samples.append(wall_ms)
        cuda_samples.append(cuda_ms)
        final_hidden = hidden
        if log_every_run:
            print(
                f"  [{mode} {spec.case_id}] measure {index + 1}/{iterations} "
                f"wall={wall_ms:.3f} ms cuda={cuda_ms:.3f} ms",
                flush=True,
            )

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    wall_summary = summarize_samples(wall_samples)
    cuda_summary = summarize_samples(cuda_samples)
    median_cuda_ms = cuda_summary["median_ms"]
    result = {
        "case_id": spec.case_id,
        "mode": mode,
        "spec": asdict(spec),
        "input": {
            key: inputs[key]
            for key in (
                "requested_word_lengths", "text_word_counts", "valid_lengths",
                "encoded_length", "valid_tokens", "padded_tokens", "padding_ratio",
                "attention_padding_waste_ratio",
            )
        },
        "first_call": {"wall_ms": first_wall_ms, "cuda_ms": first_cuda_ms},
        "wall": {**wall_summary, "samples_ms": wall_samples},
        "cuda": {**cuda_summary, "samples_ms": cuda_samples},
        "documents_per_second": spec.batch_size * 1_000 / median_cuda_ms,
        "valid_tokens_per_second": inputs["valid_tokens"] * 1_000 / median_cuda_ms,
        "padded_tokens_per_second": inputs["padded_tokens"] * 1_000 / median_cuda_ms,
        "peak_allocated_delta_mib": mib(max(0, peak_allocated - base_allocated)),
        "peak_reserved_delta_mib": mib(max(0, peak_reserved - base_reserved)),
        "hidden_shape": list(final_hidden.shape),
        "hidden_dtype": str(final_hidden.dtype),
    }
    return result, final_hidden


def encoder_modules(encoder) -> List[Tuple[str, torch.nn.Module]]:
    modules = []
    embeddings = getattr(encoder, "embeddings", None)
    if embeddings is not None:
        modules.append(("embeddings", embeddings))
    stack = getattr(encoder, "encoder", None)
    layers = getattr(stack, "layer", None)
    if layers is not None:
        modules.extend((f"layer_{index:02d}", layer) for index, layer in enumerate(layers))
    return modules


def profile_layers(
    encoder,
    inputs: Mapping[str, Any],
    warmup_steps: int,
    steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    modules = encoder_modules(encoder)
    if not modules:
        return {"available": False, "reason": "encoder modules were not recognized"}

    for _ in range(warmup_steps):
        hidden = extract_hidden(encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ))
        del hidden
    synchronize(device)

    event_pairs: Dict[str, List[List[Optional[torch.cuda.Event]]]] = defaultdict(list)
    handles = []

    def make_pre_hook(name):
        def pre_hook(_module, _args):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            event_pairs[name].append([start, None])
        return pre_hook

    def make_post_hook(name):
        def post_hook(_module, _args, _output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            event_pairs[name][-1][1] = end
        return post_hook

    for name, module in modules:
        handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
        handles.append(module.register_forward_hook(make_post_hook(name)))

    total_pairs = []
    try:
        for _ in range(steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            hidden = extract_hidden(encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ))
            end.record()
            total_pairs.append((start, end))
            del hidden
        synchronize(device)
    finally:
        for handle in handles:
            handle.remove()

    total_samples = [float(start.elapsed_time(end)) for start, end in total_pairs]
    total_median = statistics.median(total_samples)
    rows = []
    attributed = 0.0
    for name, _ in modules:
        samples = [
            float(start.elapsed_time(end))
            for start, end in event_pairs[name]
            if end is not None
        ]
        row = {"name": name, **summarize_samples(samples), "samples_ms": samples}
        row["share_of_encoder"] = row["median_ms"] / total_median
        attributed += row["median_ms"]
        rows.append(row)
    return {
        "available": True,
        "warmup_steps": warmup_steps,
        "steps": steps,
        "encoder": {**summarize_samples(total_samples), "samples_ms": total_samples},
        "modules": rows,
        "unattributed_median_ms": max(0.0, total_median - attributed),
    }


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def event_device_time_us(event: Any, self_time: bool = True) -> float:
    names = (
        ("self_device_time_total", "self_cuda_time_total")
        if self_time else
        ("device_time_total", "cuda_time_total")
    )
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def profiler_summaries(profiler, operator_limit: int, kernel_limit: int):
    operator_rows = []
    averages = profiler.key_averages(group_by_input_shape=True)
    for event in averages:
        self_device_us = event_device_time_us(event, self_time=True)
        device_us = event_device_time_us(event, self_time=False)
        if self_device_us <= 0.0 and device_us <= 0.0:
            continue
        operator_rows.append({
            "name": str(event.key),
            "calls": int(event.count),
            "self_cuda_ms": self_device_us / 1_000,
            "total_cuda_ms": device_us / 1_000,
            "self_cpu_ms": float(getattr(event, "self_cpu_time_total", 0.0)) / 1_000,
            "self_cuda_memory_mib": mib(int(getattr(event, "self_device_memory_usage", 0))),
            "input_shapes": getattr(event, "input_shapes", None),
        })
    operator_rows.sort(key=lambda row: row["self_cuda_ms"], reverse=True)

    kernels: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "total_cuda_ms": 0.0, "maximum_cuda_ms": 0.0}
    )
    for event in profiler.events():
        for kernel in getattr(event, "kernels", ()):
            name = str(getattr(kernel, "name", "unknown kernel"))
            duration_ms = float(getattr(kernel, "duration", 0.0)) / 1_000
            kernels[name]["calls"] += 1
            kernels[name]["total_cuda_ms"] += duration_ms
            kernels[name]["maximum_cuda_ms"] = max(
                kernels[name]["maximum_cuda_ms"], duration_ms
            )
    kernel_rows = []
    for name, row in kernels.items():
        calls = int(row["calls"])
        kernel_rows.append({
            "name": name,
            "calls": calls,
            "total_cuda_ms": row["total_cuda_ms"],
            "average_cuda_ms": row["total_cuda_ms"] / calls,
            "maximum_cuda_ms": row["maximum_cuda_ms"],
        })
    kernel_rows.sort(key=lambda row: row["total_cuda_ms"], reverse=True)
    return operator_rows[:operator_limit], kernel_rows[:kernel_limit]


def capture_trace(
    encoder,
    inputs: Mapping[str, Any],
    case_id: str,
    mode: str,
    path: Path,
    steps: int,
    with_stack: bool,
    device: torch.device,
    operator_limit: int,
    kernel_limit: int,
) -> Dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile, record_function

    path.parent.mkdir(parents=True, exist_ok=True)
    synchronize(device)
    with torch.inference_mode(), profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
        with_flops=True,
    ) as profiler:
        for index in range(steps):
            range_name = f"gliner2_encoder/{mode}/{case_id}/step_{index + 1}"
            with nvtx_range(range_name), record_function(range_name):
                hidden = extract_hidden(encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                ))
            del hidden
            profiler.step()
        synchronize(device)
    profiler.export_chrome_trace(str(path))
    operators, kernels = profiler_summaries(
        profiler, operator_limit=operator_limit, kernel_limit=kernel_limit
    )

    print("\nTOP CUDA OPERATORS")
    print(f"{'operator':<56} {'calls':>8} {'self ms':>12} {'total ms':>12}")
    for row in operators[:30]:
        print(
            f"{row['name'][:56]:<56} {row['calls']:>8} "
            f"{row['self_cuda_ms']:>12.3f} {row['total_cuda_ms']:>12.3f}"
        )
    print("\nTOP CUDA KERNELS")
    print(f"{'kernel':<72} {'calls':>8} {'total ms':>12} {'avg us':>12}")
    for row in kernels[:30]:
        print(
            f"{row['name'][:72]:<72} {row['calls']:>8} "
            f"{row['total_cuda_ms']:>12.3f} "
            f"{row['average_cuda_ms'] * 1_000:>12.3f}"
        )
    return {
        "kind": "torch",
        "path": str(path),
        "case_id": case_id,
        "mode": mode,
        "steps": steps,
        "operators": operators,
        "kernels": kernels,
    }


def capture_nvtx_replay(
    encoder,
    inputs: Mapping[str, Any],
    case_id: str,
    mode: str,
    steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run NVTX-annotated forwards for an external Nsight Systems capture."""
    synchronize(device)
    with torch.inference_mode():
        for index in range(steps):
            range_name = f"gliner2_encoder/{mode}/{case_id}/step_{index + 1}"
            with nvtx_range(range_name):
                hidden = extract_hidden(encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                ))
            del hidden
        synchronize(device)
    return {
        "kind": "nvtx",
        "path": None,
        "case_id": case_id,
        "mode": mode,
        "steps": steps,
        "operators": [],
        "kernels": [],
    }


def dynamo_counters() -> Dict[str, Dict[str, int]]:
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}
    return {
        str(category): {str(key): int(value) for key, value in counter.items()}
        for category, counter in counters.items()
        if counter
    }


def inductor_metrics() -> Dict[str, Any]:
    try:
        from torch._inductor import metrics
    except Exception:
        return {}
    names = (
        "generated_kernel_count", "generated_cpp_vec_kernel_count",
        "ir_nodes_pre_fusion", "num_bytes_accessed",
    )
    return {
        name: getattr(metrics, name)
        for name in names
        if isinstance(getattr(metrics, name, None), (int, float, str, bool))
    }


def compare_modes(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["mode"]] = row
    comparisons = []
    for case_id, modes in sorted(by_case.items()):
        if "eager" not in modes or "compile" not in modes:
            continue
        eager = modes["eager"]["cuda"]["median_ms"]
        compiled = modes["compile"]["cuda"]["median_ms"]
        comparisons.append({
            "case_id": case_id,
            "eager_median_cuda_ms": eager,
            "compile_median_cuda_ms": compiled,
            "compile_speedup": eager / compiled,
            "parity": modes["compile"].get("eager_parity"),
        })
    return comparisons


def scaling_analysis(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        spec = row["spec"]
        key = (
            row["mode"], spec["batch_size"], spec["schema_size"],
            spec["padding_profile"],
        )
        groups[key].append(row)
    analyses = []
    for key, group in groups.items():
        ordered = sorted(group, key=lambda row: row["input"]["encoded_length"])
        pairs = []
        for left, right in zip(ordered, ordered[1:]):
            left_length = left["input"]["encoded_length"]
            right_length = right["input"]["encoded_length"]
            if right_length <= left_length:
                continue
            left_ms = left["cuda"]["median_ms"]
            right_ms = right["cuda"]["median_ms"]
            exponent = math.log(right_ms / left_ms) / math.log(right_length / left_length)
            pairs.append({
                "from_case": left["case_id"],
                "to_case": right["case_id"],
                "from_encoded_length": left_length,
                "to_encoded_length": right_length,
                "latency_scaling_exponent": exponent,
            })
        if pairs:
            analyses.append({
                "mode": key[0],
                "batch_size": key[1],
                "schema_size": key[2],
                "padding_profile": key[3],
                "pairs": pairs,
            })
    return analyses


def recommendations(
    rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    trace: Optional[Mapping[str, Any]],
) -> List[str]:
    notes = []
    if rows:
        worst_padding = max(rows, key=lambda row: row["input"]["attention_padding_waste_ratio"])
        waste = worst_padding["input"]["attention_padding_waste_ratio"]
        if waste >= 0.20:
            notes.append(
                f"Length-bucket batches: {worst_padding['case_id']} wastes {waste:.1%} "
                "of padded quadratic attention work. Bucket on the final encoded length "
                "(text plus schema), not document words alone."
            )
    if comparisons:
        failed_parity = [
            row["case_id"] for row in comparisons
            if row.get("parity") and not row["parity"].get("allclose", False)
        ]
        if failed_parity:
            notes.append(
                f"Do not enable the compiled encoder yet: eager parity failed for "
                f"{len(failed_parity)} case(s), including {failed_parity[0]}."
            )
        else:
            speedups = [row["compile_speedup"] for row in comparisons]
            median_speedup = statistics.median(speedups)
            if median_speedup >= 1.03:
                notes.append(
                    f"Keep torch.compile for the encoder: median measured speedup is "
                    f"{median_speedup:.3f}x across matched cases."
                )
            else:
                notes.append(
                    f"torch.compile is not a broad encoder win in this sweep "
                    f"(median {median_speedup:.3f}x). Inspect per-case results before "
                    "paying compilation and cache costs."
                )
    if trace:
        kernels = trace.get("kernels", [])
        total_calls = sum(row["calls"] for row in kernels)
        total_ms = sum(row["total_cuda_ms"] for row in kernels)
        if total_calls and total_ms * 1_000 / total_calls < 15.0:
            notes.append(
                f"The traced kernels average {total_ms * 1_000 / total_calls:.1f} us "
                "across the reported kernel set, indicating launch/fusion overhead may "
                "matter. Compare eager and compiled traces before replacing attention."
            )
        operator_names = " ".join(row["name"].lower() for row in trace.get("operators", [])[:20])
        if "softmax" in operator_names or "bmm" in operator_names:
            notes.append(
                "Attention operators are prominent in the trace. Use the length-scaling "
                "results to determine whether sequence/schema shortening or a validated "
                "DeBERTa-specific fused kernel is worth pursuing."
            )
        if "layer_norm" in operator_names or "native_layer_norm" in operator_names:
            notes.append(
                "Layer normalization appears among the top CUDA operators; compilation "
                "and pointwise fusion are the lowest-risk optimization path for it."
            )
    if not notes:
        notes.append(
            "No single bottleneck crossed the built-in heuristics. Start with the top "
            "CUDA operators and kernels in the JSON/trace and optimize only a parity-tested path."
        )
    return notes


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Profile the GLiNER2 Transformers encoder on CUDA.",
    )
    parser.add_argument("--model", default="fastino/gliner2-base-v1")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--batch-sizes", type=parse_positive_csv, default=parse_positive_csv("1,4,16"))
    parser.add_argument(
        "--text-lengths", type=parse_positive_csv, default=parse_positive_csv("32,128,320"),
        help="comma-separated approximate document word counts",
    )
    parser.add_argument("--schema-sizes", type=parse_positive_csv, default=parse_positive_csv("4,16,32"))
    parser.add_argument(
        "--padding-profiles",
        type=lambda value: parse_choice_csv(value, PADDING_FACTORS),
        default=["uniform", "mixed"],
    )
    parser.add_argument(
        "--execution-modes",
        type=lambda value: parse_choice_csv(value, ("eager", "compile")),
        default=["eager", "compile"],
    )
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument(
        "--compile-static", action="store_true",
        help="compile for static shapes instead of the repository's dynamic=True path",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--layer-steps", type=int, default=5)
    parser.add_argument("--trace-steps", type=int, default=3)
    parser.add_argument("--trace-case", default="largest", help="case id or 'largest'")
    parser.add_argument("--trace-mode", choices=("eager", "compile"), default="eager")
    parser.add_argument(
        "--trace-kind", choices=("torch", "nvtx"), default="torch",
        help="torch writes a Chrome trace; nvtx is for an external nsys capture",
    )
    parser.add_argument("--trace", type=Path, default=Path("benchmarks/encoder_cuda_trace.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/encoder_cuda_profile.json"))
    parser.add_argument("--corpus", type=Path, help="plain text or JSONL with a 'text' field")
    parser.add_argument("--operator-limit", type=int, default=100)
    parser.add_argument("--kernel-limit", type=int, default=100)
    parser.add_argument("--with-stack", action="store_true", help="include Python stacks in the trace (large)")
    parser.add_argument("--quiet-runs", action="store_true", help="suppress each warmup/measurement line")
    parser.add_argument("--allow-tf32", action="store_true", help="allow TF32 for FP32 matrix multiplication")
    args = parser.parse_args(argv)
    for name in (
        "warmup", "iterations", "layer_steps", "trace_steps", "operator_limit", "kernel_limit"
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return parser, args


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser, args = parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("this profiler requires a CUDA device")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("this CUDA device does not support BF16")
    if args.trace_mode not in args.execution_modes:
        parser.error("--trace-mode must also be present in --execution-modes")

    # The legacy flag belongs to the old optional backend. This profiler must
    # measure the ordinary Transformers encoder, regardless of the caller's shell.
    legacy_flash = os.environ.pop("USE_FLASHDEBERTA", None)
    if legacy_flash:
        print("Ignoring USE_FLASHDEBERTA: this script profiles the Transformers encoder.")

    from gliner2 import GLiNER2

    device = torch.device("cuda")
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    print(f"Loading {args.model} on CPU, casting to {args.dtype}, then moving to CUDA...", flush=True)
    model = GLiNER2.from_pretrained(args.model).eval()
    model.to(dtype=dtype)
    model.to(device)
    eager_encoder = model.encoder.eval()
    encoder_config = eager_encoder.config
    max_positions = int(getattr(encoder_config, "max_position_embeddings", 0) or 0)
    print(
        f"device={torch.cuda.get_device_name(device)} capability="
        f"{torch.cuda.get_device_capability(device)} encoder={type(eager_encoder).__name__} "
        f"layers={getattr(encoder_config, 'num_hidden_layers', '?')} "
        f"hidden={getattr(encoder_config, 'hidden_size', '?')} "
        f"heads={getattr(encoder_config, 'num_attention_heads', '?')} "
        f"max_positions={max_positions or 'unspecified'}",
        flush=True,
    )

    corpus = load_corpus(args.corpus)
    specs = [
        CaseSpec(batch_size, text_words, schema_size, padding_profile)
        for batch_size in args.batch_sizes
        for text_words in args.text_lengths
        for schema_size in args.schema_sizes
        for padding_profile in args.padding_profiles
    ]
    case_inputs: Dict[str, Dict[str, Any]] = {}
    skipped = []
    print(f"Preparing {len(specs)} real GLiNER2 input cases...", flush=True)
    for spec in specs:
        inputs = build_case_inputs(model, corpus, spec, device)
        if max_positions and inputs["encoded_length"] > max_positions:
            reason = (
                f"encoded length {inputs['encoded_length']} exceeds configured "
                f"max_position_embeddings={max_positions}"
            )
            skipped.append({"case_id": spec.case_id, "reason": reason})
            print(f"  SKIP {spec.case_id}: {reason}", flush=True)
            del inputs
            continue
        case_inputs[spec.case_id] = inputs
        print(
            f"  {spec.case_id}: encoded={inputs['encoded_length']} "
            f"valid={inputs['valid_lengths']} padding={inputs['padding_ratio']:.1%} "
            f"attention_waste={inputs['attention_padding_waste_ratio']:.1%}",
            flush=True,
        )
    valid_specs = [spec for spec in specs if spec.case_id in case_inputs]
    if not valid_specs:
        parser.error("all cases exceeded the encoder's configured maximum length")

    trace_spec = max(
        valid_specs,
        key=lambda spec: spec.batch_size * case_inputs[spec.case_id]["encoded_length"] ** 2,
    ) if args.trace_case == "largest" else next(
        (spec for spec in valid_specs if spec.case_id == args.trace_case), None
    )
    if trace_spec is None:
        parser.error(
            f"unknown or skipped --trace-case {args.trace_case!r}; valid ids: "
            + ", ".join(spec.case_id for spec in valid_specs)
        )
    print(f"Layer/trace case: {trace_spec.case_id}", flush=True)

    with torch.inference_mode():
        print("\nEAGER PER-LAYER DIAGNOSTIC", flush=True)
        layer_profile = profile_layers(
            eager_encoder, case_inputs[trace_spec.case_id], args.warmup,
            args.layer_steps, device
        )
    if layer_profile.get("available"):
        print(f"{'module':<16} {'median ms':>12} {'p95 ms':>12} {'share':>10}")
        for row in layer_profile["modules"]:
            print(
                f"{row['name']:<16} {row['median_ms']:>12.3f} "
                f"{row['p95_ms']:>12.3f} {row['share_of_encoder']:>9.1%}"
            )

    encoders = {"eager": eager_encoder}
    if "compile" in args.execution_modes:
        if not hasattr(torch, "compile"):
            parser.error("this PyTorch build has no torch.compile")
        print(
            f"\nCreating compiled encoder: dynamic={not args.compile_static} "
            f"mode={args.compile_mode}",
            flush=True,
        )
        try:
            from torch._dynamo.utils import counters
            counters.clear()
        except Exception:
            pass
        try:
            from torch._inductor import metrics
            metrics.reset()
        except Exception:
            pass
        encoders["compile"] = torch.compile(
            eager_encoder,
            dynamic=not args.compile_static,
            mode=args.compile_mode,
        )

    rows = []
    for mode in args.execution_modes:
        encoder = encoders[mode]
        print(f"\n=== {mode.upper()} SWEEP ({len(valid_specs)} cases) ===", flush=True)
        for case_index, spec in enumerate(valid_specs, start=1):
            inputs = case_inputs[spec.case_id]
            print(
                f"\n[{case_index}/{len(valid_specs)}] {spec.case_id} "
                f"shape=({spec.batch_size}, {inputs['encoded_length']})",
                flush=True,
            )
            with torch.inference_mode():
                row, final_hidden = measure_case(
                    encoder, inputs, spec, mode, args.warmup, args.iterations,
                    device, log_every_run=not args.quiet_runs,
                )
                if mode == "compile":
                    eager_hidden = extract_hidden(eager_encoder(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    ))
                    tolerance = {
                        "fp16": {"atol": 2e-3, "rtol": 1e-2},
                        "bf16": {"atol": 4e-3, "rtol": 2e-2},
                        "fp32": {"atol": 1e-5, "rtol": 1e-4},
                    }[args.dtype]
                    difference = (eager_hidden.float() - final_hidden.float()).abs()
                    row["eager_parity"] = {
                        "allclose": bool(torch.allclose(
                            eager_hidden.float(), final_hidden.float(), **tolerance
                        )),
                        "max_abs_error": float(difference.max().item()),
                        "mean_abs_error": float(difference.mean().item()),
                        "tolerance": tolerance,
                    }
                    del eager_hidden, difference
                del final_hidden
            rows.append(row)
            print(
                f"  RESULT median={row['cuda']['median_ms']:.3f} ms "
                f"p95={row['cuda']['p95_ms']:.3f} ms "
                f"docs/s={row['documents_per_second']:.1f} "
                f"valid_tokens/s={row['valid_tokens_per_second']:.0f} "
                f"peak_alloc_delta={row['peak_allocated_delta_mib']:.1f} MiB",
                flush=True,
            )

    trace_inputs = case_inputs[trace_spec.case_id]
    print(
        f"\nCapturing {args.trace_kind} {args.trace_mode} CUDA trace for "
        f"{trace_spec.case_id} "
        f"({args.trace_steps} steps)...",
        flush=True,
    )
    if args.trace_kind == "torch":
        trace = capture_trace(
            encoders[args.trace_mode], trace_inputs, trace_spec.case_id,
            args.trace_mode, args.trace, args.trace_steps, args.with_stack, device,
            args.operator_limit, args.kernel_limit,
        )
    else:
        trace = capture_nvtx_replay(
            encoders[args.trace_mode], trace_inputs, trace_spec.case_id,
            args.trace_mode, args.trace_steps, device,
        )

    comparisons = compare_modes(rows)
    scaling = scaling_analysis(rows)
    notes = recommendations(rows, comparisons, trace)
    print("\nOPTIMIZATION CANDIDATES")
    for index, note in enumerate(notes, start=1):
        print(f"  {index}. {note}")

    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"
    device_properties = torch.cuda.get_device_properties(device)
    result = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_mib": mib(device_properties.total_memory),
            "multiprocessor_count": device_properties.multi_processor_count,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "model": {
            "checkpoint": args.model,
            "architecture": model.architecture,
            "encoder_class": type(eager_encoder).__name__,
            "parameter_count": sum(parameter.numel() for parameter in eager_encoder.parameters()),
            "dtype": args.dtype,
            "config": {
                key: getattr(encoder_config, key, None)
                for key in (
                    "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "max_position_embeddings", "relative_attention",
                    "position_buckets", "max_relative_positions", "pos_att_type",
                    "_attn_implementation",
                )
            },
        },
        "settings": {
            "batch_sizes": args.batch_sizes,
            "text_lengths": args.text_lengths,
            "schema_sizes": args.schema_sizes,
            "padding_profiles": args.padding_profiles,
            "execution_modes": args.execution_modes,
            "compile_dynamic": not args.compile_static,
            "compile_mode": args.compile_mode,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "layer_steps": args.layer_steps,
            "trace_steps": args.trace_steps,
            "trace_kind": args.trace_kind,
        },
        "skipped_cases": skipped,
        "cases": rows,
        "mode_comparisons": comparisons,
        "length_scaling": scaling,
        "layer_profile": layer_profile,
        "compile_diagnostics": {
            "dynamo_counters": dynamo_counters(),
            "inductor_metrics": inductor_metrics(),
        } if "compile" in args.execution_modes else None,
        "trace": trace,
        "recommendations": notes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nJSON report: {args.output}")
    if args.trace_kind == "torch":
        print(f"Chrome CUDA trace: {args.trace}")
    else:
        print("NVTX replay complete; the external profiler owns the trace artifact.")
    print(
        "Nsight Systems example:\n"
        "  nsys profile -t cuda,nvtx,osrt,cudnn,cublas --force-overwrite=true "
        "-o encoder_nsys python benchmarks/profile_encoder_cuda.py "
        "--execution-modes eager --trace-mode eager --trace-kind nvtx --quiet-runs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
