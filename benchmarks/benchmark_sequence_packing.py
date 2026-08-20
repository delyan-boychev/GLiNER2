#!/usr/bin/env python3
"""Alternating A/B benchmark for inference sequence packing.

Measures public-API wall time and CUDA event time separately, verifies
formatted-output parity before timing, and reports median/p90/p95 plus peak
CUDA allocation/reservation. Defaults: 10 warmups and 50 measured calls per
path. Use ``--help`` to select workload, precision, and compile mode.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from gliner2 import GLiNER2, PackingConfig


LABELS = [
    "company", "person", "product", "location", "date", "event", "price",
    "organization", "technology", "job title", "country", "city", "currency",
    "award", "language", "quantity", "duration", "industry", "facility",
    "software", "hardware", "version", "chemical", "disease", "treatment",
    "law", "court", "team", "sport", "book", "film", "artist",
]
SEEDS = [
    "Apple CEO Tim Cook announced the iPhone 15 in Cupertino.",
    "Google introduced Gemini features in Mountain View.",
    "Microsoft released a new Azure service in Seattle.",
    "Nvidia presented new hardware at GTC in San Jose.",
    "Amazon acquired a software company in London.",
    "Meta launched a product on September 12, 2025.",
    "Tesla opened a facility in Berlin.",
    "Adobe released new design software.",
]


def csv_ints(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(len(ordered) - 1, index))]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def make_workload(name: str, batch_size: int) -> List[str]:
    if name == "short_uniform":
        return [SEEDS[index % len(SEEDS)] for index in range(batch_size)]
    if name == "short_uneven":
        return [SEEDS[index % len(SEEDS)] * (1 + index % 5)
                for index in range(batch_size)]
    if name == "long_uniform":
        return [(SEEDS[index % len(SEEDS)] + " ") * 18
                for index in range(batch_size)]
    if name == "one_long_many_short":
        texts = [SEEDS[index % len(SEEDS)] for index in range(batch_size)]
        if texts:
            texts[-1] = (SEEDS[-1] + " ") * 24
        return texts
    raise ValueError(f"unknown workload {name!r}")


def timed_call(model, texts, labels, config, device) -> Tuple[float, Optional[float], int, int]:
    start_event = end_event = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    synchronize(device)
    started = time.perf_counter()
    model.batch_extract_entities(
        texts, labels, batch_size=len(texts), packing_config=config
    )
    if end_event is not None:
        end_event.record()
    synchronize(device)
    wall = time.perf_counter() - started
    if start_event is None:
        return wall, None, 0, 0
    return (
        wall,
        start_event.elapsed_time(end_event) / 1000.0,
        torch.cuda.max_memory_allocated(device),
        torch.cuda.max_memory_reserved(device),
    )


def summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "median_ms": statistics.median(values) * 1000.0,
        "p90_ms": percentile(values, 0.90) * 1000.0,
        "p95_ms": percentile(values, 0.95) * 1000.0,
    }


def benchmark(model, texts, labels, packing, device, warmups, iterations) -> Dict:
    normal = PackingConfig(enabled=False)
    baseline = model.batch_extract_entities(
        texts, labels, batch_size=len(texts), packing_config=normal
    )
    packed = model.batch_extract_entities(
        texts, labels, batch_size=len(texts), packing_config=packing
    )
    if baseline != packed:
        raise AssertionError("formatted-output mismatch before benchmark")

    for iteration in range(warmups):
        order = (normal, packing) if iteration % 2 == 0 else (packing, normal)
        for config in order:
            timed_call(model, texts, labels, config, device)

    measurements = {
        "baseline": {"wall": [], "cuda": [], "allocated": [], "reserved": []},
        "packed": {"wall": [], "cuda": [], "allocated": [], "reserved": []},
    }
    packing_stats = None
    for iteration in range(iterations):
        order = (("baseline", normal), ("packed", packing))
        if iteration % 2:
            order = tuple(reversed(order))
        for name, config in order:
            wall, cuda_time, allocated, reserved = timed_call(
                model, texts, labels, config, device
            )
            measurements[name]["wall"].append(wall)
            if cuda_time is not None:
                measurements[name]["cuda"].append(cuda_time)
                measurements[name]["allocated"].append(allocated)
                measurements[name]["reserved"].append(reserved)
            if name == "packed":
                packing_stats = getattr(model, "_last_packing_stats", None)

    result = {}
    for name in ("baseline", "packed"):
        entry = summary(measurements[name]["wall"])
        entry["documents_per_second"] = (
            len(texts) / statistics.median(measurements[name]["wall"])
        )
        if measurements[name]["cuda"]:
            entry["cuda"] = summary(measurements[name]["cuda"])
            entry["peak_allocated_mb"] = max(measurements[name]["allocated"]) / 2**20
            entry["peak_reserved_mb"] = max(measurements[name]["reserved"]) / 2**20
        result[name] = entry
    result["speedup"] = result["baseline"]["median_ms"] / result["packed"]["median_ms"]
    result["packing_stats"] = asdict(packing_stats) if packing_stats else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fastino/gliner2-base-v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--compile-mode", choices=(
        "eager", "default", "max-autotune-no-cudagraphs"
    ), default="eager")
    parser.add_argument("--batch-sizes", type=csv_ints, default=[4, 8, 16, 32])
    parser.add_argument("--query-counts", type=csv_ints, default=[4, 16, 32, 64, 128])
    parser.add_argument("--workloads", default=(
        "short_uniform,short_uneven,long_uniform,one_long_many_short"
    ))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device(device_name)
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("CPU FP16 is not a representative supported benchmark")
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this CUDA device does not support BF16")

    model = GLiNER2.from_pretrained(args.model).to(device=device, dtype=dtype).eval()
    if args.compile_mode != "eager":
        model.encoder = torch.compile(
            model.encoder, dynamic=True, mode=args.compile_mode
        )

    report = {
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "compile_mode": args.compile_mode,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "conditions": {},
    }
    packing = PackingConfig(enabled=True)
    workloads = [item.strip() for item in args.workloads.split(",") if item.strip()]
    for batch_size in args.batch_sizes:
        for query_count in args.query_counts:
            labels = [LABELS[index % len(LABELS)] + f" {index}"
                      for index in range(query_count)]
            for workload in workloads:
                key = f"{workload}/b{batch_size}/q{query_count}"
                print(f"benchmarking {key}", flush=True)
                report["conditions"][key] = benchmark(
                    model,
                    make_workload(workload, batch_size),
                    labels,
                    packing,
                    device,
                    args.warmups,
                    args.iterations,
                )
                print(json.dumps(report["conditions"][key], indent=2), flush=True)

    payload = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
