#!/usr/bin/env python3
"""CUDA parity and E2E benchmark for Transformers vs FlashDeBERTa.

Each backend runs in a fresh process with the same checkpoint, dtype, batches,
warmup count, and compile setting.  The parent compares intermediate tensors,
threshold decisions, formatted documents, padding/batch invariance, latency,
throughput, and peak CUDA memory, then applies the initial acceptance gates.

Examples:
    python benchmarks/compare_flashdeberta.py --dtype both
    python benchmarks/compare_flashdeberta.py --dtype fp16 --compile
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


SEED_DOCUMENTS = [
    "Apple CEO Tim Cook announced the iPhone 15 Pro in Cupertino on September 12, 2023.",
    "Google CEO Sundar Pichai introduced Gemini at a conference in Mountain View.",
    "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque in 1975.",
    "Nvidia CEO Jensen Huang presented a new accelerator during GTC in San Jose.",
    "Amazon announced that its cloud division would open a region in Malaysia next year.",
]

TOLERANCES = {
    "fp16": {"atol": 2e-3, "rtol": 1e-2},
    "bf16": {"atol": 4e-3, "rtol": 2e-2},
}


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(values: Sequence[float]) -> Dict[str, float]:
    return {
        "median_ms": statistics.median(values) * 1000,
        "p90_ms": percentile(values, 0.90) * 1000,
        "p95_ms": percentile(values, 0.95) * 1000,
    }


def synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def make_text(tokenizer, target_tokens: int, salt: int = 0) -> str:
    seed = " ".join(SEED_DOCUMENTS[salt:] + SEED_DOCUMENTS[:salt])
    text = seed
    while len(tokenizer.encode(text)) < target_tokens + 8:
        text += " " + seed
    words = text.split()
    lo, hi = 1, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.encode(" ".join(words[:mid]))) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def build_schemas(model) -> Dict[str, Any]:
    ner = model.create_schema().entities(
        ["company", "person", "product", "location", "date"]
    )
    classification = model.create_schema().classification(
        "business impact", ["positive", "negative", "neutral"]
    )
    relations = model.create_schema().relations(
        ["CEO_of", "founded_by", "located_in", "announced_on"]
    )
    structure = model.create_schema()
    structure.structure("announcement").field("company").field("product").field(
        "person"
    ).field("location").field("date")
    mixed = model.create_schema().entities(
        ["company", "person", "product", "location", "date", "event"]
    )
    mixed.classification("sentiment", ["positive", "negative", "neutral"])
    mixed.structure("launch").field("company").field("product").field(
        "location"
    ).field("date")
    mixed.relations(["CEO_of", "located_in", "announced_on"])
    return {
        "ner": ner,
        "classification": classification,
        "relations": relations,
        "structure": structure,
        "mixed": mixed,
    }


def heterogeneous_texts(tokenizer, length: int, batch_size: int) -> List[str]:
    # Alternating lengths produce high and low padding ratios in the same sweep.
    ratios = (1.0, 0.25, 0.70, 0.12, 0.45, 0.90, 0.33, 0.60)
    return [
        make_text(tokenizer, max(16, int(length * ratios[i % len(ratios)])), salt=i % 5)
        for i in range(batch_size)
    ]


def flatten_embedding_result(value) -> List[Any]:
    import torch

    tensors = []

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensors.append(item.detach().cpu())
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return tensors


class SnapshotCollector:
    def __init__(self, model):
        self.model = model
        self.tensors: Dict[str, List[Any]] = {
            "encoder_hidden_states": [],
            "text_embeddings": [],
            "schema_embeddings": [],
            "raw_task_logits": [],
            "raw_span_logits": [],
            "probabilities": [],
            "threshold_crossings": [],
        }
        self.handles = []
        self.original_extract_embeddings = None
        self.original_span_logits = None

    @staticmethod
    def _tensor(output):
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def __enter__(self):
        import torch

        def encoder_hook(module, args, output):
            tensor = self._tensor(output).detach().cpu()
            self.tensors["encoder_hidden_states"].append(tensor)

        def task_hook(module, args, output):
            tensor = self._tensor(output).detach().cpu()
            self.tensors["raw_task_logits"].append(tensor)
            self.tensors["probabilities"].append(torch.sigmoid(tensor.float()))
            self.tensors["threshold_crossings"].append(
                (torch.sigmoid(tensor.float()) >= 0.5).cpu()
            )

        self.handles.append(self.model.encoder.register_forward_hook(encoder_hook))
        self.handles.append(self.model.classifier.register_forward_hook(task_hook))

        processor = self.model.processor
        self.original_extract_embeddings = processor.extract_embeddings_from_batch

        def extract_embeddings(*args, **kwargs):
            result = self.original_extract_embeddings(*args, **kwargs)
            text_states, schema_states = result
            self.tensors["text_embeddings"].extend(
                flatten_embedding_result(text_states)
            )
            self.tensors["schema_embeddings"].extend(
                flatten_embedding_result(schema_states)
            )
            return result

        processor.extract_embeddings_from_batch = extract_embeddings

        if hasattr(self.model, "_compute_span_logits"):
            self.original_span_logits = self.model._compute_span_logits

            def span_logits(*args, **kwargs):
                output = self.original_span_logits(*args, **kwargs)
                detached = output.detach().cpu()
                self.tensors["raw_span_logits"].append(detached)
                probs = torch.sigmoid(detached.float())
                self.tensors["probabilities"].append(probs)
                self.tensors["threshold_crossings"].append(probs >= 0.5)
                return output

            self.model._compute_span_logits = span_logits
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.model.processor.extract_embeddings_from_batch = (
            self.original_extract_embeddings
        )
        if self.original_span_logits is not None:
            self.model._compute_span_logits = self.original_span_logits


def make_encoder_batch(model, texts: List[str], schema):
    schema_dicts, _ = model._build_schema_dicts_and_metadata([schema] * len(texts))
    if getattr(model, "_inference_collator", None) is None:
        from gliner2.training.trainer import ExtractorCollator

        model._inference_collator = ExtractorCollator(
            model.processor, is_training=False, architecture=model.architecture
        )
    batch = model._inference_collator(list(zip(texts, schema_dicts)))
    dtype = next(model.parameters()).dtype
    return batch.to(torch_device(), dtype if dtype != __import__("torch").float32 else None)


def torch_device():
    import torch

    return torch.device("cuda")


def measure_condition(
    model,
    texts: List[str],
    schema,
    warmup: int,
    measure: int,
    *,
    prefix: str,
) -> Dict[str, Any]:
    import torch

    batch = make_encoder_batch(model, texts, schema)
    for _ in range(warmup):
        with torch.inference_mode():
            model.encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            model.batch_extract(texts, schema, batch_size=len(texts))
    synchronize()

    encoder_times = []
    with torch.inference_mode():
        for _ in range(measure):
            synchronize()
            started = time.perf_counter()
            model.encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            synchronize()
            encoder_times.append(time.perf_counter() - started)

    torch.cuda.reset_peak_memory_stats()
    e2e_times = []
    with torch.inference_mode():
        for _ in range(measure):
            synchronize()
            started = time.perf_counter()
            model.batch_extract(texts, schema, batch_size=len(texts))
            synchronize()
            e2e_times.append(time.perf_counter() - started)

    tokens = sum(len(model.processor.tokenizer.encode(text)) for text in texts)
    median = statistics.median(e2e_times)
    result = {
        "encoder": summarize(encoder_times),
        "e2e": summarize(e2e_times),
        "documents_per_second": len(texts) / median,
        "tokens_per_second": tokens / median,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "encoder_samples_seconds": encoder_times,
        "e2e_samples_seconds": e2e_times,
    }
    print(
        f"[{prefix}] enc median={result['encoder']['median_ms']:.2f} ms | "
        f"e2e median={result['e2e']['median_ms']:.2f} ms "
        f"p90={result['e2e']['p90_ms']:.2f} ms p95={result['e2e']['p95_ms']:.2f} ms | "
        f"{result['documents_per_second']:.2f} docs/s | "
        f"peak={result['peak_allocated_mb']:.1f} MiB",
        flush=True,
    )
    return result


def consistency_checks(model, tokenizer, schema) -> List[str]:
    issues = []
    texts = heterogeneous_texts(tokenizer, 512, 4)
    batched = model.batch_extract(texts, schema, batch_size=4, include_spans=True)
    singles = [
        model.batch_extract([text], schema, batch_size=1, include_spans=True)[0]
        for text in texts
    ]
    if batched != singles:
        issues.append("batch-versus-single formatted output mismatch")

    anchor = texts[0]
    partners = [make_text(tokenizer, 32, 1), make_text(tokenizer, 384, 2)]
    anchor_outputs = [
        model.batch_extract([anchor, partner], schema, batch_size=2, include_spans=True)[0]
        for partner in partners
    ]
    if anchor_outputs[0] != anchor_outputs[1]:
        issues.append("padding/cross-sample contamination detected")
    return issues


def worker(args) -> None:
    import torch
    from gliner2 import GLiNER2

    if not torch.cuda.is_available():
        raise RuntimeError("FlashDeBERTa comparison requires CUDA")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    worker_prefix = f"{args.backend}/{args.dtype}"
    print(
        f"[{worker_prefix}] loading {args.model} on CUDA "
        f"(compile={args.compile})",
        flush=True,
    )
    model = GLiNER2.from_pretrained(
        args.model,
        map_location="cuda",
        dtype=dtype,
        encoder_backend=args.backend,
        compile=args.compile,
    )
    model.eval()
    print(
        f"[{worker_prefix}] loaded backend={model.encoder_backend}; "
        f"GPU={torch.cuda.get_device_name()}; "
        f"reason={model.encoder_backend_reason}",
        flush=True,
    )
    schemas = build_schemas(model)
    tokenizer = model.processor.tokenizer

    parity_specs = [
        ("ner", 64, 4),
        ("classification", 128, 2),
        ("relations", 256, 2),
        ("structure", 512, 1),
        ("mixed", 512, 4),
    ]
    snapshots = {}
    for name, length, batch_size in parity_specs:
        texts = heterogeneous_texts(tokenizer, length, batch_size)
        with SnapshotCollector(model) as collector, torch.inference_mode():
            formatted = model.batch_extract(
                texts, schemas[name], batch_size=batch_size, include_spans=True
            )
        snapshots[f"{name}_l{length}_b{batch_size}"] = {
            "formatted": formatted,
            "tensors": collector.tensors,
        }

    conditions = {}
    for name in ("ner", "classification", "relations", "structure", "mixed"):
        for length in args.lengths:
            for batch_size in args.batch_sizes:
                texts = heterogeneous_texts(tokenizer, length, batch_size)
                key = f"{name}_l{length}_b{batch_size}"
                conditions[key] = measure_condition(
                    model,
                    texts,
                    schemas[name],
                    args.warmup,
                    args.measure,
                    prefix=f"{worker_prefix} {key}",
                )

    consistency_issues = consistency_checks(model, tokenizer, schemas["mixed"])
    payload = {
        "backend": model.encoder_backend,
        "backend_reason": model.encoder_backend_reason,
        "dtype": args.dtype,
        "compile": args.compile,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "conditions": conditions,
        "consistency_issues": consistency_issues,
        "snapshots": snapshots,
    }
    torch.save(payload, args.artifact)


def compare_tensors(standard, flash, dtype: str) -> List[str]:
    import torch

    failures = []
    tolerance = TOLERANCES[dtype]
    for case, standard_snapshot in standard["snapshots"].items():
        flash_snapshot = flash["snapshots"][case]
        if standard_snapshot["formatted"] != flash_snapshot["formatted"]:
            failures.append(f"{case}: formatted output mismatch")
        for name, standard_tensors in standard_snapshot["tensors"].items():
            flash_tensors = flash_snapshot["tensors"][name]
            if len(standard_tensors) != len(flash_tensors):
                failures.append(
                    f"{case}/{name}: tensor count {len(standard_tensors)} != "
                    f"{len(flash_tensors)}"
                )
                continue
            for index, (expected, actual) in enumerate(
                zip(standard_tensors, flash_tensors)
            ):
                if expected.shape != actual.shape:
                    failures.append(
                        f"{case}/{name}[{index}]: shape {tuple(expected.shape)} "
                        f"!= {tuple(actual.shape)}"
                    )
                    continue
                if name == "threshold_crossings":
                    if not torch.equal(expected, actual):
                        failures.append(f"{case}/{name}[{index}]: crossing mismatch")
                elif not torch.allclose(expected.float(), actual.float(), **tolerance):
                    max_error = (expected.float() - actual.float()).abs().max().item()
                    failures.append(
                        f"{case}/{name}[{index}]: max abs error {max_error:.6g}"
                    )
    return failures


def acceptance_report(standard, flash, dtype: str) -> Dict[str, Any]:
    parity_failures = compare_tensors(standard, flash, dtype)
    consistency = standard["consistency_issues"] + flash["consistency_issues"]
    performance_failures = []
    speedups = []
    rows = {}
    for key, baseline in standard["conditions"].items():
        accelerated = flash["conditions"][key]
        base_median = baseline["e2e"]["median_ms"]
        flash_median = accelerated["e2e"]["median_ms"]
        speedup = base_median / flash_median
        speedups.append(speedup)
        samples = baseline["e2e_samples_seconds"]
        med_seconds = statistics.median(samples)
        mad = statistics.median(abs(value - med_seconds) for value in samples)
        noise = max(0.02, (1.4826 * mad / med_seconds) if med_seconds else 0.02)
        p95_ratio = accelerated["e2e"]["p95_ms"] / baseline["e2e"]["p95_ms"]
        allocated_ratio = (
            accelerated["peak_allocated_mb"] / baseline["peak_allocated_mb"]
        )
        reserved_ratio = (
            accelerated["peak_reserved_mb"] / baseline["peak_reserved_mb"]
        )
        if speedup <= 1.0 + noise:
            performance_failures.append(
                f"{key}: {speedup:.3f}x median speedup does not exceed "
                f"{noise:.1%} measured-noise gate"
            )
        if p95_ratio > 1.05:
            performance_failures.append(f"{key}: p95 regression is {p95_ratio:.3f}x")
        if max(allocated_ratio, reserved_ratio) > 1.05:
            performance_failures.append(
                f"{key}: peak-memory regression is "
                f"{max(allocated_ratio, reserved_ratio):.3f}x"
            )
        rows[key] = {
            "standard": baseline,
            "flashdeberta": accelerated,
            "median_speedup": speedup,
            "p95_ratio": p95_ratio,
            "allocated_memory_ratio": allocated_ratio,
            "reserved_memory_ratio": reserved_ratio,
        }
    accepted = not parity_failures and not consistency and not performance_failures
    return {
        "accepted_for_auto": accepted,
        "dtype": dtype,
        "tolerances": TOLERANCES[dtype],
        "formatted_and_tensor_failures": parity_failures,
        "batch_padding_leakage_failures": consistency,
        "performance_failures": performance_failures,
        "overall_median_speedup": statistics.median(speedups),
        "conditions": rows,
    }


def run_backend(args, backend: str, dtype: str, artifact: Path) -> None:
    command = [
        sys.executable,
        __file__,
        "--worker",
        "--backend", backend,
        "--dtype", dtype,
        "--model", args.model,
        "--warmup", str(args.warmup),
        "--measure", str(args.measure),
        "--artifact", str(artifact),
        "--lengths", *[str(value) for value in args.lengths],
        "--batch-sizes", *[str(value) for value in args.batch_sizes],
    ]
    if args.compile:
        command.append("--compile")
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fastino/gliner2-base-v1")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=20)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048]
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--output", default="benchmarks/flashdeberta_comparison.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--backend", choices=("transformers", "flashdeberta"), help=argparse.SUPPRESS
    )
    parser.add_argument("--artifact", help=argparse.SUPPRESS)
    return parser.parse_args()


def print_speedup_tables(reports: Dict[str, Any]) -> None:
    """Print per-dtype, per-schema batch x doc-length median-speedup matrices."""
    for dtype, report in reports.items():
        rows = report["conditions"]
        lengths = sorted({int(key.split("_l")[1].split("_b")[0]) for key in rows})
        batches = sorted({int(key.split("_b")[1]) for key in rows})
        schemas = sorted({key.split("_l")[0] for key in rows})
        print(f"\n=== {dtype}: median speedup (standard / flashdeberta, >1x is faster) ===")
        for schema in schemas:
            print(f"\n  {schema}")
            print(f"  {'bs\\len':<8}" + "".join(f"{length:>8}" for length in lengths))
            for batch in batches:
                cells = []
                for length in lengths:
                    speedup = rows[f"{schema}_l{length}_b{batch}"]["median_speedup"]
                    cells.append(f"{speedup:>7.2f}x")
                print(f"  {batch:<8}" + "".join(cells))


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args)
        return

    import torch

    dtypes = ("fp16", "bf16") if args.dtype == "both" else (args.dtype,)
    reports = {}
    with tempfile.TemporaryDirectory(prefix="gliner2-flashdeberta-") as directory:
        for dtype in dtypes:
            standard_path = Path(directory) / f"transformers-{dtype}.pt"
            flash_path = Path(directory) / f"flashdeberta-{dtype}.pt"
            run_backend(args, "transformers", dtype, standard_path)
            run_backend(args, "flashdeberta", dtype, flash_path)
            standard = torch.load(standard_path, map_location="cpu", weights_only=True)
            flash = torch.load(flash_path, map_location="cpu", weights_only=True)
            reports[dtype] = acceptance_report(standard, flash, dtype)

    print_speedup_tables(reports)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps({key: value["accepted_for_auto"] for key, value in reports.items()}, indent=2))
    print(f"Full comparison written to {output}")
    if not all(report["accepted_for_auto"] for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
