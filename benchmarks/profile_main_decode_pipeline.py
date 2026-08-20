#!/usr/bin/env python3
"""Profile main's CUDA inference pipeline and synchronized entity decoder.

This is intentionally a plain-entity inference workload.  The optimized path
uses the production synchronization-collapsed decoder.  It changes when device
values are read by Python:

* predicted counts are transferred together instead of via one ``.item()`` per
  document;
* already-computed span probabilities are packed and transferred once;
* the existing entity decoder runs on those bit-identical CPU values.

The script asserts exact formatted-output parity before reporting timings.  It
does not modify the model and does not benchmark sequence packing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from gliner2 import GLiNER2
from gliner2.training.trainer import ExtractorCollator


BASE_TEXTS = (
    "Apple CEO Tim Cook met Microsoft executives in London on Monday.",
    "OpenAI announced a new model in San Francisco.",
    "Angela Merkel visited Berlin and met representatives from Siemens.",
    "NVIDIA released new GPU products during its conference in California.",
    "Google acquired a startup founded by John Smith in New York.",
    "Tesla discussed production at the company's Texas factory.",
    "Amazon Web Services opened a new data center in Frankfurt.",
    "Meta researchers published a new AI paper with Stanford University.",
)
DEFAULT_LABELS = "person,organization,location,date,product"


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return ordered[index]


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_lengths(value: str) -> List[int]:
    lengths = [int(item) for item in parse_csv(value)]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be positive integers")
    return lengths


def make_text(source: str, target_words: int) -> str:
    words = source.rstrip(".!?").split()
    repeated = (words * ((target_words + len(words) - 1) // len(words)))[:target_words]
    return " ".join(repeated) + "."


def synchronize() -> None:
    torch.cuda.synchronize()


def memory_now() -> Tuple[int, int]:
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def measure(
    fn: Callable[[], Any],
    *,
    cuda_region: bool,
) -> Tuple[Any, Dict[str, float]]:
    synchronize()
    base_allocated, base_reserved = memory_now()
    torch.cuda.reset_peak_memory_stats()
    start_event = end_event = None
    if cuda_region:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    start = time.perf_counter()
    output = fn()
    if end_event is not None:
        end_event.record()
    synchronize()
    wall_ms = (time.perf_counter() - start) * 1_000
    cuda_ms = start_event.elapsed_time(end_event) if start_event is not None else 0.0
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return output, {
        "wall_ms": wall_ms,
        "cuda_ms": cuda_ms,
        "peak_allocated_delta_mib": max(0, peak_allocated - base_allocated) / 2**20,
        "peak_reserved_delta_mib": max(0, peak_reserved - base_reserved) / 2**20,
    }


def raw_default_decode(
    model,
    batch,
    token_embs,
    schema_embs,
    span_info,
    metadata,
    threshold: float,
    include_confidence: bool,
    include_spans: bool,
):
    results = []
    for index in range(len(batch)):
        results.append(model._extract_sample(
            token_embs=token_embs[index],
            schema_embs=schema_embs[index],
            schema_tokens_list=batch.schema_tokens_list[index],
            task_types=batch.task_types[index],
            text_tokens=batch.text_tokens[index],
            original_text=batch.original_texts[index],
            schema=batch.original_schemas[index],
            start_mapping=batch.start_mappings[index],
            end_mapping=batch.end_mappings[index],
            threshold=threshold,
            metadata=metadata[index],
            include_confidence=include_confidence,
            include_spans=include_spans,
            span_info=span_info[index],
        ))
    return results


def raw_optimized_decode(
    model,
    batch,
    token_embs,
    schema_embs,
    span_info,
    metadata,
    threshold: float,
    include_confidence: bool,
    include_spans: bool,
):
    """Run the production synchronization-collapsed decoder."""
    del token_embs
    return model._extract_from_batch_sync_collapsed(
        batch, schema_embs, span_info, threshold, metadata,
        include_confidence, include_spans,
    )


def format_batch(model, raw, metadata, include_confidence):
    return [
        model.format_results(
            result,
            include_confidence,
            metadata[index].get("relation_order", []),
            metadata[index].get("classification_tasks", []),
        )
        for index, result in enumerate(raw)
    ]


def summarize(samples: Sequence[Dict[str, float]]) -> Dict[str, float]:
    wall = [sample["wall_ms"] for sample in samples]
    cuda = [sample["cuda_ms"] for sample in samples]
    return {
        "median_wall_ms": statistics.median(wall),
        "p90_wall_ms": percentile(wall, 0.90),
        "p95_wall_ms": percentile(wall, 0.95),
        "median_cuda_ms": statistics.median(cuda),
        "peak_allocated_delta_mib": max(
            sample["peak_allocated_delta_mib"] for sample in samples
        ),
        "peak_reserved_delta_mib": max(
            sample["peak_reserved_delta_mib"] for sample in samples
        ),
        "wall_samples_ms": wall,
        "cuda_samples_ms": cuda,
    }


def output_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="fastino/gliner2-base-v1")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--execution-mode", choices=("compile", "eager"), default="compile")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--text-lengths", type=parse_lengths,
        default=parse_lengths("16,32,48,64,80,96,112,128"),
        help="comma-separated approximate word lengths, cycled over the batch",
    )
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("this profiler requires CUDA")
    if args.batch_size <= 0 or args.warmup <= 0 or args.iterations <= 0:
        parser.error("batch-size, warmup, and iterations must be positive")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("this CUDA device does not support BF16")

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    texts = [
        make_text(BASE_TEXTS[index % len(BASE_TEXTS)], args.text_lengths[index % len(args.text_lengths)])
        for index in range(args.batch_size)
    ]
    labels = parse_csv(args.labels)
    if not labels:
        parser.error("--labels must contain at least one label")

    print(
        f"device={torch.cuda.get_device_name()} dtype={args.dtype} "
        f"execution={args.execution_mode} batch={args.batch_size} labels={len(labels)}"
    )
    model = GLiNER2.from_pretrained(
        args.model, map_location="cuda"
    ).to(device="cuda", dtype=dtype).eval()
    if args.execution_mode == "compile":
        model.compile()

    schema = model.create_schema().entities(labels)
    schemas = [schema] * len(texts)
    schema_dicts, metadata = model._build_schema_dicts_and_metadata(schemas)
    dataset = list(zip(texts, schema_dicts))
    collator = ExtractorCollator(
        model.processor, is_training=False, architecture=model.architecture
    )
    include_confidence = True
    include_spans = True

    def prepare():
        batch_cpu = collator(dataset)
        batch = batch_cpu.to("cuda", dtype if dtype != torch.float32 else None)
        encoded = model.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        ).last_hidden_state
        token_embs, schema_embs = model.processor.extract_embeddings_from_batch(
            encoded, batch.input_ids, batch
        )
        span_info = model.compute_span_rep_batched(token_embs)
        return batch, token_embs, schema_embs, span_info

    def run_pipeline(decoder):
        batch_cpu = collator(dataset)
        batch = batch_cpu.to("cuda", dtype if dtype != torch.float32 else None)
        encoded = model.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        ).last_hidden_state
        token_embs, schema_embs = model.processor.extract_embeddings_from_batch(
            encoded, batch.input_ids, batch
        )
        span_info = model.compute_span_rep_batched(token_embs)
        raw = decoder(
            model, batch, token_embs, schema_embs, span_info, metadata,
            args.threshold, include_confidence, include_spans,
        )
        return format_batch(model, raw, metadata, include_confidence)

    # Compile all dynamic subgraphs and warm both decoder paths.
    with torch.inference_mode():
        for _ in range(args.warmup):
            default_output = run_pipeline(raw_default_decode)
            optimized_output = run_pipeline(raw_optimized_decode)
            if default_output != optimized_output:
                raise AssertionError("formatted output mismatch during warmup")
        synchronize()

        prepared = prepare()
        batch, token_embs, schema_embs, span_info = prepared
        print(
            f"encoded_shape={tuple(batch.input_ids.shape)} "
            f"word_lengths={batch.text_word_counts}"
        )

        stage_samples: Dict[str, List[Dict[str, float]]] = {
            name: [] for name in (
                "preprocess", "transfer", "encoder", "route", "span_rep",
                "default_decode", "optimized_decode", "default_format",
                "optimized_format",
            )
        }
        last_default = last_optimized = None
        for iteration in range(args.iterations):
            batch_cpu, timing = measure(lambda: collator(dataset), cuda_region=False)
            stage_samples["preprocess"].append(timing)
            batch, timing = measure(
                lambda: batch_cpu.to(
                    "cuda", dtype if dtype != torch.float32 else None
                ),
                cuda_region=True,
            )
            stage_samples["transfer"].append(timing)
            encoded, timing = measure(
                lambda: model.encoder(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                ).last_hidden_state,
                cuda_region=True,
            )
            stage_samples["encoder"].append(timing)
            routed, timing = measure(
                lambda: model.processor.extract_embeddings_from_batch(
                    encoded, batch.input_ids, batch
                ),
                cuda_region=True,
            )
            stage_samples["route"].append(timing)
            token_embs, schema_embs = routed
            span_info, timing = measure(
                lambda: model.compute_span_rep_batched(token_embs),
                cuda_region=True,
            )
            stage_samples["span_rep"].append(timing)

            decode_order = (
                ("default_decode", raw_default_decode),
                ("optimized_decode", raw_optimized_decode),
            )
            if iteration % 2:
                decode_order = tuple(reversed(decode_order))
            decoded = {}
            for name, decoder in decode_order:
                decoded[name], timing = measure(
                    lambda decoder=decoder: decoder(
                        model, batch, token_embs, schema_embs, span_info,
                        metadata, args.threshold, include_confidence,
                        include_spans,
                    ),
                    cuda_region=True,
                )
                stage_samples[name].append(timing)

            last_default, timing = measure(
                lambda: format_batch(
                    model, decoded["default_decode"], metadata,
                    include_confidence,
                ),
                cuda_region=False,
            )
            stage_samples["default_format"].append(timing)
            last_optimized, timing = measure(
                lambda: format_batch(
                    model, decoded["optimized_decode"], metadata,
                    include_confidence,
                ),
                cuda_region=False,
            )
            stage_samples["optimized_format"].append(timing)
            if last_default != last_optimized:
                raise AssertionError(
                    f"formatted output mismatch at measured iteration {iteration}"
                )

        full_samples = {"default": [], "optimized": []}
        functions = {
            "default": lambda: run_pipeline(raw_default_decode),
            "optimized": lambda: run_pipeline(raw_optimized_decode),
        }
        for iteration in range(args.iterations):
            names = ("default", "optimized") if iteration % 2 == 0 else ("optimized", "default")
            outputs = {}
            for name in names:
                outputs[name], timing = measure(functions[name], cuda_region=True)
                full_samples[name].append(timing)
            if outputs["default"] != outputs["optimized"]:
                raise AssertionError(
                    f"full-pipeline output mismatch at iteration {iteration}"
                )

    stage_summary = {
        name: summarize(samples) for name, samples in stage_samples.items()
    }
    full_summary = {
        name: summarize(samples) for name, samples in full_samples.items()
    }
    default_ms = full_summary["default"]["median_wall_ms"]
    optimized_ms = full_summary["optimized"]["median_wall_ms"]
    full_summary["speedup"] = default_ms / optimized_ms

    print("\nDETAILED STAGES")
    print("stage                 wall med    CUDA med    peak alloc    peak reserve")
    for name, row in stage_summary.items():
        print(
            f"{name:<20} {row['median_wall_ms']:>8.3f} ms "
            f"{row['median_cuda_ms']:>8.3f} ms "
            f"{row['peak_allocated_delta_mib']:>9.2f} MiB "
            f"{row['peak_reserved_delta_mib']:>10.2f} MiB"
        )

    print("\nFULL PIPELINE")
    for name in ("default", "optimized"):
        row = full_summary[name]
        print(
            f"{name:<10} median={row['median_wall_ms']:.3f} ms "
            f"p90={row['p90_wall_ms']:.3f} ms "
            f"p95={row['p95_wall_ms']:.3f} ms "
            f"docs/s={args.batch_size * 1_000 / row['median_wall_ms']:.2f} "
            f"peak_alloc={row['peak_allocated_delta_mib']:.2f} MiB"
        )
    print(f"speedup={full_summary['speedup']:.3f}x")
    digest = output_digest(last_default)
    print(f"formatted_parity=exact sha256={digest}")

    if args.trace:
        from torch.profiler import ProfilerActivity, profile, record_function

        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode(), profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            batch_cpu = collator(dataset)
            with record_function("stage_transfer"):
                batch = batch_cpu.to(
                    "cuda", dtype if dtype != torch.float32 else None
                )
            with record_function("stage_encoder"):
                encoded = model.encoder(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                ).last_hidden_state
            with record_function("stage_route"):
                token_embs, schema_embs = model.processor.extract_embeddings_from_batch(
                    encoded, batch.input_ids, batch
                )
            with record_function("stage_span_rep"):
                span_info = model.compute_span_rep_batched(token_embs)
            with record_function("stage_default_decode"):
                raw_default_decode(
                    model, batch, token_embs, schema_embs, span_info, metadata,
                    args.threshold, include_confidence, include_spans,
                )
            with record_function("stage_optimized_decode"):
                raw_optimized_decode(
                    model, batch, token_embs, schema_embs, span_info, metadata,
                    args.threshold, include_confidence, include_spans,
                )
        synchronize()
        profiler.export_chrome_trace(str(args.trace))
        print("\nCUDA OPERATOR PROFILE")
        print(profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=40
        ))
        print(f"Chrome trace: {args.trace}")

    result = {
        "device": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "execution_mode": args.execution_mode,
        "batch_size": args.batch_size,
        "encoded_shape": list(batch.input_ids.shape),
        "text_word_counts": batch.text_word_counts,
        "labels": labels,
        "threshold": args.threshold,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "formatted_parity": "exact",
        "output_sha256": digest,
        "stages": stage_summary,
        "full_pipeline": full_summary,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
