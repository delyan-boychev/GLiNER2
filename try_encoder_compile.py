#!/usr/bin/env python3
"""Small CUDA smoke test for the GLiNER2 encoder with torch.compile.

Run from the repository root:

    python try_encoder_compile.py

The script loads the checkpoint on CPU, keeps only its Transformers encoder,
moves that encoder to CUDA, and compares three execution paths:

* ``legacy``: ordinary eager PyTorch;
* ``compiled-default``: the repository's current ``torch.compile`` call; and
* ``reduce-overhead``: compilation with CUDA graphs where supported.

Every shape prints its first-call time, steady-state latency, and pairwise
max/mean absolute differences for compiled-default vs legacy, reduce-overhead
vs legacy, and reduce-overhead vs compiled-default. There is deliberately no
numerical threshold or pass/fail decision. A runtime failure is reported
without hiding the exception or stopping the remaining cases.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
import traceback
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
    ("legacy", None),
    ("compiled-default", "default"),
    ("reduce-overhead", "reduce-overhead"),
)
NUMERICAL_COMPARISONS = (
    ("compiled-default", "legacy"),
    ("reduce-overhead", "legacy"),
    ("reduce-overhead", "compiled-default"),
)


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
) -> tuple[float, torch.Tensor]:
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
    return statistics.median(samples), output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Compare the eager legacy, repository-default compiled, and "
            "reduce-overhead GLiNER2 encoder paths at several CUDA shapes."
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
        "--static",
        action="store_true",
        help="use dynamic=False; the default intentionally tests shape changes on one dynamic graph",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="require a single graph so graph breaks become explicit failures",
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
        f"Paths: {', '.join(name for name, _ in EXECUTION_PATHS)} | "
        f"dynamic={not args.static} fullgraph={args.fullgraph} | "
        f"shapes={shape_string(args.shapes)}",
        flush=True,
    )

    execution_paths = []
    for name, compile_mode in EXECUTION_PATHS:
        if compile_mode is None:
            execution_paths.append((name, encoder))
            continue
        compile_kwargs = {
            "dynamic": not args.static,
            "fullgraph": args.fullgraph,
        }
        # Omitting mode exactly mirrors the repository's existing encoder
        # compilation path. Calling it mode="default" is equivalent, but this
        # keeps the legacy measurement faithful to the production call site.
        if compile_mode != "default":
            compile_kwargs["mode"] = compile_mode
        execution_paths.append((name, torch.compile(encoder, **compile_kwargs)))

    failures = 0
    completed = 0
    skipped = 0
    total = len(args.shapes)
    with torch.inference_mode():
        for case_number, (batch, length) in enumerate(args.shapes, start=1):
            case = f"B={batch} L={length}"
            print(f"\n[{case_number}/{total}] {case}", flush=True)
            if max_positions and length > max_positions:
                print(f"  SKIP: length exceeds max_position_embeddings={max_positions}", flush=True)
                skipped += 1
                continue

            input_ids, attention_mask, valid_lengths = make_inputs(
                batch,
                length,
                vocab_size,
                pad_token_id,
                args.padding,
                device,
            )
            print(
                f"  input_ids={tuple(input_ids.shape)} mask_valid={valid_lengths}",
                flush=True,
            )

            legacy_ms = None
            outputs = {}
            for path_name, path_encoder in execution_paths:
                print(f"\n  --- {path_name} ---", flush=True)
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

                    path_ms, path_output = median_latency(
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
                    if path_name == "legacy":
                        legacy_ms = path_ms
                    speedup = legacy_ms / path_ms if legacy_ms is not None else None
                    speedup_text = (
                        f" | vs_legacy={speedup:.2f}x" if speedup is not None else ""
                    )
                    print(
                        f"  RESULT {path_name}: first={first_ms:.3f} ms | "
                        f"median={path_ms:.3f} ms{speedup_text}",
                        flush=True,
                    )
                    completed += 1
                except Exception as exc:  # one mode must not hide the others
                    failures += 1
                    print(
                        f"  ERROR {path_name}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)

            print("\n  --- numerical differences ---", flush=True)
            for left_name, right_name in NUMERICAL_COMPARISONS:
                if left_name not in outputs or right_name not in outputs:
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
                print(
                    f"  {left_name} vs {right_name}: "
                    f"max_abs={max_error:.6g} | mean_abs={mean_error:.6g}",
                    flush=True,
                )
                del difference

            del outputs
            del input_ids, attention_mask

    print(
        f"\nSUMMARY: completed_paths={completed} runtime_failures={failures} "
        f"skipped_shapes={skipped} total_shapes={total}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
