"""Task-level parity smoke test for the Triton DeBERTa-v2/v3 encoder.

Runs an official DeBERTa-v2 model fine-tuned on MNLI twice:
  1. untouched Hugging Face reference
  2. same checkpoint with only the DeBERTa encoder replaced by our inference backend

It compares task predictions/logits/probabilities and the final encoder hidden state.

Run from the repository root, for example:
    python parity_pretrained_mnli.py

Optional:
    python parity_pretrained_mnli.py --dtype fp32
    python parity_pretrained_mnli.py --backend optimized
"""

from __future__ import annotations

import argparse
import inspect

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from attention.encoder import enable_deberta_v2_inference


EXAMPLES = [
    (
        "A dog is running through a field.",
        "An animal is running.",
        "ENTAILMENT",
    ),
    (
        "A man is sleeping on the couch.",
        "The man is awake and standing.",
        "CONTRADICTION",
    ),
    (
        "A woman is reading a book.",
        "The book is about astronomy.",
        "NEUTRAL",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="microsoft/deberta-v2-xlarge-mnli",
        help="HF sequence-classification checkpoint.",
    )
    parser.add_argument(
        "--backend",
        choices=("triton", "optimized"),
        default="triton",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "fp32"),
        default="fp16",
    )
    parser.add_argument(
        "--bucket",
        type=int,
        default=64,
        help="Fixed padded sequence length / prepared encoder bucket.",
    )
    parser.add_argument(
        "--fp32-precision",
        choices=("strict", "fast"),
        default="strict",
    )
    return parser.parse_args()


def enable_backend(
    backbone: torch.nn.Module,
    *,
    backend: str,
    bucket: int,
    fp32_precision: str,
) -> None:
    """Support both the current API and the planned always-fused-QKV API."""

    kwargs = {
        "backend": backend,
        "sequence_lengths": [bucket],
        "fp32_precision": fp32_precision,
    }

    # In the current code fuse_qkv is still an option. Force it on.
    # Once the option is removed and QKV fusion is unconditional, this branch
    # simply disappears automatically.
    if "fuse_qkv" in inspect.signature(enable_deberta_v2_inference).parameters:
        kwargs["fuse_qkv"] = True

    enable_deberta_v2_inference(backbone, **kwargs)


@torch.inference_mode()
def run_model(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    return (
        output.logits.detach().float().cpu(),
        output.hidden_states[-1].detach().float().cpu(),
    )


def load_model(
    model_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    return model.to(device=device).eval()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this parity test")

    device = torch.device("cuda")
    dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print(f"model:   {args.model}")
    print(f"backend: {args.backend}")
    print(f"dtype:   {dtype}")
    print(f"gpu:     {torch.cuda.get_device_name(device)}")
    print(f"bucket:  {args.bucket}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    premises = [premise for premise, _, _ in EXAMPLES]
    hypotheses = [hypothesis for _, hypothesis, _ in EXAMPLES]

    encoded = tokenizer(
        premises,
        hypotheses,
        padding="max_length",
        truncation=True,
        max_length=args.bucket,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device=device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }

    # ------------------------------------------------------------------
    # Untouched Hugging Face reference.
    # ------------------------------------------------------------------
    print("Running Hugging Face reference...")
    reference = load_model(args.model, device=device, dtype=dtype)
    id2label = {
        int(index): label
        for index, label in reference.config.id2label.items()
    }

    reference_logits, reference_hidden = run_model(reference, inputs)
    reference_probs = reference_logits.softmax(dim=-1)

    del reference
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Same checkpoint, replacing only the DeBERTa encoder.
    # ------------------------------------------------------------------
    print(f"Running {args.backend} encoder...")
    candidate = load_model(args.model, device=device, dtype=dtype)

    base_model_prefix = candidate.base_model_prefix
    backbone = getattr(candidate, base_model_prefix)

    enable_backend(
        backbone,
        backend=args.backend,
        bucket=args.bucket,
        fp32_precision=args.fp32_precision,
    )

    candidate_logits, candidate_hidden = run_model(candidate, inputs)
    candidate_probs = candidate_logits.softmax(dim=-1)

    # ------------------------------------------------------------------
    # Task-level parity.
    # ------------------------------------------------------------------
    logit_error = (reference_logits - candidate_logits).abs()
    prob_error = (reference_probs - candidate_probs).abs()
    hidden_error = (reference_hidden - candidate_hidden).abs()

    reference_prediction = reference_logits.argmax(dim=-1)
    candidate_prediction = candidate_logits.argmax(dim=-1)

    print()
    print("=" * 88)
    print("MNLI task predictions")
    print("=" * 88)

    all_predictions_match = True
    for index, (premise, hypothesis, expected) in enumerate(EXAMPLES):
        ref_id = int(reference_prediction[index])
        cand_id = int(candidate_prediction[index])
        ref_label = id2label[ref_id]
        cand_label = id2label[cand_id]

        all_predictions_match &= ref_id == cand_id

        ref_conf = float(reference_probs[index, ref_id])
        cand_conf = float(candidate_probs[index, cand_id])

        print(f"\ncase {index + 1}: expected semantic class ~ {expected}")
        print(f"  premise:    {premise}")
        print(f"  hypothesis: {hypothesis}")
        print(f"  reference:  {ref_label:<14} p={ref_conf:.8f}")
        print(f"  {args.backend:<10}: {cand_label:<14} p={cand_conf:.8f}")
        print(
            "  max logit delta: "
            f"{float(logit_error[index].max()):.8g}"
        )
        print(
            "  max prob delta:  "
            f"{float(prob_error[index].max()):.8g}"
        )

    print()
    print("=" * 88)
    print("Numerical parity")
    print("=" * 88)
    print(f"predictions identical:       {all_predictions_match}")
    print(f"logits max abs error:        {float(logit_error.max()):.8g}")
    print(f"logits mean abs error:       {float(logit_error.mean()):.8g}")
    print(f"probabilities max abs error: {float(prob_error.max()):.8g}")
    print(f"probabilities mean abs err:  {float(prob_error.mean()):.8g}")
    print(f"hidden max abs error:        {float(hidden_error.max()):.8g}")
    print(f"hidden mean abs error:       {float(hidden_error.mean()):.8g}")

    if not all_predictions_match:
        raise SystemExit(
            "Task-level parity FAILED: at least one predicted MNLI label changed"
        )

    print("\nTask-level parity PASSED: all predicted labels are identical.")


if __name__ == "__main__":
    main()
