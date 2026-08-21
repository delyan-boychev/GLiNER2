# Standalone DeBERTa attention experiment

`original.py` is the auditable baseline. Its attention, layer, convolution,
mask expansion, relative-position construction, and encoder loop are copied
from Hugging Face Transformers 4.57.6 under the upstream Apache-2.0 license.
The few Transformers infrastructure dependencies are replaced locally, so the
baseline needs only PyTorch. Its state-dict keys and encoder output have been
checked exactly against the upstream classes.

`optimized.py` is the prepared PyTorch implementation. `triton_attention.py`
is the CUDA fused implementation. `encoder.py` replaces only self-attention and
keeps the copied/Hugging Face output, residual, FFN, and convolution modules.
The encoder fast path receives a 2-D padding mask directly and does not create
the baseline `[B, 1, L, L]` mask or `[L, L]` relative-position tensor.

## CUDA benchmark

Attention-only, with FP16 and FP32 eager/compiled modes:

```bash
python -m attention.benchmark_cuda --scope attention
```

Full 12-layer encoder:

```bash
python -m attention.benchmark_cuda \
  --scope encoder \
  --output encoder_attention_cuda_results.json
```

Triton autotuning is enabled by default. Triton prints the tuning time and
winning launch configuration for every new length/head/precision key. Compare
the exact same kernel with and without launch autotuning by running:

```bash
python -m attention.benchmark_cuda \
  --implementations original,triton \
  --triton-autotune \
  --output triton_autotuned.json

python -m attention.benchmark_cuda \
  --implementations original,triton \
  --no-triton-autotune \
  --output triton_manual_launch.json
```

The compiled bucket factory uses `isolate_recompiles=True` on PyTorch 2.13+.
Older PyTorch versions receive a distinct cloned code object per bucket. This
avoids the Dynamo failure where separately compiled length closures share the
default eight-entry per-code-object recompilation budget.

Hostile numerical validation (raw errors, no thresholds):

```bash
python -m attention.validate_cuda
```

Add `--executions eager,compile`, `--head-dims 32,64,128`, or
`--batches 1,2,4,8` for the full matrix. The default validation already covers
FP16, BF16, strict FP32, boundary lengths through 2048, no/left/right/heavy/all
padding, and both DeBERTa positional terms.

## Prepared encoder buckets

```python
from attention import compile_deberta_buckets, enable_deberta_v2_inference

backbone.eval()
enable_deberta_v2_inference(
    backbone,
    backend="triton",
    sequence_lengths=[32, 64, 128, 256, 512],
)

examples = {
    length: (hidden_for_length[length], mask_for_length[length])
    for length in [32, 64, 128, 256, 512]
}
compiled_buckets = compile_deberta_buckets(backbone.encoder, examples=examples)
output = compiled_buckets[128](hidden_states, attention_mask)
```

The supplied examples execute compilation and Triton autotuning during startup.
Inference then dispatches to an already prepared fixed-length bucket; only the
batch dimension is dynamic.
