"""Capture deterministic LoRA compatibility fixtures from GLiNER2 1.3.2.

Run from the repository root after installing the published reference wheel:

    python tests/fixtures/compat/capture_lora_oracle.py \
        --reference-path /tmp/gliner2-ref-132
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def _load_reference(reference_path: Path):
    reference_path = reference_path.resolve()
    sys.path = [
        str(reference_path),
        *[
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() != REPO_ROOT
        ],
    ]
    import gliner2

    if gliner2.__version__ != "1.3.2":
        raise RuntimeError(
            "Oracle capture requires gliner2==1.3.2, found "
            f"{gliner2.__version__!r} at {Path(gliner2.__file__).resolve()}"
        )
    return gliner2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, required=True)
    args = parser.parse_args()

    reference = _load_reference(args.reference_path)

    import peft
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora.layer import LoraLayer
    from gliner2.training.lora import _resolve_targets, save_lora_adapter

    class TinyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = nn.Linear(8, 8, bias=False)
            self.key = nn.Linear(8, 8, bias=False)
            self.value = nn.Linear(8, 8, bias=False)
            self.dense = nn.Linear(8, 8, bias=False)
            self.other = nn.Linear(8, 8, bias=False)

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = TinyEncoder()
            self.classifier = nn.Linear(8, 4, bias=False)
            self.span_rep = nn.Linear(8, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            encoded = self.encoder.query(x) + self.encoder.value(x)
            return self.classifier(encoded)

    torch.manual_seed(42)
    model = TinyModel()
    config = LoraConfig(
        r=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
        target_modules=_resolve_targets(model, ["encoder"]),
        bias="none",
    )
    peft_model = get_peft_model(model, config)

    # PEFT initializes lora_B to zero. Use deterministic non-zero values so
    # the oracle proves adapter behavior rather than only base-model parity.
    with torch.no_grad():
        layer_index = 1
        for _, module in peft_model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            for matrix in module.lora_A.values():
                values = torch.linspace(
                    -0.03 * layer_index,
                    0.03 * layer_index,
                    matrix.weight.numel(),
                    dtype=matrix.weight.dtype,
                )
                matrix.weight.copy_(values.reshape_as(matrix.weight))
            for matrix in module.lora_B.values():
                values = torch.linspace(
                    0.02 * layer_index,
                    -0.02 * layer_index,
                    matrix.weight.numel(),
                    dtype=matrix.weight.dtype,
                )
                matrix.weight.copy_(values.reshape_as(matrix.weight))
            layer_index += 1

    generator = torch.Generator().manual_seed(314159)
    input_batch = torch.randn(3, 8, generator=generator)
    with torch.no_grad():
        expected = peft_model(input_batch)

    legacy_weights: dict[str, torch.Tensor] = {}
    for name, module in peft_model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        clean = name.replace("base_model.model.", "").replace("base_model.", "")
        for matrix in module.lora_A.values():
            legacy_weights[f"{clean}.lora_A"] = matrix.weight.detach().cpu()
        for matrix in module.lora_B.values():
            legacy_weights[f"{clean}.lora_B"] = matrix.weight.detach().cpu()

    adapter_dir = HERE / "legacy_adapter_golden"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        save_lora_adapter(peft_model, adapter_dir)

    config_path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(config_path.read_text())
    adapter_config["created_at"] = "2026-04-21T00:29:03.781247Z"
    config_path.write_text(json.dumps(adapter_config, indent=2) + "\n")

    torch.save(input_batch, HERE / "input_batch.pt")
    torch.save(expected, HERE / "legacy_forward_outputs.pt")
    torch.save(legacy_weights, HERE / "lora_weights.pt")

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = {
        "reference_package": "gliner2",
        "reference_version": reference.__version__,
        "reference_path": str(Path(reference.__file__).resolve()),
        "torch_version": torch.__version__,
        "peft_version": peft.__version__,
        "capture_script_sha256": script_hash,
        "seed_model": 42,
        "seed_input": 314159,
    }
    (HERE / "oracle_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
