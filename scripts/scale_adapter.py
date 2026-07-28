"""Produce a LoRA adapter whose contribution is scaled by a constant factor.

Motivation. Post-training on a closed fault catalogue drives the probability of
never-demonstrated diagnosis labels to zero: measured at 0 emissions in 599 (SFT) and 626
(DPO) submissions, against 5.0% for the base model. The labels are still in the tool enum
and the system prompt -- the model simply stopped being able to reach them. That is
label-prior collapse, and it is why every post-trained variant scores below base on the
held-out fault families.

The adapter's contribution to a linear layer is

    delta_W = (lora_alpha / r) * B @ A

so multiplying `lora_B` by s scales the whole update by s, exactly and without retraining.
s = 0 recovers the base model, s = 1 the trained one. Sweeping s therefore traces the
specialization/generalization frontier with one knob, and the resulting adapters can all be
served simultaneously by the existing multi-LoRA setup.

This edits only `lora_B`; scaling both factors would apply s twice.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

WEIGHTS = "adapter_model.safetensors"


def scale_adapter(src: Path, dst: Path, scale: float) -> int:
    """Copy `src` to `dst` with every lora_B tensor multiplied by `scale`."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("checkpoint-*", "ref"))

    tensors = load_file(str(src / WEIGHTS))
    touched = 0
    for name, tensor in tensors.items():
        # peft names these `...lora_B.weight` (or `.default.weight`); matching on the
        # substring keeps this robust to the naming variants across peft versions.
        if "lora_B" in name:
            tensors[name] = tensor * scale
            touched += 1
    if not touched:
        raise SystemExit(f"no lora_B tensors found in {src / WEIGHTS}; refusing to write a "
                         "silently unscaled adapter")

    save_file(tensors, str(dst / WEIGHTS), metadata={"format": "pt"})
    # Record the provenance so a scaled adapter can never be mistaken for a trained one.
    (dst / "scaling.json").write_text(json.dumps(
        {"source_adapter": str(src), "scale": scale, "scaled_tensors": touched}, indent=2))
    return touched


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="trained adapter directory")
    p.add_argument("--out-root", type=Path, required=True, help="where scaled copies go")
    p.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--name", default=None, help="basename for outputs (default: src name)")
    args = p.parse_args()

    base = args.name or args.src.name
    for s in args.scales:
        tag = f"{base}-s{str(s).replace('.', '')}"
        dst = args.out_root / tag
        n = scale_adapter(args.src, dst, s)
        print(f"scale={s:<5} -> {dst}  ({n} lora_B tensors scaled)")


if __name__ == "__main__":
    main()
