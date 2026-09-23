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
import re
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

WEIGHTS = "adapter_model.safetensors"


def _layer_of(name: str) -> int | None:
    """The transformer block index a parameter belongs to, if it has one."""
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def scale_adapter(src: Path, dst: Path, scale: float,
                  layers: range | None = None) -> int:
    """Copy `src` to `dst` with lora_B multiplied by `scale`.

    With `layers`, only blocks in that range are scaled and the rest keep their trained
    value. Uniform scaling shows *that* update magnitude moves a model along the
    capability/generalization trade; scaling a band shows *where* in the network the
    movement comes from, which is a different question and the one that distinguishes a
    mechanism from a knob. A parameter with no block index (an embedding or head module,
    if one is ever targeted) is left untouched under banded scaling rather than being
    silently swept into the first band.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("checkpoint-*", "ref"))

    tensors = load_file(str(src / WEIGHTS))
    touched = 0
    skipped_out_of_band = 0
    for name, tensor in tensors.items():
        # peft names these `...lora_B.weight` (or `.default.weight`); matching on the
        # substring keeps this robust to the naming variants across peft versions.
        if "lora_B" in name:
            if layers is not None:
                idx = _layer_of(name)
                if idx is None or idx not in layers:
                    skipped_out_of_band += 1
                    continue
            tensors[name] = tensor * scale
            touched += 1
    if not touched:
        raise SystemExit(f"no lora_B tensors found in {src / WEIGHTS}; refusing to write a "
                         "silently unscaled adapter")

    save_file(tensors, str(dst / WEIGHTS), metadata={"format": "pt"})
    # Record the provenance so a scaled adapter can never be mistaken for a trained one.
    (dst / "scaling.json").write_text(json.dumps({
        "source_adapter": str(src),
        "scale": scale,
        "scaled_tensors": touched,
        "layers": None if layers is None else [layers.start, layers.stop],
        "unscaled_out_of_band": skipped_out_of_band,
    }, indent=2))
    return touched


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="trained adapter directory")
    p.add_argument("--out-root", type=Path, required=True, help="where scaled copies go")
    p.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--name", default=None, help="basename for outputs (default: src name)")
    p.add_argument("--layers", default=None, metavar="START:STOP",
                   help="scale only blocks [START, STOP). Default: every block.")
    args = p.parse_args()

    layers = None
    tag_suffix = ""
    if args.layers:
        start, _, stop = args.layers.partition(":")
        layers = range(int(start), int(stop))
        tag_suffix = f"-L{start}_{stop}"

    base = args.name or args.src.name
    for s in args.scales:
        tag = f"{base}-s{str(s).replace('.', '')}{tag_suffix}"
        dst = args.out_root / tag
        n = scale_adapter(args.src, dst, s, layers=layers)
        where = "all blocks" if layers is None else f"blocks {layers.start}-{layers.stop - 1}"
        print(f"scale={s:<5} {where:<18} -> {dst}  ({n} lora_B tensors scaled)")


if __name__ == "__main__":
    main()
