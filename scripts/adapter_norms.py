"""The size of a LoRA update, per module and per layer.

Why this exists
---------------
Attenuating an adapter recovers held-out capability, and it does so on two independently
trained adapter families. The natural reading is that the *magnitude* of the update
controls where a model sits on the capability/generalization trade, rather than anything
specific to one objective or checkpoint. That reading is currently an inference from a
dose-response curve: scaling s down helps, therefore size matters.

It is testable directly. The update a LoRA adapter contributes to a linear layer is

    dW = (alpha / r) * B @ A

so its Frobenius norm is computable without running the model, and `||dW|| / ||W_base||`
gives the fraction of the base weight the update moves. If magnitude is what positions a
model on the frontier, then adapters trained by different objectives should sit at
different magnitudes in a way that tracks their in-distribution/held-out position -- and
scaling by s should move the norm by exactly s, which is also a check that the scaling
tool does what it claims.

What this does NOT establish. Norm is a summary. Two updates of equal norm can point in
different directions, and this project already measured that weight distance is a poor
proxy for behaviour: two runs of the same configuration sat 138% apart in relative lora_B
norm while agreeing to 0.0005 in train loss and 0.2 pp in evaluation. So a norm that
tracks position on the frontier is evidence; a norm that does not is not a refutation.
That asymmetry is the reason this prints the per-layer distribution rather than one
number -- a difference concentrated in a few layers and a difference spread across all of
them are different claims about the mechanism.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def _layer_of(name: str) -> int | None:
    """The transformer block index a module belongs to, if it has one."""
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _module_of(name: str) -> str:
    """q_proj / k_proj / ... from a peft parameter name."""
    m = re.search(r"\.([a-z_]+_proj)\.", name)
    return m.group(1) if m else "other"


def adapter_update_norms(adapter: Path) -> dict:
    """Frobenius norm of dW = (alpha/r) B@A, per (layer, module).

    Never materialises dW. For B (d x r) and A (r x k),

        ||B@A||_F^2 = tr((B@A)^T (B@A)) = tr(A^T B^T B A) = tr((B^T B)(A A^T))

    and both factors are r x r. With r = 32 that is a 32x32 trace instead of a
    4096 x 4096 product, per pair, of which an 8B adapter has hundreds -- the difference
    between seconds and being killed by a timeout, which is how the first version of this
    ended. The identity is exact, not an approximation.
    """
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scaling = alpha / r

    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    # pair lora_A with its lora_B by the module path they share
    pairs: dict[str, dict[str, object]] = defaultdict(dict)
    for name, t in tensors.items():
        if "lora_A" in name:
            pairs[name.replace("lora_A", "*")]["A"] = t
        elif "lora_B" in name:
            pairs[name.replace("lora_B", "*")]["B"] = t

    rows = []
    for key, ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            continue
        a, b = ab["A"].double(), ab["B"].double()   # r x r Grams are tiny; fp64 is free
        # tr((B^T B)(A A^T)), both r x r -- see the docstring.
        gram = (b.T @ b) @ (a @ a.T)
        fro = scaling * float(torch.clamp(torch.diagonal(gram).sum(), min=0.0).sqrt())
        rows.append({
            "name": key,
            "layer": _layer_of(key),
            "module": _module_of(key),
            "fro": fro,
            "shape": (b.shape[0], a.shape[1]),
        })
    total = math.sqrt(sum(r_["fro"] ** 2 for r_ in rows))
    return {"adapter": str(adapter), "r": r, "alpha": alpha, "scaling": scaling,
            "n_pairs": len(rows), "total_fro": total, "rows": rows}


def update_difference_norm(a1: Path, a2: Path) -> dict:
    """||dW_1 - dW_2||_F, without materialising either update.

    The same rank-space identity extends to a difference by stacking the factors:

        s1*B1@A1 - s2*B2@A2 = [s1*B1  -s2*B2] @ [A1 ; A2]

    so the concatenated factors are (d x 2r) and (2r x k) and the Gram trace is 2r x 2r.

    This is the measurement that separates "this objective barely moved the weights" from
    "these two adapters are the same file". Two runs whose totals agree to three decimals
    in every module are either both sitting on their shared initialisation or are the
    same checkpoint twice, and a norm on its own cannot tell those apart.
    """
    import torch
    from safetensors.torch import load_file

    def factors(a: Path):
        cfg = json.loads((a / "adapter_config.json").read_text())
        sc = cfg["lora_alpha"] / cfg["r"]
        t = load_file(str(a / "adapter_model.safetensors"))
        out = {}
        for name in t:
            if "lora_A" not in name:
                continue
            key = name.replace("lora_A", "*").replace(".default", "")
            b_name = name.replace("lora_A", "lora_B")
            if b_name in t:
                out[key] = (t[name].double(), t[b_name].double(), sc)
        return out

    f1, f2 = factors(a1), factors(a2)
    shared = sorted(set(f1) & set(f2))
    rows = []
    for key in shared:
        a_1, b_1, s_1 = f1[key]
        a_2, b_2, s_2 = f2[key]
        bc = torch.cat([s_1 * b_1, -s_2 * b_2], dim=1)      # d x 2r
        ac = torch.cat([a_1, a_2], dim=0)                   # 2r x k
        gram = (bc.T @ bc) @ (ac @ ac.T)
        fro = float(torch.clamp(torch.diagonal(gram).sum(), min=0.0).sqrt())
        rows.append({"name": key, "layer": _layer_of(key),
                     "module": _module_of(key), "fro": fro})
    return {"a": str(a1), "b": str(a2), "shared_pairs": len(shared),
            "only_in_a": len(set(f1) - set(f2)), "only_in_b": len(set(f2) - set(f1)),
            "diff_fro": math.sqrt(sum(r["fro"] ** 2 for r in rows)), "rows": rows}


def _summarise(res: dict) -> None:
    rows = res["rows"]
    print(f"\n=== {Path(res['adapter']).name} ===")
    print(f"  r={res['r']} alpha={res['alpha']} scaling={res['scaling']:g} "
          f"pairs={res['n_pairs']}")
    print(f"  ||dW||_F over the whole adapter: {res['total_fro']:.3f}")

    by_module: dict[str, list[float]] = defaultdict(list)
    for r_ in rows:
        by_module[r_["module"]].append(r_["fro"])
    print(f"  {'module':<12} {'n':>4} {'mean':>9} {'max':>9}  share of total energy")
    tot_sq = sum(r_["fro"] ** 2 for r_ in rows) or 1.0
    for mod, vals in sorted(by_module.items(), key=lambda kv: -sum(v * v for v in kv[1])):
        share = sum(v * v for v in vals) / tot_sq
        print(f"  {mod:<12} {len(vals):>4} {sum(vals)/len(vals):>9.3f} {max(vals):>9.3f}"
              f"  {share:>6.1%}")

    by_layer: dict[int, float] = defaultdict(float)
    for r_ in rows:
        if r_["layer"] is not None:
            by_layer[r_["layer"]] += r_["fro"] ** 2
    if by_layer:
        layers = sorted(by_layer)
        norms = [math.sqrt(by_layer[i]) for i in layers]
        lo, hi = min(norms), max(norms)
        print(f"  per-layer ||dW||: min {lo:.3f} (layer {layers[norms.index(lo)]})  "
              f"max {hi:.3f} (layer {layers[norms.index(hi)]})  ratio {hi/max(lo,1e-9):.1f}x")
        # a coarse shape, so "concentrated in a few layers" is visible without a plot
        width = 48
        step = max(1, len(layers) // width)
        bar = "".join("▁▂▃▄▅▆▇█"[min(7, int(7 * (norms[i] - lo) / max(hi - lo, 1e-9)))]
                      for i in range(0, len(layers), step))
        print(f"  layer profile (low->high index): {bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapters", nargs="+", type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--diff", nargs=2, type=Path, metavar=("A", "B"), default=None,
                    help="also report ||dW_A - dW_B||, and it relative to each")
    args = ap.parse_args()

    out = []
    for a in args.adapters:
        if not (a / "adapter_model.safetensors").exists():
            print(f"skip {a}: no adapter_model.safetensors")
            continue
        res = adapter_update_norms(a)
        _summarise(res)
        out.append(res)

    if len(out) > 1:
        print("\n" + "=" * 72)
        print("total ||dW||_F, relative to the first adapter listed\n")
        ref = out[0]["total_fro"] or 1.0
        for res in out:
            name = Path(res["adapter"]).name
            print(f"  {name:<34} {res['total_fro']:>9.3f}   {res['total_fro']/ref:>6.3f}x")
        print("\n  A scaled copy must come out at exactly its scale factor. If it does not,")
        print("  scale_adapter.py is not doing what the frontier sweep assumes it does.")

    if args.diff:
        a1, a2 = args.diff
        d = update_difference_norm(a1, a2)
        n1 = adapter_update_norms(a1)["total_fro"]
        n2 = adapter_update_norms(a2)["total_fro"]
        print("\n" + "=" * 72)
        print("difference between two updates\n")
        print(f"  A = {Path(d['a']).name}   ||dW_A|| = {n1:.3f}")
        print(f"  B = {Path(d['b']).name}   ||dW_B|| = {n2:.3f}")
        print(f"  shared pairs {d['shared_pairs']}"
              f"  (only in A: {d['only_in_a']}, only in B: {d['only_in_b']})")
        print(f"\n  ||dW_A - dW_B||_F = {d['diff_fro']:.3f}")
        print(f"  relative to A: {d['diff_fro'] / max(n1, 1e-9):.4f}"
              f"    relative to B: {d['diff_fro'] / max(n2, 1e-9):.4f}")
        if d["diff_fro"] / max(n1, 1e-9) < 1e-4:
            print("\n  These are the same update. Two differently-named adapters holding "
                  "the same\n  weights is a lineage defect, not a result.")
        else:
            print("\n  Distinct updates. The relative figure is how far one moved from "
                  "the other,\n  as a fraction of its own size.")

    if args.json_out:
        # rows are large; keep the summary and drop the per-pair detail
        slim = [{k: v for k, v in r.items() if k != "rows"} | {
            "by_module": {m: sum(x["fro"] ** 2 for x in r["rows"] if x["module"] == m) ** 0.5
                          for m in {x["module"] for x in r["rows"]}}} for r in out]
        args.json_out.write_text(json.dumps(slim, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
