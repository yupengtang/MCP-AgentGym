"""The capability/generalization frontier, across independently trained models.

What this is for
----------------
The project's headline is that post-training buys in-catalogue capability and spends
held-out capability, and that scaling the LoRA update at serving time moves a model back
along that trade. Both halves were measured on a single training run. The project's own
variance budget then established that two independently trained models differ by 3.1 pp
and that a single DPO run's gain (+4.18 pp once, -0.27 pp the other time) is not a
property of the method. A single-run frontier is open to exactly that objection.

So this reads several seeds and asks three questions the single-run version cannot:

  1. Does the trade-off *reproduce*? Every seed must show in-distribution up and held-out
     down at s=1 relative to s=0, or the effect belongs to one run.
  2. Does attenuation *recover* held-out capability in every seed, monotonically?
  3. Do the seeds trace the *same* frontier, or does each run have its own? If the
     spread between seeds at a given s is inside the 3.1 pp floor, the frontier is a
     property of the procedure; if it is outside, position along it is run-specific.

Pooling across seeds is reported too, because it is the only way to get an n large
enough for the small scales to separate -- but the per-seed direction test is the claim,
and pooling can hide a seed that disagrees. Both are printed for that reason.

Refuses to print a partial grid, for the reason recorded in pool2_report.py: a table with
a silently missing arm reads as a complete result and is the defect that produced this
project's first retraction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPLITS = ["in_distribution", "unseen_fault"]
# The scale each served variant corresponds to. `base` is s=0 by construction: it is the
# untrained model, served with no adapter.
SCALES: list[tuple[str, float]] = [
    ("base", 0.0), ("s025", 0.25), ("s05", 0.5), ("s075", 0.75), ("s10", 1.0),
]
# Independently trained models differing only in seed (reports/EXPERIMENTS.md).
SEED_FLOOR_PP = 3.1


def _z(pa: float, na: int, pb: float, nb: int) -> float:
    """Two-proportion z. Unpooled SE, matching pool2_report."""
    if not na or not nb:
        return 0.0
    se = sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    return (pa - pb) / se if se else 0.0


def heldout_labels(heldout_set: str) -> set[str]:
    """The diagnosis labels the withheld families should elicit.

    Derived from the taxonomy rather than hardcoded. scale_report.py hardcodes the two
    labels of the published split, which silently reports zero emissions for any other
    held-out set -- the label collapse would look total no matter what the model did.
    """
    from agentgym.incidents.faults import FAULTS_BY_NAME
    from agentgym.incidents.generator import heldout_faults

    return {FAULTS_BY_NAME[n].root_cause for n in heldout_faults(heldout_set)}


def _row(ts, labels: set[str]) -> dict:
    from agentgym.data import trajectory as T  # noqa: F401  (import cost only)

    n = len(ts)
    calls = sum(t.verification.tool_calls for t in ts)
    subs = [s.args.get("root_cause") for t in ts for s in t.steps
            if s.tool == "submit_diagnosis"]
    return {
        "n": n,
        "resolved": sum(t.verification.resolved for t in ts) / n,
        "root": sum(t.verification.root_cause_correct for t in ts) / n,
        "invalid": sum(t.verification.invalid_calls for t in ts) / max(1, calls),
        "heldout_emissions": sum(1 for s in subs if s in labels),
        "submissions": len(subs),
    }


def _load_seed(runs: Path, labels: set[str]) -> dict[tuple[str, str], dict]:
    from agentgym.data import trajectory as T

    cells: dict[tuple[str, str], dict] = {}
    for variant, _ in SCALES:
        for split in SPLITS:
            f = runs / f"{variant}-{split}-clean.jsonl"
            if f.exists() and f.stat().st_size:
                ts = T.read_jsonl(f)
                if ts:
                    cells[(variant, split)] = _row(ts, labels)
    return cells


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    runs_default = Path(os.environ.get("AGENTGYM_RUNS", "runs"))
    ap.add_argument("--runs", type=Path, default=runs_default)
    ap.add_argument("--seeds", nargs="+", default=["seed1", "realseed11", "realseed12"],
                    help="suffixes of eval-frontier-<seed> directories")
    ap.add_argument("--heldout-set", default="published")
    ap.add_argument("--allow-partial", action="store_true",
                    help="print what exists. Off by default on purpose.")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    labels = heldout_labels(args.heldout_set)
    per_seed: dict[str, dict] = {}
    missing: list[str] = []
    for seed in args.seeds:
        d = args.runs / f"eval-frontier-{seed}"
        cells = _load_seed(d, labels) if d.exists() else {}
        for variant, _ in SCALES:
            for split in SPLITS:
                if (variant, split) not in cells:
                    missing.append(f"{seed}/{variant}/{split}")
        per_seed[seed] = cells

    if missing and not args.allow_partial:
        print(f"REFUSING to render: {len(missing)} cell(s) absent.", file=sys.stderr)
        for m in missing[:12]:
            print(f"  missing {m}", file=sys.stderr)
        if len(missing) > 12:
            print(f"  ... and {len(missing) - 12} more", file=sys.stderr)
        print("\nA frontier with a missing scale reads as complete and is not. Wait for "
              "the sweeps, or pass --allow-partial to inspect progress.", file=sys.stderr)
        raise SystemExit(2)

    print(f"held-out set: {args.heldout_set}   labels: {sorted(labels)}")
    print(f"seeds: {', '.join(args.seeds)}\n")

    # ---- per-seed frontier -------------------------------------------------
    for seed, cells in per_seed.items():
        if not cells:
            continue
        print(f"=== {seed} " + "=" * (66 - len(seed)))
        print(f"{'s':>5}  {'ID resolved':>12} {'OOD resolved':>13} {'OOD root':>9} "
              f"{'OOD labels':>11} {'invalid':>8}")
        for variant, s in SCALES:
            idc, ood = cells.get((variant, "in_distribution")), cells.get((variant, "unseen_fault"))
            if not idc or not ood:
                continue
            print(f"{s:5.2f}  {_pct(idc['resolved']):>12} {_pct(ood['resolved']):>13} "
                  f"{_pct(ood['root']):>9} "
                  f"{ood['heldout_emissions']:>4}/{ood['submissions']:<6} "
                  f"{_pct(ood['invalid']):>8}")
        print()

    # ---- claim 1: does the trade-off reproduce in every seed? --------------
    print("=" * 72)
    print("CLAIM 1  the trade-off reproduces across independently trained models")
    print("         (in-distribution up AND held-out down, at s=1 against s=0)\n")
    verdicts = []
    for seed, cells in per_seed.items():
        a_id, b_id = cells.get(("base", "in_distribution")), cells.get(("s10", "in_distribution"))
        a_od, b_od = cells.get(("base", "unseen_fault")), cells.get(("s10", "unseen_fault"))
        if not all((a_id, b_id, a_od, b_od)):
            continue
        d_id = 100 * (b_id["resolved"] - a_id["resolved"])
        d_od = 100 * (b_od["resolved"] - a_od["resolved"])
        z_id = _z(b_id["resolved"], b_id["n"], a_id["resolved"], a_id["n"])
        z_od = _z(b_od["resolved"], b_od["n"], a_od["resolved"], a_od["n"])
        ok = d_id > 0 and d_od < 0
        verdicts.append(ok)
        print(f"  {seed:12} ID {d_id:+6.1f} pp (z={z_id:+5.2f})   "
              f"held-out {d_od:+6.1f} pp (z={z_od:+6.2f})   {'TRADE' if ok else 'NO TRADE'}")
    if verdicts:
        print(f"\n  -> {sum(verdicts)}/{len(verdicts)} seeds show the trade in both directions.")
        if all(verdicts):
            print("     The trade-off is a property of the procedure, not of one run.")
        else:
            print("     NOT unanimous. Say that the trade-off is run-dependent, and do not "
                  "quote a pooled number as if it described every run.")

    # ---- claim 2: does attenuation recover, monotonically, in every seed? --
    print("\n" + "=" * 72)
    print("CLAIM 2  attenuating the update recovers held-out capability\n")
    mono = []
    for seed, cells in per_seed.items():
        series = [(s, cells[(v, "unseen_fault")]["resolved"])
                  for v, s in SCALES if (v, "unseen_fault") in cells]
        if len(series) < 3:
            continue
        # held-out resolution should fall as s rises
        desc = all(b <= a + 1e-9 for (_, a), (_, b) in zip(series, series[1:], strict=False))
        lo, hi = series[0][1], series[-1][1]
        c0, c1 = cells[("base", "unseen_fault")], cells[("s10", "unseen_fault")]
        z = _z(lo, c0["n"], hi, c1["n"])
        mono.append(desc)
        print(f"  {seed:12} s=0 {_pct(lo)} -> s=1 {_pct(hi)}   "
              f"{100 * (lo - hi):+6.1f} pp recovered (z={z:+5.2f})   "
              f"{'monotone' if desc else 'NOT monotone'}")
    if mono:
        print(f"\n  -> monotone in {sum(mono)}/{len(mono)} seeds.")

    # ---- claim 3: one frontier, or one per run? ---------------------------
    print("\n" + "=" * 72)
    print("CLAIM 3  do the seeds trace the same frontier?")
    print(f"         spread at each s, against the {SEED_FLOOR_PP} pp independent-run floor\n")
    print(f"{'s':>5}  {'split':<17} {'mean':>7} {'spread':>8}   verdict")
    for variant, s in SCALES:
        for split in SPLITS:
            vals = [c[(variant, split)]["resolved"] for c in per_seed.values()
                    if (variant, split) in c]
            if len(vals) < 2:
                continue
            spread = 100 * (max(vals) - min(vals))
            inside = spread <= SEED_FLOOR_PP
            print(f"{s:5.2f}  {split:<17} {_pct(sum(vals) / len(vals)):>7} "
                  f"{spread:7.1f} pp   "
                  f"{'within the floor' if inside else 'EXCEEDS the floor'}")
    print("\n  Points whose spread is within the floor are indistinguishable between runs:")
    print("  the seeds are on one frontier there. Points outside it mean position along")
    print("  the frontier is itself run-dependent, and must be reported per seed.")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {s: {f"{v}/{sp}": c for (v, sp), c in cells.items()}
             for s, cells in per_seed.items()}, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
