"""Report the adapter-scale sweep: the specialization/generalization frontier.

Scaling a LoRA update by `s` interpolates between the base model (s=0) and the trained
policy (s=1) without retraining, so one sweep shows whether the in-distribution gain and
the held-out-fault regression are separable or are two views of the same weight update.

Two rules this script enforces rather than trusts:

* **Read the sweep internally.** Cells here are drawn with a different `--limit` than the
  published evaluation, so they cover a different subset of incidents. Comparing a sweep
  cell against a table in README would reintroduce exactly the population confound that
  produced a retention ratio above 1.00 earlier in this project. Every comparison printed
  below is s-against-s within this sweep.
* **Refuse to report a partial grid.** A sweep missing cells still renders a table, and a
  frontier with a hole in it reads as a finding. `--require-complete` (default) exits
  non-zero instead.

The held-out-label count is the mechanism this sweep exists to test: post-training drove
the two never-demonstrated diagnosis labels to zero emissions in ~600 submissions, against
5.0% for base, so if attenuating the update recovers anything it should appear here first.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# The two fault families held out of training, and the labels they should elicit.
HELDOUT_LABELS = {
    "network_policy_blocks_dependency",
    "node_unhealthy_reduces_capacity",
}
SCALES = [("base", 0.0), ("dpo5s025", 0.25), ("dpo5s05", 0.5),
          ("dpo5s075", 0.75), ("dpo5s10", 1.0)]
SPLITS = ["in_distribution", "unseen_fault"]


def _load(runs: Path):
    from agentgym.data import trajectory as T

    cells: dict[tuple[str, str], list] = {}
    for variant, _ in SCALES:
        for split in SPLITS:
            f = runs / f"{variant}-{split}-clean.jsonl"
            if f.exists() and f.stat().st_size:
                cells[(variant, split)] = T.read_jsonl(f)
    return cells


def _row(ts):
    n = len(ts)
    calls = sum(t.verification.tool_calls for t in ts)
    subs = [s.args.get("root_cause") for t in ts for s in t.steps
            if s.tool == "submit_diagnosis"]
    held = sum(1 for s in subs if s in HELDOUT_LABELS)
    return {
        "n": n,
        "resolved": sum(t.verification.resolved for t in ts) / n,
        "root": sum(t.verification.root_cause_correct for t in ts) / n,
        "invalid": sum(t.verification.invalid_calls for t in ts) / max(1, calls),
        "reward": sum(t.reward for t in ts) / n,
        "heldout_emissions": held,
        "submissions": len(subs),
        "settle_timeouts": sum(getattr(t, "settle_timeouts", 0) for t in ts),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path,
                   default=Path(os.environ.get("AGENTGYM_RUNS", "runs")) / "eval-scale")
    p.add_argument("--allow-partial", action="store_true",
                   help="Report an incomplete grid. A frontier with a hole reads as a "
                        "finding, so this is off by default.")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    cells = _load(args.runs)
    expected = [(v, s) for v, _ in SCALES for s in SPLITS]
    missing = [k for k in expected if k not in cells]
    if missing and not args.allow_partial:
        print(f"incomplete sweep: {len(cells)}/{len(expected)} cells present")
        for v, s in missing:
            print(f"  missing: {v} / {s}")
        raise SystemExit(1)

    report: dict[str, dict] = defaultdict(dict)
    for split in SPLITS:
        print(f"\n## {split} (clean tools)\n")
        print(f"{'scale':>6}  {'n':>4}  {'resolved':>9}  {'root':>7}  "
              f"{'invalid':>8}  {'reward':>7}  {'held-out labels':>16}")
        for variant, s in SCALES:
            ts = cells.get((variant, split))
            if not ts:
                print(f"{s:>6.2f}  {'--':>4}  {'--':>9}")
                continue
            r = _row(ts)
            report[split][f"s={s}"] = r
            held = f"{r['heldout_emissions']}/{r['submissions']}"
            print(f"{s:>6.2f}  {r['n']:>4}  {r['resolved']:>8.1%}  {r['root']:>6.1%}  "
                  f"{r['invalid']:>7.1%}  {r['reward']:>7.2f}  {held:>16}")
            if r["settle_timeouts"]:
                print(f"        ^ {r['settle_timeouts']} settle timeout(s): this cell's "
                      "verdicts were taken against an unconverged world")

    # The frontier question, stated as the two deltas that matter, s-against-s.
    print("\n## frontier (relative to s=1.00, the unattenuated policy)\n")
    ref = {sp: report[sp].get("s=1.0") for sp in SPLITS}
    if all(ref.values()):
        print(f"{'scale':>6}  {'in-dist Δ':>10}  {'unseen-fault Δ':>15}")
        for _, s in SCALES:
            row = {sp: report[sp].get(f"s={s}") for sp in SPLITS}
            if not all(row.values()):
                continue
            d_in = row["in_distribution"]["resolved"] - ref["in_distribution"]["resolved"]
            d_uf = row["unseen_fault"]["resolved"] - ref["unseen_fault"]["resolved"]
            print(f"{s:>6.2f}  {d_in:>+9.1%}  {d_uf:>+14.1%}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
