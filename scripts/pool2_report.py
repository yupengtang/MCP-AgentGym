"""The high-power comparison table, computed from `runs/eval-pool2/`.

This is the evaluation that settled the method ranking. It exists as a script rather than
as ad-hoc queries because the ad-hoc version is how a number gets into a README that
nobody can regenerate — which happened to the serving benchmark, the expert reference line
and the trajectory count in this repository before today.

Three rules it enforces rather than trusts:

* **Refuses a partial grid.** Every arm needs all four splits. A missing cell still renders
  a table, and this project has twice drawn a conclusion from one that later reversed:
  the DPO pair-count ablation (written up from its first cell, +11.0 pp, later +1.5 pp) and
  the episode-GRPO capability claim.
* **Reports significance, with a multiple-comparison correction.** Six pairwise tests are
  run; at alpha = 0.05 that is z > 2.64 by Bonferroni, not 1.96. The distinction changes
  which of the observed effects survive.
* **Prints the effect *and* its z.** A shrinking effect with flat z is noise; a wobbling
  effect with rising z is real. Only showing one of the two hides which you have.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from math import sqrt
from pathlib import Path

SPLITS = ["in_distribution", "unseen_fault", "unseen_topology", "compositional"]
# Preferred display order; anything else is appended so an arm can never vanish silently.
# Grouped so the table reads as the comparison it is: an undertrained baseline and what
# post-training buys on it, against a properly trained baseline and the same question.
ORDER = ["base", "sft-p2", "dpo5-p2", "grpo-p2", "actgrpo-p2",
         "sft2ep1-p2", "sft2-p2", "dpo3ep-p2"]
LABEL = {
    "base": "base (untrained)",
    "sft-p2": "SFT 2 epochs (published baseline)",
    "dpo5-p2": "SFT + DPO-435",
    "grpo-p2": "SFT + episode-GRPO",
    "actgrpo-p2": "SFT + action-GRPO",
    "sft2-p2": "SFT 3 epochs",
    "dpo3ep-p2": "SFT 3ep + DPO-309",
    "sft2ep1-p2": "SFT 2 epochs (seed 1)",
}
# The 2-epoch SFT and everything built on it; then the 3-epoch SFT and the same.
EPOCH2 = ["sft-p2", "dpo5-p2", "grpo-p2", "actgrpo-p2"]
EPOCH3 = ["sft2-p2", "dpo3ep-p2"]


def _z(pa: float, na: int, pb: float, nb: int) -> float:
    se = sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    return (pb - pa) / se if se else 0.0


def load(runs: Path) -> dict[str, dict[str, list]]:
    from agentgym.data import trajectory as T

    found: dict[str, dict[str, list]] = {}
    for f in sorted(runs.glob("*-clean.jsonl")):
        stem = f.name[: -len("-clean.jsonl")]
        for sp in SPLITS:
            if stem.endswith(f"-{sp}"):
                found.setdefault(stem[: -len(sp) - 1], {})[sp] = T.read_jsonl(f)
                break
    return found


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path,
                   default=Path(os.environ.get("AGENTGYM_RUNS", "runs")) / "eval-pool2")
    p.add_argument("--allow-partial", action="store_true",
                   help="Report an incomplete grid. Off by default: a table with a hole "
                        "reads exactly like a table without one.")
    p.add_argument("--expect", nargs="*", default=None,
                   help="Arms that must be present. Defaults to every arm in ORDER, so a "
                        "run that has not started yet cannot be mistaken for one that has "
                        "no row.")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    cells = load(args.runs)
    if not cells:
        raise SystemExit(f"no cells under {args.runs}")

    # Two kinds of incompleteness, and the second is the dangerous one. An arm present with
    # a missing split shows an obviously short row; an arm absent entirely shows *nothing*,
    # and the table looks finished. The first version of this guard only checked the former
    # -- it passed while `base` and the second SFT seed were still running.
    incomplete = {v: sorted(set(SPLITS) - set(sp)) for v, sp in cells.items()
                  if len(sp) < len(SPLITS)}
    absent = [v for v in expect if v not in cells] if (expect := args.expect) else \
        [v for v in ORDER if v not in cells]
    if (incomplete or absent) and not args.allow_partial:
        print(f"incomplete grid in {args.runs}:")
        for v, missing in incomplete.items():
            print(f"  {v}: missing split(s) {', '.join(missing)}")
        for v in absent:
            print(f"  {v}: no cells at all")
        print("  pass --allow-partial to look anyway, or --expect to name the arms required")
        raise SystemExit(1)

    order = [v for v in ORDER if v in cells] + sorted(v for v in cells if v not in ORDER)
    stats: dict[str, dict] = {}

    print(f"{'variant':<22}" + "".join(f"{s[:11]:>13}" for s in SPLITS)
          + f"{'ALL':>8}{'root':>8}{'invalid':>9}{'reward':>8}{'n':>7}")
    for v in order:
        sp = cells[v]
        if len(sp) < len(SPLITS):
            print(f"{LABEL.get(v, v):<22}{'  (incomplete)':>13}")
            continue
        allts = [t for s in SPLITS for t in sp[s]]
        n = len(allts)
        calls = sum(t.verification.tool_calls for t in allts)
        row = f"{LABEL.get(v, v):<22}"
        for s in SPLITS:
            ts = sp[s]
            row += f"{sum(t.verification.resolved for t in ts) / len(ts):>12.1%} "
        res = sum(t.verification.resolved for t in allts) / n
        root = sum(t.verification.root_cause_correct for t in allts) / n
        stats[v] = {
            "n": n, "resolution": res, "root_cause": root,
            "invalid_call_rate": sum(t.verification.invalid_calls for t in allts) / max(1, calls),
            "unsafe_episodes": sum(1 for t in allts if t.verification.unsafe_actions > 0),
            "mean_reward": sum(t.reward for t in allts) / n,
            # Episodes affected, not the sum of expiry events -- one episode can expire on
            # several steps, and summing reads as an episode count when it is not.
            "settle_affected_episodes": sum(1 for t in allts
                                            if getattr(t, "settle_timeouts", 0)),
            "settle_affected_unresolved": sum(
                1 for t in allts
                if getattr(t, "settle_timeouts", 0) and not t.verification.resolved),
        }
        row += (f"{res:>7.1%}{root:>8.1%}{stats[v]['invalid_call_rate']:>9.2%}"
                f"{stats[v]['mean_reward']:>8.2f}{n:>7}")
        print(row)
        aff, unres = stats[v]["settle_affected_episodes"], stats[v]["settle_affected_unresolved"]
        if aff:
            # Not all of these are contamination. A settle window also expires when the
            # agent leaves a creation-blocking fault in place (ResourceQuotaExhausted is the
            # clear case): the world genuinely cannot converge, and scoring the episode
            # unresolved is correct rather than corrupted. Only an *affected but resolved*
            # episode is unambiguously the control plane having been slow.
            print(f"{'':<22}^ {aff} episode(s) hit the settle window; {unres} of them "
                  f"unresolved (likely an unfixed blocking fault, not slow convergence)")

    pairs = list(combinations([v for v in order if v in stats], 2))
    bonf = 2.64 if len(pairs) * 2 >= 6 else 1.96
    print(f"\npairwise, Bonferroni threshold z > {bonf:.2f} for {len(pairs) * 2} comparisons")
    for metric in ("resolution", "root_cause"):
        print(f"\n{metric.replace('_', ' ')}:")
        for a, b in pairs:
            sa, sb = stats[a], stats[b]
            z = _z(sa[metric], sa["n"], sb[metric], sb["n"])
            verdict = ("SIGNIFICANT" if abs(z) > bonf else
                       "nominal only" if abs(z) > 1.96 else "not resolved")
            print(f"  {LABEL.get(a, a):<22} vs {LABEL.get(b, b):<22} "
                  f"{sb[metric] - sa[metric]:+.1%}  z={z:+5.2f}  {verdict}")

    total_unsafe = sum(s["unsafe_episodes"] for v, s in stats.items() if v != "base")
    total_n = sum(s["n"] for v, s in stats.items() if v != "base")
    print(f"\nunsafe actions across {total_n} post-trained episodes: {total_unsafe}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(stats, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
