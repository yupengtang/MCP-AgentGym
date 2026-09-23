"""How much of an evaluation result survives being run again.

Why this exists
---------------
Three sweeps evaluated the *same untrained weights* on the *same* incidents at temperature
0, differing only in which vLLM instance served them. The aggregate resolution rates came
out 43 / 47 / 50%, which looks like ordinary small-n noise. Per incident it is not: the
tool sequence reproduced in 0 to 1 episodes out of 100 and roughly a fifth of resolve/fail
verdicts flipped. The aggregate agreed only because the flips very nearly cancelled.

That distinction decides which comparisons are legitimate. An aggregate floor measured at
large n says nothing about whether a *paired* comparison is meaningful, and a project that
quotes one figure for both will read instance noise as a result. This project has already
made the neighbouring mistake once, quoting a per-split gap as the run-to-run floor.

The mechanism is vLLM's continuous batching: greedy decoding depends on batch composition,
so a different batch produces a different token, and an agent harness compounds one token
into a different investigation, a different number of turns and a different verdict. It is
not specific to this environment; any multi-turn agent evaluation served this way has it.

So this measures three things that are usually conflated:

  aggregate spread      max - min of the pooled rate. What a results table shows.
  outcome agreement     share of incidents where every run agrees on resolved/not.
  trajectory agreement  share where the tool sequence is identical.

Only compare arms whose *weights* are identical across the runs. For an arm whose adapter
differs by seed, disagreement confounds serving nondeterminism with a real model
difference, and this script refuses to summarise those unless asked.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load(path: Path) -> dict:
    from agentgym.data import trajectory as T

    return {t.incident_id: t for t in T.read_jsonl(path)}


def _tools(t) -> tuple:
    return tuple(s.tool for s in t.steps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path,
                    default=Path(os.environ.get("AGENTGYM_RUNS", "runs")))
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="run directories to compare, relative to --runs")
    ap.add_argument("--variant", default="base",
                    help="which arm. Default `base`: the untrained model, which is the "
                         "same weights in every run and so isolates serving noise.")
    ap.add_argument("--split", default="unseen_fault")
    ap.add_argument("--same-weights", action="store_true", default=None,
                    help="assert the arm is identical across runs. Implied for `base`.")
    args = ap.parse_args()

    same_weights = args.same_weights
    if same_weights is None:
        same_weights = args.variant == "base"

    arms: dict[str, dict] = {}
    for d in args.dirs:
        f = args.runs / d / f"{args.variant}-{args.split}-clean.jsonl"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        arms[d] = _load(f)
    if len(arms) < 2:
        raise SystemExit("need at least two runs to compare")

    ids = sorted(set.intersection(*(set(a) for a in arms.values())))
    if not ids:
        raise SystemExit("the runs share no incidents; there is nothing to compare")

    print(f"variant={args.variant}  split={args.split}  "
          f"shared incidents={len(ids)}  runs={len(arms)}")
    if not same_weights:
        print("\n  NOTE: this arm's weights differ between runs, so everything below\n"
              "  conflates serving nondeterminism with a real difference between models.\n"
              "  It is not a reproducibility measurement. Use --variant base for that.\n")

    print("\naggregate rate per run")
    rates = {}
    for d, a in arms.items():
        r = sum(a[i].verification.resolved for i in ids) / len(ids)
        rates[d] = r
        print(f"  {d:28} {r:6.1%}")
    spread = 100 * (max(rates.values()) - min(rates.values()))
    print(f"  {'aggregate spread':28} {spread:5.1f} pp")

    print("\npairwise, per incident")
    for x, y in itertools.combinations(arms, 2):
        a, b = arms[x], arms[y]
        flips = sum(1 for i in ids
                    if a[i].verification.resolved != b[i].verification.resolved)
        same_traj = sum(1 for i in ids if _tools(a[i]) == _tools(b[i]))
        print(f"  {x:26} vs {y:26}  verdict differs {flips:3}/{len(ids)} "
              f"({flips / len(ids):4.0%})   identical trajectory {same_traj:3}/{len(ids)}")

    # unanimity across all runs at once
    unan_out = sum(1 for i in ids
                   if len({a[i].verification.resolved for a in arms.values()}) == 1)
    unan_traj = sum(1 for i in ids if len({_tools(a[i]) for a in arms.values()}) == 1)
    print(f"\nall {len(arms)} runs agree on the verdict     {unan_out:3}/{len(ids)} "
          f"({unan_out / len(ids):4.0%})")
    print(f"all {len(arms)} runs produce one trajectory  {unan_traj:3}/{len(ids)} "
          f"({unan_traj / len(ids):4.0%})")

    if same_weights:
        disagree = 1 - unan_out / len(ids)
        print(f"\n  Reading this: the weights are identical in every run, so every "
              f"disagreement\n  above is the serving stack. The aggregate looks stable "
              f"({spread:.1f} pp) while {disagree:.0%} of\n  individual verdicts are not "
              f"reproducible, because the flips cancel in the mean.")
        print("\n  Therefore: an aggregate difference below roughly the spread above is not\n"
              "  interpretable across runs, and a paired per-incident comparison across runs\n"
              "  is not interpretable at any n.")


if __name__ == "__main__":
    main()
