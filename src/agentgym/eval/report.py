"""Build the evaluation matrices from trajectory files.

Three tables, one per audience (see PROJECT.md §6):

  capability      how well the agent does the job
  generalization  whether it survives unfamiliar faults, topologies, and tool schemas
  systems         what it costs to run

Everything is computed from the verified trajectories; nothing here re-judges an episode.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentgym.data import trajectory as traj_mod
from agentgym.data.trajectory import Trajectory

VARIANT_ORDER = ["expert-scripted", "base", "prompted", "sft",
                 "dpo", "dpo5", "grpo"]
SUITE_ORDER = ["clean", "rename", "paraphrase", "arg_rename", "extra_required_arg",
               "retype_arg", "distractors", "server_dropout", "malformed",
               "truncation", "flaky", "combined"]
SPLIT_ORDER = ["in_distribution", "unseen_fault", "unseen_topology", "compositional"]
# Robustness is measured on one split so the only thing varying is the tool surface.
ROBUSTNESS_SPLIT = "in_distribution"


@dataclass
class Cell:
    n: int = 0
    resolved: int = 0
    root_cause: int = 0
    clean: int = 0
    unsafe: int = 0
    invalid: int = 0
    tool_calls: int = 0
    reward: float = 0.0
    wall: float = 0.0
    completion_tokens: int = 0
    truncated: int = 0

    def add(self, t: Trajectory) -> None:
        v = t.verification
        self.n += 1
        self.resolved += int(v.resolved)
        self.root_cause += int(v.root_cause_correct)
        self.clean += int(t.clean)
        self.unsafe += int(v.unsafe_actions > 0)
        self.invalid += v.invalid_calls
        self.tool_calls += v.tool_calls
        self.reward += t.reward
        self.wall += t.wall_seconds
        self.completion_tokens += t.completion_tokens
        self.truncated += getattr(t, 'truncated_turns', 0)

    def metrics(self) -> dict[str, Any]:
        if not self.n:
            return {}
        return {
            "n": self.n,
            "resolution_rate": round(self.resolved / self.n, 4),
            "root_cause_accuracy": round(self.root_cause / self.n, 4),
            "clean_rate": round(self.clean / self.n, 4),
            "unsafe_action_rate": round(self.unsafe / self.n, 4),
            "invalid_call_rate": round(self.invalid / max(1, self.tool_calls), 4),
            "mean_tool_calls": round(self.tool_calls / self.n, 2),
            "mean_reward": round(self.reward / self.n, 3),
            "mean_wall_seconds": round(self.wall / self.n, 2),
            "mean_completion_tokens": round(self.completion_tokens / self.n, 1),
            # Turns that hit the generation cap before emitting a call. A high
            # value means the budget, not the model, produced the failures.
            "truncated_turn_rate": round(self.truncated / max(1, self.tool_calls), 4),
        }


def load_all(root: Path) -> list[Trajectory]:
    """Read every trajectory under `root`, recursively."""
    out: list[Trajectory] = []
    for path in sorted(root.rglob("*.jsonl")):
        if path.name.startswith("summary"):
            continue
        try:
            out.extend(traj_mod.read_jsonl(path))
        except Exception:
            continue
    return out


def _variant_of(t: Trajectory) -> str:
    """Normalise the recorded model name into a variant label.

    Substring matching must not collapse ablation arms: a run served as `dpo5` is a
    different policy from `dpo`, and merging them would average two arms into one row
    that describes neither -- silently, and with a plausible-looking number. Any suffix
    after the family name is preserved as its own variant.
    """
    m = (t.model or "").lower()
    if "expert" in m:
        return "expert-scripted"
    for family in ("grpo", "dpo", "sft"):
        if family in m:
            # `dpo` -> "dpo"; `dpo5` / `dpo-v5` -> "dpo5" / "dpo-v5"
            suffix = m.split(family, 1)[1].strip("-_/ ")
            return f"{family}{suffix}" if suffix else family
    return "base"


def capability_matrix(trajectories: list[Trajectory]) -> dict[str, dict[str, Any]]:
    """Variant x overall performance, on clean tools only."""
    cells: dict[str, Cell] = defaultdict(Cell)
    for t in trajectories:
        if t.perturbation != "clean":
            continue
        cells[_variant_of(t)].add(t)
    order = VARIANT_ORDER + sorted(k for k in cells if k not in VARIANT_ORDER)
    return {k: cells[k].metrics() for k in order if k in cells}


def generalization_matrix(trajectories: list[Trajectory]) -> dict[str, dict[str, Any]]:
    """Variant x split, plus variant x perturbation suite. The headline table."""
    by_split: dict[str, dict[str, Cell]] = defaultdict(lambda: defaultdict(Cell))
    by_suite: dict[str, dict[str, Cell]] = defaultdict(lambda: defaultdict(Cell))
    for t in trajectories:
        variant = _variant_of(t)
        if t.perturbation == "clean":
            by_split[variant][t.split].add(t)
        by_suite[variant][t.perturbation].add(t)

    def ordered(seen: dict) -> list[str]:
        # Never drop a variant just because it is not in the hard-coded order; an
        # unlisted arm silently vanishing from a table is worse than an odd sort.
        return VARIANT_ORDER + sorted(k for k in seen if k not in VARIANT_ORDER)

    return {
        "by_split": {
            v: {s: by_split[v][s].metrics() for s in SPLIT_ORDER if s in by_split[v]}
            for v in ordered(by_split) if v in by_split
        },
        "by_perturbation": {
            v: {s: by_suite[v][s].metrics() for s in SUITE_ORDER if s in by_suite[v]}
            for v in ordered(by_suite) if v in by_suite
        },
    }


def robustness_summary(trajectories: list[Trajectory]) -> dict[str, Any]:
    """How much each perturbation costs, relative to that variant's clean baseline.

    Absolute numbers under perturbation are hard to read across variants with different
    baselines; the retention ratio is the comparable quantity.
    """
    # Scope to one split. Perturbation suites are only run on in-distribution incidents,
    # so a clean baseline averaged over all four splits (including the much harder OOD
    # ones) is a different population -- and dividing by it produces retention above
    # 1.00, i.e. "the model does better with renamed tools than with correct ones".
    # The comparison has to hold everything except the tool surface fixed.
    by_suite: dict[str, dict[str, Cell]] = defaultdict(lambda: defaultdict(Cell))
    for t in trajectories:
        if t.split != ROBUSTNESS_SPLIT:
            continue
        by_suite[_variant_of(t)][t.perturbation].add(t)

    out: dict[str, Any] = {}
    for variant, suites in by_suite.items():
        clean = suites.get("clean")
        if not clean or not clean.n or not clean.resolved:
            continue
        base_rate = clean.resolved / clean.n
        out[variant] = {
            suite: {
                "resolution_rate": round(cell.resolved / cell.n, 4),
                "retention_vs_clean": round((cell.resolved / cell.n) / base_rate, 4),
                "n": cell.n,
            }
            for suite, cell in sorted(suites.items()) if cell.n
        }
    return out


def per_fault_table(trajectories: list[Trajectory]) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Cell]] = defaultdict(lambda: defaultdict(Cell))
    for t in trajectories:
        if t.perturbation == "clean":
            cells[_variant_of(t)][t.fault].add(t)
    return {
        v: {f: c.metrics() for f, c in sorted(faults.items())}
        for v, faults in cells.items()
    }


def build(root: Path) -> dict[str, Any]:
    trajectories = load_all(root)
    return {
        "episodes_analyzed": len(trajectories),
        "capability": capability_matrix(trajectories),
        "generalization": generalization_matrix(trajectories),
        "robustness": robustness_summary(trajectories),
        "per_fault": per_fault_table(trajectories),
    }


# --- markdown rendering ------------------------------------------------------

def _md_table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = "\n".join(
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(header))) + " |"
        for r in rows
    )
    return f"{line}\n{sep}\n{body}" if rows else f"{line}\n{sep}"


def to_markdown(report: dict[str, Any]) -> str:
    parts: list[str] = ["# MCP-AgentGym evaluation\n",
                        f"Episodes analyzed: **{report['episodes_analyzed']}**\n"]

    cap = report["capability"]
    if cap:
        parts.append("\n## Agent capability (clean tool schemas)\n")
        header = ["variant", "n", "resolution", "root cause", "clean", "unsafe",
                  "invalid calls", "tool calls", "reward"]
        rows = [[v, str(m["n"]), f"{m['resolution_rate']:.1%}",
                 f"{m['root_cause_accuracy']:.1%}", f"{m['clean_rate']:.1%}",
                 f"{m['unsafe_action_rate']:.1%}", f"{m['invalid_call_rate']:.1%}",
                 f"{m['mean_tool_calls']:.1f}", f"{m['mean_reward']:.2f}"]
                for v, m in cap.items() if m]
        parts.append(_md_table(rows, header))

    gen = report["generalization"]["by_split"]
    if gen:
        parts.append("\n\n## Generalization by split (resolution rate)\n")
        splits = [s for s in SPLIT_ORDER
                  if any(s in per for per in gen.values())]
        header = ["variant", *splits]
        rows = []
        for v, per in gen.items():
            rows.append([v, *[f"{per[s]['resolution_rate']:.1%}" if s in per else "-"
                              for s in splits]])
        parts.append(_md_table(rows, header))

    rob = report["robustness"]
    if rob:
        parts.append("\n\n## Robustness to tool-schema shift (retention vs clean)\n")
        suites = [s for s in SUITE_ORDER if any(s in per for per in rob.values())]
        header = ["variant", *suites]
        rows = []
        for v, per in rob.items():
            rows.append([v, *[f"{per[s]['retention_vs_clean']:.2f}" if s in per else "-"
                              for s in suites]])
        parts.append(_md_table(rows, header))
        parts.append("\n\n_Retention is resolution rate under the suite divided by that "
                     "variant's own clean resolution rate: 1.00 means the perturbation "
                     "cost nothing._\n")

    return "".join(parts)


def main() -> None:
    import argparse

    from agentgym import config as cfg

    p = argparse.ArgumentParser(description="Build the AgentGym evaluation report")
    p.add_argument("--runs", type=Path, default=cfg.RUNS_DIR)
    p.add_argument("--out", type=Path, default=cfg.ARTIFACTS_DIR / "evaluation")
    args = p.parse_args()

    report = build(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    (args.out / "report.md").write_text(to_markdown(report))
    print(to_markdown(report))
    print(f"\nwrote {args.out}/report.json and report.md")


if __name__ == "__main__":
    main()
