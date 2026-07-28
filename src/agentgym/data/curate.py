"""Turn trajectories into training data.

Three products, all derived from the same verified trajectories:

  SFT   assistant turns from successful, clean episodes, with loss masked to the
        assistant tokens only
  DPO   preference pairs over the *same incident*, so the contrast is the behaviour and
        not the scenario
  GRPO  incident prompts for on-policy rollout

Curation is deliberately aggressive. A verified environment makes it cheap to throw away
anything questionable, and quality dominates quantity at this scale.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentgym.data.trajectory import Trajectory
from agentgym.harness.prompts import system_prompt

# A pair needs a real quality gap, or the preference signal is noise.
MIN_REWARD_MARGIN = 0.8
# Cap how many examples any single fault family may contribute, so the common families
# do not swamp the rare ones.
MAX_PER_FAULT_FRACTION = 0.18


@dataclass
class CurationReport:
    total: int = 0
    kept: int = 0
    dropped_unsuccessful: int = 0
    dropped_unclean: int = 0
    dropped_duplicate: int = 0
    dropped_eval_split: int = 0
    dropped_capped: int = 0
    dropped_too_short: int = 0
    per_fault: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _messages_for(t: Trajectory) -> list[dict[str, str]]:
    """Rebuild the conversation exactly as the policy saw it, plus its own turns."""
    from agentgym.harness.prompts import incident_brief_from_trajectory

    # SFT data comes only from clean-perturbation trajectories, so the catalog the model
    # is trained on is the unperturbed registry -- exactly what it is served at eval time
    # under the "clean" suite.
    from agentgym.mcp.tools import REGISTRY

    catalog = [t.openai_schema() for t in REGISTRY.values()]
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt(catalog)},
        {"role": "user", "content": incident_brief_from_trajectory(t)},
    ]
    for step in t.steps:
        assistant = _assistant_text(step)
        msgs.append({"role": "assistant", "content": assistant})
        if step.tool != "submit_diagnosis":
            msgs.append({"role": "user", "content": f"TOOL RESULT:\n{step.observation}"})
    return msgs


def _assistant_text(step) -> str:
    """Canonical assistant turn: brief reasoning, then one tool call as JSON.

    Using a single stable format for demonstrations matters more than matching any
    particular chat template -- the model has to learn one calling convention, and the
    gateway accepts this one.
    """
    call = {"name": step.presented_tool or step.tool, "arguments": step.args}
    thought = (step.thought or "").strip()
    body = json.dumps(call, indent=2)
    return f"{thought}\n\n```json\n{body}\n```" if thought else f"```json\n{body}\n```"


def build_sft(
    trajectories: Iterable[Trajectory],
    *,
    eval_incident_ids: set[str] | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], CurationReport]:
    """Select demonstration trajectories and render them as chat examples."""
    report = CurationReport()
    eval_ids = eval_incident_ids or set()
    rng = random.Random(seed)

    candidates: list[Trajectory] = []
    seen_signatures: set[str] = set()

    for t in trajectories:
        report.total += 1
        # Never train on anything an evaluation split will later measure.
        if t.split != "in_distribution" or t.incident_id in eval_ids:
            report.dropped_eval_split += 1
            continue
        if not t.success:
            report.dropped_unsuccessful += 1
            continue
        if not t.clean:
            report.dropped_unclean += 1
            continue
        if len([s for s in t.steps if s.tool]) < 3:
            report.dropped_too_short += 1
            continue
        sig = t.signature()
        if sig in seen_signatures:
            report.dropped_duplicate += 1
            continue
        seen_signatures.add(sig)
        candidates.append(t)

    # Rebalance so no single fault family dominates.
    rng.shuffle(candidates)
    cap = max(1, int(len(candidates) * MAX_PER_FAULT_FRACTION))
    per_fault: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for t in candidates:
        if per_fault[t.fault] >= cap:
            report.dropped_capped += 1
            continue
        per_fault[t.fault] += 1
        examples.append({
            "incident_id": t.incident_id,
            "fault": t.fault,
            "topology": t.topology,
            "reward": t.reward,
            "messages": _messages_for(t),
        })

    report.kept = len(examples)
    report.per_fault = dict(sorted(per_fault.items()))
    return examples, report


def build_dpo(
    trajectories: Iterable[Trajectory],
    *,
    eval_incident_ids: set[str] | None = None,
    min_margin: float = MIN_REWARD_MARGIN,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build preference pairs from differing attempts at the same incident.

    Pairing within an incident is what makes the preference informative: both sides face
    the identical cluster, so the only thing that differs is how the agent behaved.
    """
    eval_ids = eval_incident_ids or set()
    by_incident: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        if t.split != "in_distribution" or t.incident_id in eval_ids:
            continue
        by_incident[t.incident_id].append(t)

    pairs: list[dict[str, Any]] = []
    # Diagnose *why* an incident yields no pair. Without this the only observable is a
    # small pair count, which is equally consistent with "too few samples" and "the policy
    # is too consistent to disagree with itself" -- two problems with opposite fixes.
    stats = {"incidents_considered": 0, "incidents_with_pair": 0,
             "pairs": 0, "mean_margin": 0.0,
             "rejected_single_attempt": 0,
             "rejected_no_reward_spread": 0,
             "rejected_margin_too_small": 0,
             "rejected_best_did_not_solve": 0}
    margins: list[float] = []

    for incident_id, group in by_incident.items():
        if len(group) < 2:
            stats["rejected_single_attempt"] += 1
            continue
        stats["incidents_considered"] += 1
        ordered = sorted(group, key=lambda t: t.reward, reverse=True)
        best, worst = ordered[0], ordered[-1]
        margin = best.reward - worst.reward
        if margin == 0.0:
            # Every attempt scored identically: the policy did not vary, so there is no
            # preference to learn. More samples per incident will not help; more entropy
            # (or harder incidents) will.
            stats["rejected_no_reward_spread"] += 1
            continue
        if margin < min_margin:
            stats["rejected_margin_too_small"] += 1
            continue
        if not best.success:
            stats["rejected_best_did_not_solve"] += 1
            continue
        pairs.append({
            "incident_id": incident_id,
            "fault": best.fault,
            "margin": round(margin, 3),
            "prompt": _messages_for(best)[:2],
            "chosen": _messages_for(best)[2:],
            "rejected": _messages_for(worst)[2:],
            "chosen_reward": best.reward,
            "rejected_reward": worst.reward,
        })
        margins.append(margin)
        stats["incidents_with_pair"] += 1

    stats["pairs"] = len(pairs)
    stats["mean_margin"] = round(sum(margins) / len(margins), 3) if margins else 0.0
    return pairs, stats


def contamination_check(
    sft_examples: list[dict[str, Any]],
    eval_incident_ids: set[str],
) -> dict[str, Any]:
    """Assert no evaluation incident leaked into training."""
    train_ids = {e["incident_id"] for e in sft_examples}
    overlap = train_ids & eval_incident_ids
    return {
        "train_examples": len(sft_examples),
        "train_incidents": len(train_ids),
        "eval_incidents": len(eval_incident_ids),
        "overlap": len(overlap),
        "clean": len(overlap) == 0,
        "overlapping_ids": sorted(overlap)[:20],
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


__all__ = [
    "build_sft", "build_dpo", "contamination_check", "CurationReport",
    "write_jsonl", "read_jsonl",
]
