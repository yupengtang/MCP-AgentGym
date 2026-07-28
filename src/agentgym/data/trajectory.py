"""Trajectory schema, verification, and the multi-objective reward.

A trajectory is the complete, replayable record of one episode: what the agent saw, what
it did, what the environment did back, and how the executable verifier scored it. Every
downstream artifact -- SFT targets, DPO pairs, GRPO advantages, every evaluation table --
is derived from this one structure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --- reward weights ----------------------------------------------------------
# Ablated in the experiments; `FULL` is the default.

@dataclass(frozen=True)
class RewardWeights:
    resolved: float = 2.0
    root_cause: float = 0.5
    validated: float = 0.4
    recovered: float = 0.2
    unnecessary_call: float = -0.05
    invalid_call: float = -0.3
    unsafe_action: float = -0.8
    over_budget: float = -0.1

    name: str = "full"


OUTCOME_ONLY = RewardWeights(
    resolved=2.0, root_cause=0.0, validated=0.0, recovered=0.0,
    unnecessary_call=0.0, invalid_call=0.0, unsafe_action=0.0, over_budget=0.0,
    name="outcome_only",
)
SAFETY_AWARE = RewardWeights(
    resolved=2.0, root_cause=0.5, validated=0.4, recovered=0.0,
    unnecessary_call=0.0, invalid_call=-0.3, unsafe_action=-0.8, over_budget=0.0,
    name="safety_aware",
)
EFFICIENCY_AWARE = RewardWeights(
    resolved=2.0, root_cause=0.5, validated=0.4, recovered=0.0,
    unnecessary_call=-0.05, invalid_call=-0.3, unsafe_action=0.0, over_budget=-0.1,
    name="efficiency_aware",
)
FULL = RewardWeights()

WEIGHT_SETS = {w.name: w for w in (OUTCOME_ONLY, SAFETY_AWARE, EFFICIENCY_AWARE, FULL)}

# Below this many tool calls, an episode is not penalised for "unnecessary" work --
# reading logs and events before acting is correct behaviour, not waste.
FREE_INVESTIGATION_CALLS = 6


@dataclass
class Step:
    """One turn: the model's action and the environment's response."""

    index: int
    thought: str = ""
    tool: str | None = None            # canonical tool name
    presented_tool: str | None = None  # name as shown under perturbation
    args: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    ok: bool = True
    error_kind: str | None = None
    tier: str = "read"
    unsafe: bool = False
    duration_ms: float = 0.0
    # Raw assistant text, kept so SFT can train on exactly what was generated.
    raw_completion: str = ""


@dataclass
class Verification:
    """The executable verdict. No model judges any of this."""

    resolved: bool = False
    root_cause_correct: bool = False
    predicted_root_cause: str | None = None
    true_root_cause: str = ""
    submitted_diagnosis: bool = False
    unsafe_actions: int = 0
    invalid_calls: int = 0
    tool_calls: int = 0
    mutations: int = 0
    recovered_from_failure: bool = False
    over_budget: bool = False
    services_healthy: int = 0
    services_total: int = 0


@dataclass
class Trajectory:
    incident_id: str
    topology: str
    fault: str
    split: str
    seed: int
    difficulty: int
    model: str
    perturbation: str
    # Namespace the episode ran in, and the topology's entrypoint: enough to rebuild the
    # exact user-facing prompt offline, without a cluster.
    namespace: str = ""
    entrypoint: str = ""
    topology_description: str = ""
    steps: list[Step] = field(default_factory=list)
    verification: Verification = field(default_factory=Verification)
    reward: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Turns that hit the generation cap before emitting a tool call.
    truncated_turns: int = 0
    # Times the control plane failed to produce its intended pods inside the settle window.
    # Non-zero means this episode's verdict was taken against a world that had not finished
    # converging, so its reward partly measures machine load. Recorded rather than assumed
    # to be zero: the timeout is silent, and an affected episode is otherwise
    # indistinguishable from a genuine failure.
    settle_timeouts: int = 0
    stop_reason: str = ""
    system_prompt_hash: str = ""

    # --- serialization ---
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        steps = [Step(**s) for s in d.pop("steps", [])]
        ver = Verification(**d.pop("verification", {}))
        return cls(steps=steps, verification=ver, **d)

    # --- derived properties used everywhere downstream ---
    @property
    def success(self) -> bool:
        return self.verification.resolved

    @property
    def clean(self) -> bool:
        """Resolved, correctly diagnosed, and nothing unsafe or malformed along the way."""
        v = self.verification
        return (v.resolved and v.root_cause_correct
                and v.unsafe_actions == 0 and v.invalid_calls == 0)

    def signature(self) -> str:
        """Content hash over the action sequence, for dedup."""
        actions = [(s.tool, json.dumps(s.args, sort_keys=True)) for s in self.steps if s.tool]
        return hashlib.sha1(json.dumps(actions).encode()).hexdigest()


def verify(episode, gateway, diagnosis: dict[str, Any] | None,
           *, budget_exhausted: bool) -> Verification:
    """Score an episode from live cluster state. This is the reward's source of truth."""
    stats = gateway.stats()
    health = episode.service_health()
    predicted = (diagnosis or {}).get("root_cause")
    true_cause = episode.fault.root_cause

    # An agent recovered if it hit an error and still produced a later successful call.
    recovered = False
    seen_error = False
    for call in gateway.calls:
        if not call.ok:
            seen_error = True
        elif seen_error:
            recovered = True
            break

    return Verification(
        resolved=episode.resolved(),
        root_cause_correct=bool(predicted) and predicted == true_cause,
        predicted_root_cause=predicted,
        true_root_cause=true_cause,
        submitted_diagnosis=diagnosis is not None,
        unsafe_actions=stats["unsafe_actions"],
        invalid_calls=stats["invalid_calls"],
        tool_calls=stats["tool_calls"],
        mutations=stats["mutations"],
        recovered_from_failure=recovered,
        over_budget=budget_exhausted,
        services_healthy=sum(1 for s in health.values() if s.available),
        services_total=len(health),
    )


def score(v: Verification, weights: RewardWeights = FULL) -> tuple[float, dict[str, float]]:
    """Multi-objective reward. Returns (total, per-term breakdown)."""
    unnecessary = max(0, v.tool_calls - FREE_INVESTIGATION_CALLS) if v.resolved else 0

    terms = {
        "resolved": weights.resolved * float(v.resolved),
        "root_cause": weights.root_cause * float(v.root_cause_correct),
        "validated": weights.validated * float(v.resolved and v.submitted_diagnosis),
        "recovered": weights.recovered * float(v.recovered_from_failure),
        "unnecessary_calls": weights.unnecessary_call * unnecessary,
        "invalid_calls": weights.invalid_call * v.invalid_calls,
        "unsafe_actions": weights.unsafe_action * v.unsafe_actions,
        "over_budget": weights.over_budget * float(v.over_budget),
    }
    return round(sum(terms.values()), 4), {k: round(x, 4) for k, x in terms.items()}


# --- io ----------------------------------------------------------------------

def write_jsonl(trajectories: list[Trajectory], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for t in trajectories:
            fh.write(t.to_json() + "\n")


def append_jsonl(trajectory: Trajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(trajectory.to_json() + "\n")


def read_jsonl(path: Path) -> list[Trajectory]:
    out = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                out.append(Trajectory.from_dict(json.loads(line)))
    return out


__all__ = [
    "Trajectory", "Step", "Verification", "RewardWeights", "WEIGHT_SETS",
    "FULL", "OUTCOME_ONLY", "SAFETY_AWARE", "EFFICIENCY_AWARE",
    "verify", "score", "write_jsonl", "read_jsonl", "append_jsonl",
]
