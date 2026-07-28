"""Typed agent state and the action protocol the harness drives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class Phase(str, Enum):
    ORIENT = "orient"          # gather topology and symptoms
    INVESTIGATE = "investigate"  # narrow to a root cause
    REMEDIATE = "remediate"    # apply a fix
    VALIDATE = "validate"      # confirm the fix worked
    DONE = "done"


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    thought: str = ""
    # Exact text the model generated, kept so SFT trains on the real completion.
    raw: str = ""


@dataclass
class Finish:
    root_cause: str
    summary: str
    thought: str = ""
    raw: str = ""


Action = ToolCall | Finish


@dataclass
class AgentState:
    """Everything the policy may condition on. Serializable for checkpoint/resume."""

    incident_id: str
    namespace: str
    phase: Phase = Phase.ORIENT
    step: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Rolling notes the harness maintains: what has been observed, what has been tried.
    observed_services: dict[str, str] = field(default_factory=dict)
    attempted_fixes: list[str] = field(default_factory=list)
    consecutive_errors: int = 0
    last_error: str | None = None
    budget_remaining: int = 0
    diagnosis: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "namespace": self.namespace,
            "phase": self.phase.value,
            "step": self.step,
            "messages": self.messages,
            "observed_services": self.observed_services,
            "attempted_fixes": self.attempted_fixes,
            "consecutive_errors": self.consecutive_errors,
            "budget_remaining": self.budget_remaining,
            "diagnosis": self.diagnosis,
        }

    @classmethod
    def restore(cls, d: dict[str, Any]) -> AgentState:
        state = cls(incident_id=d["incident_id"], namespace=d["namespace"])
        state.phase = Phase(d.get("phase", "orient"))
        state.step = d.get("step", 0)
        state.messages = d.get("messages", [])
        state.observed_services = d.get("observed_services", {})
        state.attempted_fixes = d.get("attempted_fixes", [])
        state.consecutive_errors = d.get("consecutive_errors", 0)
        state.budget_remaining = d.get("budget_remaining", 0)
        state.diagnosis = d.get("diagnosis")
        return state


class Policy(Protocol):
    """Anything that can choose the next action. Implemented by the scripted expert,
    the LLM policy, and the baselines."""

    name: str

    def reset(self, state: AgentState) -> None: ...

    def act(self, state: AgentState, tool_schemas: list[dict[str, Any]]) -> Action: ...


__all__ = ["AgentState", "Phase", "ToolCall", "Finish", "Action", "Policy"]
