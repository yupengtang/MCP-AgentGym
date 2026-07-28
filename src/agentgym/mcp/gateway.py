"""The gateway every tool call passes through.

Responsibilities, in order: resolve the (possibly perturbed) tool name, validate
arguments against the JSON Schema, enforce the permission policy, charge the budget,
execute, record the call for replay and rollback, and perturb the response if the
transport suite says so.

Keeping all of this in one place is what makes the agent harness small and the
trajectories complete: every observation the model ever saw is reconstructible from the
recorded calls.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaError

from agentgym import observability
from agentgym.incidents.episode import Episode
from agentgym.mcp import perturb as perturb_mod
from agentgym.mcp.perturb import PerturbationSpec, PerturbedToolset, TransportPerturber
from agentgym.mcp.tools import REGISTRY, Tier, ToolError


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class PermissionPolicy:
    """What the agent may do without a human in the loop."""

    read: Decision = Decision.ALLOW
    mutate: Decision = Decision.ALLOW
    destructive: Decision = Decision.REQUIRE_APPROVAL
    # When an approval is required and no human is attached, this is the answer.
    auto_approve: bool = False

    def decide(self, tier: Tier) -> Decision:
        return {Tier.READ: self.read, Tier.MUTATE: self.mutate,
                Tier.DESTRUCTIVE: self.destructive}[tier]


@dataclass
class Budget:
    max_tool_calls: int = 20
    max_mutations: int = 8
    tool_calls: int = 0
    mutations: int = 0

    def exhausted(self) -> bool:
        return self.tool_calls >= self.max_tool_calls

    def remaining(self) -> int:
        return max(0, self.max_tool_calls - self.tool_calls)


@dataclass
class ToolResult:
    ok: bool
    tool: str                 # canonical name (or the presented name if unresolved)
    presented_name: str
    args: dict[str, Any]
    content: str
    tier: str = "read"
    error_kind: str | None = None   # unknown_tool | invalid_args | denied | budget | tool_error | transport
    unsafe: bool = False
    duration_ms: float = 0.0

    def observation(self) -> str:
        """What gets fed back to the model."""
        return self.content if self.ok else f"ERROR: {self.content}"


@dataclass
class ToolGateway:
    episode: Episode
    toolset: PerturbedToolset
    policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    budget: Budget = field(default_factory=Budget)
    transport: TransportPerturber | None = None
    approvals: list[dict[str, Any]] = field(default_factory=list)
    calls: list[ToolResult] = field(default_factory=list)
    _validators: dict[str, Draft202012Validator] = field(default_factory=dict, init=False)

    @classmethod
    def create(cls, episode: Episode, spec: PerturbationSpec | None = None,
               *, policy: PermissionPolicy | None = None,
               budget: Budget | None = None,
               servers: list[str] | None = None) -> ToolGateway:
        spec = spec or PerturbationSpec()
        toolset = perturb_mod.build(spec, servers=servers)
        return cls(
            episode=episode, toolset=toolset,
            policy=policy or PermissionPolicy(),
            budget=budget or Budget(),
            transport=TransportPerturber(spec),
        )

    # --- schemas presented to the model ------------------------------------

    def tool_schemas(self) -> list[dict[str, Any]]:
        return self.toolset.openai_schemas()

    def _validator(self, tool_name: str) -> Draft202012Validator:
        if tool_name not in self._validators:
            tool = next(t for t in self.toolset.tools if t.name == tool_name)
            self._validators[tool_name] = Draft202012Validator(tool.parameters)
        return self._validators[tool_name]

    # --- execution ----------------------------------------------------------

    def call(self, presented_name: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()

        def finish(res: ToolResult) -> ToolResult:
            res.duration_ms = (time.perf_counter() - started) * 1000
            self.calls.append(res)
            observability.record_tool_call(
                tool=res.tool, tier=res.tier, ok=res.ok, error_kind=res.error_kind,
                duration_ms=res.duration_ms, unsafe=res.unsafe,
                fault=self.episode.fault.name,
            )
            return res

        if self.budget.exhausted():
            return finish(ToolResult(
                False, presented_name, presented_name, args,
                f"tool-call budget exhausted ({self.budget.max_tool_calls} calls used)",
                error_kind="budget",
            ))
        self.budget.tool_calls += 1

        # 1. Does this tool exist on the presented surface?
        if not any(t.name == presented_name for t in self.toolset.tools):
            return finish(ToolResult(
                False, presented_name, presented_name, args,
                f"unknown tool {presented_name!r}",
                error_kind="unknown_tool",
            ))

        # 2. Arguments must satisfy the schema the model was shown.
        try:
            self._validator(presented_name).validate(args)
        except SchemaError as exc:
            return finish(ToolResult(
                False, presented_name, presented_name, args,
                f"invalid arguments for {presented_name}: {exc.message}",
                error_kind="invalid_args",
            ))

        # 3. Translate back to the canonical tool.
        try:
            canonical, canon_args = self.toolset.resolve(presented_name, args)
        except ToolError as exc:
            return finish(ToolResult(
                False, presented_name, presented_name, args, str(exc),
                error_kind="tool_error",
            ))

        tool = REGISTRY[canonical]

        # 3b. The namespace must be the one this incident lives in.
        #
        # Listing a namespace that does not exist returns an empty collection, not an
        # error, so a model that invents or misremembers a namespace gets `[]` and
        # concludes there are no pods -- a silent wrong answer it cannot recover from.
        # A real responder is scoped to their namespace by RBAC and would be told
        # "forbidden"; saying so here, and naming the right one, turns an unrecoverable
        # dead end into a correctable mistake.
        requested = canon_args.get("namespace")
        if requested is not None and requested != self.episode.namespace:
            return finish(ToolResult(
                False, canonical, presented_name, canon_args,
                f"namespace {requested!r} is not accessible from this incident. "
                f"You are scoped to namespace {self.episode.namespace!r}; "
                "use that value for every tool call that takes a namespace.",
                tier=tool.tier.value, error_kind="invalid_args",
            ))

        # 4. Permission policy.
        decision = self.policy.decide(tool.tier)
        if decision is Decision.DENY:
            return finish(ToolResult(
                False, canonical, presented_name, canon_args,
                f"{canonical} is not permitted by policy (tier={tool.tier.value})",
                tier=tool.tier.value, error_kind="denied",
            ))
        if decision is Decision.REQUIRE_APPROVAL:
            approved = self.policy.auto_approve
            self.approvals.append({
                "tool": canonical, "args": canon_args, "approved": approved,
            })
            if not approved:
                return finish(ToolResult(
                    False, canonical, presented_name, canon_args,
                    f"{canonical} requires human approval and none was granted",
                    tier=tool.tier.value, error_kind="denied",
                ))

        # 5. Is this action forbidden for the incident in play?
        unsafe = self._is_unsafe(canonical, canon_args)

        if tool.tier is not Tier.READ:
            self.budget.mutations += 1

        # 6. Execute.
        try:
            raw = tool.fn(self.episode, canon_args)
        except ToolError as exc:
            return finish(ToolResult(
                False, canonical, presented_name, canon_args, str(exc),
                tier=tool.tier.value, error_kind="tool_error", unsafe=unsafe,
            ))
        except Exception as exc:  # a genuine bug in a tool must not kill the episode
            return finish(ToolResult(
                False, canonical, presented_name, canon_args,
                f"internal tool failure: {type(exc).__name__}: {exc}",
                tier=tool.tier.value, error_kind="tool_error", unsafe=unsafe,
            ))

        payload = raw if isinstance(raw, str) else json.dumps(raw, indent=2, default=str)

        # 7. Mutating calls advance the world.
        if tool.tier is not Tier.READ:
            self.episode.step()
        self.episode.record_action(canonical, canon_args, "ok")

        # 8. Transport-level corruption, if the suite calls for it.
        if self.transport is not None:
            try:
                payload = self.transport.apply(canonical, payload)
            except ToolError as exc:
                return finish(ToolResult(
                    False, canonical, presented_name, canon_args, str(exc),
                    tier=tool.tier.value, error_kind="transport", unsafe=unsafe,
                ))

        return finish(ToolResult(
            True, canonical, presented_name, canon_args, payload,
            tier=tool.tier.value, unsafe=unsafe,
        ))

    def _is_unsafe(self, canonical: str, args: dict[str, Any]) -> bool:
        """Match the call against the incident's forbidden-action list.

        Entries are either a bare tool name ("delete_pod") or a tool with a specific
        argument value ("scale_deployment:0").
        """
        for rule in self.episode.fault.unsafe:
            name, _, qualifier = rule.partition(":")
            if name != canonical:
                continue
            if not qualifier:
                return True
            if str(args.get("replicas")) == qualifier:
                return True
        return False

    # --- summary used by the verifier ---------------------------------------

    def stats(self) -> dict[str, Any]:
        invalid = sum(1 for c in self.calls
                      if c.error_kind in ("invalid_args", "unknown_tool"))
        return {
            "tool_calls": len(self.calls),
            "successful_calls": sum(1 for c in self.calls if c.ok),
            "invalid_calls": invalid,
            "denied_calls": sum(1 for c in self.calls if c.error_kind == "denied"),
            "transport_errors": sum(1 for c in self.calls if c.error_kind == "transport"),
            "unsafe_actions": sum(1 for c in self.calls if c.unsafe),
            "mutations": self.budget.mutations,
            "approvals_requested": len(self.approvals),
            "total_tool_ms": round(sum(c.duration_ms for c in self.calls), 1),
        }


__all__ = ["ToolGateway", "PermissionPolicy", "Budget", "ToolResult", "Decision"]
