"""The agent harness.

Written from scratch rather than assembled from a framework, because the failure
handling *is* the contribution: budgets, schema validation, permission gates, recovery
from tool errors, and a complete replayable record.

The loop is deliberately explicit:

    observe -> policy.act -> validate -> permit -> execute -> advance world -> verify

Everything the policy saw and everything the environment did is written into the
`Trajectory`, so a run can be replayed offline without a cluster.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentgym import observability
from agentgym.data import trajectory as traj_mod
from agentgym.data.trajectory import Step, Trajectory
from agentgym.harness.prompts import incident_brief, system_prompt, system_prompt_hash
from agentgym.harness.state import AgentState, Finish, Phase, Policy, ToolCall
from agentgym.incidents.episode import Episode
from agentgym.mcp.gateway import ToolGateway

# After this many consecutive failed calls the harness injects a recovery hint. Without
# it, a model that has lost the tool schema will burn its whole budget repeating the
# same malformed call.
RECOVERY_AFTER_ERRORS = 3


@dataclass
class HarnessConfig:
    max_steps: int = 20
    reward_weights: traj_mod.RewardWeights = field(default_factory=lambda: traj_mod.FULL)
    checkpoint_dir: Path | None = None
    # Emit a recovery hint when the agent gets stuck in an error loop.
    recovery_enabled: bool = True
    verbose: bool = False


@dataclass
class AgentHarness:
    episode: Episode
    gateway: ToolGateway
    policy: Policy
    config: HarnessConfig = field(default_factory=HarnessConfig)

    state: AgentState = field(init=False)

    def __post_init__(self) -> None:
        self.state = AgentState(
            incident_id=self.episode.namespace,
            namespace=self.episode.namespace,
            budget_remaining=self.gateway.budget.remaining(),
        )

    # --- main loop ----------------------------------------------------------

    def run(self, *, incident_id: str, split: str = "in_distribution",
            model: str = "unknown", perturbation: str = "clean",
            difficulty: int = 1) -> Trajectory:
        """Run one episode to completion and return its verified trajectory."""
        with observability.span(
            "episode", incident_id=incident_id, fault=self.episode.fault.name,
            topology=self.episode.topology.name, split=split, model=model,
            perturbation=perturbation, difficulty=difficulty,
        ) as episode_span:
            return self._run(incident_id, split, model, perturbation,
                             difficulty, episode_span)

    def _run(self, incident_id: str, split: str, model: str, perturbation: str,
             difficulty: int, episode_span: Any) -> Trajectory:
        started = time.perf_counter()
        ep = self.episode

        trajectory = Trajectory(
            incident_id=incident_id,
            topology=ep.topology.name,
            fault=ep.fault.name,
            split=split,
            seed=ep.seed,
            difficulty=difficulty,
            model=model,
            perturbation=perturbation,
            namespace=ep.namespace,
            entrypoint=ep.topology.entrypoint,
            topology_description=ep.topology.description,
            system_prompt_hash=system_prompt_hash(),
        )

        # The tool catalog goes in the system prompt, not the OpenAI `tools` field: the
        # chat template is a plain message renderer, so a separate field would be dropped
        # before the model saw it. This is also what makes the perturbation suites real --
        # a renamed tool is renamed *in the prompt*.
        self.state.messages = [
            {"role": "system", "content": system_prompt(self.gateway.tool_schemas())},
            {"role": "user", "content": incident_brief(ep)},
        ]
        self.policy.reset(self.state)

        diagnosis: dict[str, Any] | None = None
        stop_reason = "max_steps"

        for step_index in range(self.config.max_steps):
            self.state.step = step_index
            self.state.budget_remaining = self.gateway.budget.remaining()

            if self.gateway.budget.exhausted():
                stop_reason = "budget_exhausted"
                break

            try:
                action = self.policy.act(self.state, self.gateway.tool_schemas())
            except Exception as exc:
                # A policy that crashes ends the episode; it does not corrupt the record.
                trajectory.steps.append(Step(
                    index=step_index, thought="", observation=f"policy error: {exc}",
                    ok=False, error_kind="policy_error",
                ))
                stop_reason = "policy_error"
                break

            if isinstance(action, Finish):
                diagnosis = {"root_cause": action.root_cause, "summary": action.summary}
                self.state.diagnosis = diagnosis
                trajectory.steps.append(Step(
                    index=step_index, thought=action.thought,
                    tool="submit_diagnosis", presented_tool="submit_diagnosis",
                    args=diagnosis, observation="diagnosis recorded", ok=True,
                    raw_completion=getattr(action, "raw", ""),
                ))
                stop_reason = "submitted"
                self.state.phase = Phase.DONE
                break

            result = self.gateway.call(action.name, action.args)
            self._record(trajectory, step_index, action, result)
            self._update_state(action, result)

            if self.config.checkpoint_dir is not None:
                self._checkpoint(trajectory)

        wall = time.perf_counter() - started

        # Let any rollout the agent triggered finish before judging it. Reward should
        # measure whether the incident was fixed, not whether the agent happened to check
        # health at the right moment.
        ep.settle_world()

        # The verifier reads the live cluster; it never consults the transcript.
        verification = traj_mod.verify(
            ep, self.gateway, diagnosis,
            budget_exhausted=self.gateway.budget.exhausted(),
        )
        reward, breakdown = traj_mod.score(verification, self.config.reward_weights)

        trajectory.verification = verification
        trajectory.reward = reward
        trajectory.reward_breakdown = breakdown
        trajectory.wall_seconds = round(wall, 2)
        trajectory.stop_reason = stop_reason
        trajectory.prompt_tokens = getattr(self.policy, "prompt_tokens", 0)
        trajectory.completion_tokens = getattr(self.policy, "completion_tokens", 0)
        trajectory.truncated_turns = getattr(self.policy, "truncations", 0)

        observability.record_episode(
            fault=trajectory.fault, split=trajectory.split,
            resolved=verification.resolved, reward=reward, wall_seconds=wall,
        )
        observability.set_attributes(
            episode_span,
            resolved=verification.resolved,
            root_cause_correct=verification.root_cause_correct,
            reward=reward, tool_calls=verification.tool_calls,
            unsafe_actions=verification.unsafe_actions,
            stop_reason=stop_reason,
        )
        return trajectory

    # --- bookkeeping --------------------------------------------------------

    def _record(self, trajectory: Trajectory, index: int, action: ToolCall, result) -> None:
        trajectory.steps.append(Step(
            index=index,
            thought=action.thought,
            tool=result.tool if result.ok or result.error_kind != "unknown_tool" else None,
            presented_tool=action.name,
            args=action.args,
            observation=result.observation(),
            ok=result.ok,
            error_kind=result.error_kind,
            tier=result.tier,
            unsafe=result.unsafe,
            duration_ms=round(result.duration_ms, 2),
            raw_completion=getattr(action, "raw", ""),
        ))
        if self.config.verbose:
            mark = "ok " if result.ok else "ERR"
            print(f"  [{index:2d}] {mark} {action.name}({_short(action.args)}) "
                  f"-> {result.observation()[:90]}")

    def _update_state(self, action: ToolCall, result) -> None:
        state = self.state
        # The assistant turn must be recorded in the SAME shape the model was trained on:
        # reasoning followed by a fenced JSON call, all inside `content`.
        #
        # Putting the call in a separate `tool_calls` field looks tidier and is what the
        # OpenAI schema suggests, but the chat template used for training and serving
        # renders `content` only -- so the call would be dropped from the history. From
        # turn two onward the model would then see a transcript in which every previous
        # assistant turn is prose with no tool call, and dutifully continue that pattern.
        # The failure is invisible on turn one, which is what makes it so easy to miss.
        state.messages.append({
            "role": "assistant",
            "content": _assistant_turn(action, result),
        })
        state.messages.append({
            "role": "user",
            "content": f"TOOL RESULT:\n{result.observation()}",
        })

        if result.ok:
            state.consecutive_errors = 0
            state.last_error = None
            if result.tier != "read":
                state.attempted_fixes.append(f"{result.tool}:{_short(result.args)}")
                state.phase = Phase.VALIDATE
            elif state.phase is Phase.ORIENT:
                state.phase = Phase.INVESTIGATE
        else:
            state.consecutive_errors += 1
            state.last_error = result.content
            if (self.config.recovery_enabled
                    and state.consecutive_errors >= RECOVERY_AFTER_ERRORS):
                state.messages.append({
                    "role": "user",
                    "content": (
                        f"You have had {state.consecutive_errors} failed tool calls in a "
                        "row. Re-read the available tool names and their required "
                        "arguments in the tool schema, then make one simple, valid call. "
                        "If a tool you expected is missing, it has been renamed -- find "
                        "the closest equivalent in the schema."
                    ),
                })
                state.consecutive_errors = 0

    def _checkpoint(self, trajectory: Trajectory) -> None:
        assert self.config.checkpoint_dir is not None
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.checkpoint_dir / f"{self.state.incident_id}.json"
        path.write_text(json.dumps({
            "state": self.state.snapshot(),
            "steps": [s.__dict__ for s in trajectory.steps],
        }, default=str))


def _assistant_turn(action: ToolCall, result) -> str:
    """Render an assistant turn exactly as `data.curate._assistant_text` does.

    These two renderings are the same contract seen from opposite ends -- one builds the
    training target, the other builds the inference history -- so any divergence between
    them is a train/serve skew. `tests/test_prompt_contract.py` asserts they agree.
    """
    call = {"name": action.name, "arguments": action.args}
    body = json.dumps(call, indent=2)
    thought = (action.thought or "").strip()
    return f"{thought}\n\n```json\n{body}\n```" if thought else f"```json\n{body}\n```"


def _short(args: dict[str, Any], limit: int = 60) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return s if len(s) <= limit else s[: limit - 3] + "..."


__all__ = ["AgentHarness", "HarnessConfig"]
