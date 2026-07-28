"""Run incidents through a policy and write trajectories.

One worker owns one control plane and streams incidents through it, so the ~13s control
plane start is amortised over hundreds of episodes. Workers are shard-addressed
(`shard i of n`), which is all a Slurm job array needs to fan out.
"""

from __future__ import annotations

import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentgym.data import trajectory as traj_mod
from agentgym.data.trajectory import Trajectory
from agentgym.harness.expert import ExpertPolicy
from agentgym.harness.harness import AgentHarness, HarnessConfig
from agentgym.harness.state import Policy
from agentgym.incidents.generator import Incident
from agentgym.incidents.gym import AgentGym
from agentgym.mcp.gateway import Budget, PermissionPolicy, ToolGateway
from agentgym.mcp.perturb import SUITES, PerturbationSpec

PolicyFactory = Callable[[Any, Any], Policy]


def expert_factory(episode, gateway) -> Policy:
    return ExpertPolicy(episode=episode, toolset=gateway.toolset)


def llm_factory(backend, adapter: str | None = None, temperature: float = 0.0) -> PolicyFactory:
    from agentgym.harness.llm_policy import LLMPolicy

    def make(episode, gateway) -> Policy:
        return LLMPolicy(backend=backend, adapter=adapter, temperature=temperature)

    return make


@dataclass
class RolloutConfig:
    max_steps: int = 24
    max_tool_calls: int = 24
    perturbation: str = "clean"
    reward_weights: str = "full"
    model_name: str = "expert"
    auto_approve_destructive: bool = False
    gym_offset: int = 0
    verbose: bool = False


def run_incidents(
    incidents: list[Incident],
    policy_factory: PolicyFactory,
    config: RolloutConfig,
    out_path: Path | None = None,
    *,
    on_result: Callable[[Trajectory], None] | None = None,
) -> list[Trajectory]:
    """Execute every incident in one gym, appending trajectories as they complete."""
    spec: PerturbationSpec = SUITES.get(config.perturbation, SUITES["clean"])
    weights = traj_mod.WEIGHT_SETS.get(config.reward_weights, traj_mod.FULL)
    results: list[Trajectory] = []

    with AgentGym(offset=config.gym_offset) as gym:
        for i, incident in enumerate(incidents):
            started = time.perf_counter()
            try:
                with gym.run_episode(incident) as episode:
                    # Perturbation is seeded per incident so a suite is reproducible but
                    # not identical across every scenario.
                    seeded = PerturbationSpec(
                        kinds=spec.kinds, severity=spec.severity, seed=incident.seed
                    )
                    gateway = ToolGateway.create(
                        episode, seeded,
                        policy=PermissionPolicy(
                            auto_approve=config.auto_approve_destructive
                        ),
                        budget=Budget(max_tool_calls=config.max_tool_calls),
                    )
                    harness = AgentHarness(
                        episode=episode,
                        gateway=gateway,
                        policy=policy_factory(episode, gateway),
                        config=HarnessConfig(
                            max_steps=config.max_steps,
                            reward_weights=weights,
                            verbose=config.verbose,
                        ),
                    )
                    trajectory = harness.run(
                        incident_id=incident.id,
                        split=incident.split,
                        model=config.model_name,
                        perturbation=config.perturbation,
                        difficulty=incident.difficulty,
                    )
            except Exception as exc:
                # One bad episode must never take down a several-hour job array.
                trajectory = Trajectory(
                    incident_id=incident.id, topology=incident.topology,
                    fault=incident.fault, split=incident.split, seed=incident.seed,
                    difficulty=incident.difficulty, model=config.model_name,
                    perturbation=config.perturbation,
                    stop_reason=f"episode_error: {type(exc).__name__}: {exc}",
                )
                if config.verbose:
                    traceback.print_exc()

            trajectory.wall_seconds = round(time.perf_counter() - started, 2)
            results.append(trajectory)
            if out_path is not None:
                traj_mod.append_jsonl(trajectory, out_path)
            if on_result is not None:
                on_result(trajectory)

            if config.verbose or (i + 1) % 25 == 0:
                solved = sum(1 for t in results if t.success)
                print(f"[{i+1}/{len(incidents)}] solved={solved} "
                      f"({solved/len(results):.1%}) last={trajectory.fault} "
                      f"reward={trajectory.reward}", flush=True)
    return results


def shard(items: list[Any], index: int, total: int) -> list[Any]:
    """Deterministic contiguous-stride shard, so every item is covered exactly once."""
    if total <= 1:
        return items
    return [x for i, x in enumerate(items) if i % total == index]


def worker_identity() -> tuple[int, int]:
    """(shard index, shard count) from the Slurm array environment."""
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    lo = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
    hi = int(os.environ.get("SLURM_ARRAY_TASK_MAX", str(idx)))
    return idx - lo, (hi - lo + 1)


def summarize(trajectories: list[Trajectory]) -> dict[str, Any]:
    if not trajectories:
        return {}
    n = len(trajectories)
    ok = [t for t in trajectories if t.success]
    return {
        "episodes": n,
        "resolution_rate": round(len(ok) / n, 4),
        "root_cause_accuracy": round(
            sum(1 for t in trajectories if t.verification.root_cause_correct) / n, 4),
        "clean_rate": round(sum(1 for t in trajectories if t.clean) / n, 4),
        "mean_reward": round(sum(t.reward for t in trajectories) / n, 4),
        "mean_tool_calls": round(
            sum(t.verification.tool_calls for t in trajectories) / n, 2),
        "unsafe_action_rate": round(
            sum(1 for t in trajectories if t.verification.unsafe_actions > 0) / n, 4),
        "invalid_call_rate": round(
            sum(t.verification.invalid_calls for t in trajectories)
            / max(1, sum(t.verification.tool_calls for t in trajectories)), 4),
        "mean_wall_seconds": round(sum(t.wall_seconds for t in trajectories) / n, 2),
        "errored_episodes": sum(1 for t in trajectories
                                if t.stop_reason.startswith("episode_error")),
    }


__all__ = [
    "RolloutConfig", "run_incidents", "shard", "worker_identity",
    "summarize", "expert_factory", "llm_factory",
]
