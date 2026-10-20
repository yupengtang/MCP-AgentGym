#!/usr/bin/env bash
# A three-minute walkthrough of MCP-AgentGym.
#
# Starts a real Kubernetes control plane, breaks a live application, watches a scripted
# SRE agent diagnose and repair it through MCP tools, and shows the executable verifier
# scoring the result. Everything here runs on CPU in about a minute.
#
#   bash scripts/demo.sh            # OOMKilled on the batch-pipeline topology
#   FAULT=NetworkPolicyDeny bash scripts/demo.sh
#   PERTURBATION=rename bash scripts/demo.sh   # same incident, renamed tools
#   STEP_DELAY=0.6 bash scripts/demo.sh        # pause between tool calls (for recordings)

set -euo pipefail
cd "${AGENTGYM_REPO:-$(dirname "$0")/..}"
source env.sh >/dev/null

FAULT="${FAULT:-OOMKilled}"
TOPOLOGY="${TOPOLOGY:-batch-pipeline}"
PERTURBATION="${PERTURBATION:-clean}"
SEED="${SEED:-4242}"
export STEP_DELAY="${STEP_DELAY:-0}"

python - "$FAULT" "$TOPOLOGY" "$PERTURBATION" "$SEED" <<'PY'
import os, sys, textwrap, time

fault_name, topology, perturbation, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

from agentgym.harness.expert import ExpertPolicy
from agentgym.harness.harness import AgentHarness, HarnessConfig
from agentgym.incidents import generator
from agentgym.incidents.generator import Incident
from agentgym.incidents.gym import AgentGym
from agentgym.mcp.gateway import Budget, ToolGateway
from agentgym.mcp.perturb import SUITES, PerturbationSpec

def summarize(observation, width=66):
    # One line per tool result, so the run reads as a sequence of decisions rather than JSON.
    import json
    try:
        data = json.loads(observation)
    except (TypeError, ValueError):
        text = " ".join(str(observation).split())
        return text[:width] + ("..." if len(text) > width else "")
    if isinstance(data, list):
        first = data[0] if data else {}
        keys = ", ".join(f"{k}={first[k]}" for k in list(first)[:2]) if isinstance(first, dict) else ""
        return f"{len(data)} items" + (f"  (first: {keys})" if keys else "")
    if isinstance(data, dict):
        text = ", ".join(f"{k}={v}" for k, v in list(data.items())[:3] if not isinstance(v, (dict, list)))
        return text[:width] + ("..." if len(text) > width else "")
    return str(data)[:width]

def short(value, width=44):
    text = repr(value)
    return text if len(text) <= width else text[:width - 4] + "...'"

def rule(title):
    print(f"\n\033[1m{'=' * 78}\n{title}\n{'=' * 78}\033[0m")

rule("1. Starting a REAL Kubernetes control plane (etcd + apiserver + controllers)")
t0 = time.time()
with AgentGym(offset=95) as gym:
    print(f"   up in {time.time() - t0:.1f}s  ->  {gym.control_plane.server_url}")
    print("   " + gym.control_plane.kubectl("get", "--raw", "/healthz").stdout.strip()
          + "  (server version "
          + gym.control_plane.kubectl("version", "-o", "yaml").stdout.split("gitVersion:")[-1].split()[0]
          + ")")

    # Not every fault fits every application: OOMKilled needs a service with a
    # meaningful working set and no KV cache, NetworkPolicyDeny needs a dependency
    # edge, and so on. Pick a topology that actually accepts this fault.
    matrix = generator.compatibility_matrix()
    if fault_name not in matrix.get(topology, []):
        alternatives = [t for t, faults in sorted(matrix.items()) if fault_name in faults]
        if not alternatives:
            raise SystemExit(f"no topology accepts {fault_name}")
        print(f"\n   note: `{topology}` has no service that {fault_name} can break; "
              f"using `{alternatives[0]}` instead")
        topology = alternatives[0]

    _, fault, params = generator.build(topology, fault_name, seed)
    incident = Incident(id=f"demo-{fault_name.lower()}", topology=topology,
                        fault=fault_name, seed=seed, split="demo",
                        difficulty=fault.difficulty, root_cause=fault.root_cause,
                        params=params)

    rule(f"2. Injecting fault: {fault_name} into `{topology}`")
    print(textwrap.fill(fault.summary, 78, initial_indent="   ", subsequent_indent="   "))

    with gym.run_episode(incident) as ep:
        print(f"\n   namespace: {ep.namespace}")
        print("\n   what an operator would see:\n")
        print(textwrap.indent(
            gym.control_plane.kubectl("get", "pods", "-n", ep.namespace).stdout.rstrip(),
            "     "))
        print(f"\n   verifier says resolved = \033[31m{ep.resolved()}\033[0m")

        rule(f"3. Agent works the incident through MCP tools (perturbation={perturbation})")
        base = SUITES[perturbation]
        gw = ToolGateway.create(
            ep, PerturbationSpec(base.kinds, base.severity, seed=seed),
            budget=Budget(max_tool_calls=28),
        )
        print(f"   {len(gw.toolset.tools)} tools presented across 4 MCP servers\n")
        harness = AgentHarness(
            episode=ep, gateway=gw,
            policy=ExpertPolicy(episode=ep, toolset=gw.toolset),
            config=HarnessConfig(max_steps=28, verbose=False),
        )
        traj = harness.run(incident_id=incident.id, split="demo",
                           model="expert-scripted", perturbation=perturbation,
                           difficulty=incident.difficulty)
        for step in traj.steps:
            if step.tool is None:
                continue
            args = ", ".join(f"{k}={short(v)}" for k, v in step.args.items()
                             if k not in ("namespace", "summary"))
            colour = "33" if step.tier != "read" else "36"
            status = "ok " if step.ok else "[31mERR[0m"
            print(f"   [{step.index:>2}] {status} [{colour}m{step.presented_tool or step.tool}[0m"
                  f"({args})")
            print(f"        -> {summarize(step.observation)}")
            time.sleep(float(os.environ.get("STEP_DELAY", "0")))

        rule("4. Cluster state after remediation")
        print(textwrap.indent(
            gym.control_plane.kubectl("get", "pods", "-n", ep.namespace).stdout.rstrip(),
            "     "))

        rule("5. Executable verification (no model judges any of this)")
        v = traj.verification
        colour = "32" if v.resolved else "31"
        print(f"   resolved            \033[{colour}m{v.resolved}\033[0m")
        print(f"   root cause          {v.predicted_root_cause}")
        print(f"   ground truth        {v.true_root_cause}  "
              f"-> correct = {v.root_cause_correct}")
        print(f"   services healthy    {v.services_healthy}/{v.services_total}")
        print(f"   tool calls          {v.tool_calls}   "
              f"(invalid {v.invalid_calls}, unsafe {v.unsafe_actions})")
        parts = ", ".join(f"{k} {v:+g}" for k, v in traj.reward_breakdown.items() if v)
        print(f"\n   reward = \033[1m{traj.reward}\033[0m   ({parts})")
PY
