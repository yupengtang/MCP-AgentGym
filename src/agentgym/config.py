"""Global paths and tunables, all overridable from the environment (see env.sh)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _p(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).expanduser()


AGENTGYM_HOME = _p("AGENTGYM_HOME", "/storage/ice-shared/mgt6655/storage/mcp-agentgym")
AGENTGYM_REPO = _p("AGENTGYM_REPO", str(Path(__file__).resolve().parents[2]))
BIN_DIR = _p("AGENTGYM_BIN", str(AGENTGYM_HOME / "bin"))
RUNTIME_DIR = _p("AGENTGYM_RUNTIME", f"{os.environ.get('TMPDIR', '/tmp')}/{os.environ.get('USER', 'user')}/agentgym-runtime")
RUNS_DIR = _p("AGENTGYM_RUNS", str(AGENTGYM_HOME / "runs"))
DATA_DIR = _p("AGENTGYM_DATA", str(AGENTGYM_HOME / "data"))
ARTIFACTS_DIR = _p("AGENTGYM_ARTIFACTS", str(AGENTGYM_HOME / "artifacts"))

BASE_MODEL = os.environ.get("AGENTGYM_BASE_MODEL", "Qwen/Qwen3-8B")
TEACHER_MODEL = os.environ.get("AGENTGYM_TEACHER_MODEL", "Qwen/Qwen3-32B")


@dataclass(frozen=True)
class PortAllocation:
    """Every episode gets a disjoint port block so N control planes coexist on a node."""

    offset: int
    etcd_client: int = field(init=False)
    etcd_peer: int = field(init=False)
    apiserver: int = field(init=False)
    controller_manager: int = field(init=False)
    scheduler: int = field(init=False)

    def __post_init__(self) -> None:
        # 10 ports per episode, starting above the ephemeral-port floor we care about.
        base = 20000 + self.offset * 10
        object.__setattr__(self, "etcd_client", base + 0)
        object.__setattr__(self, "etcd_peer", base + 1)
        object.__setattr__(self, "apiserver", base + 2)
        object.__setattr__(self, "controller_manager", base + 3)
        object.__setattr__(self, "scheduler", base + 4)


# --- Simulated cluster shape -------------------------------------------------

@dataclass(frozen=True)
class NodeSpec:
    name: str
    cpu: int          # cores
    memory_gi: int
    gpu: int = 0
    gpu_type: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    taints: tuple[tuple[str, str, str], ...] = ()  # (key, value, effect)


DEFAULT_NODES: tuple[NodeSpec, ...] = (
    NodeSpec("node-cpu-1", cpu=16, memory_gi=64, labels={"node-role": "worker", "tier": "cpu"}),
    NodeSpec("node-cpu-2", cpu=16, memory_gi=64, labels={"node-role": "worker", "tier": "cpu"}),
    NodeSpec(
        "node-gpu-1", cpu=32, memory_gi=256, gpu=4, gpu_type="a100",
        labels={"node-role": "worker", "tier": "gpu", "accelerator": "nvidia-a100"},
        taints=(("nvidia.com/gpu", "present", "NoSchedule"),),
    ),
)

# Images the simulated registry knows about. Anything else -> ImagePullBackOff.
KNOWN_IMAGES: frozenset[str] = frozenset({
    "vllm/vllm-openai:v0.6.0",
    "vllm/vllm-openai:v0.6.1",
    "ghcr.io/agentgym/api-service:1.4.2",
    "ghcr.io/agentgym/api-service:1.4.3",
    "ghcr.io/agentgym/worker:2.1.0",
    "ghcr.io/agentgym/gateway:0.9.1",
    "redis:7.2-alpine",
    "postgres:16-alpine",
    "nginx:1.25-alpine",
})

# StorageClasses the simulated provisioner will bind. A PVC asking for anything else
# stays Pending forever, which is exactly what an unbound volume looks like in
# production and is the basis of the PVCUnbound fault family.
KNOWN_STORAGE_CLASSES: frozenset[str] = frozenset({"standard", "fast-ssd", "gp3"})

# How long the control plane may take to come up before we call it a failure.
CONTROL_PLANE_TIMEOUT_S = float(os.environ.get("AGENTGYM_CP_TIMEOUT", "90"))
