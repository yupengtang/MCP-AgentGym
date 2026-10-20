# MCP-AgentGym

### A Verifiable Kubernetes Incident Gym for Post-Training and Serving Tool-Using Agents

> MCP-AgentGym is a deterministic, resettable Kubernetes incident environment backed by a **real
> Kubernetes API server**. It generates executable, programmatically-verified tool-use trajectories;
> post-trains an open-weight model on them with SFT, DPO, and GRPO; serves the result through a
> pluggable vLLM/SGLang inference layer; and measures whether any of it survives contact with
> renamed tools, changed schemas, and failing backends.

---

## 0. What this document is

This is the design spec and the contract for what "done" means. It supersedes the original
brainstorm (`project_idea.txt`, kept for provenance). Section 1 records the four design changes
made against that brainstorm and why each one matters. Those changes are the difference between a
project that can be finished and one that stalls at 70%.

---

## 1. Four changes from the original plan

### 1.1 The environment is a simulator on a real API server, not a real workload cluster

**The original plan assumed real pods crashing in a real cluster.** That is wrong on two independent
grounds, and the second one would still be wrong on a cloud budget:

1. *It is not buildable on the target hardware.* The shared HPC cluster gives no root, no Docker, and no
   `/etc/subuid` range, so rootless Podman cannot map multiple UIDs and neither `kind` nor `k3s`
   can start.
2. *It is the wrong environment for post-training even where it is buildable.* Post-training needs
   **thousands of resettable, seeded, parallel episodes with verifiable rewards**. A real cluster
   gives episodes that take minutes, never reset cleanly, and are nondeterministic, so the same
   action yields different rewards on different runs, which is fatal for DPO preference pairs and
   for GRPO advantage estimates.

**The resolution.** Split the stack at the API boundary, which is exactly where the agent operates
anyway:

| Layer | Real or simulated | Why |
| --- | --- | --- |
| `etcd` + `kube-apiserver` + `kube-controller-manager` + `kube-scheduler` | **Real**, run as user-space processes | Real object semantics, real CRDs, real RBAC, real admission, real watch/reconcile. The MCP tools and the operator talk to a genuine Kubernetes API. |
| kubelet / container runtime | **Simulated** (`kubelet_sim`) | Pod status is *computed* from declarative workload physics: resource fit, image validity, config/secret presence, probe config. Deterministic and instant. |
| Application traffic, health, metrics, logs | **Simulated** (`mesh`, `metrics`, `logs`) | Health and SLOs are a pure function of cluster state, so the verifier is executable and exact. |

This is not a shortcut around the hard part; it *is* the hard part. It buys sub-second resets,
hundreds of parallel episodes per node, bit-exact reproducibility from a seed, and rewards that are
computed rather than judged. Section 8 validates that the simulator's fault semantics match a real
cluster.

### 1.2 Tool-schema robustness is promoted to the headline result

The original doc buried dynamic tool changes in a subsection. It is the most differentiating idea in
the plan and becomes a first-class, systematically parameterized evaluation axis (§6). Every agent
demo can call a tool that exists; almost none keep working when the tool is renamed, gains a
required argument, or starts returning malformed JSON. That is the result worth owning.

### 1.3 GRPO moves from "optional stretch goal" into the core

The original treated GRPO as a nice-to-have at the end. Once §1.1 gives a deterministic simulator
with executable rewards, on-policy RL becomes *the natural thing to do* rather than an expensive
add-on, because the reward function already exists and the environment already resets. It is also the
single highest-signal item for research-engineer and post-training roles. It ships in the core.

### 1.4 Scope is cut so the core actually finishes

An unfinished project with fifteen half-features interviews far worse than a finished one with six
deep ones. Explicitly **cut**: React dashboard (replaced by a static HTML report + a small FastAPI
status page), Terraform, cloud deployment, MinIO/S3, Kafka, Spark, Airflow, Ray, service mesh.
Grafana is kept only because Prometheus is already there and the dashboards are nearly free.

---

## 2. The closed loop

```
        ┌────────────────────────────────────────────────────────────────┐
        │                                                                │
        ▼                                                                │
  Incident generator ──► Kubernetes incident environment                 │
  (seeded, 14 fault      (real apiserver + simulated kubelet/mesh)       │
   families)                        │                                    │
                                    ▼                                    │
                          MCP gateway (4 servers, 36 typed tools)         │
                          + schema-shift perturbation layer               │
                                    │                                    │
                                    ▼                                    │
                          Agent harness (plan → select → execute          │
                          → verify → recover → stop)                      │
                                    │                                    │
                                    ▼                                    │
                          Executable verifier + multi-objective reward    │
                                    │                                    │
                                    ▼                                    │
                          Trajectory store (curation, dedup,              │
                          rejection sampling, contamination checks)       │
                                    │                                    │
                                    ▼                                    │
                          QLoRA SFT ──► DPO ──► GRPO                      │
                                    │                                    │
                                    ▼                                    │
                          vLLM / SGLang serving + failover gateway        │
                                    │                                    │
                                    ▼                                    │
                          Evaluation (ID / OOD / schema-shift / systems)  │
                                    │                                    │
                                    └────► new failure cases ────────────┘
```

The loop is the point. Each arrow is a real artifact, and the last arrow closes it: failures found in
evaluation become next round's training data.

---

## 3. Environment

### 3.1 Control plane

`agentgym.cluster.controlplane` starts a real, single-tenant Kubernetes control plane as ordinary
user processes: `etcd` on a private data dir, `kube-apiserver` with a self-signed CA and RBAC
enabled, plus `kube-controller-manager` and `kube-scheduler`. No root, no containers, no ports below
1024. Each worker gets its own control plane on its own port range, so N episodes run in parallel on
one node with zero cross-talk.

### 3.2 Workload physics (`kubelet_sim`)

A controller watches Pods and drives their status from declared workload properties. It is a
deterministic state machine, not a random fault generator:

| Condition | Resulting pod state |
| --- | --- |
| `requests` exceed allocatable on every node | `Pending` / `Unschedulable` |
| image tag not in the simulated registry | `ImagePullBackOff` |
| `limits.memory` < declared working set | `OOMKilled` → `CrashLoopBackOff` |
| referenced ConfigMap key missing/invalid | `CrashLoopBackOff` (config error) |
| referenced Secret missing | `CreateContainerConfigError` |
| readiness probe path/port wrong, or `initialDelaySeconds` < boot time | `Running` but never `Ready` |
| nodeSelector/taint unsatisfiable | `Pending` |
| namespace `ResourceQuota` exceeded | admission rejection on scale-up |
| PVC with no matching StorageClass | `Pending` (unbound) |

### 3.3 Service mesh, metrics, logs

`mesh` computes Endpoints from label selectors and pod readiness, then runs a synthetic traffic model
over the topology to produce request success rate and latency, so "is the service healthy" is a
computed fact. `metrics` emits Prometheus-shaped series derived from that state (p50/p95 latency,
error rate, restart count, GPU memory, KV-cache utilization, queue depth). `logs` emits seeded,
templated container logs consistent with the injected fault, so reading logs is genuinely
informative rather than decorative. Kubernetes `Events` are written to the real API server.

### 3.4 Fault catalog (14 families)

`OOMKilled` · `ImagePullBackOff` · `BadConfigMap` · `MissingSecret` · `ServiceSelectorMismatch` ·
`ReadinessProbeMisconfig` · `ResourceQuotaExhausted` · `PVCUnbound` · `NodeSelectorUnschedulable` ·
`HPAThrash` · `KVCacheExhaustion` · `NetworkPolicyDeny` · `RolloutStuck` · `CascadingDependencyDown`

Each family declares, as code: the manifest mutation that injects it, the ground-truth root cause
label, an **executable resolution predicate**, a **safety policy** (forbidden actions), and a
difficulty tier. An *incident instance* is `(family, topology, parameters, seed)`; the generator
produces 1,000+ instances with held-out splits (§6).

---

## 4. MCP layer

Four MCP servers over the real protocol, 36 typed tools, every argument schema-validated:

- **`mcp-kubernetes`**: `list_pods`, `describe_pod`, `get_pod_logs`, `list_events`,
  `get_deployment`: `patch_deployment_resources`*, `scale_deployment`*, `set_image`*,
  `rollout_restart`*, `delete_pod`†
- **`mcp-observability`**: `query_metrics` (PromQL-lite), `get_service_health`, `get_slo_status`,
  `trace_request`: `list_alerts`
- **`mcp-config`**: `get_configmap`, `patch_configmap`*, `get_secret_meta` (metadata only, and never
  returns values), `get_service`, `patch_service_selector`*, `get_networkpolicy`,
  `patch_networkpolicy`*, `get_resourcequota`, `patch_resourcequota`*
- **`mcp-knowledge`**: `search_runbooks`, `get_runbook`, `get_service_topology`,
  `get_recent_changes` (deploy history, which often contains the actual culprit)

`*` = mutating, `†` = destructive (requires approval under the default permission policy).
Terminal action: `submit_diagnosis(root_cause, summary)`.

### 4.1 Perturbation layer

A gateway-level transform, parameterized by kind and severity, applied at episode start:
tool **rename**, description **paraphrase**, **added required argument**, argument **retyping**,
argument **nesting/reordering**, **distractor tools** (k near-duplicates), tool **dropout**,
**malformed responses**: response **truncation**, injected **latency/timeout**, and **flaky**
transient failures. This layer generates the OOD suite and is never seen during training unless a
split explicitly enables it.

---

## 5. Harness, verifier, reward

The harness is written from scratch (no `create_react_agent`) because the failure handling *is* the
contribution: typed `AgentState`, dynamic tool discovery, schema validation before dispatch,
parallel read-only calls, retry/timeout/backoff, token and tool-call budgets, checkpoint/resume,
permission tiers with human-approval gates, a rollback journal, full OpenTelemetry tracing, and
deterministic replay from a stored trajectory.

The **verifier** is executable, not a judge: it combines the resolution predicate over live cluster
state, simulated health and SLO checks, root-cause comparison against the ground-truth label, and
safety-policy violations. The reward is multi-objective:

```
score = 2.0·resolved + 0.5·root_cause_correct + 0.4·validation_passed + 0.2·recovered_from_failure
      − 0.05·unnecessary_calls − 0.3·invalid_calls − 0.8·unsafe_actions − 0.1·excessive_latency
```

Weights are ablated (outcome-only / safety-aware / efficiency-aware / full) in §6.

---

## 6. Evaluation

Three matrices, one per audience, all produced by the same runner.

**Agent capability**: Resolution rate · Root-cause accuracy · Tool-selection F1 · Valid-argument
rate · Recovery rate · Unsafe-action rate, across Base / Prompted / SFT / SFT+DPO / SFT+GRPO.

**Generalization**: In-distribution · Unseen fault family · Compositional (2 simultaneous faults) ·
Schema shift (§4.1) · MCP server failure. This is the headline table.

**Systems**: Throughput · TTFT · p95 episode latency · GPU utilization · KV-cache hit rate ·
failover recovery time · GPU-hours and cost per 1k episodes, across Transformers / vLLM / SGLang /
multi-LoRA / autoscaled configurations.

Splits are held out by *fault family* and by *topology*, and a contamination check asserts no
evaluation incident seed appears in any training trajectory.

---

## 7. Serving

One `InferenceBackend` interface with `VLLMBackend`, `SGLangBackend`, and `APIModelBackend`
(teacher / upper bound). The harness never learns which is underneath. Four experiments:
shared-prefix scaling (agent turns 2/5/10/20 × concurrency 1/8/32/64), structured tool calling
(unconstrained vs. JSON-schema-constrained vs. model-specific parser), multi-LoRA serving
(base + SFT + DPO adapters in one batch), and backend failover through a circuit-breaking gateway.

The honest claim until the numbers are in is *"benchmarked vLLM and SGLang under multi-turn
shared-prefix agent workloads"*, `outperforms` gets written only if measured.

---

## 8. Sim-to-real fidelity

The simulator is validated, not assumed. A subset of incident families is replayed against a real
single-node Kubernetes cluster (k3s on a compute node, or a managed cluster if available), and we
report per-family agreement on observable signals: pod phase and reason, event sequence, endpoint
readiness, and whether the same remediation resolves it. Disagreements are reported rather than
hidden. A fidelity table with known gaps is far more credible than a claim of realism.

---

## 9. Infrastructure

**`IncidentRun` operator** (`agentgym.io/v1`): a real controller against the real API server,
reconciling: create namespace → apply topology → inject fault → launch agent Job → collect
trajectory → verify → update `status` → clean up. Demonstrates reconciliation, idempotency,
finalizers, owner references, and retry/backoff.

**Slurm** handles batch: trajectory generation via job arrays, QLoRA SFT / DPO / GRPO on H100/H200,
and distributed evaluation, with checkpoint/resume throughout.

**Observability**: OpenTelemetry traces spanning harness → MCP → cluster, Prometheus metrics,
Grafana dashboards, MLflow experiment tracking, and GitHub Actions running the end-to-end test.

The two-scheduler story is genuine here: *Slurm for batch GPU training and large-scale evaluation,
Kubernetes for persistent serving, stateful agent execution, and isolated incident environments.*

---

## 10. Definition of done

- [x] 4 MCP servers, **36** typed tools, real MCP protocol (8 compliance tests)
- [x] 14 fault families, 1,200 seeded incidents, held-out splits
- [x] Real Kubernetes control plane, deterministic reset, parallel episodes
- [x] **11,541** executed and programmatically verified trajectories
- [x] Verifiable episode reward + verifiable action reward, ordering asserted by tests
- [x] `IncidentRun` operator reconciling end to end (idempotency, finalizer, ownerRefs)
- [x] Helm chart with RBAC and NetworkPolicy; OTel + Prometheus + MLflow + CI
- [x] Expert reference line across all 12 perturbation suites × 4 splits
- [x] Experiment report, README, 3-minute demo script
- [x] 140 tests green, ruff clean, Helm lints
- [x] QLoRA SFT trained; DPO and GRPO trainers written, reward verified
- [x] Base / SFT / DPO evaluation matrices across 4 splits and 7 tool-schema suites
- [x] vLLM prefix-cache / structured-output / multi-LoRA benchmarks
- [x] DPO run on mined preference pairs, twice, as a pair-count ablation (88 vs 435)
- [x] Episode-level GRPO trained and evaluated *(597/600 groups usable, 149 optimizer
      steps; 44.2% at n = 2,200, inside the 3.1 pp seed floor)*
- [x] Action-level GRPO trained and evaluated *(43.0% at n = 2,200; also inside the floor)*
- [x] Variance budget measured, learning rate, seed, and same-configuration
      nondeterminism separated, which is what makes the method effects readable
- [ ] Several seeds per method arm *(the grid establishes that post-training works and
      cannot establish which method is best: every arm is one training run, and the
      run-to-run floor between independently trained models is 3.1 pp)*
- [ ] SGLang **benchmarked against a live server** *(`SGLangBackend` is implemented
      and its interchangeability with `VLLMBackend` is asserted by
      `tests/test_backend_pluggability.py`: identical request payload, identical
      parsing, no reimplementation. Not benchmarked: `sglang` needs a third
      environment and the home quota on this cluster is exhausted.)*
- [ ] Node-layer sim-to-real fidelity *(process layer validated 5/5 against real
      processes, real `RLIMIT_AS` and real HTTP/TCP; the container layer has no
      real-runtime comparison and no fidelity claim is made for it)*
