# MCP-AgentGym Design Document

**Status:** working draft · **Version:** 0.1 · **Audience:** contributors and reviewers

A verifiable Kubernetes incident environment for post-training and serving tool-using
agents.

This document specifies what the system is, why it is built this way, and where its
boundaries are. [README.md](README.md) is the entry point and carries measured results;
[reports/EXPERIMENTS.md](reports/EXPERIMENTS.md) is the experiment log. This file is the
design of record.

---

## 1. Motivation

Post-training a tool-using agent requires an environment with four properties
simultaneously:

| Property | Why it is required |
| --- | --- |
| **Resettable** | Preference pairs and on-policy advantages compare *behaviours* on the *same* state. Without a clean reset, the scenario becomes a confound. |
| **Deterministic** | The same action sequence must produce the same outcome, or the reward is noise and the gradient follows it. |
| **Parallel** | Useful post-training needs thousands of episodes. Sequential minutes-long episodes make the data budget unreachable. |
| **Verifiable** | Reward must be *computed* from state, not judged by a model. A judged reward is a second model's opinion with unknown bias. |

A production Kubernetes cluster has none of them. Episodes take minutes, namespaces do
not reset cleanly, timing is nondeterministic, and "is it fixed?" has no programmatic
answer. The standard workaround, asking a strong model to grade transcripts, replaces a
measurement with an opinion.

MCP-AgentGym resolves this by **splitting the stack at the API boundary**, which is
exactly where an agent operates.

---

## 2. Design goals and non-goals

### Goals

- **G1.** Every reward term is computed from live cluster state, a JSON Schema, or a
  recorded call log. No model grades another model.
- **G2.** An episode is reproducible from `(topology, fault family, seed)` alone.
- **G3.** The agent's tool surface is a real MCP surface, drivable by any standard MCP
  client.
- **G4.** Episode cost is low enough (seconds, CPU-only) that evaluation is run routinely
  rather than at milestones.
- **G5.** Tool-schema robustness is a first-class, systematically parameterized axis, not
  an anecdote.

### Non-goals

- **N1.** Not a Kubernetes simulator for capacity planning or scheduler research. Fidelity
  is scoped to what an incident-response agent can observe.
- **N2.** Not a general agent framework. The harness is deliberately specific to this
  environment.
- **N3.** Not a benchmark claiming to predict production SRE performance. It measures
  behaviour in a defined environment; §9 states the gap explicitly.
- **N4.** Not a frontier-scale training run. Model sizes and budgets are chosen so results
  are reproducible on one GPU.

---

## 3. Architecture

```
                        ┌──────────────────────────────────────┐
                        │            Incident Generator        │
                        │  (topology × fault × seed → splits)  │
                        └───────────────────┬──────────────────┘
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                          Kubernetes Incident Environment                       │
│                                                                                │
│   REAL              etcd · kube-apiserver · controller-manager · scheduler     │
│   ─────────────────────────────────────────────────────────────────────────    │
│   SIMULATED         kubelet (workload physics) · CSI provisioner ·             │
│                     service mesh · metrics · logs · events                     │
└───────────────────────────────┬───────────────────────────────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────────────┐
        │   MCP Gateway · schema validation · permission tiers   │
        │   budgets · perturbation layer · call log              │
        └───┬───────────────┬───────────────┬───────────────┬───┘
            ▼               ▼               ▼               ▼
      mcp-kubernetes  mcp-observability  mcp-config   mcp-knowledge
                                ▼
        ┌───────────────────────────────────────────────────────┐
        │   Agent Harness · typed state · recovery · replay      │
        │   Policy: scripted expert │ LLM │ baseline             │
        └───────────────────────────┬───────────────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │   Executable Verifier → multi-objective reward         │
        └───────────────────────────┬───────────────────────────┘
                                    ▼
              Trajectory Store → curation → SFT / DPO / GRPO
                                    ▼
              vLLM serving → evaluation matrices → new failures ─┐
                                    ▲                            │
                                    └────────────────────────────┘
```

### 3.1 The real/simulated split

| Layer | Implementation | Rationale |
| --- | --- | --- |
| etcd, kube-apiserver, controller-manager, scheduler | **Real** (v1.31.4, unprivileged user processes) | Genuine object semantics, CRDs, RBAC, admission, watch/reconcile. Deployment→ReplicaSet→Pod, StatefulSet ordinals, scheduling predicates, taints and `Unschedulable` are all authentic. |
| MCP protocol | **Real** (reference `mcp` library, JSON-RPC) | Any standard MCP client can drive the environment. |
| kubelet / container runtime | **Simulated** | Pod status is computed from declared workload physics. Deterministic and instant. |
| CSI provisioner | **Simulated** | Binds PVCs whose StorageClass exists; leaves others Pending. |
| Application traffic, health, metrics, logs | **Simulated** | Health becomes a pure function of cluster state, which is what makes the verifier exact. |

Container runtimes are unavailable on the reference hardware (no root, no Docker, no
`/etc/subuid` for rootless Podman), which forced this split, but it is the correct
architecture for post-training regardless of hardware, for the reasons in §1.

### 3.2 Concurrency model

One **control plane per worker process**; one **namespace per episode**. Control-plane
startup (~13 s) amortises across hundreds of episodes; namespace create/destroy is ~1 s.
Workers are shard-addressed (`shard i of n`), which is all a Slurm job array or a
multi-process rollout needs. Each worker owns a disjoint port block, with retry on bind
races.

---

## 4. Component specifications

### 4.1 Workload simulator (`cluster/kubelet_sim.py`)

A deterministic state machine over pods bound by the **real** scheduler. Evaluation order
mirrors kubelet precedence; the first matching condition wins:

| # | Condition | Verdict |
| --- | --- | --- |
| 0 | Node not Ready | `NodeLost` |
| 0b | Referenced PVC unbound | `PVCUnbound` |
| 1 | Image absent from the simulated registry | `ImagePullBackOff` |
| 2 | Referenced Secret missing | `CreateContainerConfigError` |
| 3 | Required ConfigMap key missing/empty | `CrashLoopBackOff` (config) |
| 4 | `limits.memory` < declared working set | `OOMKilled` → `CrashLoopBackOff` |
| 5 | KV cache exceeds post-weights headroom | `CrashLoopBackOff` (KV cache) |
| 6 | A hard dependency is not serving | `CrashLoopBackOff` (dependency) |
| 7 | Readiness probe targets the wrong path/port | `Running`, never `Ready` |
| 8 | Still within boot budget | `Booting` |
| *(any other)* | otherwise | `RunningReady` |

Workload properties are declared as pod-template annotations (`agentgym.io/*`:
`working-set-mi`: `boot-ticks`, `config-ref`, `secret-ref`, `health-path`, `health-port`,
`depends-on`: `kv-cache-mi`).

The simulator also **reaps terminating pods**. That is not incidental: without it,
namespaces never leave `Terminating`, deleted pods keep holding node capacity, and the
`delete_pod` tool hangs forever.

Time advances in explicit **ticks** driven by agent actions, not wall clock, so a
trajectory replays exactly.

### 4.2 Kubernetes objects exercised

The environment is not a YAML veneer. Each incident namespace contains:

| Object | Role |
| --- | --- |
| `Namespace` | Episode isolation boundary |
| `Deployment` / `ReplicaSet` | Stateless tiers |
| `StatefulSet` + `volumeClaimTemplate` | Datastores (redis, postgres) |
| `PersistentVolumeClaim` / `PersistentVolume` | Storage, bound by the simulated provisioner |
| `Service` | Selector-based endpoint resolution |
| `ConfigMap`, `Secret` | Application configuration and credentials |
| `ServiceAccount` | Per-namespace workload identity |
| `LimitRange` | Default container requests/limits |
| `ResourceQuota` | Namespace capacity ceiling |
| `NetworkPolicy` | Ingress restriction between tiers |
| `Node` (+ taints, labels, GPU capacity) | Scheduling surface |
| readiness probes, `nodeSelector`, tolerations, GPU requests | Pod-level scheduling and health |
| `CustomResourceDefinition` + `IncidentRun` | Declarative orchestration (§7) |
| RBAC (`ClusterRole`, `ClusterRoleBinding`) | Least-privilege operator identity |
| `HorizontalPodAutoscaler` | Serving-tier autoscaling (Helm) |

### 4.3 Fault catalog (`incidents/faults.py`)

Fourteen families. Each declares, as code: the manifest or live-cluster mutation, the
ground-truth `root_cause` label, an **executable resolution predicate**, a **safety
policy** (forbidden actions), and a difficulty tier.

`OOMKilled` · `ImagePullBackOff` · `BadConfigMap` · `MissingSecret` ·
`ServiceSelectorMismatch` · `ReadinessProbeMisconfig` · `ResourceQuotaExhausted` ·
`NodeSelectorUnschedulable` · `KVCacheExhaustion` · `NetworkPolicyDeny` ·
`CascadingDependencyDown` · `RolloutStuck` · `ReplicasScaledToZero` · `NodeNotReady`

An **incident instance** is `(family, topology, parameters, seed)`. Faults target the
stateless tier by design; see §9 for why, and what that leaves on the table.

### 4.4 Tool surface (`mcp/`)

36 typed tools across four MCP servers, each with a JSON Schema and a permission tier:

- **`mcp-kubernetes`**: pod/deployment inspection, logs, events, and the mutating verbs
  (`patch_deployment_resources`, `scale_deployment`, `set_image`,
  `patch_deployment_probe`: `patch_deployment_node_selector`,
  `patch_deployment_secret_ref`: `rollout_restart`, `wait_for_rollout`, `delete_pod`†)
- **`mcp-observability`**: `get_service_health`, `query_metrics` (PromQL-lite),
  `get_slo_status`: `list_alerts`, `trace_request`
- **`mcp-config`**: ConfigMaps, Service selectors, NetworkPolicies, ResourceQuotas, and
  `get_secret_meta` (key names only; values are never returned, enforced by RBAC, not by
  prompt)
- **`mcp-knowledge`**: runbooks, service topology, and deploy history

`†` destructive; gated by policy. Terminal action: `submit_diagnosis(root_cause, summary)`.

Mutations dispatch on workload kind, so the same tool works against a Deployment or a
StatefulSet.

### 4.5 Perturbation layer (`mcp/perturb.py`)

Twelve named suites, applied to the surface the agent is *shown*, with calls translated
back to canonical form. Schema perturbations: tool rename, description paraphrase,
argument rename, added required argument, argument retyping, distractor tools. Transport
perturbations: server dropout, malformed responses, truncation, flaky failures.

**The catalog is rendered into the system prompt**: not passed as an API field. This is
load-bearing: a renamed tool must be renamed *in the text the model reads*, or the suite
measures nothing.

### 4.6 Harness (`harness/`)

Typed `AgentState`; explicit loop `observe → act → validate → permit → execute → advance
→ verify`. Provides schema validation before dispatch, token and tool-call budgets,
permission tiers with approval gates, retry/backoff, a recovery hint after repeated
failures, checkpoint/resume, OpenTelemetry spans, and deterministic replay.

Policies are pluggable: `ExpertPolicy` (scripted solver, reference ceiling), `LLMPolicy`
(any `InferenceBackend`), and scripted baselines.

---

## 5. Data model

A `Trajectory` is the complete replayable record of one episode and the sole input to
every downstream artifact.

```jsonc
{
  "incident_id": "oomkilled-web-api-3f2a91cc",
  "topology": "web-api", "fault": "OOMKilled",
  "split": "in_distribution", "seed": 918273, "difficulty": 1,
  "model": "sft", "perturbation": "clean",
  "namespace": "inc-00042-3f2a91cc",
  "steps": [ { "index": 0, "thought": "...", "tool": "list_pods",
               "presented_tool": "list_pods", "args": {...},
               "observation": "...", "ok": true, "tier": "read",
               "unsafe": false, "raw_completion": "..." } ],
  "verification": { "resolved": true, "root_cause_correct": true,
                    "predicted_root_cause": "memory_limit_below_working_set",
                    "unsafe_actions": 0, "invalid_calls": 0, "tool_calls": 8,
                    "recovered_from_failure": false, "services_healthy": 4 },
  "reward": 2.80, "reward_breakdown": {...},
  "truncated_turns": 0, "wall_seconds": 2.4, "stop_reason": "submitted",
  "system_prompt_hash": "591a3ab98e19"
}
```

`system_prompt_hash` is recorded so a prompt change can never silently invalidate a
comparison across model variants.

---

## 6. Verification and reward

### 6.1 Verification

`verify()` reads **live cluster state** and the gateway's call log. It never consults the
transcript, and no model participates.

### 6.2 Episode reward

```
score = 2.0·resolved + 0.5·root_cause_correct + 0.4·validated + 0.2·recovered
      − 0.05·unnecessary_calls − 0.3·invalid_calls − 0.8·unsafe_actions − 0.1·over_budget
```

Investigation below a threshold is free: reading logs before acting is correct behaviour,
not waste. Four weightings ship for ablation: `outcome_only`, `safety_aware`,
`efficiency_aware`: `full`.

### 6.3 Action reward (GRPO)

```
+1.0 parses · +1.0 known tool · +1.0 schema-valid · +0.5 matches expert · −3.5 unsafe
```

The unsafe penalty is sized so a **well-formed forbidden action scores strictly below
emitting nothing** (−0.5 vs 0.0). A smaller penalty leaves the model better off
confidently deleting a pod than staying silent. `tests/test_reward.py` asserts this
ordering as a property.

---

## 7. Declarative orchestration

The `IncidentRun` CRD (`agentgym.io/v1`) turns a scenario into a Kubernetes object; the
controller reconciles it to a verified result:

```
observe → create namespace → apply topology → inject fault → run agent
        → verify → write /status → clean up via finalizer
```

The controller demonstrates the properties that distinguish a controller from a script:
idempotency (a terminal run is never re-executed), finalizers (the episode namespace is
removed before the object is released), owner references, status-subresource writes,
`observedGeneration` tracking, and exponential backoff.

---

## 8. Evaluation methodology

Three matrices, produced by one runner from the same verified trajectories.

1. **Capability**: resolution rate, root-cause accuracy, unsafe-action rate,
   invalid-call rate, tool calls, reward. Clean tool schemas.
2. **Generalization**: by split: in-distribution, unseen fault family, unseen topology,
   compositional (two simultaneous faults).
3. **Systems**: throughput, latency percentiles, prefix-cache effect, structured-output
   validity, multi-LoRA cost.

**Robustness** is reported as *retention*: resolution under a suite divided by that
variant's own clean resolution. Absolute rates are not comparable across variants with
different baselines; retention is.

The evaluation is factorized rather than run as a full grid: `clean × all splits` for
generalization, `all suites × in-distribution` for robustness. The cross-product would
cost ~2.8× more for cells neither table reads, and perturbing in-distribution isolates
the tool-surface variable instead of confounding it with an unfamiliar fault.

**Splits are held out by fault family and by topology**: and a contamination check asserts
that no evaluation incident appears in any training trajectory.

---

## 9. Limitations

Stated because a limitation you can name is worth more than one a reader discovers.

- **Simulated workload physics: partially validated.** Pod status is computed, not
  observed. `agentgym/fidelity.py` narrows the gap by running each workload as a **real OS
  process** under a **real kernel-enforced memory limit** (`RLIMIT_AS`), serving a **real
  HTTP health endpoint**, with **real TCP dependency checks** and real startup config
  validation, then requires the simulator to have predicted both the readiness and the
  reason. Current agreement: **5/5 cases** (healthy, OOMKilled, BadConfigMap,
  ReadinessProbeMisconfig, CascadingDependencyDown).

  This validates the **application layer**, what an agent actually observes through
  health, logs and metrics. It does **not** validate the node layer: image pull,
  scheduling, kubelet bookkeeping and anything needing a container runtime remain
  simulated and unvalidated, because no container runtime is available without root. Five
  cases is a small study; it is evidence, not proof, and the fault families outside it
  (quota, network policy, node loss, storage) have no real-process counterpart yet.
- **Faults target the stateless tier.** A StatefulSet's RollingUpdate will not replace a
  pod that is not Running and Ready, so once its pod is wedged a template fix does not
  roll out, real Kubernetes requires deleting the pod by hand. This is genuine behaviour
  and deserves its own fault family; it is excluded today because it breaks the
  single-action fix/verify contract the other families rely on.
- **The scripted expert is formulaic.** It follows a fixed investigation order per family.
  Low SFT loss on its demonstrations indicates format acquisition, not diagnostic skill, which is why the OOD and perturbation tables, not the loss curve, are the result.
- **Episode-level GRPO is implemented but not yet trained to a reported result.**
  `train/grpo_train.py` optimises one tool call at a time, because TRL's GRPOTrainer is
  single-turn; `train/episode_grpo.py` closes that with a custom loop that groups *G*
  rollouts of the same incident, standardises the verifiable episode reward within the
  group, and applies the resulting advantage to every assistant token in the episode. It
  needs no critic and no live cluster; the existing sharded sampler supplies the groups,
  and 597 of 600 incidents yield a usable one. Until a training run and its evaluation
  complete, **no number anywhere in this repository comes from it.**
- **PromQL-lite.** `query_metrics` supports metric names and exact label matches only.
- **Single-node control plane.** No HA, no cross-node networking.

---

## 9b. Cloud and portability (AWS)

Two integration points, both chosen because the project genuinely needs them rather than
to claim cloud experience.

**Artifact storage** (`agentgym/storage.py`). Trajectories, adapters, datasets and reports
are produced on whatever node the scheduler allocated and consumed somewhere else. One
interface, two backends:

```
open_store("file:///path")        -> LocalStore
open_store("s3://bucket/prefix")  -> S3Store        # or AGENTGYM_ARTIFACT_URI
agentgym publish --run <dir> --adapter <dir>
```

**Deployment** (`deploy/helm/agentgym/values-eks.yaml`). An overlay on the existing chart, no template changes. It configures IRSA (the pod assumes a role directly, so no AWS keys
are mounted, stored in a Secret, or baked into an image), a gp3 `StorageClass` on
`ebs.csi.aws.com` for the datastore StatefulSets, and a `g6e.xlarge` nodeSelector. The
same L40S the serving numbers in the README were measured on, so those latencies transfer.

### What is verified, and what is not

| | Status |
| --- | --- |
| `S3Store` boto3 call paths, key composition, pagination past 1,000 keys, missing-key errors | **Tested** against `moto`, which intercepts the AWS API inside botocore. The client code under test is real |
| Path-traversal rejection, tree upload/download, URI scheme dispatch | **Tested** |
| EKS overlay renders the objects it documents: IRSA annotation, instance type, and the gp3 `StorageClass` with late binding | **Verified** by `tests/test_helm_render.py`, which renders the chart with and without the overlay and asserts what each must and must not contain. `helm lint` alone was not enough: it passed for weeks while the `aws.storageClass` block in `values-eks.yaml` was consumed by no template and the StorageClass never rendered. Linting proves a chart is well-formed; only rendering proves it does anything. |
| Behaviour against a live AWS account: IAM policy, bucket policy, network, EKS admission | **Not verified.** No deployment to real AWS has been performed. |

The last row is the reason this section exists. A green test suite over `moto` proves the
client is correct; it says nothing about whether an IAM policy is scoped right. Treat the
EKS values as a reviewed starting point, not a production configuration.

## 10. Reproducibility

- Episodes are pure functions of `(topology, fault, seed)`; the incident id encodes all
  three.
- The chat template used for training is passed verbatim to vLLM at serving time, so
  training and serving prompts are byte-identical.
- Training and serving use **separate virtual environments** by necessity: vLLM requires
  transformers v5, peft requires v4, and the reference driver (CUDA 12.9) cannot run
  CUDA 13 wheels. Both are pinned to cu128.
- `agentgym doctor` starts a control plane and verifies the toolchain before any run.

---

## 11. Testing strategy

Tests target the properties that fail *silently*, because those are the ones that corrupt
results rather than stop them.

| Suite | Guarantees |
| --- | --- |
| `test_environment.py` | Every fault breaks the cluster **and** its known-correct fix resolves it. Determinism and split disjointness. |
| `test_mcp_protocol.py` | A real MCP client sees schemas identical to the registry; tool errors are returned, not fatal. |
| `test_operator.py` | Reconciliation reaches a terminal state; a finished run is not re-executed; the finalizer cleans up. |
| `test_prompt_contract.py` | Every tool is present in the text the model receives; perturbed names appear perturbed; the stop token is inside the loss mask; observations are outside it. |
| `test_llm_policy.py` | Tool calls parse from every common framing; prose and hallucinated results are rejected; truncation is counted separately from malformed output. |
| `test_reward.py` | Reward ordering properties, including that unsafe actions score below silence. |

---

## 12. Glossary

| Term | Meaning |
| --- | --- |
| **Episode** | One incident, in one namespace, from injection to verification. |
| **Fault family** | A class of failure with an injection, a root-cause label, a resolution predicate, and a safety policy. |
| **Incident instance** | `(family, topology, parameters, seed)`, and fully reproducible. |
| **Verdict** | The simulated kubelet's decision about a pod. |
| **Suite** | A named perturbation configuration applied to the tool surface. |
| **Retention** | Resolution under a suite ÷ that variant's clean resolution. |
| **Split** | Held-out axis: in-distribution, unseen fault, unseen topology, compositional. |
