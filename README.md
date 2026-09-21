# MCP-AgentGym

**A verifiable Kubernetes incident gym for post-training and serving tool-using agents.**

MCP-AgentGym runs a **real Kubernetes control plane** — etcd, apiserver, controller-manager,
scheduler — as unprivileged user processes, and replaces the kubelet with a deterministic
workload simulator. Incidents are injected from a seeded catalogue of 14 fault families, an
agent works them through 36 typed MCP tools, and an executable verifier reads live cluster
state to decide whether the incident was actually fixed. No model judges anything.

That gives the three properties post-training needs and a real cluster cannot provide:
**sub-second resets, hundreds of parallel episodes per node, and rewards that are computed
rather than graded.** On top of it the project trains Qwen3-8B with SFT, DPO and two
flavours of GRPO, serves the result through vLLM, and measures what survives.

📊 **[Results and methodology →](https://yupengtang.github.io/MCP-AgentGym/)**  ·
📐 [Design document](DESIGN.md) · 🧪 [Experiment log](reports/EXPERIMENTS.md) ·
🗂 [Design decisions](PROJECT.md)

---

## The result: post-training on a closed catalogue is a trade, not an improvement

`n = 2,200` per arm, on 5,000 incidents no model has seen (zero id overlap with training).
Bonferroni threshold for the 42 comparisons reported is `z > 3.0`.

| | in-distribution | **held-out faults** | unseen topology | compositional | **all** | invalid calls |
|---|---|---|---|---|---|---|
| base, untrained | 33.4% | **65.8%** | 24.6% | 22.4% | 36.3% | **26.3%** |
| SFT (2 epochs) | 59.4% | 40.6% | 28.2% | 33.4% | 42.1% | 0.35% |
| best trained (SFT 3ep + DPO) | **71.3%** | 38.2% | **52.0%** | **40.8%** | **52.5%** | 0.32% |

```
what training buys   resolution          +16.2 pp        z = +10.96
                     root cause          +14.2 pp        z =  +9.57
                     invalid tool calls  26.3% → 0.3%    ~100×

what it costs        held-out families   −27.6 pp        z = −19.06
```

Both sides clear significance by a wide margin, and both dwarf every difference between
post-training methods. **The untrained model is the best agent in this project on fault
families it was never trained for** — 65.8% against 33–41% for everything trained — and the
worst everywhere else.

**The mechanism is measurable.** Fine-tuning drives the two never-demonstrated diagnosis
labels to **0 emissions in 599 submissions** (base: 25/501, 5.0%), though both sit in the
`submit_diagnosis` enum and the system prompt on every episode. The models are not blind to
those labels; they can no longer reach them. What they emit instead is confidently wrong and
drawn from the training catalogue — `readiness_probe_targets_wrong_endpoint` for 17 of 32
`NodeNotReady` incidents. That is **label-prior collapse**, not degraded reasoning, and the
distinction decides the fix: degraded reasoning needs better data, a collapsed prior needs
the update attenuated.

**And the trade is reversible after training, at serving time.** A LoRA update enters each
layer as `ΔW = (α/r)·B@A`, so scaling `lora_B` by `s` scales the update exactly. `s = 0` is
the base model, `s = 1` the trained one, and every value between is servable from one engine
by multi-LoRA:

| s | held-out resolved | root cause | held-out labels emitted |
|---|---|---|---|
| 1.00 (trained) | 36.7% | 0.0% | **0 / 114** |
| 0.75 | 50.8% | 6.7% | 8 / 112 |
| **0.50** | **72.5%** | **20.0%** | **24 / 56** |

Monotone at every step, both adjacent comparisons significant (z = 3.55, z = 2.22),
**z = 5.97 end-to-end**, and no measurable in-distribution cost (67.2% vs 65.6%, z = 0.27).
**It replicates on a second, independently trained adapter family to within 1.6 pp** — the
SFT adapter recovers 35.8% → 70.0% (z = 5.65). Two policies trained by different objectives
on different data collapse the same way and recover the same way, which is the strongest
available evidence that the cause is update *magnitude* rather than anything incidental to
one checkpoint.

One knob, applied after training has finished, moves a model along the specialization axis.

## The variance budget: what actually moves the number

Every source of variation this project can measure, all at `n = 2,200` per arm on the same
5,000 fresh incidents:

| source | Δ resolution | z |
|---|---|---|
| **learning rate, 5e-5 → 1e-4** | **+14.23 pp** | **+9.54** |
| DPO on baseline A | +4.18 pp | +2.83 |
| **training seed, 11 → 12** | **−3.09 pp** | **−2.06** |
| DPO on baseline B | −0.27 pp | −0.18 |
| nondeterminism, *same* seed | −0.18 pp | −0.12 |

**Tuning the learning rate mattered more than every post-training method tested**, by a
margin nothing else approaches — and the published baseline had been trained at the slower
one. Same seed, same corpus, same epochs: +14.2 pp resolution, +10.4 pp root cause, invalid
calls 0.59% → 0.01%.

**The seed floor is 3.1 pp, which is the same size as DPO's effect.** That is why DPO helped
substantially on one baseline (+4.18 pp, z = 2.83) and not at all on another (−0.27 pp), when
both baselines were identically configured and scored within 0.2 pp of each other. The
quantity being measured is comparable to the noise in what it is measured against, so a
single run cannot resolve it.

**Two floors, and using the wrong one is the easy mistake:** 0.2 pp between two runs of the
*same* configuration, 3.1 pp between independently trained models. Method comparisons are
the second case. This document quoted the first as *the* floor for part of one session, which
understated the relevant noise sixteenfold.

## The complete grid

All eight arms, `n = 2,200` each. Bonferroni threshold for the 56 comparisons is `z > 3.0`.

| variant | resolved | root cause | invalid calls |
|---|---|---|---|
| **SFT 2 epochs, seed 1 — no post-training** | **57.7%** | 48.9% | **0.01%** |
| SFT 3 epochs + DPO-309 | 52.5% | **51.8%** | 0.32% |
| SFT 3 epochs | 50.3% | 50.6% | 0.19% |
| SFT 2 epochs + DPO-435 | 44.3% | 41.5% | 0.18% |
| SFT 2 epochs + episode-GRPO | 44.2% | 40.5% | 0.28% |
| SFT 2 epochs + action-GRPO | 43.0% | 38.5% | 0.28% |
| SFT 2 epochs, seed A — the published baseline | 42.1% | 36.9% | 0.35% |
| base, untrained | 36.3% | 37.6% | 26.32% |

**The best agent here is a plain supervised fine-tune that happened to train at the better
learning rate.** It beats every post-trained variant and has the cleanest tool use in the
project at 0.01% invalid calls. The top row is not a post-training result; it is the absence
of one.

## What this project does not establish

**Whether DPO helps, as a number.** With the learning rate matched and the floor measured,
the question was asked properly — train DPO from a specific SFT, compare against *that* SFT.
Two pairs, both baselines identically configured, scoring 57.7% and 57.5%:

| | Δ resolution | z | Δ root cause | z |
|---|---|---|---|---|
| pair 1 | **+4.18 pp** | +2.83 | **+5.41 pp** | +3.60 |
| pair 2 | −0.27 pp | −0.18 | +0.05 pp | +0.03 |

**They disagree, and it is not evaluation noise** — the floor is 0.2 pp and pair 1's effect is
twenty times that. The same recipe on two indistinguishable baselines gives a large gain once
and nothing the other time, so the variability is in *training*, not measurement. DPO can help
by a wide margin; what cannot be quoted is a number. Two pairs also cannot support a sign
test, and none is offered.

**Any ordering among SFT / DPO / episode-GRPO / action-GRPO.** Each was trained once, so their
differences sit inside the spread one recipe produces by reseeding. Establishing that any of
them helps needs several seeds per arm, which has not been done here.

**Anything about a second base model.** Everything is Qwen3-8B. Mistral is cached in Mistral
format on the machine this ran on, with `HF_HUB_OFFLINE=1`, so HF-format weights could not be
fetched.

**Live AWS behaviour.** The S3 artifact store is asserted against `moto`; the Helm EKS overlay
renders and is diffed in tests. Neither has been run against a real account.
`scripts/verify_aws.py` does that and has not been executed.

## Why believe any of it

The discipline that produced the results above also retracted five of this project's earlier
claims — including one that was published here, contradicted by a later run, and withdrawn
with its trajectories quarantined rather than deleted:

| retracted claim | what it actually was |
|---|---|
| "DPO +8.2 pp" | +2.1 pp under power; never significance-tested when first published. Winner's curse. |
| "preference pair count matters" | A 437-pair replication scored identically to the 88-pair run. |
| "episode-GRPO damages tool-call format" | An adapter initialised two versions behind its own training data. |
| "one SFT epoch beats every method" | Confounded; the epoch alone is −4.1 pp. |
| "seed variance is 15.6 pp" | Learning rate. **No launcher ever forwarded `--seed`, so every adapter this project produced trained at seed 0.** |

Every increase in rigour moved the answer toward "no measurable difference" — until the
learning rate was matched, at which point DPO reappeared at +4.2 pp on one baseline and
nothing on another. Both halves of that belong in the record, and both are in
[reports/EXPERIMENTS.md](reports/EXPERIMENTS.md) with the defect that caused each retraction
and the assertion that now prevents it.

**The guards are in the code, not the prose.** The report script refuses to render a partial
grid. A test asserts every tool count quoted in the docs matches the registry, another that
the serving numbers are recomputable from their raw artifacts. Training records a weight digest
of its base adapter, so lineage is checkable — that is how the withdrawn GRPO run was caught.
`build-datasets` refuses to overwrite the corpora the published results were trained on. And
`tests/test_launcher_plumbing.py` asserts that every result-bearing argument a trainer accepts
is reachable from its launcher, because for most of this project's life `--seed` was not.

---

## How it works

Post-training an agent needs thousands of **resettable, seeded, parallel** episodes with
**verifiable** rewards. A real Kubernetes cluster gives none of those: episodes take minutes,
never reset cleanly, and are nondeterministic, so the same action yields different rewards on
different runs — fatal for DPO preference pairs and GRPO advantages alike. So the stack is
split at the API boundary, which is exactly where an agent operates anyway.

```
Incident generator ──► Kubernetes incident environment ──► MCP gateway (36 typed tools)
 (14 fault families,     (real apiserver + simulated       + schema-shift perturbation
  seeded, held-out        kubelet/mesh/telemetry)                     │
  splits)                                                            ▼
       ▲                                                    Agent harness
       │                                            (plan → act → verify → recover)
       │                                                             │
       │                                                             ▼
       │                                          Executable verifier + reward
       │                                                             │
       │                                                             ▼
       └──── new failure cases ◄──── Evaluation ◄──── SFT / DPO / GRPO ◄── trajectories
```

| Layer | Real / simulated | Consequence |
|---|---|---|
| etcd, kube-apiserver, controller-manager, scheduler | **Real** (v1.31.4, user processes) | Real CRDs, RBAC, admission, watch/reconcile. Deployment→ReplicaSet→Pod is genuine, as are scheduling, taints and `Unschedulable`. |
| MCP protocol | **Real** (`mcp` reference library, JSON-RPC) | A standard MCP client drives this environment unchanged. |
| kubelet / container runtime | **Simulated** | Pod status computed from declared workload physics. Deterministic and instant. |
| Traffic, health, metrics, logs | **Simulated** | Health is a pure function of cluster state, so the verifier is exact. |
| The scripted expert | **Scripted, not a model** | Reference ceiling and SFT demonstration source. Always reported as a scripted policy. |

Container runtimes were unavailable on the target hardware — no root, no Docker, no
`/etc/subuid` range for rootless Podman — which is what forced this split. But as argued in
[PROJECT.md §1.1](PROJECT.md), it is the correct architecture for post-training regardless of
hardware.

## Verified results

Everything below was produced by running the code in this repository. Model rows regenerate
with `python -m agentgym.eval.report`, which reads `runs/eval/`.

### The environment is correct

The foundation everything rests on: for all **14 fault families**, the injected fault provably
breaks the cluster *and* the known-correct remediation provably fixes it, with the executable
verifier recognising both states.

```
tests/test_environment.py ......... 19 passed   (each fault breaks, then fixes)
tests/test_llm_policy.py .......... 18 passed   (parsing, recovery, end-to-end fix)
tests/test_episode_grpo.py ........ 16 passed   (advantage algebra, loss, KL, masking)
tests/test_prompt_contract.py ..... 12 passed   (the model can see every tool)
tests/test_storage.py ............. 10 passed   (artifact store, S3 paths via moto)
tests/test_reward.py .............. 10 passed   (reward ordering properties)
tests/test_cli_contract.py ........ 10 passed   (every command's --help resolves)
tests/test_report_variants.py ...... 8 passed   (no two variants share a table row)
tests/test_mcp_protocol.py ......... 8 passed   (real MCP client ↔ real server)
tests/test_launcher_plumbing.py .... 7 passed   (every trainer arg reaches its launcher)
tests/test_helm_render.py .......... 6 passed   (chart renders, EKS overlay diffs)
tests/test_fidelity.py ............. 6 passed   (sim vs real processes and limits)
tests/test_operator.py ............. 5 passed   (reconcile, idempotency, finalizer)
tests/test_backend_pluggability.py . 5 passed   (vLLM / SGLang / hosted interface)
                                    140 passed
```

The prompt-contract suite exists because of the worst bug in this project: tool schemas were
passed as an OpenAI API field that the project's own chat template ignored, so the model was
being asked to call tools it had never been shown. Nothing raised. It would have silently
voided both the base-model comparison and the entire schema-shift result. The invariant is now
asserted — whatever the gateway presents must be literally present in the text the model
receives.

Sim-to-real fidelity is checked where it is checkable: 5/5 on the process layer (healthy,
`OOMKilled`, `BadConfigMap`, `ReadinessProbeMisconfig`, `CascadingDependencyDown`), agreeing
on **both** readiness and reason.

### Scale and cost

| | |
|---|---|
| Incidents generated | 1,200 base pool + a 5,000-incident prefix-stable evaluation pool |
| Trajectories executed and verified | **11,541** |
| Control plane start | ~13 s, amortised over hundreds of episodes |
| Median episode wall time | **2.4 s** |
| GPU cost of the environment | **zero** — CPU only |
| QLoRA SFT (Qwen3-8B, 3 epochs, 614 examples) | 1 h 29 m on one L40S |
| Unsafe actions across 3,840 post-trained episodes | **0** |

### The scripted expert reference line

The expert is the environment's *solver*: it has ground truth and resolves tool names through
the schema it is shown. It is the ceiling a model could reach by reading the schema perfectly
every turn.

| variant | n | resolution | root cause | unsafe | invalid calls | tool calls | reward |
|---|---|---|---|---|---|---|---|
| expert-scripted | 4,010 | 91.4% | 100.0% | 0.0% | 0.0% | 8.0 | 2.60 |

By split: in-distribution 92.9%, unseen fault 97.8%, unseen topology 95.0%, **compositional
50.8%**. The compositional number is the informative one — the expert reasons about a single
root cause, so when two independent faults are injected at once it fixes one and leaves the
other. Single-fault reasoning does not compose, and the environment measures that directly
rather than asserting it.

### Robustness to tool-schema shift

Retention = resolution under the suite ÷ that variant's own clean resolution, both on
in-distribution incidents, so the tool surface is the only thing that varies. `n = 64` per cell.

| variant | clean | rename | arg_rename | +required arg | distractors | **server dropout** | combined |
|---|---|---|---|---|---|---|---|
| expert (ceiling) | 1.00 | 1.04 | 1.04 | 1.02 | 1.02 | **0.74** | 1.04 |
| base | 1.00 | 0.82 | 1.00 | 0.77 | 0.77 | **0.50** | 0.83¹ |
| SFT | 1.00 | 1.12 | **1.21** | 0.76 | 1.15 | 0.56 | 0.82 |
| **SFT + DPO** | 1.00 | 1.09 | 1.18 | **0.88** | **1.29** | **0.62** | **0.91** |

¹ base `combined` is `n = 7`; that cell was cut off when the job hit its 8-hour wall clock.

Both post-trained variants are far more robust than base to a **renamed** tool (1.09–1.12 vs
0.82), a **renamed argument** (1.18–1.21 vs 1.00) and **distractor** tools (1.15–1.29 vs 0.77).
They learned to *read* the schema they are given rather than recall tool names from training —
the intended effect, and the clearest evidence the gain is not memorisation.

The floor is still the floor. Nobody handles a tool that **disappears**: even a policy reading
the schema perfectly retains only 0.74, because a withdrawn tool cannot be called by anyone.

### Serving: what prefix caching is worth to an agent workload

One L40S, Qwen3-8B, with the real 36-tool system prompt. The synthetic conversation deepens
exactly as an episode does, so the shared prefix grows while new tokens per turn stay small.

| turns | prompt tokens | concurrency | p50 **without** prefix cache | p50 **with** | speedup |
|---|---|---|---|---|---|
| 2 | 3,735 | 32 | 14,123 ms | 3,250 ms | 4.3× |
| 5 | 6,768 | 32 | 27,511 ms | 3,833 ms | 7.2× |
| 10 | 11,823 | 32 | 49,239 ms | 4,687 ms | 10.5× |
| 20 | 21,943 | 32 | 84,418 ms | **6,576 ms** | **12.8×** |

Throughput at 20 turns / 32 concurrent: **0.26 → 4.67 req/s (18×)**. Read as cost per unit of
prompt, ms per 1k prompt tokens goes 3,781 → 3,847 uncached (flat — every turn re-reads the
whole conversation) against 870 → **300** cached. **Prefix caching does not merely help agent
workloads; its benefit compounds with conversation depth**, which is the defining property of
one.

Multi-LoRA on one engine: base 2,292 ms p50 against the SFT adapter's 1,075 ms. The adapter is
2.1× faster because it emits a short tool call where the base emits a long `<think>` block — a
latency win that comes from behaviour, not from the engine.

---

## Quickstart

```bash
source env.sh                          # paths, venvs, HF cache
bash scripts/fetch_k8s_binaries.sh     # etcd + kube-* (once, ~330 MB)
python -m agentgym.cli doctor          # starts a control plane and verifies it

python -m agentgym.cli gen-incidents --n 1200 --seed 11
python -m agentgym.cli rollout --policy expert --limit 20 --verbose
python -m agentgym.cli stats $AGENTGYM_RUNS/expert-clean
python -m agentgym.cli build-datasets  # SFT/DPO corpora + contamination check
```

`build-datasets` refuses to overwrite an existing dataset directory, because the published
results were trained on specific corpora and silently regenerating them would invalidate
every number above.

Batch work on Slurm:

```bash
sbatch slurm/rollout_expert.sbatch            # 1,200 trajectories, CPU, ~5 min wall
sbatch slurm/expert_perturbation_sweep.sbatch # 12 perturbation suites × 4 splits
sbatch slurm/train_sft.sbatch                 # QLoRA SFT on one L40S
sbatch --export=ALL,VARIANT=sft,ADAPTER=<path> slurm/serve_and_eval.sbatch
python -m agentgym.eval.report                # rebuild all matrices
```

Site-specific paths (shared storage, node-local scratch, a shared HF cache) go in
`env.local.sh` — copy `env.local.sh.example`. It is never committed.

## Repository map

```
src/agentgym/
  cluster/       control plane (real), kubelet sim, service mesh, telemetry
  incidents/     14 fault families, topologies, generator, episode, gym
  mcp/           36 typed tools, 4 MCP servers, perturbation layer, gateway
  harness/       agent loop, typed state, prompts, scripted expert, LLM policy
  backends/      InferenceBackend: vLLM / SGLang / hosted API
  data/          trajectory schema, verifier, reward, curation
  train/         QLoRA SFT, DPO, action-GRPO, episode-GRPO
  eval/          report matrices, serving benchmarks
  operator/      IncidentRun CRD + controller
deploy/helm/     serving stack, RBAC, NetworkPolicy, EKS overlay
slurm/           job arrays for rollout, sweep, training, evaluation
tests/           140 tests: environment contract, MCP protocol, operator, plumbing
```

**The pieces worth reading first:**

- [`cluster/kubelet_sim.py`](src/agentgym/cluster/kubelet_sim.py) — the deterministic workload
  state machine, and why pod reaping is the kubelet's job
- [`incidents/faults.py`](src/agentgym/incidents/faults.py) — 14 families, each with an
  executable resolution predicate and a forbidden-action set
- [`mcp/perturb.py`](src/agentgym/mcp/perturb.py) — the schema-shift layer
- [`data/trajectory.py`](src/agentgym/data/trajectory.py) — verification and the reward
- [`train/episode_grpo.py`](src/agentgym/train/episode_grpo.py) — group-relative advantages
  over whole episodes, no critic
- [`operator/controller.py`](src/agentgym/operator/controller.py) — reconciliation, finalizers,
  owner references

## Reward

**Episode reward** — what SFT data is filtered on and what evaluation reports:

```
score = 2.0·resolved + 0.5·root_cause_correct + 0.4·validated + 0.2·recovered
      − 0.05·unnecessary_calls − 0.3·invalid_calls − 0.8·unsafe_actions − 0.1·over_budget
```

**Action reward** — what action-level GRPO optimises, one tool call at a time:

```
+1.0 parses · +1.0 known tool · +1.0 schema-valid · +0.5 matches expert · −3.5 unsafe
```

Every term is read from live cluster state, a JSON Schema, or the gateway's call log. No model
judges any of it.

The `−3.5` is sized deliberately so a *well-formed forbidden* action scores below emitting
nothing parseable (−0.5 vs 0.0). A smaller penalty would leave the model better off confidently
deleting a pod than staying quiet — the wrong lesson for an agent holding production
credentials. `tests/test_reward.py` asserts that ordering.

**Episode-level GRPO.** Action-level optimisation cannot credit-assign across a whole incident,
so `train/episode_grpo.py` groups *G* rollouts of the **same** incident and standardises the
episode reward within the group:

```
A_i = (R_i − mean(R_group)) / (std(R_group) + ε)
```

The group mean replaces a learned value function — no critic — and the advantage weights every
assistant token in that episode. Over the 2,400 rollouts already sampled for DPO it keeps **597
of 600 groups (2,388 episodes)** against **435 DPO pairs from the same data**: 5.5× the training
signal, because a preference pair keeps one best/worst couple above a margin while a group uses
the whole reward distribution.

Trained for 149 optimizer steps. KL to the frozen SFT reference drifts 0.011 → 0.022 (max
0.038), and 21 of 2,388 episodes hit the 12,288-token cap — 0.88%, against 1.2% predicted from
a length sample before the cap was chosen. The policy-gradient term is near-evenly signed across
logged steps (15 negative, 14 positive, mean −0.14), which is the *correct* signature for this
objective rather than a problem: advantages are zero-mean within a group, so over complete
groups the expected term sits near zero and the learning signal is the correlation between
advantage and log-probability. "Loss went down" is the wrong health check here.

At full power it lands at 44.2% resolution against the matched SFT's 42.1% — inside the seed
floor, like every other method in the grid.

## Known limitations

Stated because a limitation you can name is worth more than one a reader finds.

- **The expert is formulaic.** SFT reaches eval-loss 0.0215 partly because scripted
  demonstrations follow a fixed investigation order per fault family. Low loss here indicates
  format acquisition, not diagnostic skill — which is why the OOD and perturbation tables, not
  the loss curve, are the result.
- **Simulated workload physics.** Pod status is computed, not observed. Fidelity is verified on
  the process layer (5/5) and asserted per fault family; it is not a claim about a production
  cluster.
- **Post-training regresses on held-out fault families** — 38.2% against base's 65.8% at
  n = 2,200. Diagnosed to label-prior collapse and reversible at serving time, but the sweep
  does not locate an optimum, and the held-out axis is two fault families.
- **Every method arm is a single training run.** With a 3.1 pp seed floor and method effects of
  2–5 pp, the grid establishes that post-training works and does not establish which method is
  best.
- **One base model.** Qwen3-8B throughout.
- **PromQL-lite.** `query_metrics` supports metric names and exact label matches only.
- **Single-node control plane.** No HA, no multi-tenancy, no cross-node networking.
- **Compositional incidents inherit the primary fault's safety policy**, so unsafe-action
  accounting is slightly conservative on that split.

## Documents

| | |
|---|---|
| [DESIGN.md](DESIGN.md) | Design of record: architecture, components, data model, verification, evaluation methodology, limitations |
| [reports/EXPERIMENTS.md](reports/EXPERIMENTS.md) | Experiment log: every result, every retraction, and the engineering notes worth keeping |
| [PROJECT.md](PROJECT.md) | The four decisions taken against the original brainstorm, and why |

## License

MIT — see [LICENSE](LICENSE).
