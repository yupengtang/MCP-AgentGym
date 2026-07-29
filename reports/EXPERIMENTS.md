# MCP-AgentGym — experiment log

Methodology and results. Every number here was produced by running the code in this
repository on PACE ICE; nothing is estimated. Tables are regenerated with
`python -m agentgym.eval.report`.

---

## 1. Environment correctness

**Claim.** For every fault family, the injected fault breaks the cluster, the
known-correct remediation fixes it, and the executable verifier recognises both states.

**Method.** `tests/test_environment.py` parameterises over all 14 families. For each: a
fresh namespace, the topology applied, the fault injected, `assert not ep.resolved()`;
then the remediation applied through the same mutation methods the MCP write-tools use,
the world advanced until pod state stabilises, `assert ep.resolved()`.

**Result.** 14/14 families pass in both directions, plus determinism and split-disjointness
checks — **16 passed**.

This is the load-bearing result. A fault that does not bite makes its reward vacuous; a
fault that cannot be fixed makes its reward unreachable. Two bugs found this way:

- **The taint bug.** The API server stamps `node.kubernetes.io/not-ready:NoSchedule` on
  every new Node and relies on the node-lifecycle-controller to lift it. That controller
  is disabled here (it would evict simulated pods), so nothing removed the taint and
  *nothing was ever schedulable*. The simulator now owns the taint, because it owns node
  health.
- **The reaper bug.** Nothing confirmed pod deletion, so namespaces never left
  `Terminating` and deleted pods kept holding node capacity. After ~13 episodes a worker's
  nodes were full and every new pod stayed `Pending`. Confirming termination is the
  kubelet's job; the simulator now does it. This also fixed `delete_pod`, which would
  otherwise have hung forever.

Both were only visible because the contract test asserts on *cluster state* rather than
on the agent's transcript.

## 2. MCP protocol compliance

**Claim.** The tools are MCP tools, not functions with an MCP-shaped docstring.

**Method.** `tests/test_mcp_protocol.py` connects the reference `mcp` client to each of
the four servers over a real JSON-RPC transport, lists tools, and compares every
advertised name, description, and `inputSchema` against the registry.

**Result.** **7 passed** — schemas match exactly for all four servers; a tool error is
returned as an error result rather than killing the session, and the session stays usable
afterwards.

## 3. Operator

**Claim.** A real controller, with the properties that make controllers correct.

**Method.** `tests/test_operator.py` drives `IncidentRunController` against the real API
server: reconcile to terminal state, re-reconcile a finished run, delete and verify the
finalizer cleaned up.

**Result.** **5 passed** — CRD installs with a status subresource; a run reaches
`Succeeded` with `resolved`, `rootCauseCorrect`, `toolCalls` on `/status`; a second
reconcile pass acts on nothing (`acted == 0`, timestamps unchanged); the finalizer
removes the episode namespace before releasing the object.

## 4. Scale and cost

| Quantity | Measured |
|---|---|
| Incidents generated | 1,200 across 4 splits |
| Trajectories executed and verified | 11,541 |
| — of which back the reported tables | 5,383 |
| Control-plane start | 11–13 s |
| Median episode wall time | 2.4 s |
| Full 1,200-episode rollout | ~5 min wall (16-way CPU array) |
| Environment GPU cost | zero |
| QLoRA SFT, Qwen3-8B, 3 epochs, 614 examples | 1 h 29 m, one L40S |

Composition of the 11,541, so the figure is auditable rather than impressive: 5,383 in the
evaluation tree (the trajectories every table above is computed from), 2,558 expert
large-scale rollouts, 2,400 DPO/GRPO sampling episodes, and 1,200 expert clean rollouts.
The concatenated copies that sharded jobs leave behind, and the sampling set reused by
episode-GRPO, are counted once rather than twice.

An episode costs ~2.4 s on a CPU core. That is the number that makes post-training
viable: the same workload on a real cluster would be minutes per episode and would not
reset cleanly.

## 5. Scripted expert reference line

The expert has ground truth and resolves tool names through the schema it is shown. It is
the ceiling for an agent that reads the schema perfectly every turn, and it is reported as
a scripted policy throughout — never as a model.

| variant | n | resolution | root cause | unsafe | invalid calls | tool calls | reward |
|---|---|---|---|---|---|---|---|
| expert-scripted | 1,452 | 90.4% | 100.0% | 0.0% | 0.0% | 7.5 | 2.60 |

### 5.1 By split

| split | resolution |
|---|---|
| in-distribution | 95.7% |
| unseen fault | 97.8% |
| unseen topology | 95.0% |
| **compositional** | **50.8%** |

The expert reasons about one root cause. Given two independent simultaneous faults it
fixes one and stops — so compositional resolution lands near half. This is a designed
property being measured, not a defect: it quantifies exactly how far single-fault
reasoning carries.

### 5.2 Robustness to tool-schema shift

Retention = resolution under the suite ÷ that variant's own clean resolution.

| suite | retention |
|---|---|
| clean | 1.00 |
| rename | 0.96 |
| paraphrase | 0.96 |
| arg_rename | 0.96 |
| extra_required_arg | 0.95 |
| retype_arg | 0.95 |
| distractors | 0.95 |
| **server_dropout** | **0.74** |
| malformed | 0.95 |
| truncation | 0.95 |
| flaky | 0.95 |
| combined | 0.96 |

Two things follow, and both matter for reading the model results:

1. **The perturbations are really applied.** `server_dropout` costs 26% even to a perfect
   schema-reader, because a withdrawn tool cannot be called by anyone. That is the
   irreducible floor.
2. **Schema shift is survivable in principle.** Renames, paraphrases, argument renames,
   retyping, and distractors cost a perfect reader only ~4–5%. So when a model loses more
   than that, the loss is a fact about the model — it stopped reading the schema and
   started calling tools from memory — not an artifact of an impossible task.

## 6. Supervised fine-tuning

**Setup.** Qwen3-8B, QLoRA (4-bit NF4, LoRA r=32, α=64, all attention and MLP
projections), 614 curated demonstrations, 3 epochs, lr 1e-4 cosine, effective batch 8,
8,192-token context, one L40S.

**Loss masking.** Only assistant tokens contribute. Trajectories are dominated by long
JSON tool outputs; training on those would teach the model to *generate* cluster state
instead of querying for it. Qwen3's stock template has no `{% generation %}` markers, so
the project ships its own template — and passes the identical file to vLLM via
`--chat-template`, making the training and serving prompts byte-identical.

**Data hygiene.** 1,200 trajectories → 646 kept: 526 dropped as evaluation-split, 28 as
unresolved, and the remainder capped so no fault family exceeds 18% of the set.
Contamination check: **0 overlap** between the 646 training incidents and the 526
evaluation incidents.

**Result.** train_loss 0.110, eval_loss 0.034, token accuracy 98.9%.

**Reading it honestly.** An eval loss of 0.034 is *not* evidence of diagnostic skill. The
expert is formulaic — a fixed investigation order per fault family — so a low loss mostly
means the format was learned. Whether anything transferable was learned is what §5.2's
perturbation suites and the held-out splits are for, and those are the numbers to trust.

## 6b. RETRACTED — the "SFT is worse" result was a bug in this repository

**This section originally reported that SFT made the agent ten times worse and presented
it as a finding about training. That conclusion was wrong.** The cause was a
train/inference history mismatch in the harness (§ engineering notes, "the harness showed
the model a transcript it was never trained on"). Fixing it moved the same adapter from
3.1% to 29.3% resolution with no retraining.

The original analysis is kept below, unedited, because the reasoning that led to a wrong
conclusion is worth preserving: every hypothesis in the table *was* checked and correctly
ruled out, and the conclusion still turned out to be wrong — because the one thing not on
the list was the harness itself. The measurement apparatus was never a suspect.

The lesson is not about SFT. It is that a result which moves in the direction you expect
gets less scrutiny than one that does not, and "the fine-tuned model overfit and degraded"
is exactly the kind of story that feels explanatory enough to stop investigating.

---

### Original (retracted) analysis

**Setup.** Qwen3-8B, QLoRA r=32, 614 curated expert demonstrations, 3 epochs. Evaluated
against the base model on 256 in-distribution episodes with clean tool schemas.

| variant | resolution | root cause | invalid call rate | mean calls | mean reward |
|---|---|---|---|---|---|
| base (Qwen3-8B) | **30.9%** | 33.2% | 35.4% | 12.4 | −0.43 |
| SFT (v3) | **3.1%** | 0.4% | **89.8%** | 18.5 | −5.02 |

Fine-tuning made the model roughly **ten times worse** at the task it was fine-tuned for.

**What the SFT model actually emits.** It produces the *reasoning* half of the training
target and never the tool call:

```
Look at pod-level state to find which workloads are unhealthy and whether they are
restarting. Cluster events will show why they died. Cluster events will show why
they died.
```

Note the verbatim repetition and the blending of phrasing from different demonstration
steps — the signature of a degenerate policy, not of a model that learned a format.

**What was ruled out.** Each of these was checked rather than assumed:

| Hypothesis | Check | Result |
| --- | --- | --- |
| Training targets lack the JSON call | 5,233 assistant turns inspected | 5,233/5,233 contain a ```json block |
| Examples truncated at `max_length` | Tokenized the whole dataset | max 6,844 vs cap 8,192 — 0% truncated |
| Loss mask excludes the JSON | Decoded the mask on real content | mask = full content **+** `<\|im_end\|>` |
| Adapter not served | vLLM engine log and adapter config | LoRA loaded, r=32, correct base |
| Generation cap cutting the call | `truncated_turns` tracked separately | 167 of 4,736 turns — not the cause |

**Why this matters more than a good number would.** Training metrics were excellent:
`eval_loss 0.0334`, token accuracy **98.9%**. Under teacher forcing the model predicts
every next token almost perfectly, because the true prefix is always supplied. Free-running
it never has to *decide* to emit the call — and it doesn't. The gap between 98.9% token
accuracy and 3.1% task success is the entire argument for this project: **an agent cannot
be evaluated by its loss curve.** Only executed, verified rollouts reveal this, and they
have to be cheap enough to run routinely or nobody runs them.

The likely remedies — more diverse demonstrations, lower learning rate, fewer epochs,
mixing in on-policy data — are stated as hypotheses, not conclusions; none has been tested
yet. The result stands as measured.

## 6c. DPO, and two ways of reading the same number

**The result.** DPO on 88 preference pairs, mined from four sampled attempts at the same
incident and ranked by the executable reward (mean margin 2.61), on top of the SFT
adapter:

| | resolution | root cause | invalid calls | reward |
| --- | --- | --- | --- | --- |
| base | 32.8% | 39.1% | 26.3% | 0.10 |
| SFT | 40.6% | 35.2% | 0.6% | 1.08 |
| SFT + DPO (88 pairs) | 47.3% | 40.6% | 0.1% | 1.27 |
| SFT + DPO (435 pairs) | **48.8%** | **44.1%** | **0.1%** | **1.32** |

**The near-miss.** On in-distribution incidents with clean tools, SFT and DPO score
*identically* (53.1%). Reading that cell alone gives "DPO produced no gain" — a conclusion
that survives a sanity check, sounds like a familiar result, and is wrong. The gain is on
the held-out axes: **+23.4 pp on unseen topologies**, +3.1 pp on compositional incidents,
and improvement on every perturbation suite where SFT was weakest (required-argument
0.76 → 0.88, server dropout 0.56 → 0.62, combined 0.82 → 0.91).

That is a mechanistically sensible place for it to land: the preference pairs are drawn
from the model's *own failures*, so they carry recovery behaviour that demonstrations of
the happy path never show. But the point here is methodological — this is the second time
in this project that a correct set of numbers produced an inverted conclusion because the
wrong slice was read first (the earlier one compared SFT against the expert ceiling rather
than against base). Splitting the evaluation along four held-out axes is what made the
difference recoverable.

### Diagnosing a pair shortage without guessing

The first DPO run mined only 88 pairs from 120 incidents, and two hypotheses explained
that equally well: too few incidents sampled, or a policy too self-consistent to disagree
with itself. They have opposite fixes — sample more, or add entropy — and the pair count
alone cannot distinguish them.

Rather than pick one, the miner was instrumented to record *why* each incident failed to
yield a pair. One run settled it:

```
incidents_considered        120
incidents_with_pair          88     73% pair rate
mean_margin                2.609    high-information contrasts
rejected_no_reward_spread     1     only ONE incident had zero variance
rejected_margin_too_small    30
```

The consistency hypothesis was **disproved**: the policy varies plenty, and the pairs it
does produce are strongly separated. The shortage was purely a sampling-volume problem —
and the reason volume was low was a defect, not a budget: the sampling phase ran with
`--shard-total 1` while the evaluation had been sharded eight ways weeks earlier. It never
errored; it was just slow, which is the easiest symptom to misattribute to the model.

Sharding the sampler took 480 episodes to **2,400**, and the pair count from 88 to **435**
— within 1% of the 440 projected from the measured 73% rate, which is itself a check that
the diagnosis was right.

The generalisable lesson: **when one observable is consistent with two opposite diagnoses,
the correct move is not to choose, but to make the system report one more fact.** Four
counters settled in a single run what would otherwise have been an expensive guess.

### The pair-count ablation, and a second near-miss

Because both DPO runs used the same method, starting adapter and evaluation, the pair
count is the only variable — a clean ablation. Resolution rate, clean tools, `n = 64` per
cell:

| variant | pairs | in-dist. | unseen fault | unseen topology | compositional | all splits |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | — | 53.1% | 29.7% | 32.8% | 46.9% | 40.6% |
| SFT + DPO | 88 | 53.1% | 29.7% | **56.2%** | **50.0%** | 47.3% |
| SFT + DPO | 435 | **64.1%** | **31.2%** | 51.6% | 48.4% | **48.8%** |

**5× the preference data bought +11.0 pp in-distribution and nothing out of it** — +1.5 pp
overall, with two held-out splits slightly *down*. The extra pairs sharpened the policy on
the distribution they were mined from. They did not buy generalisation.

This nearly went into the record backwards. The evaluation writes one cell at a time, and
the first to land was in-distribution, showing +11.0 pp resolution, +10.9 pp root cause and
0.0% invalid calls. That is a decisive-looking win, it confirmed the hypothesis that
motivated the work, and it was written up as one before the remaining fifteen cells
finished. The full sweep reduced it to +1.5 pp. **The third inverted conclusion in this
project, and the first caused by reading a partial result rather than the wrong slice of a
complete one** — which is why `serve_and_eval.sbatch` now fails the job on any empty cell,
and why no number enters a table before its sweep is complete.

The schema-shift view carries a related trap. Retention is normalised to each variant's
*own* clean baseline, and the 435-pair run's baseline rose 11 pp — so its retention falls
even on suites where it resolves strictly more incidents. In absolute terms:

| suite | DPO-88 | DPO-435 | Δ |
| --- | --- | --- | --- |
| clean | 53.1% | 64.1% | +10.9 |
| rename | 57.8% | 60.9% | +3.1 |
| arg_rename | 62.5% | 65.6% | +3.1 |
| distractors | 68.8% | 70.3% | +1.6 |
| server_dropout | 32.8% | 35.9% | +3.1 |
| combined | 48.4% | 54.7% | +6.3 |
| **extra_required_arg** | **46.9%** | **34.4%** | **−12.5** |

Six up, one down — and the regression is on the suite that adds a newly *required*
argument. That is the failure mode a policy sharpened on 435 examples of the argument
conventions it already knew would be expected to have, and the retention table hides it
behind a moved denominator. **A normalised metric is the right default for comparing
variants with different baselines, and the wrong one for deciding whether a variant
regressed.** Both views are now reported.

## 6d. The unseen-fault regression is label-prior collapse

Every post-trained variant scores below the base model on held-out fault families — 29.7%
(SFT), 29.7% (DPO-88), 31.2% (DPO-435) against **48.4%** for base. This is the only place
in the project where post-training makes the agent *worse*, so it is worth knowing exactly
what broke rather than filing it under "catastrophic forgetting" and moving on.

The first extra fact reframes it. On the same episodes:

| variant | resolution | **root cause** | mean tool calls | **mean reward** |
| --- | --- | --- | --- | --- |
| base | 48.4% | 14.1% | 13.3 | **−0.11** |
| SFT | 29.7% | 0.0% | 11.0 | +0.63 |
| DPO-435 | 31.2% | 1.6% | 11.6 | +0.65 |

Base resolves more unfamiliar incidents while earning a *negative* reward and identifying
the cause 14% of the time: it flails through 13.3 calls, accumulates invalid ones, and
sometimes restarts its way into a fix it cannot explain. That is worth knowing, but it does
not explain the post-trained models' **0.0%**.

The second fact does. Counting every diagnosis the models ever submitted, across all four
splits:

```
                emissions of the two held-out labels
base            25 / 501 submissions   5.0%
SFT              0 / 599 submissions   0.0%
DPO-88           0 / 626 submissions   0.0%
DPO-435          1 / 616 submissions   0.2%
```

The post-trained policies **never** emit `network_policy_blocks_dependency` or
`node_unhealthy_reduces_capacity` — not once in ~600 opportunities. Both labels are in the
`submit_diagnosis` enum and in the system prompt on every single episode; the models are
not unable to see them, they are unable to reach them. What they emit instead is
confidently wrong and drawn from the training catalogue —
`readiness_probe_targets_wrong_endpoint` for 17 of 32 `NodeNotReady` episodes.

So the regression is not degraded reasoning. It is **label-prior collapse**: fine-tuning on
a closed catalogue drives P(label) to zero for classes with no demonstrations, and the
argmax can no longer select them at any evidence strength. The distinction matters because
the two diagnoses have different fixes — degraded reasoning needs better data, a collapsed
prior needs the update attenuated or regularised.

### Attenuating the update, without retraining

A LoRA update enters each layer as `delta_W = (lora_alpha / r) * B @ A`, so multiplying
`lora_B` by `s` scales the entire update by exactly `s`. `s = 0` is the base model, `s = 1`
the trained one, and intermediate values interpolate — no retraining, and every scale can
be served at once by the multi-LoRA path already built for the serving benchmark.

`scripts/scale_adapter.py` produces the scaled copies (verified exact: `lora_B` scaled,
`lora_A` untouched, `|B|` sums in a 1:2:3 ratio across s = 0.25/0.5/0.75), and
`slurm/adapter_scale_sweep.sbatch` evaluates s ∈ {0, 0.25, 0.5, 0.75, 1} on the
in-distribution and held-out-fault splits. One knob, one engine load, and it answers
directly whether the in-distribution gain and the held-out regression are separable or are
two views of the same weight update.

### Result: halving the update recovers the regression at no measurable in-distribution cost

Ten cells, `n = 48` each, clean tools, `python scripts/scale_report.py`:

| s | in-dist resolved | in-dist root | unseen-fault resolved | unseen-fault root | invalid calls (in-dist / unseen) | held-out labels emitted |
|---|---|---|---|---|---|---|
| 0.00 (base) | 27.1% | 45.8% | 54.2% | 18.8% | 27.0% / 31.8% | 10/42, 5/46 |
| 0.25 | 39.6% | 52.1% | 39.6% | 10.4% | 19.8% / 25.1% | 2/46, 5/46 |
| **0.50** | **66.7%** | **77.1%** | **62.5%** | **27.1%** | 9.1% / 5.3% | **0/41, 13/24** |
| 0.75 | 47.9% | 68.8% | 43.8% | 10.4% | 5.1% / 2.2% | 0/47, 5/43 |
| 1.00 (trained) | 62.5% | 66.7% | 27.1% | 2.1% | 0.0% / 0.0% | 0/48, 1/46 |

**s = 0.5 dominates the unattenuated policy.** Against s = 1.00 it gains **+35.4 pp
unseen-fault resolution** (z = 3.73) and **+25.0 pp unseen-fault root cause** (z = 3.71),
while in-distribution resolution differs by +4.2 pp — statistically indistinguishable
(z = 0.43). Halving the LoRA update recovers most of the held-out-fault regression and
costs nothing measurable on the distribution the policy was trained for.

The mechanism moves with it, which is what makes this a fix rather than a coincidence. On
held-out incidents the trained policy emits a correct held-out label **once in 46
submissions**; at s = 0.5 it emits **13 in 24**, and those 13 are exactly its 13 correct
root causes. The collapsed prior is recoverable by attenuation because the labels were
never removed from the model — only pushed below every competitor by fine-tuning on a
closed catalogue.

**What the n = 48 sweep did not establish.** The frontier is *not* monotone: s = 0.75 sits
below both neighbours on both splits. Neither dip clears significance at this size
(z = 1.87 and 1.90 against s = 0.5), so that sweep resolves the direction but not the
optimum. Rather than gloss it, a second sweep was run at the largest `n` the splits allow.

### Confirmation at n = 128: a dose-response, and one prediction that failed

`s ∈ {0.5, 0.75, 1.0}`, `n = 128` in-distribution and `n = 120` on held-out faults (the
split ceiling), `runs/eval-scale-n128`:

| s | in-dist resolved | unseen-fault resolved | unseen-fault root | held-out labels emitted (unseen) |
|---|---|---|---|---|
| 0.50 | 67.2% | **72.5%** | **20.0%** | **24/56 (42.9%)** |
| 0.75 | 57.0% | 50.8% | 6.7% | 8/112 (7.1%) |
| 1.00 | 65.6% | 36.7% | 0.0% | **0/114 (0.0%)** |

**On held-out faults the relationship is monotone and every step is significant**:
72.5% → 50.8% → 36.7%, with z = 3.55 between the first pair, z = 2.22 between the second,
and z = 5.97 end-to-end. Root-cause accuracy and held-out label emission move in lockstep
and monotonically — and at s = 1.00 the trained policy emits a correct held-out label
**zero times in 114 submissions**.

A dose-response across three levels is the strongest evidence available here that the
effect is caused by update magnitude rather than by anything incidental to one checkpoint.
And the in-distribution cost of the best setting remains unmeasurable: s = 0.5 against
s = 1.0 is +1.6 pp, z = 0.27.

**The prediction that failed.** The confirmation was justified by a power calculation
saying n = 128 would resolve the s = 0.75 in-distribution dip at z ≈ 3.1. It did not. The
dip replicated in *direction* but roughly halved in size — 18.8 pp at n = 48, 10.2 pp at
n = 128 — and now sits at z = 1.69, still short of significance. Textbook regression to the
mean: the calculation assumed the n = 48 effect size was the true one, which is exactly the
caveat recorded when the job was submitted, and it is the caveat that came true. **The
in-distribution non-monotonicity remains unresolved, and the honest reading is that the
first sweep overestimated it.**

So the claim the project makes is the one the data supports: attenuation monotonically buys
held-out-fault performance at no measurable in-distribution cost. It does not claim to have
located an optimum, and the wobble at s = 0.75 in-distribution is reported as unexplained
rather than smoothed away.

**Two internal validity checks, both passed.** Invalid-call rate is cleanly monotone in s —
27.0 → 19.8 → 9.1 → 5.1 → 0.0% in-distribution, 31.8 → 25.1 → 5.3 → 2.2 → 0.0% on held-out
faults. That is the quantity the adapter most directly controls, and it scales smoothly
exactly as an exact linear interpolation of the update predicts. Separately, the s = 1.00
arm reproduces the published DPO-435 row (62.5% vs 64.1% in-distribution, 27.1% vs 31.2%
unseen fault, both within noise) and the s = 0.00 arm reproduces base — so the scaling
machinery and the serving path are the same ones that produced the tables above.

**The held-out fault axis is narrow, and that bounds every claim made on it.** The
`unseen_fault` split contains **120 incidents across exactly two fault families**
(`NetworkPolicyDeny`, `NodeNotReady`). 120 is a hard ceiling, discovered by asking for 128
and receiving 120 with all eight shards reporting evenly and no failures — so no
unseen-fault comparison anywhere in this document can be strengthened by sampling more.
Widening it means generating incidents for more held-out families, which changes the split
definition and invalidates comparison against every number already published here. Stated
because "held out" sounds broader than two families are.

**Contamination, reported rather than absorbed.** Three cells logged control-plane settle
timeouts: one each in the two s = 0.00 cells and s = 0.25/in-distribution, and **five of 48
in s = 0.25/unseen-fault** (~10%). Those episodes had their verdicts taken against a world
that had not finished converging. The counter that reveals this was added hours before this
sweep ran; every earlier table in this document was computed without it, and reads 0 only
because the field did not exist.

## 6e. Episode-level GRPO

`grpo_train.py` optimises one *action*: TRL's GRPOTrainer samples a group of completions
for a single prompt, so the prompt is the conversation up to some step and the completion
is the next tool call. The reward is genuinely verifiable, but it cannot express what makes
agent RL hard — an episode is twenty-odd calls whose value is known only at the end.

`src/agentgym/train/episode_grpo.py` closes that. The group is *G* full rollouts of the
same incident, the reward is the episode's verifiable reward — the same number §5 reports —
and the advantage is group-relative:

```
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

which is GRPO's defining move: the group mean replaces a learned value function, so there
is no critic to train. That advantage multiplies every assistant token in episode *i*,
making the credit assignment episode-level rather than per-action.

**It extracts far more signal from the same rollouts than DPO does.** Run over the 2,400
episodes already sampled for DPO:

```
trajectories               2400
groups_seen                 600
groups_kept                 597      99.5%
dropped_no_reward_spread      3
episodes_kept              2388
mean_group_std             0.856
mean_abs_advantage         0.896
reward_range        [-1.10, 3.05]
```

**435 DPO pairs versus 2,388 GRPO episodes across 597 groups from identical data** — 5.5×
the training signal, because a preference pair keeps only a best/worst couple that clears a
margin, while a group uses the entire reward distribution.

Two implementation details are load-bearing and both are pinned by tests:

* **Length normalisation.** Dividing by the assistant-token count stops the objective
  preferring long episodes whenever the advantage is positive — a bias that surfaces as the
  agent padding its reasoning to farm reward. `test_loss_is_length_normalised` asserts that
  doubling episode length at equal per-token likelihood leaves the loss unchanged.
* **Degenerate groups are dropped, not zeroed.** When every rollout of an incident earns
  the same reward the advantage is 0/0; keeping those as zeros silently shrinks the
  effective batch while the step count claims otherwise — the same fail-open shape as the
  sampling bug above. They are removed and counted (3 of 600 here).

One bug was caught before it ever ran: the policy forward was written inside
`model.disable_adapter()`, which would have trained the adapter on gradients computed
without it. It raises nothing and still produces a model — so
`test_policy_forward_runs_with_adapter_enabled` now pins it.

15 tests cover the advantage algebra (group-scoped, zero-mean, unit-scale, ordering
preserved), the loss direction under positive and negative advantage, length invariance,
KL non-negativity and finiteness, group adjacency, and mask correctness.
`slurm/train_episode_grpo.sbatch` runs the full sample-then-train loop; passing
`SAMPLES_FILE` reuses existing rollouts, since the DPO sampler already drew four attempts
per incident from this same policy at temperature.

### Three defects between "tests pass" and "it trains"

All 15 tests passed before the first run, and the first three attempts still failed. Each
failure is a different way for a unit test to be true and useless.

**1. The tests had a 8-token vocabulary.** The loss computed
`log_softmax(logits.float())`, materialising a `[B, T, V]` float32 tensor. At Qwen3's
151,669-token vocabulary and a 12k context that is ~7.5 GB for the forward and as much
again retained for the backward — an OOM on a 44 GB L40S at batch size 1. The tests
exercise the same code path with `V = 8`, where the tensor is 200 bytes. Switching to
`cross_entropy(reduction="none")`, which is fused and recomputes softmax in the backward
rather than storing it, fixed it; **all 15 tests passed unchanged, which is the evidence
the rewrite is equivalent** and also the evidence they could never have caught it.

**2. The KL reference was the wrong policy — and the docstring said so.** The reference
was `model.disable_adapter()`, which yields the *base* model, while the policy starts from
the SFT adapter. So KL(SFT ‖ base) was already enormous before any update:

```
step 5    pg = -0.8598    kl = 32472.04    loss = 1298.02
```

At β = 0.04 the penalty exceeded the policy gradient by three orders of magnitude: the run
was optimising *return to base*, not *improve the policy*. Nothing errored — the loss fell
steadily. The module docstring claimed the KL was "measured against the policy being
improved", which is precisely what disabling the adapter does not do; the prose and the
code disagreed and only the prose was right. The SFT adapter is now loaded twice —
`default` trains, `reference` is frozen — and `grpo_loss` takes a `ref_context` that
switches between them. After the fix, `step 5: pg = -0.8594, kl = 0.0108`. The new test
asserts the *relationship* the bug violated: `loss == pg_loss` at step 0.

The old KL test passed throughout, because it handed the stub identical logits for policy
and reference. It tested the estimator's algebra, not whether the right policy was passed
to it.

**3. Truncation was silent.** `_collate` truncated at `max_length` without a word. A cut
episode keeps its full group-relative advantage but loses its later turns — frequently the
one that resolved the incident — so the gradient attributes +A to a prefix that did not
earn it. Truncated and zero-assistant-token examples are now counted into
`train_log.json`, with a warning above 25%. `max_length` was then chosen by measurement
rather than habit: over 250 sampled episodes the median is 6,282 tokens, p95 8,225, max
13,064, so 12,288 truncates 1.2% where 8,192 would truncate 5.6%.

A fourth was caught by inspection rather than failure: the training loop does not shuffle,
which is load-bearing. Advantages are zero-mean *within* a group, so group members staying
adjacent is what lets each accumulation window span whole groups and cancel the baseline
inside it — the variance reduction GRPO exists to buy. The ordering was correct but
accidental; `test_group_members_stay_adjacent` now pins it.

### Episode-level GRPO: a negative capability result, and convergent evidence for the mechanism

All four clean-split cells complete (`n = 64` each; the six perturbation cells are still
running and no robustness number is claimed yet).

| variant | in-dist | unseen fault | unseen topo | compositional | **all** | invalid calls | reward |
|---|---|---|---|---|---|---|---|
| base | 34.4% | 48.4% | 29.7% | 18.8% | 32.8% | 26.3% | 0.10 |
| SFT | 53.1% | 29.7% | 32.8% | 46.9% | 40.6% | 0.6% | 1.08 |
| SFT+DPO-435 | 64.1% | 31.2% | 51.6% | 48.4% | **48.8%** | 0.2% | **1.32** |
| **SFT+episode-GRPO** | 45.3% | 40.6% | 23.4% | 32.8% | **35.5%** | **15.2%** | 0.39 |

**As a capability method at these settings, it did not work.** 35.5% overall against SFT's
40.6% and DPO's 48.8%, with mean tool calls up from 10.2 to 13.0 and reward down from 1.32
to 0.39. The clearest damage is to tool-call validity: **0.2% → 15.2% invalid calls**,
three-quarters of the way back to the untrained base model. Advantage-weighted likelihood
pushes *down* on low-reward episodes, and those episodes are full of well-formed tool calls
that simply did not resolve the incident — so the format SFT taught is collateral.

This was checked for defects before being reported as a result. The saved adapter differs
from its SFT initialisation on every sampled tensor (mean relative change 0.54%, max 1.02%),
so training genuinely moved the policy; the KL of 0.011 → 0.022 was measured on the
*training* rollouts and bounds nothing on unseen evaluation prompts.

**But it independently reversed the label-prior collapse.** Emissions of the two
never-demonstrated diagnosis labels, across all splits:

```
base            16 / 225 submissions   7.1%
SFT              0 / 244                0.0%
DPO-88           0 / 254                0.0%
DPO-435          1 / 250                0.4%
episode-GRPO     9 / 165                5.5%
```

and on held-out faults its root-cause accuracy is **14.1% — exactly base's 14.1%**, against
0.0% for SFT and 1.6% for DPO-435.

This matters more than the capability number. Two interventions with nothing in common —
**scaling the LoRA update in weight space**, and **on-policy RL against a verifiable
reward** — both restore a prior that supervised and preference training had driven to zero.
That is convergent evidence that the collapse is caused by *optimising hard toward a closed
catalogue*, rather than by anything specific to one objective, one checkpoint, or one
weight-space direction. Either result alone would be suggestive; together they identify the
cause.

Both also pay for it, in the same currency. Attenuation trades in-distribution sharpness;
GRPO trades tool-call validity. Nothing here recovers held-out coverage for free.

**Not claimed.** Episode-GRPO's unseen-fault edge over DPO-435 (40.6% vs 31.2%) does *not*
reach significance at `n = 64` (z = 1.11). One run at one learning rate, one KL coefficient,
one epoch, on rollouts from the SFT policy — this is evidence that the method as configured
degrades capability, not that episode-level GRPO cannot work. The obvious next moves are a
smaller step size and a KL reference that constrains the *evaluation* distribution rather
than the training one.

### The attenuation effect replicates across adapter families

The dose-response was established on DPO-435. Whether it was a property of *closed-catalogue
fine-tuning* or of that one checkpoint was the open question — and the honest answer mattered,
because the convergent-evidence argument that would have settled it turned out to rest on a
wrong-base artifact.

Scaling the SFT adapter instead — a different objective, different data, independently
trained — reproduces it almost exactly. Both at the held-out split's 120-incident ceiling,
same harness, each with its own `s = 1.0` reference arm:

| | DPO-435 | SFT-v4 |
|---|---|---|
| unseen-fault resolved, s = 1.0 | 36.7% | 35.8% |
| unseen-fault resolved, s = 0.5 | **72.5%** | **70.0%** |
| gain | **+35.8 pp** (z = 5.97) | **+34.2 pp** (z = 5.64) |
| unseen-fault root cause, s = 1.0 → 0.5 | 0.0% → 20.0% | 0.8% → 18.3% (z = 4.82) |
| held-out labels emitted, s = 1.0 | 0/114 | 1/116 |
| held-out labels emitted, s = 0.5 | 24/56 | 22/55 |
| in-distribution cost | +1.6 pp (z = 0.27) | −3.1 pp (z = −0.51) |

Two adapters trained by different methods, collapsed to the same near-zero label prior, and
recovered by the same weight-space intervention to within 1.6 pp of each other on every
measured quantity. Both gains are roughly **3× the ~11 pp measurement floor** established by
the DPO replication, which is why they are reportable where the pair-count ablation was not.

**This is what the withdrawn convergent-evidence claim should have been.** That argument
paired attenuation with episode-level GRPO and was wrong, because the GRPO run had trained
from a stale base; the corrected runs emit 0/253 and 0/248 held-out labels and recover
nothing. The replacement is not two mechanisms agreeing but one mechanism reproducing on two
independently-trained policies — weaker as a causal story, far stronger as evidence, and it
does not depend on a result that has since been retracted.

The mechanism claim now stands as: **fine-tuning hard on a closed catalogue drives
never-demonstrated labels below every competitor, and attenuating the update recovers them,
monotonically, on every adapter tested.**

### What the replication settled: three corrections, two of them to published claims

The DPO replication and the two corrected episode-GRPO runs landed together. All four clean
splits complete, `n = 64` per cell.

**1. The pair-count ablation does not replicate — retracted.** The published +11.0 pp
in-distribution gain from 5× the preference data is run-to-run variance. A second run at 437
pairs scores **53.1%**, identical to the 88-pair run and 11 pp below the 435-pair run it
replicates. The original difference was never significance-tested; it is z = 1.27. Between 88
and ~435 pairs there is no detectable capability effect. DPO over SFT survives — all three
runs beat SFT on unseen topologies (56.2 / 51.6 / 50.0 against 32.8) — but the pair-count
conclusion does not.

**2. The measurement floor is ~11 pp, and that governs the whole document.** Two runs of an
identical recipe differ by 11 pp in-distribution and 2.7 pp overall. At `n = 64` per cell,
*no single-run difference below roughly 15 pp is interpretable*. This is the most useful
thing the replication bought, and it should have been established before any ablation was
reported rather than after. The frontier dose-response clears it — 35.8 pp, monotone across
three levels, z = 5.97 — which is now a load-bearing distinction rather than a stylistic one.

**3. Episode-GRPO does not damage capability; the earlier finding was the wrong base.** From
`sft-qwen3-8b-v4` it reaches **44.9%** against SFT's 40.6%, inside the range two DPO runs
span, with **0.2% invalid calls**. The withdrawn run reported 35.5% and a 0.2% → 15.2%
invalid-call regression, explained at the time by advantage-weighted likelihood eroding
SFT's format. That explanation was mechanistically plausible and completely wrong: it
described an adapter initialised two versions behind the rollouts it trained on.

**And the convergent-evidence claim is withdrawn.** The wrong-base run appeared to
independently restore the collapsed label prior (9/165 emissions, against base's 7.1%); the
corrected runs emit **0/253** and **0/248**. Episode-GRPO does not recover the prior. The
label-prior mechanism rests on attenuation alone.

That last one is worth sitting with. The convergent argument was the strongest claim in this
document — two interventions sharing no mechanism, both restoring a prior that training had
zeroed, therefore the cause is identified. It was built on an artifact, and the artifact was
persuasive precisely because it agreed with a result that was independently true.

### The data pipeline replicates to within 0.5%

Rerunning DPO end-to-end at matched settings — 600 incidents, 4 samples each, temperature
0.9, base `sft-qwen3-8b-v4`, an independent sampling pass — reproduces the published
pipeline almost exactly:

| | published | replication |
|---|---|---|
| incidents sampled | 600 | 600 |
| episodes rolled out | 2,400 | 2,400 |
| **preference pairs mined** | **435** | **437** |
| mean reward margin | 2.609 | 2.594 |

Two pairs apart on 437, and a margin difference of 0.015. That is not a claim about DPO's
*effect* — the trained adapters still have to be evaluated — but it does establish that the
sampling and mining stages are stable across independent runs, so a difference in downstream
capability between two DPO runs cannot be attributed to the data pipeline drifting under
them. The variance estimate on the +8.2 pp headline is the next thing this answers.

### The lineage digest paid for itself within the hour

The instrumentation added after the episode-GRPO withdrawal — recording a hash of the base
adapter's *weights* into every training output — caught its second case immediately.

A second DPO run had been submitted to put a variance estimate on the headline +8.2 pp,
before the sbatch default was corrected. Its saved reference hashes to `2c9d616c7c6eff12`,
which is `sft-qwen3-8b-v2`, not the `31388a5ef825d248` that `dpo-qwen3-8b-v5` and
`sft-qwen3-8b-v4` share. Wrong base, same defect as before, caught in one line rather than
by comparing file timestamps and sampled tensors.

It was wrong a second way, which the digest could not catch and arithmetic could: it sampled
600 episodes and mined **86 pairs**, where the published run sampled 2,400 and mined **435**.
The cause was `LIMIT=150` against the published run's 600 incidents. A "replication" at a
quarter of the sampling budget is confounded with pair count — the *one* variable the
ablation established as dominant — so it would have answered a question nobody asked while
looking like an answer to the one that was. `LIMIT` now defaults to the published 600 with
the reason recorded beside it.

Both invalid runs are quarantined rather than deleted, and 5547595 replicates at matched
settings.

### Withdrawn runs are quarantined, not deleted

`eval.report` reads `runs/eval/` recursively, so the withdrawn episode-GRPO trajectories
rendered as an ordinary capability row — same shape, same formatting, nothing marking them
as retracted. Regenerating the tables would have republished them. That is the same failure
mode as the stale-file report earlier in this log, and it is why the retraction needed a
filesystem action and not just a paragraph.

Withdrawn runs move to `runs/eval-withdrawn/<name>/` with a README recording what they
measured, why they are invalid, and what supersedes them. Deleting them would make the
retraction unverifiable; leaving them in place would make it ineffective.

## 7. Open items

Stated as open rather than quietly omitted.

- **Episode-level GRPO is trained; its evaluation is in flight.** 149 optimizer steps over
  2,388 episodes in 597 groups, KL to the frozen SFT reference drifting 0.011 → 0.022,
  truncation 0.88% against 1.2% predicted. No capability number from it is claimed until
  the ten-cell evaluation completes.
- **Action-level GRPO** (`grpo_train.py`) is implemented but its trained policy has not
  been evaluated in the tables above.
- **Node-layer sim-to-real fidelity.** `src/agentgym/fidelity.py` validates the process
  layer against real processes, real `RLIMIT_AS` kernel enforcement and real HTTP/TCP
  (5/5 agreement on readiness *and* reason). The container/kubelet layer has no real-runtime
  comparison, because no container runtime is available on this cluster — so no fidelity
  claim is made for it.
- **AWS.** S3 code paths are tested against `moto`, and the EKS overlay is validated with
  `helm lint`/`template`. Live EKS/IAM/network verification is pending
  `scripts/verify_aws.py` being run where credentials exist.

---

## Engineering notes worth keeping

Seven problems that cost real time and had non-obvious answers.

**CORRECTED — two harnesses that never disagreed; I compared two different adapters.** Testing whether the
attenuation finding generalises to the SFT adapter produced a contradiction: the arm serving
the *published* SFT adapter did not reproduce the *published* SFT row. On the same 64
incidents, same adapter, temperature 0.0 — root cause **0.0% vs 20.3%**, invalid calls
**0.1% vs 16.0%**.

**The first diagnosis was wrong.** The harnesses differ in step budget — 20 published
against 24 in the sweep — and the sweep's failures were hallucinated tools
(`get_statefulset`, `create_configmap`, `get_namespace`, none in the 36-tool registry),
which fits a policy running past the end of its ideas. Rerunning the sweep at
`--max-steps 20` should then have reconciled them.

It did not. At 20 the sweep still gave 45.3% resolved / 23.4% root cause / 18.3% invalid,
essentially unchanged from 24 and still nothing like the published 29.7% / 0.0% / 0.1%. The
step budget explained none of it.

**The actual cause: there are four SFT adapters, and the two runs served different ones.**
`artifacts/` holds `sft-qwen3-8b`, `-v2`, `-v3` and `-v4`. The published `sft` trajectories
were written at 01:05 on 2026-07-28; `-v4` was written at 00:51 the same morning. The sweep
served `-v2`, which `train_dpo.sbatch` and `train_episode_grpo.sbatch` both take as their
default base. `v2` and `v4` are not near each other — sampled `lora_B` tensors differ by
**121% relative**, none identical.

So the two runs were never measuring the same weights, and the "harnesses disagree"
conclusion was an artifact of my own comparison. The system prompt hash is identical across
both (`591a3ab98e19`), which had already ruled out the prompt-contract class of bug and
should have pointed here sooner.

Two things follow that are worth more than the retraction. **The step-budget change is kept
anyway**: matching the published harness is correct regardless of what explained this, and a
sweep that is not comparable to the published tables is a trap left armed. And **a real
lineage question is now open** — if the published `sft` row is `v4` while DPO and
episode-GRPO were trained from `v2`, then every "DPO over SFT" comparison in this document
spans two different base adapters. Job 5547208 serves both against the same incidents to
settle which produced the published row; nothing downstream is claimed until it lands.

**The frontier dose-response survives this, and not by luck of the numbers.** Every arm of
that sweep shared one harness, and the report script was written from the start to compare
`s` against `s` internally, refusing cross-harness comparison — a rule adopted to prevent a
*population* confound from differing `--limit`. It happened to also block a confound nobody
anticipated. The `s = 1.0` arm's agreement with the published DPO-435 row now reads as
fortunate rather than guaranteed.

**The SFT generalisation test does not survive**, and is withheld rather than reported: a
remedy cannot be evaluated against a baseline that is not the published baseline. `MAX_STEPS`
is now a variable defaulting to 20, and a second defect surfaced alongside — the scaled-GRPO
sweep had been submitted without an `s = 1.0` reference arm, on the assumption the published
row could stand in for one. Both were resubmitted.

**The serial-rollout bug, written twice.** The adapter-scale sweep was submitted with an
unsharded rollout loop and measured 3 episodes in 18 minutes — about 64 hours for the full
sweep, against roughly 4 once sharded eight ways. This is exactly the defect described
above as the DPO sampler's `--shard-total 1`, in the same repository that documents it,
committed by the same author who wrote that section. It is worth keeping rather than
quietly fixing, because the reason it recurs is structural: **a missing shard flag produces
no error, no warning and no wrong number — only slowness**, and slowness reads as a
property of the model or the cluster rather than as a bug in the harness. Correctness bugs
get caught by assertions; throughput bugs need someone to hold a stopwatch against an
expectation. The sweep script now shards by default and prints the episode count per cell,
so a collapsed batch shows up as a number rather than as patience.

**A test that measured machine load.** `test_fault_breaks_then_fixes` applied each fault's
known-correct fix and then advanced a fixed 24 ticks before asserting resolution. It passed
alone and in its own file, and failed roughly once per full-suite run — on a *different*
fault each time. The cause is the hybrid design itself: ticks are deterministic, but pods
are created asynchronously by the real controller-manager, so under load fewer pods exist
per tick and the fixed budget expires mid-convergence. The failing verdicts showed it
plainly — `redis-0` still `Booting`, everything downstream `DependencyUnavailable`, and a
rollout with two live ReplicaSets. The fix was to settle on the same rule the reward uses
(`settle_world()`: not converged while anything is transient) instead of a tick count.
Same class as the rest of this document — the assertion was true, it was just answering
"is the machine fast enough" rather than "does the fix work".

**…and the fix was right but insufficient.** Two clean full-suite runs later it flaked
again, on `MissingSecret`. The actual root cause was one level down: `Episode._settle()`
polls until the real controllers have produced their intended pods, with a 20-second
deadline — and on expiry the `while` loop simply exited and the function returned. Every
tick's settle could therefore give up silently underneath the new convergence rule, letting
the world advance with pods the controller-manager had not created. The incident then reads
unresolved, and **at the point where the reward is computed that is indistinguishable from
a genuine failure.**

This is the most consequential instance of the pattern in the repository, because it is not
confined to a test: every rollout calls `step()`, every `step()` calls `_settle()`, and any
episode affected would contribute a wrong reward to training data or an evaluation table
with nothing anywhere recording it. `_settle` now returns whether it settled and counts
expiries; the count travels into the `Trajectory` and into the evaluation report, where a
non-zero value means that table describes the machine as much as the agent. The environment
test skips with an explicit message when it hits one, rather than asserting a fix failure
it cannot substantiate.

Trajectories recorded before this field existed read 0. That is the absence of a record,
not evidence that no timeouts occurred, and it is stated that way in the code and in the
commit rather than quietly enjoyed.

**No container runtime.** PACE ICE has no root, no Docker, and no `/etc/subuid` range, so
rootless Podman cannot map multiple UIDs and neither `kind` nor `k3s` will start. Running
`etcd` + `kube-apiserver` + `kube-controller-manager` + `kube-scheduler` directly as user
processes sidesteps the runtime entirely and keeps the API semantics that matter. Two
flags were needed that are easy to miss: the scheduler and controller-manager exit or
log-spam without `--authentication-kubeconfig`/`--authorization-kubeconfig`, and the
requestheader CA flags are required or both components error several times a second about
a missing CA bundle.

**CUDA version skew.** The driver is 575.57 (CUDA 12.9). The default `vllm` pulled
torch 2.11+**cu130**, which silently reports `cuda.is_available() == False` and surfaces
as a confusing "your setup doesn't support bf16/gpu" from the Trainer. Pinning
torch 2.8.0+cu128 fixes it — and then a stale torchvision built against 2.11 breaks
`transformers` imports with `operator torchvision::nms does not exist`.

**The tool catalog never reached the model.** The most dangerous bug in the project, and
it produced no error at all. Tool schemas were passed as the OpenAI `tools` API field,
but the project ships its own chat template — a plain message renderer that ignores the
`tools` variable. So the schemas were silently dropped before the prompt was built, and
the model was being asked to call tools it had never been shown.

Nothing crashed. A fine-tuned model would still have scored reasonably, because it
memorised tool names from demonstrations — which is precisely what makes this dangerous:
the failure hides behind a plausible number. Two experiments would have been quietly
meaningless:

- the base-model comparison, since an untrained model cannot call tools it cannot see, and
- **the entire schema-shift suite**, since a renamed tool never appeared in the prompt,
  so there was nothing for the model to adapt to.

Caught by rendering the template with `tools=` populated and grepping for a tool name:

```
>>> does 'list_pods' appear in the prompt? False
```

The fix moves the catalog into the system prompt for both training and serving, so a
perturbed schema is visibly perturbed *in the prompt*. Expert results are unaffected — the
scripted policy resolves names through the toolset object, not the prompt — so §5 stands;
the SFT run was redone.

The lesson generalises: an instrumentation path that fails open is worse than one that
fails loudly, and prompt construction deserves an assertion, not an assumption.

**The parser discarded valid tool calls and blamed the model.** The first base-model
cells came back at 0/2 resolved with 37 of 48 calls invalid. The number pointed the way
everyone expects an untuned model to point, which is exactly why it was worth checking.
The raw completions showed the model emitting perfectly good calls:

```
</think>

tool
{ "name": "get_recent_changes", "arguments": { "namespace": "inc-00002" } }
```

The bare-object fallback was `\{[^{}]*"(?:name|tool)"\s*:.*?\}`. The lazy `.*?}` stops at
the **first** closing brace — the one ending `arguments` — so the extracted text is
missing its final brace and fails to parse. Every call with a nested argument object was
dropped, which is nearly every call.

Regex cannot do this: lazy truncates nested objects, greedy swallows everything between
the first `{` and the last `}`. Replaced with a brace-balanced, string-aware scan, plus
support for `<tool_call>` tags and for objects appearing after `</think>`. Of the 19
"unparseable" completions, **10 were recovered as real tool calls**.

The other 9 were genuine model failures, and worth naming because they are distinct
behaviours: narrating an intention without acting, emitting what a tool *would have
returned* instead of calling it, and stopping mid-`<think>`. Two tests guard against
over-correcting — prose is not an action, and a hallucinated result object (no `name`
key) must be rejected rather than dispatched.

**The benchmark measured a request shape production never sends.** vLLM rejected the
first benchmark run: `"auto" tool choice requires --enable-auto-tool-choice`. The
benchmark was still passing `tools` as an API field while the harness had moved the
catalog into the system prompt and stopped sending one. A benchmark whose requests differ
from the system it is meant to characterise is worse than no benchmark.

**The robustness table divided by the wrong denominator.** Retention came out at 1.46 for
SFT under tool renaming — the model apparently solving *more* incidents with renamed tools
than with correct ones. Physically impossible, and the table looked entirely normal.

The `clean` baseline averaged over all four splits, including the much harder OOD ones,
while the perturbation suites ran only on in-distribution incidents. Numerator and
denominator were different populations. Scoping both to one split fixed it, and the real
numbers are more interesting than the fake ones: SFT is *more* rename-robust than base
(1.12 vs 0.82) but markedly more brittle under server dropout (0.56 vs the expert's
irreducible 0.74).

Worth noting that this bug produced a *flattering* result. Those are the hardest to catch,
because nothing about them prompts a second look. What caught it was asking what a
retention above 1.00 would have to mean.

**The harness showed the model a transcript it was never trained on.** The most expensive
bug in the project, and the one that nearly shipped as a scientific claim.

Training targets render an assistant turn as reasoning followed by a fenced JSON call, all
inside `content`. The harness instead recorded the reasoning in `content` and the call in a
separate `tool_calls` field — the shape the OpenAI schema suggests, and the tidier of the
two. But the chat template used for training *and* serving renders `content` only, so the
call was dropped from the history.

From turn two onward the model saw a transcript in which every previous assistant turn was
prose with no tool call, and continued that pattern. **Turn one was always perfect**, which
is why a single-turn probe with `peft` showed the adapter reproducing its target exactly
and sent the investigation toward training instead of serving.

Effect of the one-line fix, same adapter, no retraining:

| | before | after |
| --- | --- | --- |
| resolution | 3.1% | **29.3%** |
| invalid tool calls | 89.8% | **17.3%** |

Guarded now by a test that pins the two renderers to each other — `_assistant_turn` (used
to build inference history) must equal `_assistant_text` (used to build training targets),
or the suite fails. They are the same contract seen from opposite ends, and nothing was
enforcing that.

**A hallucinated namespace returned success.** `list_namespaced_pod` on a namespace that
does not exist returns an empty collection, not an error. A model that invented a namespace
therefore received `[]` with `ok=True`, concluded there were no pods, and had no way to
notice. The gateway now scopes every call to the episode's namespace and, on mismatch,
returns an error that names the correct one — a real responder is scoped by RBAC and would
be told "forbidden" rather than "nothing here".

**`--limit` was applied after sharding.** Eight shards times a limit of 32 is 256 episodes
per cell, not 32 — an eightfold overrun across ten cells and two variants. It raised no
error and presented purely as slowness, which is the easiest symptom in the world to
misattribute to the model.

**Two real Kubernetes behaviours, discovered by adding real objects.** Making the
datastore tier StatefulSets with PersistentVolumeClaims broke six fault families, and
both causes turned out to be genuine Kubernetes semantics rather than simulator bugs.

*StatefulSet rollouts do not self-heal.* A StatefulSet's RollingUpdate will not replace a
pod that is not Running and Ready. Once its pod is wedged in `ImagePullBackOff`, patching
the template changes `updateRevision` and nothing else — the pod keeps the old image
indefinitely, and real Kubernetes requires deleting it by hand:

```
STS template image: postgres:16-alpine      (fix applied)
pod image         : postgres:latest-broken  (pod never replaced)
updateStrategy=RollingUpdate  currentRev != updateRev
```

This deserves its own fault family. It is excluded today because it breaks the
single-action fix/verify contract the other families rely on, so faults target the
stateless tier and the stateful tier serves as the dependency they cascade through.

*A LimitRange default below an existing request rejects the pod.* Adding a LimitRange with
`default: {cpu: 500m}` silently broke every GPU service, because those request 2 CPUs and
declare no CPU *limit*. The default is not clamped to the request — admission rejects the
pod outright and the ReplicaSet creates nothing:

```
Error creating: Pod ... is invalid: spec.containers[0].resources.requests:
  Invalid value: "2": must be less than or equal to cpu limit of 500m
```

The lesson generalises: a LimitRange default bounds containers that declare nothing; set
it below what declared containers already request and it becomes an admission-time
outage. Both of these are the kind of thing that only surfaces when the environment uses
the real objects rather than approximating them.

**Irreconcilable dependency sets.** vLLM 0.26 requires transformers v5; peft 0.19 (the
newest available) requires transformers v4. They cannot share an environment, so training
and serving get separate venvs — which is the right architecture anyway. vLLM 0.11 also
pulls transformers v5 by default and dies on `Qwen2Tokenizer has no attribute
all_special_tokens_extended`, so the serving venv pins `<5` too.
