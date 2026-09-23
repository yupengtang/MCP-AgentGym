# MCP-AgentGym, experiment log

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
checks, **16 passed**.

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

**Result.** **7 passed**, schemas match exactly for all four servers; a tool error is
returned as an error result rather than killing the session, and the session stays usable
afterwards.

## 3. Operator

**Claim.** A real controller, with the properties that make controllers correct.

**Method.** `tests/test_operator.py` drives `IncidentRunController` against the real API
server: reconcile to terminal state, re-reconcile a finished run, delete and verify the
finalizer cleaned up.

**Result.** **5 passed**, CRD installs with a status subresource; a run reaches
`Succeeded` with `resolved`, `rootCauseCorrect`, `toolCalls` on `/status`; a second
reconcile pass acts on nothing (`acted == 0`, timestamps unchanged); the finalizer
removes the episode namespace before releasing the object.

## 4. Scale and cost

| Quantity | Measured |
|---|---|
| Incidents generated | 1,200 across 4 splits |
| Trajectories executed and verified | 11,541 |
|, of which back the reported tables | 5,383 |
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
a scripted policy throughout, never as a model.

| variant | n | resolution | root cause | unsafe | invalid calls | tool calls | reward |
|---|---|---|---|---|---|---|---|
| expert-scripted | 4,010 | 91.4% | 100.0% | 0.0% | 0.0% | 8.0 | 2.60 |

### 5.1 By split

| split | resolution |
|---|---|
| in-distribution | 92.9% |
| unseen fault | 97.8% |
| unseen topology | 95.0% |
| **compositional** | **50.8%** |

The expert reasons about one root cause. Given two independent simultaneous faults it
fixes one and stops, so compositional resolution lands near half. This is a designed
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
   than that, the loss is a fact about the model (it stopped reading the schema and
   started calling tools from memory), not an artifact of an impossible task.

## 6. Supervised fine-tuning

**Setup, corrected.** This block described an SFT run that no published result uses. It
said *614 demonstrations, 3 epochs*; the adapter every result actually derives from
(`sft-qwen3-8b-v4`) trained on the **1,783-example** corpus for **2.0 epochs**, its
`train_metrics.json` records `epoch: 2.0`, `train_loss 0.0918`, `eval_loss 0.0215`, and the
corpus file predates the adapter by four hours.

Current, verified: Qwen3-8B, QLoRA (4-bit NF4, LoRA r=32, α=64, all attention and MLP
projections), **1,783 curated demonstrations, 2.0 epochs**, lr 1e-4 cosine, effective batch
8, 8,192-token context, one L40S, 4 h 32 m.

The stale figures survived because they were plausible and nothing recomputed them, the
same failure as the serving numbers and the expert reference line. It matters more here:
the epoch count is not a detail, it is the finding recorded below, since a 3-epoch SFT
outperforms every variant post-trained on this 2-epoch base.

**Loss masking.** Only assistant tokens contribute. Trajectories are dominated by long
JSON tool outputs; training on those would teach the model to *generate* cluster state
instead of querying for it. Qwen3's stock template has no `{% generation %}` markers, so
the project ships its own template, and passes the identical file to vLLM via
`--chat-template`, making the training and serving prompts byte-identical.

**Data hygiene.** 1,200 trajectories → 646 kept: 526 dropped as evaluation-split, 28 as
unresolved, and the remainder capped so no fault family exceeds 18% of the set.
Contamination check: **0 overlap** between the 646 training incidents and the 526
evaluation incidents.

**Result.** `sft-qwen3-8b-v4`, the adapter behind every published row: train_loss 0.0918,
eval_loss 0.0215. (The previously quoted 0.110 / 0.034 belonged to the superseded run.)

**Reading it honestly.** An eval loss of 0.034 is *not* evidence of diagnostic skill. The
expert is formulaic, a fixed investigation order per fault family, so a low loss mostly
means the format was learned. Whether anything transferable was learned is what §5.2's
perturbation suites and the held-out splits are for, and those are the numbers to trust.

## 6b. RETRACTED, the "SFT is worse" result was a bug in this repository

**This section originally reported that SFT made the agent ten times worse and presented
it as a finding about training. That conclusion was wrong.** The cause was a
train/inference history mismatch in the harness (§ engineering notes, "the harness showed
the model a transcript it was never trained on"). Fixing it moved the same adapter from
3.1% to 29.3% resolution with no retraining.

The original analysis is kept below, unedited, because the reasoning that led to a wrong
conclusion is worth preserving: every hypothesis in the table *was* checked and correctly
ruled out, and the conclusion still turned out to be wrong, because the one thing not on
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
steps, the signature of a degenerate policy, not of a model that learned a format.

**What was ruled out.** Each of these was checked rather than assumed:

| Hypothesis | Check | Result |
| --- | --- | --- |
| Training targets lack the JSON call | 5,233 assistant turns inspected | 5,233/5,233 contain a fenced `json` block |
| Examples truncated at `max_length` | Tokenized the whole dataset | max 6,844 vs cap 8,192, 0% truncated |
| Loss mask excludes the JSON | Decoded the mask on real content | mask = full content **+** `<\|im_end\|>` |
| Adapter not served | vLLM engine log and adapter config | LoRA loaded, r=32, correct base |
| Generation cap cutting the call | `truncated_turns` tracked separately | 167 of 4,736 turns, not the cause |

**Why this matters more than a good number would.** Training metrics were excellent:
`eval_loss 0.0334`, token accuracy **98.9%**. Under teacher forcing the model predicts
every next token almost perfectly, because the true prefix is always supplied. Free-running
it never has to *decide* to emit the call, and it doesn't. The gap between 98.9% token
accuracy and 3.1% task success is the entire argument for this project: **an agent cannot
be evaluated by its loss curve.** Only executed, verified rollouts reveal this, and they
have to be cheap enough to run routinely or nobody runs them.

The likely remedies (more diverse demonstrations, lower learning rate, fewer epochs,
mixing in on-policy data) are stated as hypotheses, not conclusions; none has been tested
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
*identically* (53.1%). Reading that cell alone gives "DPO produced no gain", a conclusion
that survives a sanity check, sounds like a familiar result, and is wrong. The gain is on
the held-out axes: **+23.4 pp on unseen topologies**, +3.1 pp on compositional incidents,
and improvement on every perturbation suite where SFT was weakest (required-argument
0.76 → 0.88, server dropout 0.56 → 0.62, combined 0.82 → 0.91).

That is a mechanistically sensible place for it to land: the preference pairs are drawn
from the model's *own failures*, so they carry recovery behaviour that demonstrations of
the happy path never show. But the point here is methodological, this is the second time
in this project that a correct set of numbers produced an inverted conclusion because the
wrong slice was read first (the earlier one compared SFT against the expert ceiling rather
than against base). Splitting the evaluation along four held-out axes is what made the
difference recoverable.

### Diagnosing a pair shortage without guessing

The first DPO run mined only 88 pairs from 120 incidents, and two hypotheses explained
that equally well: too few incidents sampled, or a policy too self-consistent to disagree
with itself. They have opposite fixes (sample more, or add entropy) and the pair count
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
does produce are strongly separated. The shortage was purely a sampling-volume problem,
and the reason volume was low was a defect, not a budget: the sampling phase ran with
`--shard-total 1` while the evaluation had been sharded eight ways weeks earlier. It never
errored; it was just slow, which is the easiest symptom to misattribute to the model.

Sharding the sampler took 480 episodes to **2,400**, and the pair count from 88 to **435**
, within 1% of the 440 projected from the measured 73% rate, which is itself a check that
the diagnosis was right.

The generalisable lesson: **when one observable is consistent with two opposite diagnoses,
the correct move is not to choose, but to make the system report one more fact.** Four
counters settled in a single run what would otherwise have been an expensive guess.

### The pair-count ablation, and a second near-miss

Because both DPO runs used the same method, starting adapter and evaluation, the pair
count is the only variable, a clean ablation. Resolution rate, clean tools, `n = 64` per
cell:

| variant | pairs | in-dist. | unseen fault | unseen topology | compositional | all splits |
| --- | --- | --- | --- | --- | --- | --- |
| SFT |, | 53.1% | 29.7% | 32.8% | 46.9% | 40.6% |
| SFT + DPO | 88 | 53.1% | 29.7% | **56.2%** | **50.0%** | 47.3% |
| SFT + DPO | 435 | **64.1%** | **31.2%** | 51.6% | 48.4% | **48.8%** |

**5× the preference data bought +11.0 pp in-distribution and nothing out of it**: +1.5 pp
overall, with two held-out splits slightly *down*. The extra pairs sharpened the policy on
the distribution they were mined from. They did not buy generalisation.

This nearly went into the record backwards. The evaluation writes one cell at a time, and
the first to land was in-distribution, showing +11.0 pp resolution, +10.9 pp root cause and
0.0% invalid calls. That is a decisive-looking win, it confirmed the hypothesis that
motivated the work, and it was written up as one before the remaining fifteen cells
finished. The full sweep reduced it to +1.5 pp. **The third inverted conclusion in this
project, and the first caused by reading a partial result rather than the wrong slice of a
complete one**, which is why `serve_and_eval.sbatch` now fails the job on any empty cell,
and why no number enters a table before its sweep is complete.

The schema-shift view carries a related trap. Retention is normalised to each variant's
*own* clean baseline, and the 435-pair run's baseline rose 11 pp, so its retention falls
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

Six up, one down, and the regression is on the suite that adds a newly *required*
argument. That is the failure mode a policy sharpened on 435 examples of the argument
conventions it already knew would be expected to have, and the retention table hides it
behind a moved denominator. **A normalised metric is the right default for comparing
variants with different baselines, and the wrong one for deciding whether a variant
regressed.** Both views are now reported.

## 6d. The unseen-fault regression is label-prior collapse

Every post-trained variant scores below the base model on held-out fault families, 29.7%
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
`node_unhealthy_reduces_capacity`: not once in ~600 opportunities. Both labels are in the
`submit_diagnosis` enum and in the system prompt on every single episode; the models are
not unable to see them, they are unable to reach them. What they emit instead is
confidently wrong and drawn from the training catalogue,
`readiness_probe_targets_wrong_endpoint` for 17 of 32 `NodeNotReady` episodes.

So the regression is not degraded reasoning. It is **label-prior collapse**: fine-tuning on
a closed catalogue drives P(label) to zero for classes with no demonstrations, and the
argmax can no longer select them at any evidence strength. The distinction matters because
the two diagnoses have different fixes, degraded reasoning needs better data, a collapsed
prior needs the update attenuated or regularised.

### Attenuating the update, without retraining

A LoRA update enters each layer as `delta_W = (lora_alpha / r) * B @ A`, so multiplying
`lora_B` by `s` scales the entire update by exactly `s`. `s = 0` is the base model, `s = 1`
the trained one, and intermediate values interpolate, no retraining, and every scale can
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
while in-distribution resolution differs by +4.2 pp, statistically indistinguishable
(z = 0.43). Halving the LoRA update recovers most of the held-out-fault regression and
costs nothing measurable on the distribution the policy was trained for.

The mechanism moves with it, which is what makes this a fix rather than a coincidence. On
held-out incidents the trained policy emits a correct held-out label **once in 46
submissions**; at s = 0.5 it emits **13 in 24**, and those 13 are exactly its 13 correct
root causes. The collapsed prior is recoverable by attenuation because the labels were
never removed from the model, only pushed below every competitor by fine-tuning on a
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
and monotonically, and at s = 1.00 the trained policy emits a correct held-out label
**zero times in 114 submissions**.

A dose-response across three levels is the strongest evidence available here that the
effect is caused by update magnitude rather than by anything incidental to one checkpoint.
And the in-distribution cost of the best setting remains unmeasurable: s = 0.5 against
s = 1.0 is +1.6 pp, z = 0.27.

**The prediction that failed.** The confirmation was justified by a power calculation
saying n = 128 would resolve the s = 0.75 in-distribution dip at z ≈ 3.1. It did not. The
dip replicated in *direction* but roughly halved in size (18.8 pp at n = 48, 10.2 pp at
n = 128) and now sits at z = 1.69, still short of significance. Textbook regression to the
mean: the calculation assumed the n = 48 effect size was the true one, which is exactly the
caveat recorded when the job was submitted, and it is the caveat that came true. **The
in-distribution non-monotonicity remains unresolved, and the honest reading is that the
first sweep overestimated it.**

So the claim the project makes is the one the data supports: attenuation monotonically buys
held-out-fault performance at no measurable in-distribution cost. It does not claim to have
located an optimum, and the wobble at s = 0.75 in-distribution is reported as unexplained
rather than smoothed away.

**Two internal validity checks, both passed.** Invalid-call rate is cleanly monotone in s,
27.0 → 19.8 → 9.1 → 5.1 → 0.0% in-distribution, 31.8 → 25.1 → 5.3 → 2.2 → 0.0% on held-out
faults. That is the quantity the adapter most directly controls, and it scales smoothly
exactly as an exact linear interpolation of the update predicts. Separately, the s = 1.00
arm reproduces the published DPO-435 row (62.5% vs 64.1% in-distribution, 27.1% vs 31.2%
unseen fault, both within noise) and the s = 0.00 arm reproduces base, so the scaling
machinery and the serving path are the same ones that produced the tables above.

**The held-out fault axis is narrow, and that bounds every claim made on it.** The
`unseen_fault` split contains **120 incidents across exactly two fault families**
(`NetworkPolicyDeny`, `NodeNotReady`). 120 is a hard ceiling, discovered by asking for 128
and receiving 120 with all eight shards reporting evenly and no failures, so no
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
agent RL hard. An episode is twenty-odd calls whose value is known only at the end.

`src/agentgym/train/episode_grpo.py` closes that. The group is *G* full rollouts of the
same incident, the reward is the episode's verifiable reward, the same number §5 reports,
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

**435 DPO pairs versus 2,388 GRPO episodes across 597 groups from identical data**: 5.5×
the training signal, because a preference pair keeps only a best/worst couple that clears a
margin, while a group uses the entire reward distribution.

Two implementation details are load-bearing and both are pinned by tests:

* **Length normalisation.** Dividing by the assistant-token count stops the objective
  preferring long episodes whenever the advantage is positive, a bias that surfaces as the
  agent padding its reasoning to farm reward. `test_loss_is_length_normalised` asserts that
  doubling episode length at equal per-token likelihood leaves the loss unchanged.
* **Degenerate groups are dropped, not zeroed.** When every rollout of an incident earns
  the same reward the advantage is 0/0; keeping those as zeros silently shrinks the
  effective batch while the step count claims otherwise, the same fail-open shape as the
  sampling bug above. They are removed and counted (3 of 600 here).

One bug was caught before it ever ran: the policy forward was written inside
`model.disable_adapter()`, which would have trained the adapter on gradients computed
without it. It raises nothing and still produces a model, so
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
again retained for the backward, an OOM on a 44 GB L40S at batch size 1. The tests
exercise the same code path with `V = 8`, where the tensor is 200 bytes. Switching to
`cross_entropy(reduction="none")`, which is fused and recomputes softmax in the backward
rather than storing it, fixed it; **all 15 tests passed unchanged, which is the evidence
the rewrite is equivalent** and also the evidence they could never have caught it.

**2. The KL reference was the wrong policy, and the docstring said so.** The reference
was `model.disable_adapter()`, which yields the *base* model, while the policy starts from
the SFT adapter. So KL(SFT ‖ base) was already enormous before any update:

```
step 5    pg = -0.8598    kl = 32472.04    loss = 1298.02
```

At β = 0.04 the penalty exceeded the policy gradient by three orders of magnitude: the run
was optimising *return to base*, not *improve the policy*. Nothing errored, the loss fell
steadily. The module docstring claimed the KL was "measured against the policy being
improved", which is precisely what disabling the adapter does not do; the prose and the
code disagreed and only the prose was right. The SFT adapter is now loaded twice (`default` trains, `reference` is frozen) and `grpo_loss` takes a `ref_context` that
switches between them. After the fix, `step 5: pg = -0.8594, kl = 0.0108`. The new test
asserts the *relationship* the bug violated: `loss == pg_loss` at step 0.

The old KL test passed throughout, because it handed the stub identical logits for policy
and reference. It tested the estimator's algebra, not whether the right policy was passed
to it.

**3. Truncation was silent.** `_collate` truncated at `max_length` without a word. A cut
episode keeps its full group-relative advantage but loses its later turns, frequently the
one that resolved the incident, so the gradient attributes +A to a prefix that did not
earn it. Truncated and zero-assistant-token examples are now counted into
`train_log.json`, with a warning above 25%. `max_length` was then chosen by measurement
rather than habit: over 250 sampled episodes the median is 6,282 tokens, p95 8,225, max
13,064, so 12,288 truncates 1.2% where 8,192 would truncate 5.6%.

A fourth was caught by inspection rather than failure: the training loop does not shuffle,
which is load-bearing. Advantages are zero-mean *within* a group, so group members staying
adjacent is what lets each accumulation window span whole groups and cancel the baseline
inside it, the variance reduction GRPO exists to buy. The ordering was correct but
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
that simply did not resolve the incident, so the format SFT taught is collateral.

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

and on held-out faults its root-cause accuracy is **14.1%, exactly base's 14.1%**, against
0.0% for SFT and 1.6% for DPO-435.

This matters more than the capability number. Two interventions with nothing in common,
**scaling the LoRA update in weight space**, and **on-policy RL against a verifiable
reward**, both restore a prior that supervised and preference training had driven to zero.
That is convergent evidence that the collapse is caused by *optimising hard toward a closed
catalogue*, rather than by anything specific to one objective, one checkpoint, or one
weight-space direction. Either result alone would be suggestive; together they identify the
cause.

Both also pay for it, in the same currency. Attenuation trades in-distribution sharpness;
GRPO trades tool-call validity. Nothing here recovers held-out coverage for free.

**Not claimed.** Episode-GRPO's unseen-fault edge over DPO-435 (40.6% vs 31.2%) does *not*
reach significance at `n = 64` (z = 1.11). One run at one learning rate, one KL coefficient,
one epoch, on rollouts from the SFT policy, this is evidence that the method as configured
degrades capability, not that episode-level GRPO cannot work. The obvious next moves are a
smaller step size and a KL reference that constrains the *evaluation* distribution rather
than the training one.

### The variance budget, decomposed, and a correction to the floor I reported

Every source of variation this project can now measure, all at n = 2,200 per arm on the
same 5,000 fresh incidents:

| source | Δ resolution | z |
|---|---|---|
| **learning rate, 5e-5 → 1e-4** | **+14.23 pp** | **+9.54** |
| DPO, pair 1 | +4.18 pp | +2.83 |
| **training seed, 11 → 12** | **−3.09 pp** | **−2.06** |
| DPO, pair 2 | −0.27 pp | −0.18 |
| nondeterminism, *same* seed | −0.18 pp | −0.12 |

**A correction.** Earlier in this session the run-to-run floor was reported as 0.2 pp, and
the conclusion drawn was that 2–5 pp method effects are ten to twenty-five times the floor
and therefore readable. **That floor is the wrong one for the comparison being made.** It
was measured between two runs at the *same* seed, so it captures GPU nondeterminism only.

Two genuinely different seeds (the first such comparison in this project, possible only
after the launcher was fixed) differ by **3.1 pp (z = −2.06)**. Sixteen times the
same-seed figure, and nominally significant.

Which floor applies depends on the question:

* Comparing two runs of the *same* configuration: **0.2 pp**.
* Comparing independently trained models, which is what every method comparison does:
  **3.1 pp**.

**And 3.1 pp is the same size as DPO's effect**: +4.18 pp in one pair, −0.27 pp in the
other. That is not a coincidence; it is the explanation. The pairs disagreed because the
quantity being measured is comparable to the noise in what it is measured against.

So the original conclusion, that method comparisons here need several seeds per arm, was
**right, and this session first refuted it for a bad reason and then restored it for a good
one**. The refutation used the same-seed floor, which understates the relevant noise by a
factor of sixteen.

**The learning rate remains dominant and is now clean.** 5e-5 against 1e-4, same seed, same
corpus, same epochs: +14.23 pp resolution (z = 9.54), +10.41 pp root cause (z = 7.00), and
invalid calls 0.59% → 0.01%. It is 4.6× the seed floor and the only factor measured here
that clearly rises above it. **Tuning the learning rate mattered more than every
post-training method tested, and by a margin nothing else approaches.**

### DPO's benefit does not replicate across two indistinguishable baselines

With the learning rate matched and the floor known, the question "does DPO help" could
finally be asked properly: train DPO from a specific SFT and compare it against *that* SFT,
so nothing about the baseline differs.

Two pairs. Both SFT bases have identical configuration (seed 0, LR 1e-4, same corpus, 2
epochs) and score **57.7% and 57.5%**, 0.2 pp apart, which is the measured floor. They are
behaviourally indistinguishable.

| | base | + DPO | Δ resolution | z | Δ root cause | z |
|---|---|---|---|---|---|---|
| **pair 1** | 57.73% | 61.91% | **+4.18 pp** | **+2.83** | **+5.41 pp** | **+3.60** |
| **pair 2** | 57.55% | 57.27% | **−0.27 pp** | −0.18 | **+0.05 pp** | +0.03 |

**They disagree.** On one base DPO gives +4.2 pp resolution and +5.4 pp root cause, both
clearing significance. On the other it gives nothing, −0.27 pp and +0.05 pp, indexes
indistinguishable from zero.

This is not evaluation noise. The evaluation floor is 0.2 pp at n = 2,200, and pair 1's
effect is twenty times that. **The variability is in the training, not the measurement**:
the same DPO recipe, applied to two baselines that score within a fifth of a point of each
other, produces a large gain in one case and nothing in the other.

**What this permits saying:** DPO *can* help, by a margin well above the measurement floor.
**What it forbids:** quoting a number. A single DPO run's improvement is not a property of
the method here. It is a property of that run, and this project has now observed both
outcomes from nominally identical starting points.

Two pairs support a statement about direction agreement, and the directions **do not**
agree. Two observations also cannot support a sign test; there is no p-value to quote from
them, and one is not offered.

**Note what was avoided.** Pair 1 alone was complete for one cycle before pair 2 finished.
Written up then, it would have read: *"with the learning rate matched, DPO delivers +4.2 pp
resolution (z = 2.83) and +5.4 pp root cause (z = 3.60), the undertuned baseline had been
understating it."* Every number in that sentence is correct. The sentence is wrong. It is
the fifth time in this project that a first cell has pointed somewhere the full grid did
not, and the reason the report script refuses to render a partial grid.

### The run-to-run floor is 0.2 pp, not 15.6 pp, and it makes the method effects readable

With the launcher fixed, the floor could finally be measured. Two SFT runs with **identical
configuration** (same seed 0, same LR 1e-4, same corpus, same 2 epochs) differing only in
GPU nondeterminism:

| | run 1 | run 2 | Δ | z |
|---|---|---|---|---|
| in-distribution | 72.0% | 66.9% | −5.1 pp | |
| unseen fault | 53.2% | 56.6% | +3.4 pp | |
| unseen topology | 55.8% | 65.0% | +9.2 pp | |
| compositional | 44.2% | 38.0% | −6.2 pp | |
| **all (n = 2,200)** | **57.7%** | **57.5%** | **−0.2 pp** | **−0.12** |
| root cause | 48.9% | 49.9% | +1.0 pp | +0.63 |

**0.2 pp.** Not 15.6. The number that made every method comparison look hopeless was a
learning-rate difference wearing a seed's name.

Two things worth separating here. Per-split the runs wobble by up to 9 pp, because each
split is 500–700 episodes; pooled to 2,200 they agree to a fifth of a point. **The floor is
a property of the measurement size, not of training**, quoting a per-split difference as
run-to-run noise overstates it by an order of magnitude, which is exactly the error that
produced the 11 pp figure this project carried for weeks.

And the training is **loss-reproducible but weight-divergent**: those two adapters sit
**138% apart** in relative `lora_B` norm while their train losses agree to 0.0005 and their
evaluations to 0.2 pp. Different points in weight space, same function. A weight-space
distance is therefore not evidence of a behavioural difference, though it remains a valid
*lineage* check, which is how the withdrawn episode-GRPO run was caught.

**The consequence: method effects of 2–5 pp are ten to twenty-five times the floor and are
readable.** The earlier conclusion that no single-run comparison here is interpretable was
wrong, and wrong because of the same misattribution.

### CORRECTION. The "seed variance" was a learning rate, and the seed was never varied

The section below attributes +15.6 pp to seed variance and concludes that no single-run
method comparison in this project is interpretable. **The attribution is wrong.** The
conclusion partly survives, for a different reason.

`training_args.bin` for all three 2-epoch SFT adapters:

| adapter | seed | learning rate | train loss |
|---|---|---|---|
| `sft-qwen3-8b-v4` (published) | **0** | **5e-5** | 0.0918 |
| `sft-2ep-seed1` | **0** | **1e-4** | 0.0728 |
| `sft-2ep-seed2` | **0** | **1e-4** | 0.0733 |

Every adapter this project has ever produced trained at **seed 0**. All three launchers
accepted a `SEED` environment variable; none forwarded it to the trainer, which has taken
`--seed` and used it the whole time. Three jobs were submitted with `SEED=1` and `SEED=2`
and all of them ignored it.

So the 15.6 pp is a **learning-rate effect, 5e-5 against 1e-4**. The two runs sharing a
learning rate agree to 0.0005 in training loss, which is what a same-seed repeat should
look like.

**What this changes.**

* The dominant controllable factor is the learning rate, and **the published baseline was
  trained at half the better one**. Every method comparison in this document was measured
  against an undertuned reference, which is a more useful finding than "the noise is too
  large", because it names something a reader can act on.
* The true seed-to-seed floor is **unmeasured**. It might be small, in which case the
  method comparisons become interpretable once the learning rate is fixed; it might not.
  The first runs with a genuinely different seed are in flight.
  *(Resolved: seed 11 vs seed 12 is **3.1 pp**, z = −2.06, the same size as DPO's effect,
  which is why the paired DPO test came out a negative. See "The variance budget,
  decomposed" above.)*
* The conclusion that the method ordering is unsupported still stands, but its reason
  changes: not "seed noise swamps it" but "the baseline was undertuned and the noise floor
  is unknown."

**A second instance of the same defect, found by the same audit:** `train_dpo.sbatch`
never passed `--lr` either, so DPO's learning rate was always the trainer's default and
exporting `LR` did nothing.

`tests/test_launcher_plumbing.py` now asserts that every result-bearing argument a trainer
accepts is reachable from its launcher. Its first version was defective in exactly the way
it exists to prevent, it searched the raw file, and the comment explaining the fix contains
the string `--seed`, so deleting the real argument left it passing. It strips comments now,
and the failure was verified by deletion.

This is the fourth defect of this shape in the repository: a `--max-prompt-length` that
reached no code path, a `values-eks.yaml` block consumed by no template, a settle-timeout
counter that summed events and was read as episodes, and now a `SEED` nothing forwarded.
**None errored. Each described a system that was not the one running.**

### Final: the complete grid, and the two variables that dominate it

All eight arms complete, n = 2,200 each, on 5,000 incidents no model has seen.

| variant | resolved | root cause | invalid calls | reward |
|---|---|---|---|---|
| **SFT 2ep, seed 1, no post-training** | **57.7%** | 48.9% | **0.01%** | 1.54 |
| SFT 3ep + DPO-309 | 52.5% | **51.8%** | 0.32% | 1.47 |
| SFT 3ep | 50.3% | 50.6% | 0.19% | 1.40 |
| SFT 2ep + DPO-435 | 44.3% | 41.5% | 0.18% | 1.21 |
| SFT 2ep + episode-GRPO | 44.2% | 40.5% | 0.28% | 1.20 |
| SFT 2ep + action-GRPO | 43.0% | 38.5% | 0.28% | 1.15 |
| SFT 2ep, seed A, published baseline | 42.1% | 36.9% | 0.35% | 1.12 |
| base | 36.3% | 37.6% | 26.32% | 0.15 |

Two single-variable comparisons account for the ordering:

```
SEED    identical recipe, different seed   +15.6 pp   z = +10.48
EPOCH   same seed, 2 epochs vs 3            -7.4 pp   z =  -4.94
```

against the three post-training methods, none of which resolves:

```
+ DPO-435  +2.1 pp (z=1.41)   + episode-GRPO  +2.0 pp (z=1.34)   + action-GRPO  +0.9 pp (z=0.60)
```

**The seed is 7× the largest method effect; the epoch effect is 3.5× it and negative.** The
best agent in the project is a plain supervised fine-tune with a lucky seed, and it also has
the cleanest tool use here, 0.01% invalid calls against 0.18–0.35% for everything
post-trained.

Even the headline "post-training clears base" is seed-dependent by a factor of 3.6: the
same SFT recipe gives +5.9 pp (z = 3.99) with one seed and +21.5 pp (z = 14.60) with
another. It clears base either way (every arm does, at z = 3.99 to 14.60) but the
magnitude anyone would quote is mostly a draw.

**What this means for the project's claims.** Three survive:

1. **Post-training clears the untrained base**, on every arm, at z ≥ 3.99.
2. **It collapses invalid tool calls** from 26.3% to 0.01–0.35%, a ~100× effect no seed
   draw comes near.
3. **It trades away held-out coverage**: base is the *best* model here on fault families it
   never saw (65.8% against 33–53%), and the label-prior mechanism behind that is
   measured, reversible at serving time, and replicated across two adapter families at
   z ≈ 6.

One does not: **any ordering among SFT, DPO, episode-GRPO and action-GRPO.** All four sit
inside the spread a single recipe produces by reseeding.

The honest one-line summary of the post-training work: *it reliably teaches the tool
protocol and reliably narrows the model; which algorithm you use to do it was not
measurable here, and the run-to-run draw mattered more than the algorithm.*

### RETRACTED. The epoch effect was seed variance, and seed variance dominates everything

**The section below is wrong and is kept for the record.** It reported that an extra SFT
epoch outweighs every post-training method (+8.1 pp, z = 5.43). That comparison held the
epoch count apart from the *seed*, and the two runs differed in both.

Training a third SFT adapter, 2 epochs at the **same seed** as the 3-epoch run, separates
them. In-distribution, n = 700 per arm:

| run | epochs | seed | resolution | root cause |
|---|---|---|---|---|
| `v4` (published baseline) | 2 | A | 59.4% | 59.0% |
| `sft-2ep-seed1` | 2 | 1 | **72.0%** | **70.0%** |
| `sft-...-seed2` | 3 | 1 | 67.9% | 68.0% |

```
SEED  only (both 2 epochs, different seed):  +12.6 pp   z = +5.00   SIGNIFICANT
EPOCH only (both seed 1, 2 vs 3 epochs):      −4.1 pp   z = −1.69   negative
```

**The epoch effect is slightly negative. The +8.1 pp was the seed.**

And that is the finding, larger than anything else this project measured:

| comparison | Δ | z | kind |
|---|---|---|---|
| **SFT seed A → SFT seed B, identical recipe** | **+12.6 pp** | **5.00** | **seed** |
| SFT 2ep → 3ep, same seed | −4.1 pp | −1.69 | epoch |
| SFT → + DPO-435 | +4.0 pp | 1.54 | method |
| SFT 3ep → + DPO-309 | +3.4 pp | 1.40 | method |
| SFT → + episode-GRPO | +3.3 pp | 1.26 | method |
| SFT → + action-GRPO | −1.3 pp | −0.50 | method |

**Two SFT runs differing only in random seed are 12.6 pp apart, three times the largest
method effect, and the only comparison in the table that clears significance.**

The consequence is unavoidable: **no single-run method comparison in this project is
interpretable.** DPO, episode-GRPO and action-GRPO were each trained once. Their measured
differences from the baseline, and from each other, sit entirely inside the distribution
that one recipe produces by reseeding. Establishing that any of them helps requires several
seeds per arm and a comparison of distributions, which this project has not done.

This is the third time these numbers have been revised, each time downward and each time
because power increased or a confound was removed: +8.2 pp from DPO became +2.1 pp under
power; the residual was reattributed to a missing epoch; the epoch turns out to be seed.
The direction of travel is itself informative, **every increase in rigour has moved the
result toward "no measurable difference", which is what a null looks like when it is
approached honestly.**

What survives untouched: the environment's correctness, the base-to-post-trained gain, the
collapse of invalid tool calls from 26.3% to ~0.2%, and the label-prior finding, which was
never a single-run comparison. It is a dose-response across four attenuation levels,
replicated on two independently-trained adapters, at z ≈ 6. That design is why it survives
and the method comparisons do not.

<details><summary>The retracted section, unedited</summary>

### One SFT epoch outweighs every post-training method tested

Five complete arms at n = 2,200 on fresh incidents. `base` and `SFT 3ep + DPO` are still
running; nothing below depends on them.

| variant | in-dist | unseen fault | unseen topo | compositional | **all** | **root cause** | invalid |
|---|---|---|---|---|---|---|---|
| SFT 2 epochs, published baseline | 59.4% | 40.6% | 28.2% | 33.4% | 42.1% | 36.9% | 0.35% |
| &nbsp;&nbsp;+ DPO-435 | 63.4% | 37.4% | 33.4% | 35.2% | 44.3% | 41.5% | 0.18% |
| &nbsp;&nbsp;+ episode-GRPO | 62.7% | **43.6%** | 30.6% | 32.4% | 44.2% | 40.5% | 0.28% |
| &nbsp;&nbsp;+ action-GRPO | 58.1% | 41.8% | 31.2% | 35.0% | 43.0% | 38.5% | 0.28% |
| **SFT 3 epochs** | **67.9%** | 33.2% | **51.2%** | **41.8%** | **50.3%** | **50.6%** | 0.19% |

Bonferroni threshold for the twenty comparisons reported is z > 2.64.

```
SFT 2ep -> SFT 3ep        +8.1 pp   z = 5.43   SIGNIFICANT
SFT 2ep -> DPO-435        +2.1 pp   z = 1.43   not resolved
SFT 2ep -> episode-GRPO   +2.0 pp   z = 1.37   not resolved
SFT 2ep -> action-GRPO    +0.9 pp   z = 0.61   not resolved
```

**The only intervention with a detectable effect on resolution is an extra epoch of
supervised training.** Three post-training methods, two of them on-policy RL against a
verifiable reward, produce nothing this evaluation can distinguish from noise at n = 2,200.

Action-level GRPO contributing least (+0.9 pp) is the outcome its own training metrics
predicted: mean reward 3.22 against a 3.5 ceiling and **55% of groups with zero reward
variance**, because SFT already parses, names real tools and satisfies the schema almost
always. There was nearly nothing left in that objective to optimise, and the evaluation
agrees.

**The 3-epoch model is worse on held-out faults**: 33.2% against 40.6%. More supervised
training buys in-distribution capability and pays for it in held-out coverage, which is the
same specialization/generalization trade the attenuation sweep measured from the opposite
direction: scaling the update *down* recovered held-out labels at some in-distribution
cost, and training *longer* does the reverse. Two independent handles on one axis.

</details>

### DPO's one significant result does not survive a properly trained baseline

In-distribution cell complete for all four arms, n = 700 each, fresh incidents. The other
three splits are still running and nothing below is claimed beyond in-distribution.

| variant | resolved | root cause | invalid calls | reward |
|---|---|---|---|---|
| SFT 2 epochs, the published baseline | 59.4% | 59.0% | 0.23% | 1.62 |
| &nbsp;&nbsp;+ DPO-435 | 63.4% | **66.9%** | 0.17% | 1.76 |
| **SFT 3 epochs** | **67.9%** | **68.0%** | 0.03% | 1.89 |
| &nbsp;&nbsp;+ DPO-309 | 71.3% | 70.3% | 0.03% | 1.99 |

```
DPO on the 2-epoch base   root cause  +7.9 pp   z = 3.05   significant
DPO on the 3-epoch base   root cause  +2.3 pp   z = 0.93   not significant
```

**A third SFT epoch delivers what DPO delivers.** 3-epoch SFT alone reaches 68.0% root
cause; 2-epoch SFT *plus* DPO reaches 66.9%. And on the better base DPO's diagnostic gain
falls to +2.3 pp and stops being detectable, the single statistically significant
between-method result in this document, evaluated against a baseline that was undertrained.

This is consistent with the pair-mining counters above, which showed the signal thinning
before any adapter was trained: the 3-epoch policy yields 309 pairs against 437 because its
attempts are too alike in quality to rank. Less to disagree about, less for DPO to learn.

**What is not claimed.** DPO is not shown to be useless: it still adds +3.4 pp resolution
on the better base (z = 1.40), the direction is positive on every metric, and this is one
split. What *is* shown is that the reported benefit was measured against a baseline with
an epoch of training left in it, and most of that benefit was the missing epoch.

The honest revision of this project's headline: **post-training helps, and roughly half of
what was attributed to preference optimisation is attributable to training the supervised
baseline to convergence first.**

### A better base yields less preference signal

Mining DPO pairs from the 3-epoch SFT, at settings identical to the 2-epoch run, 600
incidents, 4 samples, temperature 0.9:

| base | pairs | rejected: margin < 0.8 | rejected: no spread | mean margin |
|---|---|---|---|---|
| SFT 2 epochs | **437** | 157 | 6 | 2.594 |
| SFT 3 epochs | **309** | **276** | 15 | 2.616 |

Pair yield falls 29% because **76% more incidents fail the margin threshold**. The pairs
that still form are no less informative, mean margin is unchanged at 2.62, there are
simply fewer of them, because a better policy's four attempts at one incident are more
alike in quality and therefore harder to rank.

This is the mechanism by which DPO's headroom shrinks as the base improves, measured
rather than asserted, and it is visible before any evaluation of the resulting adapter.
Earlier in this project the "policy too consistent to disagree with itself" hypothesis was
raised for the 2-epoch base and **disproved** by these same counters, 73% pair rate, only
one zero-spread incident. On the 3-epoch base the same instrument now supports it. The
counters did not change; the policy did.

### The SFT baseline everything is measured against is undertrained

A second SFT run, submitted to estimate seed variance, instead exposed something worse.
Its training metrics do not match the published adapter's:

| | published SFT (`v4`) | second run |
|---|---|---|
| epochs | **2.0** | **3.0** |
| train loss | 0.0918 | 0.0603 |
| eval loss | 0.0215 | 0.0193 |

**Every post-training result in this project sits on a 2-epoch base, and a 3-epoch SFT with
no post-training at all beats all of them** (n = 2,200 per arm, same pool):

| variant | resolution | root cause |
|---|---|---|
| SFT 2ep, the published baseline | 42.1% | 36.9% |
| SFT 2ep + DPO-435 | 44.3% | 41.5% |
| SFT 2ep + episode-GRPO | 44.2% | 40.5% |
| **SFT 3ep, no post-training** | **50.3%** | **50.6%** |

```
3ep SFT vs 2ep + DPO-435   resolution +6.0 pp  z = 3.99   root cause  +9.1 pp  z = 6.08
3ep SFT vs 2ep + GRPO      resolution +6.1 pp  z = 4.06   root cause +10.1 pp  z = 6.76
```

**This is confounded and is not yet a result.** The two runs differ in seed *and* epoch
count, so the gap could be either. But it is large enough to threaten the project's central
claim: if one more epoch of supervised training buys more than DPO does, then the
post-training gains reported here may be recovering capability an adequately trained
baseline would have had for free.

Two runs are in flight to settle it rather than argue it: a 2-epoch SFT at the *same* seed
as the 3-epoch run, which isolates the epoch effect from the seed effect; and DPO trained
on top of the 3-epoch SFT, which answers the question that actually matters, whether
preference optimisation adds anything to a base that is not undertrained.

*(Resolved, and then re-resolved. The epoch alone is **−4.1 pp**, so the apparent epoch
effect was a difference between adapters, and that difference was later traced to the
**learning rate**, not the seed, because no launcher ever forwarded `--seed`. Both
retractions are recorded below.)*

Note what nearly happened. The +13.8 pp root-cause gap between the two SFT runs was, for
several minutes, about to be written up as *seed variance so large it invalidates every
between-method comparison in this document*. That would have been wrong, and checking
`train_metrics.json` before writing was the only thing that caught it. The right conclusion
and the catastrophic one differed by one field in a JSON file.

### The contamination counter over-reported, and checking it was the only way to know

The settle-timeout counter added earlier today fired on the high-power run, and the first
reading looked alarming: 255 expiries across 6,600 episodes. Two things were wrong with
that reading, and both were mine.

**It was an event count, not an episode count.** One episode expires on several steps, so
summing the field reads as a share of episodes when it is nothing of the kind. The true
figure is **27 episodes, 0.4%**.

**And those 27 are not contamination.** Every one involves `ResourceQuotaExhausted`, alone
or in a compositional pair, and all 27 are unresolved. Inspecting one shows why: the agent
never called `patch_resource_quota`, so the quota kept rejecting pod creation, so `_settle`
waited for pods that could not exist and expired. The world genuinely cannot converge until
the agent fixes the blocking fault. **Scoring the episode unresolved is correct, not
corrupted**. The expiry is a symptom of the agent failing, not a cause of a wrong verdict.

So the counter conflates two situations with opposite meanings:

| expiry with... | means |
|---|---|
| episode **unresolved**, blocking fault outstanding | the agent did not fix it; the verdict is right |
| episode **resolved** despite an expiry | the control plane really was slow; the verdict is suspect |

Across all three arms, **every affected episode is in the first category** (14, 7 and 6,
all unresolved). There is no measured contamination in this run at all.

The instrument was built this morning to stop a silent failure and it immediately produced
a loud false one. It now reports affected episodes rather than expiry events, and splits
them by whether the episode resolved, because the count alone cannot distinguish "the
machine was slow" from "the agent left the quota in place", and those demand opposite
responses.

### Resolved at full power: DPO improves diagnosis, not resolution

The 5,000-incident prefix-stable pool lifted the ceiling from 601 to **2,200 per arm** on
incidents no model has seen (zero id overlap with the v1 pool or the training set). Twelve
cells, three arms:

| variant | in-dist | unseen fault | unseen topo | compositional | **all (2,200)** | **root cause** | invalid | reward |
|---|---|---|---|---|---|---|---|---|
| SFT | 59.4% | 40.6% | 28.2% | 33.4% | 42.1% | 36.9% | 0.35% | 1.12 |
| DPO-435 | **63.4%** | 37.4% | **33.4%** | **35.2%** | **44.3%** | **41.5%** | 0.18% | **1.21** |
| episode-GRPO | 62.7% | **43.6%** | 30.6% | 32.4% | 44.2% | 40.5% | 0.28% | 1.20 |

```
resolution   SFT vs DPO-435   +2.1 pp   z = 1.43   not resolved
             SFT vs GRPO      +2.0 pp   z = 1.37   not resolved
             GRPO vs DPO-435  +0.1 pp   z = 0.06   not resolved

root cause   SFT vs DPO-435   +4.6 pp   z = 3.15   significant, survives Bonferroni (z > 2.64, six comparisons)
             SFT vs GRPO      +3.6 pp   z = 2.45   nominal only
             GRPO vs DPO-435  +1.0 pp   z = 0.71   not resolved
```

**Preference optimisation measurably improves diagnostic accuracy over demonstrations
alone. It does not measurably improve resolution rate.** DPO and episode-GRPO are
indistinguishable from each other on every metric.

### The two effects behaved oppositely as power grew, and that is the finding

Both comparisons were run at four sizes. Watching them together separates a real effect
from a lucky one better than any single number:

| n per arm | resolution Δ | z | root-cause Δ | z |
|---|---|---|---|---|
| 256 | **+8.2 pp** | 1.86 | +3.4 pp | 0.79 |
| 601 | +4.3 pp | 1.50 | +5.7 pp | 2.02 |
| 700 | +4.0 pp | 1.50 | +7.9 pp | 3.02 |
| **2,200** | **+2.1 pp** | 1.40 | **+4.6 pp** | **3.11** |

The resolution effect **shrank monotonically toward zero**, 8.2 → 4.3 → 4.0 → 2.1, while
its z never moved off ~1.5. The root-cause effect wobbled between 3.4 and 7.9 but its **z
climbed with every increase in n**, which is what a real effect does: the estimate is noisy,
the evidence accumulates.

The +8.2 pp that was once this project's headline was almost entirely winner's curse. It
was never significance-tested when first reported, and four independent increases in power
walked it down to +2.1 pp and left it there.

**The general lesson, and the reason both rows are kept:** a real effect makes z grow with
n; a noise effect makes the point estimate shrink while z stays flat. One project produced
both, on the same runs, and only the sequence distinguishes them. A single well-powered
measurement would have shown DPO's root-cause gain and missed the story about the
resolution gain entirely.

### What this environment can and cannot measure

The variant ranking was rerun at the maximum size the incident pool allows: 256
in-distribution plus the 120/120/105 the other splits cap at, **601 per arm**.

| variant | in-dist | unseen fault | unseen topo | compositional | **all (601)** | root cause | invalid | reward |
|---|---|---|---|---|---|---|---|---|
| SFT | 57.8% | 33.3% | 35.0% | 47.6% | 46.6% | 41.6% | 0.30% | 1.25 |
| DPO-435 | **63.3%** | 37.5% | **40.0%** | 48.6% | **50.9%** | **47.3%** | 0.21% | **1.38** |
| episode-GRPO | 61.3% | **38.3%** | 32.5% | **52.4%** | 49.4% | 44.8% | 0.25% | 1.33 |

Every pairwise comparison is unresolved:

```
SFT  vs DPO-435       +4.3 pp   z = 1.50
SFT  vs episode-GRPO  +2.8 pp   z = 0.98
GRPO vs DPO-435       +1.5 pp   z = 0.52
```

DPO leads SFT on all four splits (sign test p = 0.0625) and its root-cause gain reaches
z = 1.99. The single comparison that touches threshold, and it would not survive a
multiple-comparison correction. No variant wins consistently: DPO takes in-distribution and
unseen topology, episode-GRPO takes unseen faults and compositional.

DPO is ahead on **all four splits** (+5.5, +4.2, +5.0, +1.0), a sign test at p = 0.0625.
Root cause just clears threshold; resolution does not. With a multiple-comparison
correction across the metrics reported here, neither would.

**The ceiling was structural, and the reason turned out to be a defect.** Resolving the
observed +4.3 pp at 80% power requires **n ≈ 2,119 per arm**; 601 is everything the four
splits contained. Growing the pool looked impossible because it renamed every existing
incident: `generate(5000)[:1200] != generate(1200)`, so every published result would have
stopped being comparable.

That is not a property of incident generation. It was three coupled choices in
`generator.generate`: split counts computed as `int(n * 0.70)`, one global `rng.shuffle`
over the whole plan, and per-incident seeds drawn from a shared stream in shuffled order.
Each makes incident *i* depend on *n*.

`version=2` makes every incident a pure function of `(seed, index)`: splits come from a
fixed 7/1/1/1 position cycle rather than a proportion of `n`, and each incident seeds its
own RNG on `f"{seed}:{i}"`. It is prefix-stable at every size tested, holds the splits at
exactly 70/10/10/10, and leaks no held-out family or topology. `version=1` is retained and
verified to still reproduce the stored 1,200 incidents byte-for-byte, so nothing already
published moves.

**The pool can now grow without discarding its own history.** An evaluation set that
cannot be extended is only discovered to be broken on the day it needs more power, which is
the day the cost of fixing it is highest.

That gives the environment a stated resolution, which is more useful than a ranking:

| effect size | example | verdict |
|---|---|---|
| ~35 pp | attenuation on held-out faults | decisive, z ≈ 6, replicated |
| ~16 pp | base → post-trained | decisive, z = 3.73 |
| ~11 pp | run-to-run variance of one recipe | the noise floor |
| ~4 pp | DPO over SFT | below what 601 episodes can separate |

Every claim in this document should be read against that table. The two findings the project
rests on sit far above the floor; the method ranking sits below it and is reported as
unresolved rather than as a result.

### The attenuation effect replicates across adapter families

The dose-response was established on DPO-435. Whether it was a property of *closed-catalogue
fine-tuning* or of that one checkpoint was the open question, and the honest answer mattered,
because the convergent-evidence argument that would have settled it turned out to rest on a
wrong-base artifact.

Scaling the SFT adapter instead (a different objective, different data, independently
trained) reproduces it almost exactly. Both at the held-out split's 120-incident ceiling,
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
independently-trained policies, weaker as a causal story, far stronger as evidence, and it
does not depend on a result that has since been retracted.

The mechanism claim now stands as: **fine-tuning hard on a closed catalogue drives
never-demonstrated labels below every competitor, and attenuating the update recovers them,
monotonically, on every adapter tested.**

### What the replication settled: three corrections, two of them to published claims

The DPO replication and the two corrected episode-GRPO runs landed together. All four clean
splits complete, `n = 64` per cell.

**1. The pair-count ablation does not replicate, retracted.** The published +11.0 pp
in-distribution gain from 5× the preference data is run-to-run variance. A second run at 437
pairs scores **53.1%**, identical to the 88-pair run and 11 pp below the 435-pair run it
replicates. The original difference was never significance-tested; it is z = 1.27. Between 88
and ~435 pairs there is no detectable capability effect. DPO over SFT survives, all three
runs beat SFT on unseen topologies (56.2 / 51.6 / 50.0 against 32.8), but the pair-count
conclusion does not.

**2. The measurement floor is ~11 pp, and that governs the whole document.** Two runs of an
identical recipe differ by 11 pp in-distribution and 2.7 pp overall. At `n = 64` per cell,
*no single-run difference below roughly 15 pp is interpretable*. This is the most useful
thing the replication bought, and it should have been established before any ablation was
reported rather than after. The frontier dose-response clears it (35.8 pp, monotone across
three levels, z = 5.97) which is now a load-bearing distinction rather than a stylistic one.

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
document, two interventions sharing no mechanism, both restoring a prior that training had
zeroed, therefore the cause is identified. It was built on an artifact, and the artifact was
persuasive precisely because it agreed with a result that was independently true.

### The data pipeline replicates to within 0.5%

Rerunning DPO end-to-end at matched settings (600 incidents, 4 samples each, temperature
0.9, base `sft-qwen3-8b-v4`, an independent sampling pass) reproduces the published
pipeline almost exactly:

| | published | replication |
|---|---|---|
| incidents sampled | 600 | 600 |
| episodes rolled out | 2,400 | 2,400 |
| **preference pairs mined** | **435** | **437** |
| mean reward margin | 2.609 | 2.594 |

Two pairs apart on 437, and a margin difference of 0.015. That is not a claim about DPO's
*effect*, the trained adapters still have to be evaluated, but it does establish that the
sampling and mining stages are stable across independent runs, so a difference in downstream
capability between two DPO runs cannot be attributed to the data pipeline drifting under
them. The variance estimate on the +8.2 pp headline is the next thing this answers.

### The lineage digest paid for itself within the hour

The instrumentation added after the episode-GRPO withdrawal, recording a hash of the base
adapter's *weights* into every training output, caught its second case immediately.

A second DPO run had been submitted to put a variance estimate on the headline +8.2 pp,
before the sbatch default was corrected. Its saved reference hashes to `2c9d616c7c6eff12`,
which is `sft-qwen3-8b-v2`, not the `31388a5ef825d248` that `dpo-qwen3-8b-v5` and
`sft-qwen3-8b-v4` share. Wrong base, same defect as before, caught in one line rather than
by comparing file timestamps and sampled tensors.

It was wrong a second way, which the digest could not catch and arithmetic could: it sampled
600 episodes and mined **86 pairs**, where the published run sampled 2,400 and mined **435**.
The cause was `LIMIT=150` against the published run's 600 incidents. A "replication" at a
quarter of the sampling budget is confounded with pair count, the *one* variable the
ablation established as dominant, so it would have answered a question nobody asked while
looking like an answer to the one that was. `LIMIT` now defaults to the published 600 with
the reason recorded beside it.

Both invalid runs are quarantined rather than deleted, and 5547595 replicates at matched
settings.

### Withdrawn runs are quarantined, not deleted

`eval.report` reads `runs/eval/` recursively, so the withdrawn episode-GRPO trajectories
rendered as an ordinary capability row, same shape, same formatting, nothing marking them
as retracted. Regenerating the tables would have republished them. That is the same failure
mode as the stale-file report earlier in this log, and it is why the retraction needed a
filesystem action and not just a paragraph.

Withdrawn runs move to `runs/eval-withdrawn/<name>/` with a README recording what they
measured, why they are invalid, and what supersedes them. Deleting them would make the
retraction unverifiable; leaving them in place would make it ineffective.

### Post-training barely moves the weights; reseeding relocates them entirely

A LoRA update is `dW = (alpha/r) B@A`, so its size is readable off the adapter without
running anything. `scripts/adapter_norms.py` computes `||dW||_F` per module and per layer,
and the difference between two updates, via `||B@A||_F^2 = tr((B^T B)(A A^T))` -- both
factors are r x r, so a 32x32 trace replaces a 4096x4096 product. In fp64, because fp32
loses three digits to cancellation across the Gram trace (verified against the direct
product: 1e-15 relative error against 5e-3).

**The scaling tool does what the frontier assumes.** The attenuated copies come out at
0.250x / 0.500x / 0.750x of the trained adapter, to three decimals. This is worth pinning
rather than assuming: a monotone bug in the scale factor produces a monotone
dose-response, so the shape of the attenuation curve is not evidence that the axis is
right. `tests/test_scale_adapter.py` now asserts the arithmetic directly.

**Distance from a method to its own starting point, as a fraction of its own size:**

| comparison | distance, as a fraction of A's own size |
|---|---|
| action-GRPO vs its SFT init (`sft-qwen3-8b-v4`) | 0.51% |
| episode-GRPO vs the same init | 0.99% |
| DPO-on-seed2 vs its SFT init | 1.73% |
| DPO-on-seed1 vs its SFT init | 1.99% |
| **SFT seed 1 vs SFT seed 11**, same recipe | **125.2%** |

Every post-training method in this project leaves the weights within 2% of where SFT put
them. Two runs of the *same* SFT recipe, differing only in seed, end up 125% apart. The
gap between those two facts is two orders of magnitude, and it is a weight-space account
of the result the grid already showed behaviourally: method effects of 2-5 pp sit inside
a 3.1 pp seed floor. In weight terms the methods are nearly stationary while reseeding
relocates the model.

**Which half of this is evidence, and which is not.** The direction matters and the two
are not symmetric:

  - The *small* numbers support a behavioural claim. An update that differs by 0.5-2%
    cannot produce a large change in the function, so "these methods change the model
    little" follows from them.
  - The *large* number supports nothing on its own. This project has already measured
    weight distance failing as a behavioural proxy, in this exact form: two
    same-configuration runs sat 138% apart in relative lora_B norm while agreeing to
    0.0005 in train loss and 0.2 pp in evaluation. The 125% here is that same phenomenon
    measured over the full update, and the two seeds it compares agree to within 3.1 pp.
    It is emphatically *not* evidence that the seeds behave differently.

So the claim is one-directional: the methods are small moves, and the seed floor is not
explained by the methods being large. Nothing here says reseeding produces a *better* or
*different-behaving* model, only a distant one.

**Where the update lives.** 83% of the update energy is in the MLP (`gate_proj` 37%,
`up_proj` 31%, `down_proj` 15%) against ~17% across all four attention projections, and
per-layer magnitude rises with depth (2.7x from block 0 to block 35, monotone in the
upper half). The update concentrates in late-layer MLP. That is suggestive next to the
label-prior collapse -- the same place a diagnosis prior would be expected to live -- but
it is a location, not a mechanism, and nothing here tests whether those are the layers
that *cause* the collapse. `scale_adapter.py --layers START:STOP` exists to ask that
question directly and has not yet been run.

**The same structural signature appears on a second architecture.** Llama-3.1-8B,
trained on the same corpus with the same recipe (2 epochs, LR 1e-4, seed 0, train loss
0.0495 against Qwen's 0.0728, eval loss 0.0202 against 0.0215), puts 81.3% of its update
energy in the MLP (gate 39.7%, up 31.4%, down 10.2%) against Qwen's 83.3% (37.1 / 31.4 /
14.8), and concentrates it in the upper blocks the same way (1.8x rise, plateau from block
15 of 32; Qwen 2.7x, peak at block 35 of 36). Two architectures with different depths and
different total update magnitudes (9.32 against 12.53) allocate that update across module
types to within two percentage points. This is a statement about where SFT on this corpus
puts its mass, not yet about behaviour: whether the trade-off itself replicates on Llama
needs the frontier sweep, which has not been run.

**The learning rate is a magnitude difference.** `sft-2ep-lr5e5` sits at 0.661x the norm
of the 1e-4 runs, and the three 1e-4 seeds cluster at 12.53 / 12.71 / 12.67 -- within
1.5% of each other. The dominant term in the variance budget (+14.23 pp, z = 9.54) is a
34% smaller update, which is consistent with magnitude positioning a model on the trade,
and is not on its own a demonstration of it.

### The frontier across three seeds, and a reproducibility problem underneath it

Three independently trained SFT adapters (seed 0/11/12, matched LR 1e-4, train losses
0.0724-0.0728), each swept at s in {0, 0.25, 0.5, 0.75, 1.0} on the v2 5,000-incident
pool, n = 100 per cell, 16 shards, max-steps 20. `scripts/frontier_report.py` refuses a
partial grid; all 30 cells completed.

| s | in-distribution (mean) | held-out (mean) | held-out spread across seeds |
|---|---|---|---|
| 0.00 (base) | 34.0% | 46.7% | 7 pp |
| 0.25 | 41.3% | 53.7% | 9 pp |
| **0.50** | 58.0% | **64.0%** | 11 pp |
| 0.75 | 56.7% | 36.3% | 7 pp |
| 1.00 (trained) | 70.7% | 38.3% | 3 pp |

**The trade reproduces in all three runs.** In-distribution rises +40.0 / +38.0 / +32.0 pp
from s=0 to s=1 (z = 6.20, 5.82, 4.78) while held-out falls -6.0 / -7.0 / -12.0 pp. The
direction is unanimous, so the trade-off is a property of the procedure rather than of one
training run, which is what the single-run version of this result could not establish. The
held-out *loss* is not individually significant at n = 100 (z = -0.87, -1.00, -1.72); the
published -27.6 pp carried z = -19.06 at n = 2,200, and nothing here contradicts it.

**Attenuation is not monotone, and there is an interior optimum.** Held-out capability
peaks at s = 0.5 in all three seeds (65 / 58 / 69%) and is *higher there than at either
end* -- above the fully trained adapter by ~26 pp and above the untrained base by ~17 pp.
At s = 0.5 in-distribution is 58.0% against base's 34.0%, so the half-scaled adapter beats
the untrained model on **both** axes at once. Two things follow, and the second is a
correction. First, "the sweep does not locate an optimum" is no longer accurate: it locates
one near s = 0.5. Second, the published curve reported s = 0.75 at 50.8%, between its
neighbours; here s = 0.75 sits at 36.3%, at or *below* s = 1.0 in all three seeds. Those
disagree, and the section below is the reason to believe the disagreement rather than
either number.

### Greedy decoding is not reproducible across engine instances, and it cascades

The base arm is the same untrained weights in all three sweeps -- no adapter is served for
it -- run on the same 100 incidents at temperature 0.0 with the same step budget. Only the
vLLM instance differs. It should be identical. It is not:

| pair | outcome flips | identical tool sequence |
|---|---|---|
| seed1 vs realseed11 | 24/100 | 0/100 |
| seed1 vs realseed12 | 21/100 | 1/100 |
| realseed11 vs realseed12 | 21/100 | 0/100 |

**The trajectory essentially never reproduces, and roughly a fifth of resolve/fail outcomes
flip.** The aggregate rates (43 / 47 / 50%) look reassuringly close only because the flips
very nearly cancel.

The mechanism is visible in a single episode. On `nodenotready-batch-pipeline-c76f61b5`,
with an identical `system_prompt_hash`, two runs agree on the first tool call and diverge
on the second (`list_deployments` against `get_service_health`), then end 20 steps and 11
steps later respectively. vLLM's continuous batching makes greedy decoding depend on batch
composition, so one differing token early becomes a different investigation, a different
number of turns, and a different verdict. An agent harness amplifies token-level
nondeterminism into outcome-level nondeterminism in a way single-turn benchmarks do not.

**What this invalidates and what it does not.** It does not invalidate within-sweep
comparisons: every variant in a sweep was served by one engine, and the frontier's shape
is a 17-26 pp effect against a ~7 pp instance spread. It does invalidate *paired,
per-incident* comparison across runs, and it explains a discrepancy that would otherwise
look like a result -- this sweep's base arm scores 43% on the same 100 held-out incidents
where the published pool2 run scored 61%, same model, same prompt hash. That gap is the
engine, not the model, and it is the reason the s = 0.75 disagreement above should not be
read as either curve being wrong.

It also means the run-to-run floor is not one number. The published 0.2 pp is a
*same-configuration, large-n aggregate* floor, and it is real at that scope. At n = 100
across engine instances the aggregate spread is ~7 pp, and the per-incident disagreement
rate is ~22% at any n. Quoting the 0.2 pp figure for a small-n or a per-incident
comparison understates the noise by more than an order of magnitude -- the same class of
mistake as quoting a per-split gap as the run-to-run floor, recorded earlier in this file.

**One secondary observation.** The trained adapter flips less than base (9-16% against
21-24%) and reproduces its tool sequence 21 times in 100 on one pair, where base manages
one. Post-training makes the policy more deterministic as well as more accurate, which is
consistent with the invalid-call collapse: a model emitting a short confident tool call has
far less room to diverge than one emitting a long reasoning block.

**Not yet done.** Every number above is n = 100 per cell. The interior optimum is a 26 pp
effect and survives the instance spread comfortably; the s = 0.75 dip is smaller and does
not. Settling the shape of the curve needs the n = 2,200 treatment the published grid got,
and a determinism check with a fixed batch size would separate batching from any remaining
environment nondeterminism.

## 7. Open items

Stated as open rather than quietly omitted.

- **Several seeds per method arm.** This is the one open item that would change a
  conclusion. Both GRPO variants are now trained and evaluated at n = 2,200 (episode-level
  44.2%, action-level 43.0%, against the matched SFT's 42.1%) and both differences sit
  inside the 3.1 pp floor between independently trained models. Every arm in the grid is a
  single training run, so the grid establishes that post-training works and cannot
  establish which method is best. Resolving it needs several seeds per arm, and the
  launcher defect that made every adapter train at seed 0 was only fixed near the end.
- **Node-layer sim-to-real fidelity.** `src/agentgym/fidelity.py` validates the process
  layer against real processes, real `RLIMIT_AS` kernel enforcement and real HTTP/TCP
  (5/5 agreement on readiness *and* reason). The container/kubelet layer has no real-runtime
  comparison, because no container runtime is available on this cluster, so no fidelity
  claim is made for it.
- **AWS.** S3 code paths are tested against `moto`, and the EKS overlay is validated with
  `helm lint`/`template`. Live EKS/IAM/network verification is pending
  `scripts/verify_aws.py` being run where credentials exist.

---

## Engineering notes worth keeping

Seven problems that cost real time and had non-obvious answers.

**CORRECTED, two harnesses that never disagreed; I compared two different adapters.** Testing whether the
attenuation finding generalises to the SFT adapter produced a contradiction: the arm serving
the *published* SFT adapter did not reproduce the *published* SFT row. On the same 64
incidents, same adapter, temperature 0.0, root cause **0.0% vs 20.3%**, invalid calls
**0.1% vs 16.0%**.

**The first diagnosis was wrong.** The harnesses differ in step budget, 20 published
against 24 in the sweep, and the sweep's failures were hallucinated tools
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
default base. `v2` and `v4` are not near each other, sampled `lora_B` tensors differ by
**121% relative**, none identical.

So the two runs were never measuring the same weights, and the "harnesses disagree"
conclusion was an artifact of my own comparison. The system prompt hash is identical across
both (`591a3ab98e19`), which had already ruled out the prompt-contract class of bug and
should have pointed here sooner.

Two things follow that are worth more than the retraction. **The step-budget change is kept
anyway**: matching the published harness is correct regardless of what explained this, and a
sweep that is not comparable to the published tables is a trap left armed. And **a real
lineage question is now open**, if the published `sft` row is `v4` while DPO and
episode-GRPO were trained from `v2`, then every "DPO over SFT" comparison in this document
spans two different base adapters. Job 5547208 serves both against the same incidents to
settle which produced the published row; nothing downstream is claimed until it lands.

**The frontier dose-response survives this, and not by luck of the numbers.** Every arm of
that sweep shared one harness, and the report script was written from the start to compare
`s` against `s` internally, refusing cross-harness comparison, a rule adopted to prevent a
*population* confound from differing `--limit`. It happened to also block a confound nobody
anticipated. The `s = 1.0` arm's agreement with the published DPO-435 row now reads as
fortunate rather than guaranteed.

**The SFT generalisation test does not survive**, and is withheld rather than reported: a
remedy cannot be evaluated against a baseline that is not the published baseline. `MAX_STEPS`
is now a variable defaulting to 20, and a second defect surfaced alongside, the scaled-GRPO
sweep had been submitted without an `s = 1.0` reference arm, on the assumption the published
row could stand in for one. Both were resubmitted.

**The serial-rollout bug, written twice.** The adapter-scale sweep was submitted with an
unsharded rollout loop and measured 3 episodes in 18 minutes, about 64 hours for the full
sweep, against roughly 4 once sharded eight ways. This is exactly the defect described
above as the DPO sampler's `--shard-total 1`, in the same repository that documents it,
committed by the same author who wrote that section. It is worth keeping rather than
quietly fixing, because the reason it recurs is structural: **a missing shard flag produces
no error, no warning and no wrong number, only slowness**, and slowness reads as a
property of the model or the cluster rather than as a bug in the harness. Correctness bugs
get caught by assertions; throughput bugs need someone to hold a stopwatch against an
expectation. The sweep script now shards by default and prints the episode count per cell,
so a collapsed batch shows up as a number rather than as patience.

**A test that measured machine load.** `test_fault_breaks_then_fixes` applied each fault's
known-correct fix and then advanced a fixed 24 ticks before asserting resolution. It passed
alone and in its own file, and failed roughly once per full-suite run, on a *different*
fault each time. The cause is the hybrid design itself: ticks are deterministic, but pods
are created asynchronously by the real controller-manager, so under load fewer pods exist
per tick and the fixed budget expires mid-convergence. The failing verdicts showed it
plainly, `redis-0` still `Booting`, everything downstream `DependencyUnavailable`, and a
rollout with two live ReplicaSets. The fix was to settle on the same rule the reward uses
(`settle_world()`: not converged while anything is transient) instead of a tick count.
Same class as the rest of this document, the assertion was true, it was just answering
"is the machine fast enough" rather than "does the fix work".

**…and the fix was right but insufficient.** Two clean full-suite runs later it flaked
again, on `MissingSecret`. The actual root cause was one level down: `Episode._settle()`
polls until the real controllers have produced their intended pods, with a 20-second
deadline, and on expiry the `while` loop simply exited and the function returned. Every
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
torch 2.8.0+cu128 fixes it, and then a stale torchvision built against 2.11 breaks
`transformers` imports with `operator torchvision::nms does not exist`.

**The tool catalog never reached the model.** The most dangerous bug in the project, and
it produced no error at all. Tool schemas were passed as the OpenAI `tools` API field,
but the project ships its own chat template, a plain message renderer that ignores the
`tools` variable. So the schemas were silently dropped before the prompt was built, and
the model was being asked to call tools it had never been shown.

Nothing crashed. A fine-tuned model would still have scored reasonably, because it
memorised tool names from demonstrations, which is precisely what makes this dangerous:
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
perturbed schema is visibly perturbed *in the prompt*. Expert results are unaffected (the
scripted policy resolves names through the toolset object, not the prompt) so §5 stands;
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
the **first** closing brace, the one ending `arguments`, so the extracted text is
missing its final brace and fails to parse. Every call with a nested argument object was
dropped, which is nearly every call.

Regex cannot do this: lazy truncates nested objects, greedy swallows everything between
the first `{` and the last `}`. Replaced with a brace-balanced, string-aware scan, plus
support for `<tool_call>` tags and for objects appearing after `</think>`. Of the 19
"unparseable" completions, **10 were recovered as real tool calls**.

The other 9 were genuine model failures, and worth naming because they are distinct
behaviours: narrating an intention without acting, emitting what a tool *would have
returned* instead of calling it, and stopping mid-`<think>`. Two tests guard against
over-correcting, prose is not an action, and a hallucinated result object (no `name`
key) must be rejected rather than dispatched.

**The benchmark measured a request shape production never sends.** vLLM rejected the
first benchmark run: `"auto" tool choice requires --enable-auto-tool-choice`. The
benchmark was still passing `tools` as an API field while the harness had moved the
catalog into the system prompt and stopped sending one. A benchmark whose requests differ
from the system it is meant to characterise is worse than no benchmark.

**The robustness table divided by the wrong denominator.** Retention came out at 1.46 for
SFT under tool renaming, the model apparently solving *more* incidents with renamed tools
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
separate `tool_calls` field, the shape the OpenAI schema suggests, and the tidier of the
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

Guarded now by a test that pins the two renderers to each other, `_assistant_turn` (used
to build inference history) must equal `_assistant_text` (used to build training targets),
or the suite fails. They are the same contract seen from opposite ends, and nothing was
enforcing that.

**A hallucinated namespace returned success.** `list_namespaced_pod` on a namespace that
does not exist returns an empty collection, not an error. A model that invented a namespace
therefore received `[]` with `ok=True`, concluded there were no pods, and had no way to
notice. The gateway now scopes every call to the episode's namespace and, on mismatch,
returns an error that names the correct one, a real responder is scoped by RBAC and would
be told "forbidden" rather than "nothing here".

**`--limit` was applied after sharding.** Eight shards times a limit of 32 is 256 episodes
per cell, not 32, an eightfold overrun across ten cells and two variants. It raised no
error and presented purely as slowness, which is the easiest symptom in the world to
misattribute to the model.

**Two real Kubernetes behaviours, discovered by adding real objects.** Making the
datastore tier StatefulSets with PersistentVolumeClaims broke six fault families, and
both causes turned out to be genuine Kubernetes semantics rather than simulator bugs.

*StatefulSet rollouts do not self-heal.* A StatefulSet's RollingUpdate will not replace a
pod that is not Running and Ready. Once its pod is wedged in `ImagePullBackOff`, patching
the template changes `updateRevision` and nothing else, the pod keeps the old image
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
declare no CPU *limit*. The default is not clamped to the request, admission rejects the
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
and serving get separate venvs, which is the right architecture anyway. vLLM 0.11 also
pulls transformers v5 by default and dies on `Qwen2Tokenizer has no attribute
all_special_tokens_extended`, so the serving venv pins `<5` too.
