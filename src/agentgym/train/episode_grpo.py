"""Episode-level GRPO: group-relative advantages over whole incidents.

`grpo_train.py` optimises one *action* at a time, because TRL's GRPOTrainer samples a group
of completions for a single prompt. That is a genuine verifiable-reward objective, but it
cannot express the thing that actually makes agent RL hard: an episode is a sequence of
twenty-odd tool calls whose value is only known at the end, and the credit for a resolved
incident belongs to the whole trajectory, not to the last call in it.

This module closes that gap. The group is G full episodes rolled out against the *same*
incident, the reward is the episode's verifiable reward -- the same number the evaluation
reports, computed from cluster state -- and the advantage is group-relative:

    A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

which is the defining move of GRPO: the group mean replaces a learned value function, so
there is no critic to train and no value-head bias to fight. That advantage is then applied
to every assistant token in episode i, which is what makes it episode-level credit
assignment rather than per-action shaping.

Loss for one episode, with tokens masked to the assistant's own turns:

    L_i = -A_i * (1 / |T_i|) * sum_{t in T_i} log pi_theta(a_t | s_<t)  +  beta * KL_i

Two details that matter more than they look:

* **Length normalisation.** Dividing by the number of assistant tokens stops the objective
  from preferring long episodes when the advantage is positive and short ones when it is
  negative -- a bias that shows up as the agent padding its reasoning to farm reward.
* **Degenerate groups are dropped, not zeroed.** If every rollout of an incident earns the
  same reward, the advantage is 0/0. Those groups carry no signal, and keeping them as
  zeros silently shrinks the effective batch while the step count says otherwise -- the
  same fail-open shape as the sampling bug in reports/EXPERIMENTS.md. They are removed and
  counted.

Rollouts come from the existing sharded sampler (the one `train_dpo.sbatch` uses), so this
trains on exactly the trajectory format the harness produces and the evaluation reads.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TEMPLATE_PATH = Path(__file__).parent / "agentgym_chat_template.jinja"

# Groups whose rewards are this close together carry no usable preference signal.
MIN_REWARD_STD = 1e-6


@dataclass
class GroupStats:
    """Why each incident group was kept or dropped. Reported, never inferred."""
    groups_seen: int = 0
    groups_kept: int = 0
    dropped_too_small: int = 0
    dropped_no_reward_spread: int = 0
    episodes_kept: int = 0
    mean_group_std: float = 0.0
    mean_abs_advantage: float = 0.0
    reward_range: tuple[float, float] = (0.0, 0.0)
    per_incident_group_size: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["reward_range"] = list(self.reward_range)
        d.pop("per_incident_group_size", None)
        return d


def group_advantages(
    trajectories: list[Any],
    *,
    min_group: int = 2,
) -> tuple[list[tuple[Any, float]], GroupStats]:
    """Assign each episode a group-relative advantage against its own incident.

    Returns `(episode, advantage)` pairs and the accounting for what was dropped.
    """
    by_incident: dict[str, list[Any]] = defaultdict(list)
    for t in trajectories:
        by_incident[t.incident_id].append(t)

    stats = GroupStats()
    out: list[tuple[Any, float]] = []
    stds: list[float] = []
    advs: list[float] = []
    lo, hi = float("inf"), float("-inf")

    for incident_id, group in by_incident.items():
        stats.groups_seen += 1
        stats.per_incident_group_size[incident_id] = len(group)
        if len(group) < min_group:
            stats.dropped_too_small += 1
            continue

        rewards = [float(t.reward) for t in group]
        lo, hi = min(lo, *rewards), max(hi, *rewards)
        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = var ** 0.5
        if std < MIN_REWARD_STD:
            # Every rollout scored identically: the group is uninformative. Dropping it is
            # correct; keeping it as zero advantage would quietly dilute every batch.
            stats.dropped_no_reward_spread += 1
            continue

        stds.append(std)
        for t, r in zip(group, rewards, strict=True):
            a = (r - mean) / (std + 1e-8)
            out.append((t, a))
            advs.append(abs(a))
        stats.groups_kept += 1
        stats.episodes_kept += len(group)

    stats.mean_group_std = round(sum(stds) / len(stds), 4) if stds else 0.0
    stats.mean_abs_advantage = round(sum(advs) / len(advs), 4) if advs else 0.0
    stats.reward_range = (round(lo, 3), round(hi, 3)) if out else (0.0, 0.0)
    return out, stats


def build_examples(scored: list[tuple[Any, float]]) -> list[dict[str, Any]]:
    """Render each episode into the *same* chat format the harness and SFT use.

    Reusing `curate._messages_for` is deliberate. The single most costly defect in this
    project was a mismatch between how history was rendered at training time and at
    inference time; the fix is not to re-derive the format here but to call the one
    function that both paths already agree on.
    """
    from agentgym.data.curate import _messages_for

    return [
        {
            "incident_id": t.incident_id,
            "fault": t.fault,
            "reward": float(t.reward),
            "advantage": advantage,
            "messages": _messages_for(t),
        }
        for t, advantage in scored
    ]


def _collate(examples, tokenizer, max_len: int, counters: dict[str, int] | None = None):
    """Tokenise with a loss mask over the assistant's own tokens only.

    Truncation is counted, not silent. An episode cut at `max_len` keeps its group-relative
    advantage but loses its later turns -- including, often, the one that resolved the
    incident -- so the gradient says "this prefix earned +A" about a prefix that did not.
    It trains fine and reports nothing, so the count is surfaced instead.
    """
    import torch

    counters = counters if counters is not None else {}
    input_ids, masks, advantages = [], [], []
    for ex in examples:
        enc = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            max_length=max_len,
            truncation=True,
        )
        ids = enc["input_ids"]
        if len(ids) >= max_len:
            counters["truncated"] = counters.get("truncated", 0) + 1
        assistant_mask = enc.get("assistant_masks")
        if assistant_mask is None or not any(assistant_mask):
            # No assistant tokens survived: the example contributes nothing but would still
            # count toward the batch. Skip it rather than divide by zero later.
            counters["no_assistant_tokens"] = counters.get("no_assistant_tokens", 0) + 1
            continue
        input_ids.append(torch.tensor(ids))
        masks.append(torch.tensor(assistant_mask))
        advantages.append(float(ex["advantage"]))

    if not input_ids:
        return None
    pad = tokenizer.pad_token_id
    width = max(len(x) for x in input_ids)
    batch_ids = torch.full((len(input_ids), width), pad, dtype=torch.long)
    batch_mask = torch.zeros((len(input_ids), width), dtype=torch.long)
    batch_attn = torch.zeros((len(input_ids), width), dtype=torch.long)
    for i, (ids, m) in enumerate(zip(input_ids, masks, strict=True)):
        batch_ids[i, : len(ids)] = ids
        batch_mask[i, : len(m)] = m
        batch_attn[i, : len(ids)] = 1
    return {
        "input_ids": batch_ids,
        "assistant_mask": batch_mask,
        "attention_mask": batch_attn,
        "advantages": torch.tensor(advantages, dtype=torch.float32),
    }


def _token_logprobs(model, ids, attn, targets):
    """Log pi(target_t | prefix) for every position, shifted.

    Uses `cross_entropy` rather than `log_softmax(...).gather(...)`. The obvious spelling
    materialises a `[B, T, V]` float32 tensor -- with Qwen3's 151k vocabulary and a 12k
    context that is ~7.5 GB for the forward and as much again held for the backward, which
    OOMs a 44 GB card on batch size 1. `cross_entropy` is fused: it recomputes softmax in
    the backward instead of storing it, and returns exactly the per-token negative log
    likelihood we want.
    """
    import torch.nn.functional as F

    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    # transpose to [B, V, T] is a view; cross_entropy wants class-dim second.
    return -F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")


def grpo_loss(model, batch, *, beta: float, use_ref: bool = True):
    """Length-normalised, advantage-weighted NLL over assistant tokens, plus KL.

    The reference policy is this same model with the LoRA adapter disabled, so the KL is
    measured against the policy being improved without holding a second copy of the weights
    in memory. The policy forward must happen with the adapter *enabled* -- computing it
    inside the `disable_adapter()` context would train the adapter on gradients taken
    without it, which converges to something, just not to the objective written down here.
    """
    import torch

    ids, attn = batch["input_ids"], batch["attention_mask"]
    mask, adv = batch["assistant_mask"], batch["advantages"]
    targets = ids[:, 1:]
    tok_mask = mask[:, 1:].float()

    chosen = _token_logprobs(model, ids, attn, targets)

    # Per-episode mean over assistant tokens: without this the objective scales with
    # episode length and the agent learns to pad rather than to solve.
    denom = tok_mask.sum(dim=1).clamp(min=1.0)
    seq_logprob = (chosen * tok_mask).sum(dim=1) / denom
    pg = -(adv.to(seq_logprob.device) * seq_logprob).mean()

    kl = torch.zeros((), device=pg.device)
    if use_ref and beta > 0:
        with torch.no_grad(), model.disable_adapter():
            ref_chosen = _token_logprobs(model, ids, attn, targets)
        # k3 estimator: non-negative and lower variance than the raw log-ratio.
        diff = ref_chosen - chosen
        per_tok = torch.exp(diff) - diff - 1.0
        kl = ((per_tok * tok_mask).sum(dim=1) / denom).mean()

    return pg + beta * kl, {"pg_loss": float(pg.detach()), "kl": float(kl.detach())}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", type=Path, required=True,
                   help="Rollouts with >=2 samples per incident (see train_dpo.sbatch).")
    p.add_argument("--model", default=os.environ.get("AGENTGYM_BASE_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--sft-adapter", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--beta", type=float, default=0.04, help="KL coefficient to the ref policy.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    # Measured over 250 sampled episodes: median 6,282 tokens, p95 8,225, max 13,064.
    # 12,288 truncates 1.2% of episodes; 8,192 would truncate 5.6%.
    p.add_argument("--max-length", type=int, default=12288)
    p.add_argument("--min-group", type=int, default=2)
    p.add_argument("--dry-run", action="store_true",
                   help="Compute advantages and report group statistics, then stop.")
    return p.parse_args()


def main() -> None:
    from agentgym.data import trajectory as T

    args = parse_args()
    src = args.trajectories
    files = sorted(src.rglob("*.jsonl")) if src.is_dir() else [src]
    trajectories = [t for f in files for t in T.read_jsonl(f)]
    if not trajectories:
        raise SystemExit(f"no trajectories under {src}")

    scored, stats = group_advantages(trajectories, min_group=args.min_group)
    print(json.dumps({"trajectories": len(trajectories), **stats.as_dict()}, indent=2))
    if not scored:
        raise SystemExit(
            "no usable groups: every incident either had fewer than "
            f"{args.min_group} rollouts or zero reward spread. Sample more attempts per "
            "incident (--samples) or raise the rollout temperature.")

    examples = build_examples(scored)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "group_stats.json").write_text(
        json.dumps({"trajectories": len(trajectories), **stats.as_dict()}, indent=2))

    if args.dry_run:
        print(f"dry run: {len(examples)} episodes across {stats.groups_kept} groups")
        return

    _train(args, examples)


def _train(args, examples: list[dict[str, Any]]) -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = TEMPLATE_PATH.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    if args.sft_adapter:
        # Start from the SFT policy and keep a frozen copy of it as the KL reference, so
        # the penalty is measured against the policy we are actually improving.
        model = PeftModel.from_pretrained(model, str(args.sft_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))

    # Episodes are ~6k tokens on average and every one needs a policy pass and a reference
    # pass, so activation memory, not parameter memory, is the binding constraint here.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = int(len(examples) * args.epochs) // (args.batch_size * args.grad_accum)
    print(f"episodes={len(examples)} optimizer_steps={max(1, steps)}")

    log: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    step = 0
    for epoch in range(max(1, int(args.epochs))):
        for i in range(0, len(examples), args.batch_size):
            batch = _collate(examples[i:i + args.batch_size], tokenizer,
                             args.max_length, counters)
            if batch is None:
                continue
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss, parts = grpo_loss(model, batch, beta=args.beta)
            (loss / args.grad_accum).backward()
            if (i // args.batch_size + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"epoch {epoch} step {step} loss={float(loss):.4f} "
                          f"pg={parts['pg_loss']:.4f} kl={parts['kl']:.4f}", flush=True)
                    log.append({"step": step, "loss": float(loss), **parts})

    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    seen = len(examples) * max(1, int(args.epochs))
    counters["episodes_seen"] = seen
    counters["truncation_rate"] = round(counters.get("truncated", 0) / max(1, seen), 4)
    (args.output_dir / "train_log.json").write_text(
        json.dumps({"steps": log, "counters": counters}, indent=2))
    print(json.dumps(counters, indent=2))
    if counters["truncation_rate"] > 0.25:
        print(f"WARNING: {counters['truncation_rate']:.1%} of episodes hit the "
              f"{args.max_length}-token cap. Their advantage is applied to a prefix that "
              "may exclude the turn that resolved the incident; raise --max-length.")
    print(f"episode-level GRPO complete -> {args.output_dir}")


if __name__ == "__main__":
    main()
