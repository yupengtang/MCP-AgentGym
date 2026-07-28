"""GRPO with verifiable action rewards.

**Scope, stated up front.** TRL's GRPO is single-turn: it samples a group of completions
for one prompt and ranks them. An incident is multi-turn, so this trains at the level of
*one action* rather than one episode: the prompt is the conversation up to some step, the
completion is the next tool call, and the reward is computed by the same machinery the
gateway uses to accept or reject a call.

That is a real, verifiable reward -- every term is decided by a parser, a JSON Schema, and
the incident's safety policy, never by a model -- and it targets exactly the failure modes
the evaluation measures: unparseable output, unknown tools, invalid arguments, and unsafe
actions. What it does *not* do is credit-assign across a whole episode; a full multi-turn
GRPO would need a custom rollout loop over live clusters, which is future work and is
labelled as such in the report.

Reward per completion:

    +1.0  parses into a tool call at all
    +1.0  names a tool that exists on the presented surface
    +1.0  arguments satisfy that tool's JSON Schema
    +0.5  matches the action the expert took at this step
    -3.5  is an action the incident's safety policy forbids

The unsafe penalty is sized so that a *well-formed* forbidden action scores strictly
below emitting nothing parseable (3.0 - 3.5 = -0.5 against 0.0). A smaller penalty would
leave the model better off confidently deleting a pod than saying nothing, which is
exactly the wrong lesson for an agent with production credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

TEMPLATE_PATH = Path(__file__).parent / "agentgym_chat_template.jinja"

# Weights for the reward terms above.
W_PARSED, W_KNOWN, W_VALID, W_EXPERT, W_UNSAFE = 1.0, 1.0, 1.0, 0.5, -3.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO for MCP-AgentGym")
    p.add_argument("--trajectories", type=Path, required=True,
                   help="Directory or file of verified trajectories to build prompts from.")
    p.add_argument("--model", default=os.environ.get("AGENTGYM_BASE_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--sft-adapter", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--num-generations", type=int, default=8,
                   help="Group size: completions sampled per prompt.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-prompt-length", type=int, default=6144)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.02, help="KL coefficient.")
    p.add_argument("--max-prompts", type=int, default=2000)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="grpo")
    return p.parse_args()


# --- reward ------------------------------------------------------------------

def make_reward_fn(tokenizer):
    """Build the reward function TRL calls with a batch of completions."""
    from jsonschema import Draft202012Validator, ValidationError

    from agentgym.harness.llm_policy import LLMPolicy
    from agentgym.mcp.tools import REGISTRY

    parser = LLMPolicy(backend=None)  # type: ignore[arg-type]  # only _to_action is used
    validators = {
        name: Draft202012Validator(tool.parameters) for name, tool in REGISTRY.items()
    }

    def reward_actions(completions: list[str], **kwargs: Any) -> list[float]:
        expert_tools = kwargs.get("expert_tool") or [None] * len(completions)
        unsafe_rules = kwargs.get("unsafe") or [()] * len(completions)
        rewards: list[float] = []

        for text, expert_tool, unsafe in zip(completions, expert_tools, unsafe_rules,
                                             strict=False):
            action = parser._to_action(text, [])
            name = getattr(action, "name", "submit_diagnosis")
            args = getattr(action, "args", None)
            if args is None:  # Finish carries root_cause/summary instead
                args = {"root_cause": getattr(action, "root_cause", ""),
                        "summary": getattr(action, "summary", "")}

            score = 0.0
            if name != "__unparseable__":
                score += W_PARSED
            tool = REGISTRY.get(name)
            if tool is not None:
                score += W_KNOWN
                try:
                    validators[name].validate(args)
                    score += W_VALID
                except ValidationError:
                    pass
                if expert_tool and name == expert_tool:
                    score += W_EXPERT
                for rule in (unsafe or ()):
                    rule_name, _, qualifier = rule.partition(":")
                    if rule_name == name and (
                        not qualifier or str(args.get("replicas")) == qualifier
                    ):
                        score += W_UNSAFE
                        break
            rewards.append(score)
        return rewards

    reward_actions.__name__ = "verifiable_action_reward"
    return reward_actions


# --- dataset -----------------------------------------------------------------

def build_prompts(trajectory_path: Path, tokenizer, *, max_prompts: int,
                  seed: int) -> list[dict[str, Any]]:
    """Turn verified trajectories into (prompt, expert action) pairs.

    Each pair is a prefix of a real episode, so the model is optimised on states it will
    actually encounter rather than on synthetic conversations.
    """
    from agentgym.data import trajectory as traj_mod
    from agentgym.data.curate import _messages_for
    from agentgym.incidents.faults import FAULTS_BY_NAME

    files = sorted(trajectory_path.glob("*.jsonl")) if trajectory_path.is_dir() \
        else [trajectory_path]
    trajectories = [t for f in files for t in traj_mod.read_jsonl(f)]
    # Only clean, in-distribution, successful episodes: a prefix from a failed run would
    # teach the model to continue down a path that did not work.
    trajectories = [t for t in trajectories
                    if t.split == "in_distribution" and t.success and t.clean
                    and t.perturbation == "clean"]

    rows: list[dict[str, Any]] = []
    for t in trajectories:
        messages = _messages_for(t)
        fault = FAULTS_BY_NAME.get(t.fault.split("+")[0])
        unsafe = list(fault.unsafe) if fault else []
        # messages alternates assistant/user after the two-message preamble; each
        # assistant turn is a decision point.
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant" or i < 2:
                continue
            prefix = messages[:i]
            step = t.steps[(i - 2) // 2] if (i - 2) // 2 < len(t.steps) else None
            rows.append({
                "prompt": tokenizer.apply_chat_template(
                    prefix, tokenize=False, add_generation_prompt=True),
                "expert_tool": (step.presented_tool or step.tool) if step else None,
                "unsafe": unsafe,
            })

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:max_prompts]


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = TEMPLATE_PATH.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = build_prompts(args.trajectories, tokenizer,
                         max_prompts=args.max_prompts, seed=args.seed)
    if not rows:
        raise SystemExit(f"no usable prompts under {args.trajectories}")
    dataset = Dataset.from_list(rows)
    print(f"GRPO prompts: {len(dataset)}", flush=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quantization_config,
        dtype=torch.bfloat16, device_map={"": 0},
    )
    peft_config = None
    if args.sft_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.sft_adapter), is_trainable=True)
        print(f"continuing from SFT adapter {args.sft_adapter}", flush=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

    config = GRPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to=_reporters(),
        run_name=args.run_name,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[make_reward_fn(tokenizer)],
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(dict(result.metrics), indent=2))
    print(json.dumps(dict(result.metrics), indent=2), flush=True)
    print(f"adapter written to {args.output_dir}", flush=True)


def _reporters() -> list[str]:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return []
    try:
        import mlflow

        mlflow.set_experiment("agentgym")
        with mlflow.start_run(run_name="_probe"):
            mlflow.log_metric("probe", 1.0)
        return ["mlflow"]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] MLflow disabled ({type(exc).__name__}: {exc})", flush=True)
        return []


if __name__ == "__main__":
    main()
