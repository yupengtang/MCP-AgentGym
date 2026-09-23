"""DPO on preference pairs mined from verified rollouts.

Pairs come from multiple attempts at the *same* incident, ranked by the executable
reward. Because both sides of a pair faced an identical cluster, the preference isolates
behaviour -- which tool was chosen, whether the fix was validated, whether anything
unsafe happened -- rather than scenario difficulty.

Starts from the SFT adapter, so DPO refines a policy that already knows the tool
protocol instead of teaching protocol and preference at once.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentgym.train.chat_template import template_for


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO for MCP-AgentGym")
    p.add_argument("--pairs-file", type=Path, required=True)
    p.add_argument("--model", default=os.environ.get("AGENTGYM_BASE_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--sft-adapter", type=Path, default=None,
                   help="SFT adapter to continue training from.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=6144)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="dpo")
    return p.parse_args()


def _flatten(messages: list[dict]) -> str:
    """Render a message list into the single completion string DPO compares."""
    return "\n\n".join(
        m["content"] for m in messages if m.get("role") == "assistant"
    )


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    rows = [json.loads(line) for line in args.pairs_file.read_text().splitlines()
            if line.strip()]
    if not rows:
        raise SystemExit(f"no preference pairs in {args.pairs_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = template_for(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render_prompt(prompt_messages: list[dict]) -> str:
        return tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

    dataset = Dataset.from_list([{
        "prompt": render_prompt(r["prompt"]),
        "chosen": _flatten(r["chosen"]),
        "rejected": _flatten(r["rejected"]),
    } for r in rows])
    print(f"preference pairs: {len(dataset)}", flush=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quantization_config,
        dtype=torch.bfloat16, device_map={"": 0},
    )
    if args.sft_adapter is not None:
        from peft import PeftModel

        # Continue from SFT and keep training the same adapter, so the DPO result is a
        # refinement of the SFT policy rather than a second, independent adapter.
        model = PeftModel.from_pretrained(model, str(args.sft_adapter), is_trainable=True)
        peft_config = None
        print(f"continuing from SFT adapter {args.sft_adapter}", flush=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        # TRL 1.9 collapsed the prompt/completion budgets into one `max_length`;
        # `max_prompt_length` and `model_adapter_name` no longer exist.
        max_length=args.max_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to=_reporters(),
        run_name=args.run_name,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
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
