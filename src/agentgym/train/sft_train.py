"""QLoRA supervised fine-tuning on verified agent trajectories.

Loss is masked to assistant tokens only. That matters more than it sounds: the tool
outputs in these conversations are long, and training on them would teach the model to
generate plausible-looking cluster state instead of querying for it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "agentgym_chat_template.jinja"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA SFT for MCP-AgentGym")
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--eval-file", type=Path, default=None)
    p.add_argument("--model", default=os.environ.get("AGENTGYM_BASE_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-4bit", action="store_true", help="Full bf16 LoRA instead of QLoRA.")
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--run-name", default="sft")
    return p.parse_args()


def _reporters() -> list[str]:
    """Enable MLflow only if it is importable *and* its backend actually accepts a run.

    The file store refuses to initialize unless MLFLOW_ALLOW_FILE_STORE is set, and it
    raises from inside a Trainer callback -- i.e. after the model is loaded and the
    dataset tokenized. Probing here turns that into a warning instead.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return []
    try:
        import mlflow

        mlflow.set_experiment("agentgym")
        with mlflow.start_run(run_name="_probe"):
            mlflow.log_metric("probe", 1.0)
        return ["mlflow"]
    except Exception as exc:  # noqa: BLE001 - any failure means "just don't track"
        print(f"[warn] MLflow disabled ({type(exc).__name__}: {exc})", flush=True)
        return []


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    def load(path: Path) -> Dataset:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if args.max_examples:
            rows = rows[: args.max_examples]
        # SFTTrainer wants the conversational schema: a `messages` column and nothing
        # else it might mistake for text.
        return Dataset.from_list([{"messages": r["messages"]} for r in rows])

    train_ds = load(args.train_file)
    eval_ds = load(args.eval_file) if args.eval_file and args.eval_file.exists() else None
    print(f"train examples: {len(train_ds)}"
          + (f" | eval: {len(eval_ds)}" if eval_ds else ""), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # The template used here is the one vLLM is given at serving time, so the model is
    # never asked to generalise across a prompt-format change.
    tokenizer.chat_template = TEMPLATE_PATH.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds else "no",
        bf16=True,
        max_length=args.max_length,
        # Only assistant tokens contribute to the loss.
        assistant_only_loss=True,
        # Packing would merge separate incidents into one sequence and let attention run
        # across episode boundaries; each conversation stays its own example.
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        seed=args.seed,
        # Experiment tracking must never be the reason a six-hour training run dies.
        report_to=_reporters(),
        run_name=args.run_name,
        dataset_num_proc=4,
    )

    trainer = SFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )

    result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = dict(result.metrics)
    if eval_ds:
        metrics.update(trainer.evaluate())
    (args.output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"adapter written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
