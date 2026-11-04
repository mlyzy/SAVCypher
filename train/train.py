#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from paths import ensure_sys_path

ensure_sys_path()

from train.data import SIRCMTLDataset, TaskAwareDataCollator
from train.mtl_lora.utils import save_pretrain
from train.qwen_mlora import (
    DEFAULT_TARGET_MODULES,
    LANGUAGE_MODEL_PREFIX,
    inject_mlora,
    load_adapter_checkpoint,
)

_WORKSPACE_MODEL = REPO_ROOT.parent / "models/Qwen__Qwen3.6-35B-A3B"
_LOCAL_MODEL = REPO_ROOT / "models/Qwen__Qwen3.6-35B-A3B"
DEFAULT_MODEL = _LOCAL_MODEL if _LOCAL_MODEL.exists() else _WORKSPACE_MODEL
DEFAULT_DATA = REPO_ROOT / "sirc_data/mtl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_DATA / "train.jsonl")
    parser.add_argument("--dev-file", type=Path, default=DEFAULT_DATA / "dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/qwen36_sirc_mlora")
    parser.add_argument("--deepspeed", type=Path, default=Path(__file__).resolve().parent / "config/ds3.json")
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length. 1024 fits Qwen3.6-35B training on 2x48GB.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override epochs with a fixed optimizer-step count; useful for smoke tests.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--eval-strategy",
        choices=("no", "epoch", "steps"),
        default="epoch",
        help="Trainer evaluation schedule; use 'no' for quick smoke tests.",
    )
    parser.add_argument(
        "--adapter-save-steps",
        type=int,
        default=50,
        help="Save a resumable adapter snapshot every N optimizer steps; 0 disables it.",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--b-experts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated exact module suffixes.",
    )
    parser.add_argument("--module-prefix", default=LANGUAGE_MODEL_PREFIX)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=-1,
        help="Local process rank injected by torchrun/DeepSpeed.",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def make_adapter_checkpoint_callback(model, output_dir: Path, save_steps: int):
    """Build a callback that saves small ZeRO-safe adapter snapshots."""
    from transformers import TrainerCallback

    class AdapterCheckpointCallback(TrainerCallback):
        @staticmethod
        def _save(destination: Path) -> None:
            save_pretrain(model, str(destination), prefix=["lora"])

        def on_step_end(self, args, state, control, **kwargs):
            if (
                save_steps > 0
                and state.global_step > 0
                and state.global_step % save_steps == 0
            ):
                destination = (
                    output_dir
                    / "adapter-checkpoints"
                    / f"step-{state.global_step:06d}"
                )
                self._save(destination)
            return control

    return AdapterCheckpointCallback()


def resolve_deepspeed_config(args: argparse.Namespace) -> dict:
    config = json.loads(args.deepspeed.read_text(encoding="utf-8"))
    config["train_micro_batch_size_per_gpu"] = args.batch_size
    config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    # Let each DeepSpeed initialization derive the global batch size from its
    # actual initialized world size. Model-load ZeRO init occurs before Trainer
    # initializes distributed state, so hard-coding WORLD_SIZE is incorrect.
    config.pop("train_batch_size", None)
    return config


def load_model_and_tokenizer(args: argparse.Namespace):
    try:
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from transformers.integrations import HfDeepSpeedConfig
    except ImportError as exc:
        raise RuntimeError(
            "Install a Qwen3.6-compatible transformers build and DeepSpeed first"
        ) from exc

    # HfDeepSpeedConfig must exist before from_pretrained so ZeRO-3 partitions
    # the 67GB checkpoint while loading rather than replicating it per process.
    # DeepSpeed's standalone ZeRO initialization sees this config before
    # TrainingArguments has a chance to resolve its "auto" placeholders.
    ds_config = resolve_deepspeed_config(args)
    args.resolved_deepspeed_config = ds_config
    zero_helper = HfDeepSpeedConfig(ds_config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    wrapped = inject_mlora(
        model,
        target_modules=tuple(
            item.strip() for item in args.target_modules.split(",") if item.strip()
        ),
        module_prefix=args.module_prefix,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        task_count=2,
        b_experts=args.b_experts,
        temperature=args.temperature,
    )
    wrapped._zero3_config_helper = zero_helper
    wrapped._mlora_adapter_config.update(
        {
            "base_model": str(args.base_model),
            "task_names": {
                "0": "text_to_sirc",
                "1": "sirc_to_cypher",
            },
        }
    )
    if args.adapter_checkpoint:
        load_adapter_checkpoint(wrapped, args.adapter_checkpoint)
    return wrapped, tokenizer


def main() -> None:
    args = parse_args()
    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("transformers is required for training") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(args)
    train_dataset = SIRCMTLDataset(
        args.train_file, tokenizer, args.max_length
    )
    dev_dataset = SIRCMTLDataset(args.dev_file, tokenizer, args.max_length)
    collator = TaskAwareDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy="no",
        bf16=True,
        tf32=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        deepspeed=args.resolved_deepspeed_config,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        callbacks=[
            make_adapter_checkpoint_callback(
                model, args.output_dir, args.adapter_save_steps
            )
        ],
    )
    trainer.train()
    final_destination = (
        args.output_dir
        / "adapter-checkpoints"
        / f"step-{trainer.state.global_step:06d}"
    )
    already_saved = (
        args.adapter_save_steps > 0
        and trainer.state.global_step % args.adapter_save_steps == 0
    )
    if not already_saved:
        save_pretrain(model, str(final_destination), prefix=["lora"])
    if int(os.environ.get("RANK", "0")) == 0:
        tokenizer.save_pretrained(final_destination / "checkpoint")
        (args.output_dir / "train_arguments.json").write_text(
            json.dumps(vars(args), default=str, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

