#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from paths import ensure_sys_path

ensure_sys_path()

from infer.sirc_syntax import check_sirc_syntax, retry_feedback
from train.data import SYSTEM_PROMPTS, make_task_input, render_prompt
from train.qwen_mlora import (
    inject_mlora,
    load_adapter_checkpoint,
    read_adapter_config,
)
from train.schema_format import format_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--schema-file", type=Path)
    parser.add_argument(
        "--use-gold-ir",
        action="store_true",
        help="Use each input record's SIRC field to diagnose task 1.",
    )
    parser.add_argument("--max-ir-tokens", type=int, default=1024)
    parser.add_argument("--max-cypher-tokens", type=int, default=512)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=1024,
        help="Fail if a complete prompt exceeds this token budget.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-sirc-attempts",
        type=int,
        default=3,
        help="First-stage generations including the initial greedy decode. "
        "Invalid SIRC is regenerated up to this many times.",
    )
    parser.add_argument(
        "--sirc-retry-temperature",
        type=float,
        default=0.8,
        help="Sampling temperature used after a SIRC syntax failure.",
    )
    return parser.parse_args()


def clean_generation(text: str) -> str:
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    text = text.strip()
    fence = re.fullmatch(r"```(?:cypher|text)?\s*(.*?)```", text, re.DOTALL | re.I)
    return fence.group(1).strip() if fence else text


def load_model(args: argparse.Namespace):
    try:
        from transformers import AutoModelForImageTextToText, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install a Qwen3.6-compatible transformers build first"
        ) from exc

    config = read_adapter_config(args.checkpoint)
    base_model = args.base_model or Path(config["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    wrapped = inject_mlora(
        model,
        target_modules=tuple(config["target_modules"]),
        module_prefix=config.get("module_prefix"),
        rank=int(config["r"]),
        alpha=int(config["lora_alpha"]),
        dropout=float(config["lora_dropout"]),
        task_count=int(config["lambda_num"]),
        b_experts=int(config["B_num"]),
        temperature=float(config["B_scale"]),
    )
    load_adapter_checkpoint(wrapped, args.checkpoint)
    wrapped.eval()
    return wrapped, tokenizer


def generate_task(
    model,
    tokenizer,
    *,
    task_id: int,
    task_input: str,
    max_new_tokens: int,
    max_input_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 0.8,
) -> str:
    record = {
        "task_id": task_id,
        "system": SYSTEM_PROMPTS[task_id],
        "input": task_input,
        "output": "_",
    }
    prompt = render_prompt(tokenizer, record)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    )
    prompt_length = int(encoded["input_ids"].shape[1])
    if prompt_length > max_input_tokens:
        raise ValueError(
            f"Task {task_id} prompt requires {prompt_length} tokens, exceeding "
            f"max_input_tokens={max_input_tokens}; schema truncation is disabled"
        )
    input_device = next(model.base_model.parameters()).device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    generate_kwargs: Dict[str, Any] = {
        "task_id": torch.tensor([task_id], device=input_device),
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    output = model.generate(**encoded, **generate_kwargs)
    generated = output[0, encoded["input_ids"].shape[1] :]
    return clean_generation(
        tokenizer.decode(generated, skip_special_tokens=True)
    )


def generate_sirc_with_syntax_check(
    model,
    tokenizer,
    *,
    question: str,
    schema: str,
    max_new_tokens: int,
    max_input_tokens: int = 1024,
    max_attempts: int = 3,
    retry_temperature: float = 0.8,
) -> Dict[str, Any]:
    """Generate SIRC, parse it, and resample if the syntax is invalid."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    base_input = make_task_input(0, schema=schema, question=question)
    attempts: List[Dict[str, Any]] = []
    chosen = ""
    chosen_check: Dict[str, Any] = {"valid": False, "errors": ["SIRCSyntaxError: empty SIRC"]}
    feedback = ""
    for index in range(max_attempts):
        task_input = (
            f"{base_input}\n\n{feedback}" if feedback else base_input
        )
        text = generate_task(
            model,
            tokenizer,
            task_id=0,
            task_input=task_input,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            do_sample=index > 0,
            temperature=retry_temperature,
        )
        check = check_sirc_syntax(text)
        attempts.append(
            {
                "attempt": index + 1,
                "sampled": index > 0,
                "sirc": text,
                "valid": check["valid"],
                "errors": list(check["errors"]),
            }
        )
        chosen = text
        chosen_check = check
        if check["valid"]:
            break
        feedback = retry_feedback(text, check["errors"])
    return {
        "sirc": chosen,
        "valid": bool(chosen_check["valid"]),
        "errors": list(chosen_check["errors"]),
        "attempts": attempts,
        "attempt_count": len(attempts),
    }


def normalize_schema(value: Any, question: str = "") -> str:
    return format_schema(value, question=question)


def read_inputs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.input_file:
        with args.input_file.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if not args.question or not args.schema_file:
        raise ValueError(
            "Provide --input-file, or both --question and --schema-file"
        )
    return [
        {
            "id": "interactive",
            "question": args.question,
            "schema": args.schema_file.read_text(encoding="utf-8"),
        }
    ]


def run_two_stage(
    model,
    tokenizer,
    source: Dict[str, Any],
    *,
    use_gold_ir: bool = False,
    max_ir_tokens: int = 1024,
    max_cypher_tokens: int = 512,
    max_input_tokens: int = 1024,
    max_sirc_attempts: int = 3,
    sirc_retry_temperature: float = 0.8,
) -> Dict[str, Any]:
    schema = normalize_schema(source["schema"], str(source["question"]))
    if use_gold_ir:
        natcypher = str(source.get("sirc") or source.get("natcypher") or "").strip()
        syntax = check_sirc_syntax(natcypher)
        sirc_meta = {
            "sirc_syntax_valid": syntax["valid"],
            "sirc_syntax_errors": syntax["errors"],
            "sirc_attempt_count": 0,
            "sirc_attempts": [],
        }
    else:
        generated = generate_sirc_with_syntax_check(
            model,
            tokenizer,
            question=str(source["question"]),
            schema=schema,
            max_new_tokens=max_ir_tokens,
            max_input_tokens=max_input_tokens,
            max_attempts=max_sirc_attempts,
            retry_temperature=sirc_retry_temperature,
        )
        natcypher = generated["sirc"]
        sirc_meta = {
            "sirc_syntax_valid": generated["valid"],
            "sirc_syntax_errors": generated["errors"],
            "sirc_attempt_count": generated["attempt_count"],
            "sirc_attempts": generated["attempts"],
        }
    cypher = generate_task(
        model,
        tokenizer,
        task_id=1,
        task_input=make_task_input(
            1, schema=schema, natcypher=natcypher
        ),
        max_new_tokens=max_cypher_tokens,
        max_input_tokens=max_input_tokens,
    )
    return {
        "id": source.get("id"),
        "question": source.get("question"),
        "predicted_sirc": natcypher,
        "predicted_natcypher": natcypher,
        "predicted_cypher": cypher,
        **sirc_meta,
    }


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args)
    sources = read_inputs(args)
    if args.limit is not None:
        sources = sources[: args.limit]
    results = [
        run_two_stage(
            model,
            tokenizer,
            source,
            use_gold_ir=args.use_gold_ir,
            max_ir_tokens=args.max_ir_tokens,
            max_cypher_tokens=args.max_cypher_tokens,
            max_input_tokens=args.max_input_tokens,
            max_sirc_attempts=args.max_sirc_attempts,
            sirc_retry_temperature=args.sirc_retry_temperature,
        )
        for source in sources
    ]

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

