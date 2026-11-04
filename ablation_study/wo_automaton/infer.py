#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_STAGE1_DIR,
    DEFAULT_STAGE2_DIR,
    generate_unconstrained,
    load_standard_lora,
    read_jsonl,
    release_model,
    render_chat,
    resolve_adapter_checkpoint,
    sirc_text,
    write_jsonl,
)
from natcypher.data import SYSTEM_PROMPTS, make_task_input
from natcypher.schema_format import format_schema


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--stage2-checkpoint", type=Path, default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-ir-tokens", type=int, default=512)
    parser.add_argument("--max-cypher-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def stage1_prompt(tokenizer, source: dict[str, Any]) -> tuple[str, str]:
    schema = format_schema(source["schema"], question=str(source["question"]))
    user = make_task_input(0, question=str(source["question"]), schema=schema)
    return render_chat(tokenizer, SYSTEM_PROMPTS[0], user), schema


def stage2_prompt(tokenizer, sirc: str, schema: str) -> str:
    user = make_task_input(1, schema=schema, natcypher=sirc)
    return render_chat(tokenizer, SYSTEM_PROMPTS[1], user)


def predict_sources(args: argparse.Namespace, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage1 = resolve_adapter_checkpoint(args.stage1_checkpoint)
    stage2 = resolve_adapter_checkpoint(args.stage2_checkpoint)
    missing = [source for source in sources if not sirc_text(source)]
    generated_sirc: dict[int, str] = {}
    if missing:
        model, tokenizer = load_standard_lora(stage1, args.base_model)
        try:
            for source in missing:
                prompt, _schema = stage1_prompt(tokenizer, source)
                generated_sirc[id(source)] = generate_unconstrained(
                    model,
                    tokenizer,
                    prompt,
                    source_id=source.get("id", source.get("source_id", "<unknown>")),
                    max_input_tokens=args.max_input_tokens,
                    max_new_tokens=args.max_ir_tokens,
                )
        finally:
            release_model(model, tokenizer)

    model, tokenizer = load_standard_lora(stage2, args.base_model)
    results = []
    try:
        for source in sources:
            schema = format_schema(
                source["schema"], question=str(source.get("question", ""))
            )
            sirc = generated_sirc.get(id(source)) or sirc_text(source)
            prompt = stage2_prompt(tokenizer, sirc, schema)
            cypher = generate_unconstrained(
                model,
                tokenizer,
                prompt,
                source_id=source.get("id", source.get("source_id", "<unknown>")),
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_cypher_tokens,
            )
            results.append(
                {
                    **source,
                    "ablation": "wo_automaton",
                    "predicted_sirc": sirc,
                    "predicted_natcypher": sirc,
                    "predicted_cypher": cypher,
                    "constrained_decoding": False,
                }
            )
    finally:
        release_model(model, tokenizer)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sources = read_jsonl(args.input_file)
    if args.limit is not None:
        sources = sources[: args.limit]
    results = predict_sources(args, sources)
    write_jsonl(args.output_file, results)


if __name__ == "__main__":
    main()
