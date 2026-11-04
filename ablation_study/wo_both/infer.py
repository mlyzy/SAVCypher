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
    DEFAULT_WO_BOTH_DIR,
    generate_unconstrained,
    load_standard_lora,
    read_jsonl,
    render_chat,
    resolve_adapter_checkpoint,
    write_jsonl,
)
from natcypher.data import make_task_input
from natcypher.schema_format import format_schema
from natcypher.train_direct_lora import SYSTEM_PROMPT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WO_BOTH_DIR)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-cypher-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def direct_prompt(tokenizer, source: dict[str, Any]) -> str:
    schema = format_schema(source["schema"], question=str(source["question"]))
    user = make_task_input(0, question=str(source["question"]), schema=schema)
    return render_chat(tokenizer, SYSTEM_PROMPT, user)


def predict_sources(args: argparse.Namespace, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checkpoint = resolve_adapter_checkpoint(args.checkpoint)
    model, tokenizer = load_standard_lora(checkpoint, args.base_model)
    results = []
    for source in sources:
        prompt = direct_prompt(tokenizer, source)
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
                "ablation": "wo_both",
                "predicted_sirc": "",
                "predicted_natcypher": "",
                "predicted_cypher": cypher,
                "constrained_decoding": False,
            }
        )
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
