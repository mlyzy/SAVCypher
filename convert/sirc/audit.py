#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Callable

from sirc.serializer import canonicalize_sirc


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    return sorted(values)[max(0, math.ceil(len(values) * quantile) - 1)]


def audit(
    source: Path, token_counter: Callable[[str], int]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    natcypher_tokens: list[int] = []
    cypher_tokens: list[int] = []
    failures: list[dict[str, object]] = []
    total = 0
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("supported") is not True:
                continue
            total += 1
            natcypher = str(record.get("sirc") or record.get("natcypher") or "")
            try:
                canonical = canonicalize_sirc(natcypher)
            except ValueError as exc:
                failures.append(
                    {"line": line_number, "id": record.get("id"), "error": str(exc)}
                )
                continue
            if canonical != natcypher:
                failures.append(
                    {
                        "line": line_number,
                        "id": record.get("id"),
                        "error": "noncanonical_serialization",
                    }
                )
            natcypher_tokens.append(token_counter(natcypher))
            cypher_tokens.append(token_counter(str(record["cypher"])))

    report = {
        "version": "0.3",
        "records": total,
        "canonical_records": total - len(failures),
        "canonical_rate": (total - len(failures)) / total if total else 0.0,
        "same_ast_unique_serialization": not failures,
        "token_count_method": getattr(token_counter, "method", "custom"),
        "natcypher": {
            "mean": mean(natcypher_tokens) if natcypher_tokens else 0.0,
            "p90": percentile(natcypher_tokens, 0.9),
        },
        "cypher": {
            "mean": mean(cypher_tokens) if cypher_tokens else 0.0,
            "p90": percentile(cypher_tokens, 0.9),
        },
    }
    report["targets_met"] = {
        "mean_below_cypher": report["natcypher"]["mean"] < report["cypher"]["mean"],
        "p90_below_cypher": report["natcypher"]["p90"] < report["cypher"]["p90"],
    }
    return report, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--failures", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

        def token_counter(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        token_counter.method = str(args.tokenizer)  # type: ignore[attr-defined]
    else:
        def token_counter(text: str) -> int:
            return len(text.split())

        token_counter.method = "whitespace"  # type: ignore[attr-defined]

    report, failures = audit(args.source, token_counter)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    if args.failures:
        args.failures.parent.mkdir(parents=True, exist_ok=True)
        with args.failures.open("w", encoding="utf-8") as destination:
            for failure in failures:
                destination.write(json.dumps(failure, ensure_ascii=False) + "\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
