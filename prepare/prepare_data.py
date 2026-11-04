#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Iterable, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import ensure_sys_path

ensure_sys_path()

from train.data import SYSTEM_PROMPTS, make_task_input
from train.schema_format import (
    SCHEMA_FORMAT_VERSION,
    audit_schema_symbols,
    estimate_tokens,
    format_schema,
)


REPO_ROOT = _REPO_ROOT
DEFAULT_SOURCE = REPO_ROOT / "sirc_data" / "text2cypher.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "sirc_data" / "mtl"
SPLITS = ("train", "dev", "test")
RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
REQUIRED_FIELDS = ("id", "dataset", "graph", "question", "schema", "cypher")
DEFAULT_DEV_EVAL_SIZE = 100
DEFAULT_SCHEMA_TOKEN_BUDGET = 1024


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_record(
    record: Any,
    line_number: int,
    schema_token_budget: int = DEFAULT_SCHEMA_TOKEN_BUDGET,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Validate and normalize one supported source record."""
    if not isinstance(record, Mapping):
        raise ValueError(f"line {line_number}: record must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"line {line_number}: missing required fields: {', '.join(missing)}"
        )

    ir_text = record.get("sirc") or record.get("natcypher")
    if not _nonempty_text(ir_text):
        raise ValueError(f"line {line_number}: sirc/natcypher must be a nonempty string")
    record = dict(record)
    record["sirc"] = str(ir_text).strip()
    record["natcypher"] = record["sirc"]

    for field in ("id", "dataset", "graph", "question", "cypher"):
        if not _nonempty_text(record[field]):
            raise ValueError(f"line {line_number}: {field} must be a nonempty string")

    if not isinstance(record["schema"], (str, Mapping)):
        raise ValueError(f"line {line_number}: schema must be a string or object")
    schema_text = format_schema(record["schema"], question=str(record["question"]))
    if not schema_text.strip():
        raise ValueError(f"line {line_number}: schema must be nonempty")
    token_count = (
        token_counter(schema_text) if token_counter else estimate_tokens(schema_text)
    )
    if token_counter and token_count > schema_token_budget:
        raise ValueError(
            f"line {line_number}: compact structural schema requires "
            f"{token_count} tokens, exceeding budget {schema_token_budget}; "
            "refusing to truncate relationships or schema symbols"
        )
    audit = audit_schema_symbols(record["schema"], schema_text, str(record["cypher"]))
    if audit["coverage"] != 1.0 or audit["referenced_coverage"] != 1.0:
        raise ValueError(
            f"line {line_number}: compact schema symbol coverage failure: "
            f"{audit['missing'] or audit['referenced_missing']}"
        )

    normalized = {field: record[field].strip() for field in REQUIRED_FIELDS if field != "schema"}
    normalized["schema"] = schema_text
    normalized["sirc"] = record["sirc"]
    normalized["natcypher"] = record["sirc"]
    normalized["_schema_metrics"] = {
        "tokens": token_count,
        "token_count_method": "model_tokenizer" if token_counter else "estimate",
        **audit,
    }
    return normalized


def read_source(
    path: Path,
    schema_token_budget: int = DEFAULT_SCHEMA_TOKEN_BUDGET,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read valid supported records, skipping records explicitly not supported."""
    records: list[dict[str, Any]] = []
    total = 0
    unsupported = 0
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"line {line_number}: record must be a JSON object")
            if raw.get("supported") is not True:
                unsupported += 1
                continue
            record = validate_record(
                raw, line_number, schema_token_budget, token_counter
            )
            if record["id"] in seen_ids:
                raise ValueError(f"line {line_number}: duplicate id {record['id']!r}")
            seen_ids.add(record["id"])
            records.append(record)

    if not records:
        raise ValueError("source contains no valid supported records")
    return records, {
        "source_records": total,
        "supported_records": len(records),
        "unsupported_records_skipped": unsupported,
    }


def assign_database_splits(
    records: Iterable[Mapping[str, Any]], seed: int
) -> dict[str, str]:
    """Assign whole databases to splits while approximating 80/10/10 by records."""
    counts = Counter(str(record["graph"]) for record in records)
    if not counts:
        raise ValueError("cannot split an empty record collection")

    databases = sorted(counts)
    random.Random(seed).shuffle(databases)
    total = sum(counts.values())
    targets = {split: total * RATIOS[split] for split in SPLITS}
    assigned_counts = {split: 0 for split in SPLITS}
    assignments: dict[str, str] = {}

    for index, database in enumerate(databases):
        empty_splits = [split for split in SPLITS if split not in assignments.values()]
        groups_remaining = len(databases) - index
        if groups_remaining == len(empty_splits):
            candidates = empty_splits
        else:
            candidates = list(SPLITS)

        group_size = counts[database]

        def score(split: str) -> tuple[float, float, int]:
            before = abs(assigned_counts[split] - targets[split])
            after = abs(assigned_counts[split] + group_size - targets[split])
            improvement = before - after
            deficit = targets[split] - assigned_counts[split]
            return improvement, deficit, -SPLITS.index(split)

        chosen = max(candidates, key=score)
        assignments[database] = chosen
        assigned_counts[chosen] += group_size

    return assignments


def make_task_records(record: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create one record per MTL direction from a normalized source record."""
    common = {
        "source_id": record["id"],
        "dataset": record["dataset"],
        "database": record["graph"],
        "schema_format": SCHEMA_FORMAT_VERSION,
    }
    task_zero = {
        **common,
        "system": SYSTEM_PROMPTS[0],
        "input": make_task_input(
            0, question=record["question"], schema=record["schema"]
        ),
        "output": record["natcypher"],
        "task_id": 0,
    }
    task_one = {
        **common,
        "system": SYSTEM_PROMPTS[1],
        "input": make_task_input(
            1, natcypher=record["natcypher"], schema=record["schema"]
        ),
        "output": record["cypher"],
        "task_id": 1,
    }
    return task_zero, task_one


def make_paired_source(record: Mapping[str, str]) -> dict[str, str]:
    return {
        "id": record["id"],
        "dataset": record["dataset"],
        "database": record["graph"],
        "question": record["question"],
        "schema": record["schema"],
        "schema_format": SCHEMA_FORMAT_VERSION,
        "sirc": record["sirc"],
        "natcypher": record["natcypher"],
        "cypher": record["cypher"],
    }


def select_stratified_sources(
    records: Iterable[Mapping[str, str]], size: int, seed: int
) -> list[dict[str, str]]:
    """Select a deterministic, database-proportional evaluation subset."""
    sources = [make_paired_source(record) for record in records]
    if size <= 0 or size >= len(sources):
        return sources

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in sources:
        grouped[source["database"]].append(source)
    total = len(sources)
    exact = {
        database: size * len(items) / total
        for database, items in grouped.items()
    }
    allocation = {
        database: min(len(grouped[database]), int(count))
        for database, count in exact.items()
    }
    remaining = size - sum(allocation.values())
    priority = sorted(
        grouped,
        key=lambda database: (
            -(exact[database] - allocation[database]),
            database,
        ),
    )
    while remaining:
        made_progress = False
        for database in priority:
            if allocation[database] < len(grouped[database]):
                allocation[database] += 1
                remaining -= 1
                made_progress = True
                if remaining == 0:
                    break
        if not made_progress:
            break

    selected = []
    for database in sorted(grouped):
        items = sorted(grouped[database], key=lambda item: item["id"])
        random.Random(f"{seed}:{database}").shuffle(items)
        selected.extend(items[: allocation[database]])
    random.Random(seed).shuffle(selected)
    return selected


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for record in records:
            destination.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
    return count


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _schema_manifest_metrics(
    records: Iterable[Mapping[str, Any]], token_budget: int
) -> dict[str, Any]:
    metrics = [record["_schema_metrics"] for record in records]
    tokens = [int(item["tokens"]) for item in metrics]
    method = (
        metrics[0]["token_count_method"] if metrics else "model_tokenizer"
    )
    total = sum(int(item["total"]) for item in metrics)
    covered = sum(int(item["covered"]) for item in metrics)
    referenced_total = sum(int(item["referenced_total"]) for item in metrics)
    referenced_covered = sum(int(item["referenced_covered"]) for item in metrics)
    return {
        "token_budget": token_budget,
        "token_count_method": method,
        "token_p50": _percentile(tokens, 0.50),
        "token_p90": _percentile(tokens, 0.90),
        "token_p99": _percentile(tokens, 0.99),
        "token_max": max(tokens, default=0),
        "over_budget_count": sum(value > token_budget for value in tokens),
        "truncation_count": 0,
        "critical_symbols_total": total,
        "critical_symbols_covered": covered,
        "critical_symbol_coverage": covered / total if total else 1.0,
        "gold_symbols_total": referenced_total,
        "gold_symbols_covered": referenced_covered,
        "gold_symbol_coverage": (
            referenced_covered / referenced_total if referenced_total else 1.0
        ),
    }


def prepare_data(
    source_path: Path,
    output_dir: Path,
    seed: int = 42,
    dev_eval_size: int = DEFAULT_DEV_EVAL_SIZE,
    schema_token_budget: int = DEFAULT_SCHEMA_TOKEN_BUDGET,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Build MTL JSONL splits, paired test sources, and a split manifest."""
    records, source_stats = read_source(
        source_path, schema_token_budget, token_counter
    )
    assignments = assign_database_splits(records, seed)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    database_source_counts = Counter(record["graph"] for record in records)

    for record in records:
        by_split[assignments[record["graph"]]].append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_details: dict[str, Any] = {}
    for split in SPLITS:
        source_records = by_split[split]
        tasks = [
            task
            for source_record in source_records
            for task in make_task_records(source_record)
        ]
        task_count = _write_jsonl(output_dir / f"{split}.jsonl", tasks)
        split_databases = sorted(
            database for database, assigned in assignments.items() if assigned == split
        )
        split_details[split] = {
            "databases": split_databases,
            "database_count": len(split_databases),
            "source_record_count": len(source_records),
            "task_record_count": task_count,
            "task_counts": {"0": len(source_records), "1": len(source_records)},
            "schema_audit": _schema_manifest_metrics(
                source_records, schema_token_budget
            ),
        }

    paired_dev = [make_paired_source(record) for record in by_split["dev"]]
    paired_test = [make_paired_source(record) for record in by_split["test"]]
    dev_eval = select_stratified_sources(
        by_split["dev"], dev_eval_size, seed
    )
    _write_jsonl(output_dir / "dev_sources.jsonl", paired_dev)
    _write_jsonl(output_dir / "dev_eval_sources.jsonl", dev_eval)
    _write_jsonl(output_dir / "test_sources.jsonl", paired_test)

    manifest = {
        "seed": seed,
        "ratios": RATIOS,
        "schema_format": SCHEMA_FORMAT_VERSION,
        "schema_token_budget": schema_token_budget,
        **source_stats,
        "database_count": len(assignments),
        "database_assignments": {
            database: {
                "split": assignments[database],
                "source_record_count": database_source_counts[database],
            }
            for database in sorted(assignments)
        },
        "splits": split_details,
        "files": {
            "train": "train.jsonl",
            "dev": "dev.jsonl",
            "paired_dev": "dev_sources.jsonl",
            "dev_eval": "dev_eval_sources.jsonl",
            "test": "test.jsonl",
            "paired_test": "test_sources.jsonl",
        },
        "dev_eval_size": len(dev_eval),
    }
    with (output_dir / "manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", "--input", dest="source", type=Path, default=DEFAULT_SOURCE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-eval-size", type=int, default=DEFAULT_DEV_EVAL_SIZE)
    parser.add_argument(
        "--schema-token-budget", type=int, default=DEFAULT_SCHEMA_TOKEN_BUDGET
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional model tokenizer for authoritative preparation-time token gates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token_counter = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required when --tokenizer is supplied"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )
        token_counter = lambda text: len(
            tokenizer.encode(text, add_special_tokens=False)
        )
    manifest = prepare_data(
        args.source,
        args.output_dir,
        args.seed,
        args.dev_eval_size,
        args.schema_token_budget,
        token_counter,
    )
    print(
        f"Wrote {manifest['supported_records']} source records "
        f"({manifest['supported_records'] * 2} task records) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
