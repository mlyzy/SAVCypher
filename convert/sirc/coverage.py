#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from typing import Any, Iterable, Optional

from sirc.compiler import compile_sirc


CONSTRUCTS = (
    "ANSWER", "GRAPH", "FILTER", "LET", "KEEP", "AGG", "PER", "HAVING",
    "SCOPE", "EXPOSE", "ORDER", "TOP", "TAKE", "SKIP", "OFFSET", "DEDUP",
    "OPTIONAL", "EXPAND", "ANY", "PATH",
)
SCHEMA_ERROR_PREFIXES = (
    "unknown_label:",
    "unknown_node_property:",
    "unknown_relationship:",
    "unknown_relationship_property:",
    "relationship_endpoint_mismatch:",
)


def schema_error_is_in_gold(error: str, gold: str) -> bool:
    """Require evidence that the original gold query has the same schema conflict."""
    if not error.startswith(SCHEMA_ERROR_PREFIXES):
        return False
    _, _, symbol = error.partition(":")
    if error.startswith(("unknown_node_property:", "unknown_relationship_property:")):
        return bool(symbol and re.search(rf"(?<!\w){re.escape(symbol)}(?!\w)", gold))
    if error.startswith("unknown_label:"):
        return bool(symbol and re.search(rf":\s*{re.escape(symbol)}(?!\w)", gold))
    relation_types = symbol.split("|")
    return bool(
        relation_types
        and all(re.search(rf":\s*{re.escape(item)}(?!\w)", gold) for item in relation_types)
    )


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield line_number, value


def fields(
    record: dict[str, Any], paired_gold: Optional[str] = None
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if record.get("sirc") or record.get("natcypher"):
        return str(record.get("sirc") or record["natcypher"]), str(record.get("schema", "")), str(record.get("cypher", ""))
    if record.get("task_id") == 0 and record.get("output"):
        input_text = str(record.get("input", ""))
        schema = input_text.split("\n\nSchema:\n", 1)[1] if "\n\nSchema:\n" in input_text else ""
        return str(record["output"]), schema, paired_gold
    return None, None, None


def execute_http_pair(executor, database: str, gold: str, compiled: str) -> tuple[str, str]:
    from evaluate_text2cypher_vllm_execution import compare_rows

    gold_rows, gold_error, gold_transport = executor.run(database, gold)
    if gold_error:
        return "gold_error", gold_error
    if gold_transport:
        return "transport_error", gold_transport
    compiled_rows, compiled_error, compiled_transport = executor.run(database, compiled)
    if compiled_error:
        return "compiled_error", compiled_error
    if compiled_transport:
        return "transport_error", compiled_transport
    equal, _, status = compare_rows(gold_rows or [], compiled_rows or [], gold, False)
    return ("match" if equal else "mismatch"), status


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--http-base-url", default="https://demo.neo4jlabs.com:7473")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    executor = None
    if args.execute:
        from evaluate_text2cypher_vllm_execution import HttpExecutor

        executor = HttpExecutor(args.http_base_url, args.timeout, args.retries, "query_v2")

    total = supported = dataset_schema_errors = comparable = equivalent = 0
    errors: Counter[str] = Counter()
    constructs: Counter[str] = Counter()
    compiled_constructs: Counter[str] = Counter()
    dataset_error_constructs: Counter[str] = Counter()
    failures = []
    execution_jobs = []
    differential_statuses: Counter[str] = Counter()
    for path in args.inputs:
        records = list(read_jsonl(path))
        paired_gold = {
            str(record.get("source_id")): str(record.get("output", ""))
            for _, record in records
            if record.get("task_id") == 1 and record.get("source_id")
        }
        for line_number, record in records:
            natcypher, schema, gold = fields(
                record, paired_gold.get(str(record.get("source_id")))
            )
            if natcypher is None:
                continue
            if args.limit is not None and total >= args.limit:
                break
            total += 1
            present_constructs = set()
            for construct in CONSTRUCTS:
                if re.search(rf"(?m)^\s*{construct}\b", natcypher):
                    constructs[construct] += 1
                    present_constructs.add(construct)
            result = compile_sirc(natcypher, schema)
            if not result.supported:
                is_schema_error = (
                    bool(gold)
                    and any(
                            schema_error_is_in_gold(error, gold)
                        for error in result.errors
                    )
                    and compile_sirc(natcypher, validate=False).supported
                )
                if is_schema_error:
                    dataset_schema_errors += 1
                    dataset_error_constructs.update(present_constructs)
                    failures.append({
                        "file": str(path), "line": line_number,
                        "id": record.get("id", record.get("source_id")),
                        "status": "dataset_schema_error",
                        "errors": result.errors, "natcypher": natcypher,
                        "gold_cypher": gold,
                    })
                    continue
                reason = result.errors[0] if result.errors else "unknown"
                errors[reason] += 1
                failures.append({
                    "file": str(path), "line": line_number,
                    "id": record.get("id", record.get("source_id")),
                    "errors": result.errors, "natcypher": natcypher,
                })
                continue
            supported += 1
            compiled_constructs.update(present_constructs)
            if executor is not None and gold:
                execution_jobs.append((
                    str(record.get("database", record.get("graph", ""))),
                    gold,
                    result.cypher,
                    {
                        "file": str(path), "line": line_number,
                        "id": record.get("id", record.get("source_id")),
                        "natcypher": natcypher, "compiled_cypher": result.cypher,
                        "gold_cypher": gold,
                    },
                ))

    if executor is not None:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(execute_http_pair, executor, database, gold, compiled): details
                for database, gold, compiled, details in execution_jobs
            }
            for future in as_completed(futures):
                details = futures[future]
                try:
                    status, detail = future.result()
                except Exception as exc:
                    status, detail = "worker_error", f"{type(exc).__name__}: {exc}"
                differential_statuses[status] += 1
                if status in {"match", "mismatch"}:
                    comparable += 1
                    equivalent += int(status == "match")
                if status != "match":
                    failures.append({**details, "status": status, "execution_detail": detail})

    categories: Counter[str] = Counter()
    for reason, count in errors.items():
        category = reason.split(":", 1)[0]
        categories[category] += count
    eligible = total - dataset_schema_errors
    compiler_unsupported = eligible - supported
    report = {
        "total": total,
        "dataset_schema_errors": dataset_schema_errors,
        "eligible": eligible,
        "compiled": supported,
        "structural_coverage": supported / total if total else 0.0,
        "compiler_coverage": supported / eligible if eligible else 0.0,
        "construct_counts": dict(sorted(constructs.items())),
        "construct_coverage": {
            name: {
                "total": constructs[name] - dataset_error_constructs[name],
                "compiled": compiled_constructs[name],
                "dataset_schema_errors": dataset_error_constructs[name],
                "coverage": (
                    compiled_constructs[name]
                    / (constructs[name] - dataset_error_constructs[name])
                    if constructs[name] > dataset_error_constructs[name]
                    else 0.0
                ),
            }
            for name in sorted(constructs)
        },
        "unsupported": compiler_unsupported,
        "unsupported_categories": dict(categories.most_common()),
        "unsupported_reasons": dict(errors.most_common()),
        "differential": {
            "enabled": bool(args.execute),
            "comparable": comparable,
            "equivalent": equivalent,
            "accuracy": equivalent / comparable if comparable else None,
            "statuses": dict(differential_statuses.most_common()),
        },
    }
    return report, failures


def main(argv=None) -> None:
    args = parse_args(argv)
    report, failures = audit(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    if args.failures:
        args.failures.parent.mkdir(parents=True, exist_ok=True)
        with args.failures.open("w", encoding="utf-8") as destination:
            for failure in failures:
                destination.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
    if not args.quiet:
        print(rendered, end="")


if __name__ == "__main__":
    main()
