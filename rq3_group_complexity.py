#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent

TEXT2CYPHER_CSV = (
    ROOT / "text2cypher/datasets/synthetic_gpt4o_demodbs/text2cypher_gpt4o.csv"
)
TEXT2CYPHER_TEST = ROOT / "natcypher_data/compact_v03_text2cypher/test_sources.jsonl"
MIND_TEST = ROOT / "natcypher_data/compact_v03_mind_the_query/test_sources.jsonl"
MIND_SPLIT_ROOT = ROOT / "Mind-the-Query/Train_Test_Splits"

BUCKETS = ("SR", "CR", "SA", "CA")

# Longer Mind-the-Query tokens first so Simple_Retrieval is not swallowed by Retrieval.
_MIND_LABELS = (
    "Complex_Aggregation",
    "Simple_Aggregation",
    "Complex_Retrieval",
    "Simple_Retrieval",
    "Evaluation_query",
)

_EXPLICIT = {
    "simple retrieval queries": "SR",
    "simple retrieval": "SR",
    "simple_retrieval": "SR",
    "complex retrieval queries": "CR",
    "complex retrieval": "CR",
    "complex_retrieval": "CR",
    "simple aggregation queries": "SA",
    "simple aggregation": "SA",
    "simple_aggregation": "SA",
    "complex aggregation queries": "CA",
    "complex aggregation": "CA",
    "complex_aggregation": "CA",
    "pathfinding queries": "CR",
    "pathfinding": "CR",
    "verbose query": "CR",
    "verbose": "CR",
    "evaluation_query": "CR",
    "evaluation query": "CR",
}


def _norm(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def map_complexity_label(raw: str) -> str:
    """Map a dataset-native label onto SR / CR / SA / CA."""
    key = _norm(raw)
    if not key:
        raise ValueError("empty complexity label")
    if key in _EXPLICIT:
        return _EXPLICIT[key]
    underscored = key.replace(" ", "_")
    if underscored in _EXPLICIT:
        return _EXPLICIT[underscored]

    has_agg = "aggregat" in key
    has_simple = bool(re.search(r"\bsimple\b", key))
    if has_simple and has_agg:
        return "SA"
    if has_simple:
        return "SR"
    if has_agg:
        return "CA"
    return "CR"


def mind_label_from_id(example_id: str) -> str:
    wrapped = f"_{example_id}_"
    for label in _MIND_LABELS:
        if f"_{label}_" in wrapped:
            return label
    return ""


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_text2cypher_types(csv_path: Path) -> dict[str, str]:
    types: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            types[f"text2cypher:{index}"] = (row.get("type") or "").strip()
    return types


def annotate_text2cypher(test_path: Path, csv_path: Path) -> list[dict]:
    types = load_text2cypher_types(csv_path)
    out = []
    missing = 0
    for record in read_jsonl(test_path):
        example_id = str(record.get("id") or "")
        raw = types.get(example_id, "")
        if not raw:
            missing += 1
            continue
        item = dict(record)
        item["source_label"] = raw
        item["complexity"] = map_complexity_label(raw)
        out.append(item)
    if missing:
        raise SystemExit(
            f"Text2Cypher: {missing} test rows had no type in {csv_path}"
        )
    return out


def annotate_mind_from_test_sources(test_path: Path) -> list[dict]:
    out = []
    missing = 0
    for record in read_jsonl(test_path):
        example_id = str(record.get("id") or "")
        raw = mind_label_from_id(example_id)
        if not raw:
            missing += 1
            continue
        item = dict(record)
        item["source_label"] = raw
        item["complexity"] = map_complexity_label(raw)
        out.append(item)
    if missing:
        raise SystemExit(
            f"Mind-the-Query: {missing} test rows had no category in id"
        )
    return out


def annotate_mind_from_official_split(split_root: Path) -> list[dict]:
    out = []
    for path in sorted(split_root.glob("**/test/**/*.json")):
        raw = path.stem
        for suffix in ("_test", "_train"):
            if raw.endswith(suffix):
                raw = raw[: -len(suffix)]
                break
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else []
        split_kind = "manual" if "Manual" in path.parts else "automated"
        database = path.parent.parent.name
        for index, row in enumerate(rows):
            question = (
                row.get("NL Question") or row.get("question") or row.get("nl_question") or ""
            )
            cypher = row.get("Cypher") or row.get("cypher") or ""
            if not question or not cypher:
                continue
            out.append(
                {
                    "id": row.get("unique_id") or f"mind:{path.stem}:{index}",
                    "dataset": "mind_the_query",
                    "database": database,
                    "split_kind": split_kind,
                    "source_path": str(path),
                    "question": question,
                    "cypher": cypher,
                    "source_label": raw,
                    "complexity": map_complexity_label(raw),
                }
            )
    return out


def dump_groups(
    records: list[dict],
    paths: dict[str, Path],
    summary_path: Path,
) -> dict:
    grouped: dict[str, list[dict]] = {bucket: [] for bucket in BUCKETS}
    source_counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        bucket = record["complexity"]
        grouped[bucket].append(record)
        source_counts[bucket][record["source_label"]] += 1

    summary = {
        "n": len(records),
        "counts": {bucket: len(grouped[bucket]) for bucket in BUCKETS},
        "source_labels": {
            bucket: dict(source_counts[bucket]) for bucket in BUCKETS
        },
        "paths": {bucket: str(paths[bucket]) for bucket in BUCKETS},
    }
    for bucket in BUCKETS:
        write_jsonl(paths[bucket], grouped[bucket])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def print_summary(name: str, summary: dict) -> None:
    print(f"\n{name}: {summary['n']} test queries")
    for bucket in BUCKETS:
        n = summary["counts"][bucket]
        labels = summary["source_labels"][bucket]
        label_str = ", ".join(f"{k}={v}" for k, v in labels.items()) or "-"
        print(f"  {bucket:2s}  {n:5d}   ({label_str})")
        print(f"       {summary['paths'][bucket]}")


def bucket_paths(args: argparse.Namespace, dataset: str) -> dict[str, Path]:
    prefix = "text2cypher" if dataset == "text2cypher" else "mind"
    defaults = args.output_dir / dataset
    return {
        bucket: getattr(args, f"{prefix}_{bucket.lower()}")
        or defaults / f"{bucket}.jsonl"
        for bucket in BUCKETS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group Text2Cypher and Mind-the-Query test queries into SR/CR/SA/CA."
    )
    parser.add_argument(
        "--text2cypher-test",
        type=Path,
        default=TEXT2CYPHER_TEST,
        help="Text2Cypher test_sources.jsonl used for evaluation.",
    )
    parser.add_argument(
        "--text2cypher-csv",
        type=Path,
        default=TEXT2CYPHER_CSV,
        help="Original Text2Cypher CSV that carries the type column.",
    )
    parser.add_argument(
        "--mind-test",
        type=Path,
        default=MIND_TEST,
        help="Mind-the-Query test_sources.jsonl used for evaluation.",
    )
    parser.add_argument(
        "--mind-official-split",
        action="store_true",
        help="Use Mind-the-Query/Train_Test_Splits official test JSON instead of compact test_sources.",
    )
    parser.add_argument(
        "--mind-split-root",
        type=Path,
        default=MIND_SPLIT_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/rq3_complexity",
        help="Default parent directory when the four per-label paths are omitted.",
    )
    parser.add_argument(
        "--text2cypher-sr",
        type=Path,
        help="Text2Cypher SR test JSONL path.",
    )
    parser.add_argument(
        "--text2cypher-cr",
        type=Path,
        help="Text2Cypher CR test JSONL path.",
    )
    parser.add_argument(
        "--text2cypher-sa",
        type=Path,
        help="Text2Cypher SA test JSONL path.",
    )
    parser.add_argument(
        "--text2cypher-ca",
        type=Path,
        help="Text2Cypher CA test JSONL path.",
    )
    parser.add_argument(
        "--mind-sr",
        type=Path,
        help="Mind-the-Query SR test JSONL path.",
    )
    parser.add_argument(
        "--mind-cr",
        type=Path,
        help="Mind-the-Query CR test JSONL path.",
    )
    parser.add_argument(
        "--mind-sa",
        type=Path,
        help="Mind-the-Query SA test JSONL path.",
    )
    parser.add_argument(
        "--mind-ca",
        type=Path,
        help="Mind-the-Query CA test JSONL path.",
    )
    parser.add_argument(
        "--text2cypher-summary",
        type=Path,
        help="Text2Cypher grouping summary JSON path.",
    )
    parser.add_argument(
        "--mind-summary",
        type=Path,
        help="Mind-the-Query grouping summary JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    text2cypher = annotate_text2cypher(args.text2cypher_test, args.text2cypher_csv)
    if args.mind_official_split:
        mind = annotate_mind_from_official_split(args.mind_split_root)
    else:
        mind = annotate_mind_from_test_sources(args.mind_test)

    t2c_paths = bucket_paths(args, "text2cypher")
    mind_paths = bucket_paths(args, "mind_the_query")
    t2c_summary = dump_groups(
        text2cypher,
        t2c_paths,
        args.text2cypher_summary or args.output_dir / "text2cypher" / "summary.json",
    )
    mind_summary = dump_groups(
        mind,
        mind_paths,
        args.mind_summary or args.output_dir / "mind_the_query" / "summary.json",
    )
    print_summary("Text2Cypher test", t2c_summary)
    print_summary("Mind-the-Query test", mind_summary)


if __name__ == "__main__":
    main()
