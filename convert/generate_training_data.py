#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import ensure_sys_path

ensure_sys_path()

from convert.dataset_adapters import load_dataset, set_data_root
from sirc.generator import ConversionResult, convert_cypher_to_sirc


DATASETS = ("cypherbench", "text2cypher", "mind_the_query", "synthcypher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert gold Cypher queries into strict SIRC targets."
    )
    parser.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="cypherbench",
        help="Dataset adapter to use (default: cypherbench).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "sirc_data",
        help="Directory for JSONL targets and summary files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root that contains text2cypher/, benchmark/, Mind-the-Query/, SynthCypher/. "
        "Defaults to SIRC_DATA_ROOT or the parent workspace.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples per dataset.",
    )
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help="Write rejected examples to the main JSONL as well.",
    )
    parser.add_argument(
        "--fail-on-unsupported",
        action="store_true",
        help="Exit non-zero if any example cannot be converted safely.",
    )
    return parser.parse_args()


def convert_dataset(
    dataset: str,
    output_dir: Path,
    limit: Optional[int] = None,
    include_unsupported: bool = False,
    data_root: Optional[Path] = None,
) -> Dict[str, object]:
    examples = load_dataset(dataset, limit=limit, data_root=data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"{dataset}.jsonl"
    rejected_path = output_dir / f"{dataset}.unsupported.jsonl"
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    supported = 0

    with target_path.open("w", encoding="utf-8") as targets, rejected_path.open(
        "w", encoding="utf-8"
    ) as rejected:
        for example in examples:
            conversion = convert_cypher_to_sirc(example.gold_cypher)
            record = make_record(dataset, example, conversion)
            encoded = json.dumps(record, ensure_ascii=False)
            if conversion.supported:
                targets.write(encoded + "\n")
                supported += 1
            else:
                rejected.write(encoded + "\n")
                if include_unsupported:
                    targets.write(encoded + "\n")
            error_counts.update(conversion.errors)
            warning_counts.update(conversion.warnings)

    summary: Dict[str, object] = {
        "dataset": dataset,
        "total": len(examples),
        "supported": supported,
        "unsupported": len(examples) - supported,
        "support_rate": supported / len(examples) if examples else 0.0,
        "errors": dict(error_counts.most_common()),
        "warnings": dict(warning_counts.most_common()),
        "output": str(target_path),
        "rejected_output": str(rejected_path),
    }
    summary_path = output_dir / f"{dataset}.summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def make_record(dataset: str, example, conversion: ConversionResult) -> Dict[str, object]:
    """Keep source supervision and conversion diagnostics in one portable row."""
    return {
        "id": example.example_id,
        "dataset": dataset,
        "graph": example.graph_key,
        "question": example.question,
        "schema": example.schema_struct or example.schema_text,
        "cypher": example.gold_cypher,
        "sirc": conversion.sirc if conversion.supported else None,
        "natcypher": conversion.sirc if conversion.supported else None,
        "supported": conversion.supported,
        "errors": conversion.errors,
        "warnings": conversion.warnings,
        "task_type": example.task_type,
        "metadata": {
            **example.metadata,
            "conversion": conversion.metadata,
        },
    }


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.data_root is not None:
        set_data_root(args.data_root)
    datasets: Iterable[str] = DATASETS if args.dataset == "all" else (args.dataset,)
    summaries: List[Dict[str, object]] = []
    for dataset in datasets:
        summary = convert_dataset(
            dataset,
            args.output_dir,
            limit=args.limit,
            include_unsupported=args.include_unsupported,
            data_root=args.data_root,
        )
        summaries.append(summary)
        print(
            f"{dataset}: {summary['supported']}/{summary['total']} supported "
            f"({summary['support_rate']:.1%})"
        )
    aggregate_path = args.output_dir / "summary.json"
    aggregate_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.fail_on_unsupported and any(item["unsupported"] for item in summaries):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
