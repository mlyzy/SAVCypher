#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_TEST_SOURCES,
    DEFAULT_WO_SIRC_DIR,
    execute_predictions,
    read_jsonl,
    write_jsonl,
)
from infer import predict_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WO_SIRC_DIR)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_TEST_SOURCES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "outputs/ablation/wo_sirc/test",
    )
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-cypher-tokens", type=int, default=256)
    parser.add_argument("--max-output-characters", type=int, default=1024)
    parser.add_argument("--candidate-limit", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip-execution", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = read_jsonl(args.input_file)
    if args.limit is not None:
        sources = sources[: args.limit]
    results = predict_sources(args, sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, results)
    if args.skip_execution:
        print(f"Wrote {predictions_path}")
        return
    summary = execute_predictions(
        results,
        source_path=args.input_file,
        output_dir=args.output_dir,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
    )
    summary["predictions"] = str(predictions_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
