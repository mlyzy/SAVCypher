#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from paths import ensure_sys_path

ensure_sys_path()
_WORKSPACE = REPO_ROOT.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.append(str(_WORKSPACE))

STEP_PATTERN = re.compile(r"step-(\d+)")
INVALID_EXECUTION_STATUSES = {
    "gold_error",
    "transport_error",
    "protocol_error",
    "worker_error",
}


def parse_sample_limit(value: str) -> Optional[int]:
    """Parse a positive sample count, with ``all`` represented by None."""
    if value.casefold() == "all":
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample limit must be a positive integer or 'all'") from exc
    if limit <= 0:
        raise argparse.ArgumentTypeError("sample limit must be a positive integer or 'all'")
    return limit


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--eval-sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-limit",
        type=parse_sample_limit,
        default=None,
        metavar="N|all",
        help="Deterministically sample N dev rows (default: all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-ir-tokens", type=int, default=1024)
    parser.add_argument("--max-cypher-tokens", type=int, default=512)
    parser.add_argument(
        "--count-empty-match",
        action="store_true",
        help="Count matching empty result sets as execution-correct.",
    )
    parser.add_argument(
        "--order-sensitive",
        action="store_true",
        help="Respect result order when the gold query contains ORDER BY.",
    )
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be greater than zero")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    return args


def checkpoint_step(checkpoint: Path) -> int:
    """Return the numeric step for a ``step-N/checkpoint`` directory."""
    match = STEP_PATTERN.fullmatch(checkpoint.parent.name)
    if checkpoint.name != "checkpoint" or match is None:
        raise ValueError(f"Not a step checkpoint directory: {checkpoint}")
    return int(match.group(1))


def discover_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """Discover immediate ``step-N/checkpoint`` directories in numeric order."""
    checkpoints = [
        step_dir / "checkpoint"
        for step_dir in checkpoints_dir.iterdir()
        if step_dir.is_dir()
        and STEP_PATTERN.fullmatch(step_dir.name)
        and (step_dir / "checkpoint").is_dir()
    ]
    checkpoints.sort(key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(
            f"No step-XXXXXX/checkpoint directories found in {checkpoints_dir}"
        )
    return checkpoints


def normalize_natcypher(value: Any) -> str:
    """Normalize presentation-only differences for exact matching."""
    text = "" if value is None else str(value)
    fence = re.fullmatch(
        r"\s*```(?:natcypher|text)?\s*(.*?)```\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence:
        text = fence.group(1)
    return re.sub(r"\s+", " ", text).strip()


def normalized_exact_match(prediction: Any, gold: Any) -> bool:
    return normalize_natcypher(prediction) == normalize_natcypher(gold)


def limit_sources(
    sources: Sequence[Mapping[str, Any]], limit: Optional[int], seed: int
) -> list[dict[str, Any]]:
    """Return a deterministic sample while retaining original file order."""
    records = [dict(source) for source in sources]
    if limit is None or limit >= len(records):
        return records
    indices = sorted(random.Random(seed).sample(range(len(records)), limit))
    return [records[index] for index in indices]


def ranking_key(summary: Mapping[str, Any]) -> tuple[float, float, float, int]:
    """Sort by execution, Jaccard, SIRC EM, then later training step."""
    return (
        float(summary.get("execution_accuracy", 0.0)),
        float(summary.get("mean_result_jaccard", 0.0)),
        float(summary.get("natcypher_exact_match", 0.0)),
        int(summary.get("step", 0)),
    )


def select_best_summary(
    summaries: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any]:
    candidates = list(summaries)
    if not candidates:
        raise ValueError("Cannot rank an empty set of checkpoint summaries")
    return max(candidates, key=ranking_key)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no examples")
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _inference_api():
    from infer.two_stage import load_model, run_two_stage
    from train.qwen_mlora import load_adapter_checkpoint

    return load_model, run_two_stage, load_adapter_checkpoint


def _execution_api():
    from evaluate_text2cypher_vllm_execution import (
        GoldCache,
        HttpExecutor,
        atomic_write,
        evaluate_row,
        make_summary,
    )

    return GoldCache, HttpExecutor, atomic_write, evaluate_row, make_summary


def generate_predictions(
    model: Any,
    tokenizer: Any,
    sources: Sequence[Mapping[str, Any]],
    *,
    step: int,
    max_ir_tokens: int,
    max_cypher_tokens: int,
) -> list[dict[str, Any]]:
    """Run the repository's two-stage generator for one loaded adapter."""
    _, run_two_stage, _ = _inference_api()
    predictions = []
    for example_index, source in enumerate(sources):
        generated = run_two_stage(
            model,
            tokenizer,
            dict(source),
            max_ir_tokens=max_ir_tokens,
            max_cypher_tokens=max_cypher_tokens,
        )
        predictions.append(
            {
                **dict(source),
                **generated,
                "example_index": example_index,
                "checkpoint_step": step,
            }
        )
    return predictions


def _execution_input(prediction: Mapping[str, Any]) -> dict[str, Any]:
    database = prediction.get("database", prediction.get("graph", ""))
    return {
        "example_index": int(prediction["example_index"]),
        "database": str(database),
        "type": prediction.get("type", prediction.get("dataset", "")),
        "question": prediction.get("question", ""),
        "reference_cypher": prediction.get("cypher", ""),
        "extracted_cypher": prediction.get("predicted_cypher", ""),
    }


def _worker_failure(row: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        **row,
        "status": "worker_error",
        "execution_correct": 0,
        "result_jaccard": 0.0,
        "gold_rows": None,
        "prediction_rows": None,
        "gold_error": "",
        "prediction_error": "",
        "transport_error": f"{type(exc).__name__}: {exc}",
    }


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    output_csv: Path,
    *,
    source_path: Path,
    workers: int,
    executor: Any,
    gold_cache: Any,
    timeout: float,
    retries: int,
    count_empty_match: bool,
    order_sensitive: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute prediction pairs in parallel and persist row and aggregate metrics."""
    _, _, atomic_write, evaluate_row, make_summary = _execution_api()
    execution_args = SimpleNamespace(
        input=source_path,
        output=output_csv,
        backend="http",
        http_api="query_v2",
        http_base_url=executor.base_url,
        bolt_uri="",
        timeout=timeout,
        retries=retries,
        count_empty_match=count_empty_match,
        order_sensitive=order_sensitive,
    )
    source_rows = [_execution_input(prediction) for prediction in predictions]
    evaluated: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluate_row, row, execution_args, gold_cache, executor): (
                row,
                prediction,
            )
            for row, prediction in zip(source_rows, predictions)
        }
        for future in as_completed(futures):
            row, prediction = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = _worker_failure(row, exc)
            gold_ir = prediction.get("sirc") or prediction.get("natcypher", "")
            pred_ir = (
                prediction.get("predicted_sirc")
                or prediction.get("predicted_natcypher", "")
            )
            result["gold_sirc"] = gold_ir
            result["gold_natcypher"] = gold_ir
            result["predicted_sirc"] = pred_ir
            result["predicted_natcypher"] = pred_ir
            result["natcypher_exact"] = int(normalized_exact_match(pred_ir, gold_ir))
            evaluated.append(result)

    atomic_write(evaluated, output_csv)
    summary = make_summary(evaluated, execution_args)
    summary["natcypher_exact_matches"] = sum(
        int(row["natcypher_exact"]) for row in evaluated
    )
    summary["natcypher_exact_match"] = (
        summary["natcypher_exact_matches"] / len(evaluated) if evaluated else 0.0
    )
    summary["valid_statuses_exclude"] = sorted(INVALID_EXECUTION_STATUSES)
    return evaluated, summary


def evaluate_checkpoint(
    model: Any,
    tokenizer: Any,
    checkpoint: Path,
    sources: Sequence[Mapping[str, Any]],
    destination: Path,
    *,
    source_path: Path,
    args: argparse.Namespace,
    executor: Any,
    gold_cache: Any,
) -> dict[str, Any]:
    step = checkpoint_step(checkpoint)
    destination.mkdir(parents=True, exist_ok=True)
    predictions = generate_predictions(
        model,
        tokenizer,
        sources,
        step=step,
        max_ir_tokens=args.max_ir_tokens,
        max_cypher_tokens=args.max_cypher_tokens,
    )
    predictions_path = destination / "predictions.jsonl"
    execution_path = destination / "execution.csv"
    write_jsonl(predictions_path, predictions)
    _, summary = evaluate_predictions(
        predictions,
        execution_path,
        source_path=source_path,
        workers=args.workers,
        executor=executor,
        gold_cache=gold_cache,
        timeout=args.timeout,
        retries=args.retries,
        count_empty_match=args.count_empty_match,
        order_sensitive=args.order_sensitive,
    )
    summary.update(
        {
            "step": step,
            "checkpoint": str(checkpoint),
            "predictions": str(predictions_path),
            "execution_csv": str(execution_path),
        }
    )
    write_json(destination / "summary.json", summary)
    return summary


def copy_best_adapter(
    checkpoint: Path, output_dir: Path, summary: Mapping[str, Any]
) -> None:
    destination = output_dir / "best-adapter-by-execution"
    shutil.copytree(checkpoint, destination / "checkpoint", dirs_exist_ok=True)
    write_json(
        destination / "best_metric.json",
        {
            "selected_checkpoint": str(checkpoint),
            "selection_order": [
                "execution_accuracy",
                "mean_result_jaccard",
                "natcypher_exact_match",
                "later_step",
            ],
            **dict(summary),
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    checkpoints = discover_checkpoints(args.checkpoints_dir)
    dev_sources = limit_sources(
        read_jsonl(args.eval_sources), args.sample_limit, args.seed
    )
    evaluation_root = args.output_dir / "execution-eval"
    evaluation_root.mkdir(parents=True, exist_ok=True)

    load_model, _, load_adapter_checkpoint = _inference_api()
    model, tokenizer = load_model(
        SimpleNamespace(checkpoint=checkpoints[0], base_model=args.base_model)
    )
    GoldCache, HttpExecutor, _, _, _ = _execution_api()
    executor = HttpExecutor(
        "https://demo.neo4jlabs.com:7473",
        args.timeout,
        args.retries,
        "query_v2",
    )
    gold_cache = GoldCache(executor)
    summaries: list[dict[str, Any]] = []
    try:
        for index, checkpoint in enumerate(checkpoints):
            if index:
                load_adapter_checkpoint(model, checkpoint)
                model.eval()
            step = checkpoint_step(checkpoint)
            print(f"Evaluating step {step} on {len(dev_sources)} dev examples")
            summaries.append(
                evaluate_checkpoint(
                    model,
                    tokenizer,
                    checkpoint,
                    dev_sources,
                    evaluation_root / f"step-{step:06d}",
                    source_path=args.eval_sources,
                    args=args,
                    executor=executor,
                    gold_cache=gold_cache,
                )
            )

        best = dict(select_best_summary(summaries))
        leaderboard = {
            "eval_sources": str(args.eval_sources),
            "sample_limit": args.sample_limit,
            "seed": args.seed,
            "selection_metric": "execution_accuracy",
            "tie_breakers": [
                "mean_result_jaccard",
                "natcypher_exact_match",
                "later_step",
            ],
            "selected_step": best["step"],
            "checkpoints": sorted(
                summaries, key=ranking_key, reverse=True
            ),
        }
        write_json(evaluation_root / "leaderboard.json", leaderboard)
        selected_checkpoint = Path(best["checkpoint"])
        copy_best_adapter(selected_checkpoint, args.output_dir, best)

        if args.test_file:
            load_adapter_checkpoint(model, selected_checkpoint)
            model.eval()
            test_sources = read_jsonl(args.test_file)
            print(
                f"Evaluating selected step {best['step']} "
                f"on {len(test_sources)} test examples"
            )
            evaluate_checkpoint(
                model,
                tokenizer,
                selected_checkpoint,
                test_sources,
                args.output_dir / "final-test",
                source_path=args.test_file,
                args=args,
                executor=executor,
                gold_cache=gold_cache,
            )
    finally:
        executor.close()

    print(f"Selected step {best['step']}: execution_accuracy={best['execution_accuracy']:.6f}")
    print(f"Wrote {evaluation_root / 'leaderboard.json'}")


if __name__ == "__main__":
    main()
