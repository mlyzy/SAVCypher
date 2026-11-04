from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from convert.models import UnifiedText2CypherExample


def default_data_root() -> Path:
    env = os.environ.get("SIRC_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    workspace = Path(__file__).resolve().parents[2]
    if (workspace / "text2cypher").exists() or (workspace / "benchmark").exists():
        return workspace
    return Path.cwd()


ROOT = default_data_root()


def set_data_root(path: Path | str) -> Path:
    global ROOT
    ROOT = Path(path).expanduser().resolve()
    return ROOT


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Dataset adapters require pandas") from exc
    return pd


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def load_text2cypher(
    data_path: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[UnifiedText2CypherExample]:
    data_path = data_path or (
        ROOT / "text2cypher/datasets/synthetic_gpt4o_demodbs/text2cypher_gpt4o.csv"
    )
    schema_path = schema_path or (
        ROOT
        / "text2cypher/datasets/synthetic_gpt4o_demodbs/text2cypher_schemas.csv"
    )
    pd = _pandas()
    data = pd.read_csv(data_path)
    schemas = pd.read_csv(schema_path)
    schema_by_db = {
        _text(row["database"]): row for row in schemas.to_dict("records")
    }
    examples = []
    for index, row in enumerate(data.to_dict("records")):
        if limit is not None and len(examples) >= limit:
            break
        graph_key = _text(row.get("database"))
        schema = schema_by_db.get(graph_key, {})
        examples.append(
            UnifiedText2CypherExample(
                example_id=f"text2cypher:{index}",
                question=_text(row.get("question")),
                gold_cypher=_text(row.get("cypher")),
                graph_key=graph_key,
                schema_text=_text(schema.get("schema")),
                schema_struct=_decode_object(schema.get("structured_schema")),
                task_type=_text(row.get("type")),
                metadata={
                    "dataset": "text2cypher",
                    "explanation": _text(row.get("explanation")),
                },
            )
        )
    return examples


def load_cypherbench(
    data_path: Optional[Path] = None,
    schema_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[UnifiedText2CypherExample]:
    data_path = data_path or (ROOT / "benchmark/test.json")
    schema_dir = schema_dir or (ROOT / "benchmark/graphs/schemas")
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    examples = []
    schemas: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if limit is not None and len(examples) >= limit:
            break
        graph_key = _text(row.get("graph"))
        if graph_key not in schemas:
            path = schema_dir / f"{graph_key}_schema.json"
            schemas[graph_key] = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            )
        template = row.get("from_template") or {}
        examples.append(
            UnifiedText2CypherExample(
                example_id=_text(row.get("qid") or f"cypherbench:{index}"),
                question=_text(row.get("nl_question")),
                gold_cypher=_text(row.get("gold_cypher")),
                graph_key=graph_key,
                schema_struct=schemas[graph_key],
                task_type=_text(template.get("match_category")),
                metadata={
                    "dataset": "cypherbench",
                    "answer_json": row.get("answer_json"),
                    "from_template": template,
                },
            )
        )
    return examples


def load_mind_the_query(
    split_root: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    split: str = "test",
    limit: Optional[int] = None,
) -> List[UnifiedText2CypherExample]:
    split_root = split_root or (ROOT / "Mind-the-Query/Train_Test_Splits")
    schema_path = schema_path or (ROOT / "Mind-the-Query/db_schema.jsonl")
    schema_by_db = {}
    with schema_path.open(encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            schema_by_db.update(data)

    examples = []
    for path in sorted(split_root.glob(f"**/{split}/**/*.json")):
        database = _infer_mind_database(path, schema_by_db)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else _find_record_list(payload)
        for index, row in enumerate(rows):
            if limit is not None and len(examples) >= limit:
                return examples
            question = _text(
                row.get("NL Question")
                or row.get("question")
                or row.get("nl_question")
            )
            cypher = _text(row.get("Cypher") or row.get("cypher"))
            if not question or not cypher:
                continue
            examples.append(
                UnifiedText2CypherExample(
                    example_id=_text(
                        row.get("unique_id") or f"mind:{path.stem}:{index}"
                    ),
                    question=question,
                    gold_cypher=cypher,
                    graph_key=database,
                    schema_struct=schema_by_db.get(database, {}),
                    task_type=path.parent.parent.name,
                    metadata={"dataset": "mind_the_query", "source_path": str(path)},
                )
            )
    return examples


def load_synthcypher(
    data_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[UnifiedText2CypherExample]:
    data_path = data_path or (
        ROOT / "SynthCypher/data/test-00000-of-00001.parquet"
    )
    pd = _pandas()
    data = pd.read_parquet(data_path)
    examples = []
    for index, row in enumerate(data.to_dict("records")):
        if limit is not None and len(examples) >= limit:
            break
        examples.append(
            UnifiedText2CypherExample(
                example_id=_text(row.get("Id") or row.get("id") or f"synth:{index}"),
                question=_text(row.get("Question") or row.get("question")),
                gold_cypher=_text(row.get("Cypher") or row.get("cypher")),
                graph_key=_text(row.get("Schema_Type") or row.get("schema_type")),
                schema_text=_text(row.get("Schema") or row.get("schema")),
                task_type=_text(
                    row.get("Question_Type") or row.get("question_type")
                ),
                metadata={"dataset": "synthcypher"},
            )
        )
    return examples


def load_dataset(
    name: str,
    limit: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> List[UnifiedText2CypherExample]:
    if data_root is not None:
        set_data_root(data_root)
    normalized = name.lower().replace("-", "_")
    loaders = {
        "text2cypher": load_text2cypher,
        "cypherbench": load_cypherbench,
        "mind_the_query": load_mind_the_query,
        "synthcypher": load_synthcypher,
    }
    if normalized not in loaders:
        raise ValueError(f"Unknown dataset: {name}")
    return loaders[normalized](limit=limit)


def attach_predictions(
    examples: Sequence[UnifiedText2CypherExample],
    execution_csv: Path,
    dataset: str,
) -> List[UnifiedText2CypherExample]:
    pd = _pandas()
    data = pd.read_csv(execution_csv, low_memory=False)
    pred_column = {
        "text2cypher": "extracted_cypher",
        "cypherbench": "pred_cypher",
        "mind_the_query": "extracted_cypher",
        "synthcypher": "prediction",
    }[dataset]
    id_columns = {
        "text2cypher": "example_index",
        "cypherbench": "qid",
        "mind_the_query": "example_index",
        "synthcypher": "example_index",
    }
    rows = data.to_dict("records")
    by_position = {index: row for index, row in enumerate(rows)}
    by_id = {_text(row.get(id_columns[dataset])): row for row in rows}
    attached = []
    for index, example in enumerate(examples):
        row = by_id.get(example.example_id) or by_position.get(index, {})
        example.prediction = _text(row.get(pred_column))
        example.metadata.update(
            {
                "execution_status": _text(row.get("status")),
                "execution_correct": row.get("execution_correct"),
                "result_jaccard": row.get("result_jaccard"),
            }
        )
        attached.append(example)
    return attached


def _decode_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not _text(value):
        return {}
    import ast

    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, dict) else {}


def _find_record_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        if any(key in value for key in ("NL Question", "question", "Cypher", "cypher")):
            return [value]
        for nested in value.values():
            found = _find_record_list(nested)
            if found:
                return found
    return []


def _infer_mind_database(path: Path, schemas: Dict[str, Any]) -> str:
    parts = {part.lower(): part for part in path.parts}
    for key in schemas:
        if key.lower() in parts or key.lower() in str(path).lower():
            return key
    return path.parent.name
