from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

try:
    from .schema_format import SCHEMA_FORMAT_VERSION, format_schema
except ImportError:
    from schema_format import SCHEMA_FORMAT_VERSION, format_schema


TASK_NAMES = {
    0: "text_to_sirc",
    1: "sirc_to_cypher",
}
SYSTEM_PROMPTS = {
    0: (
        "Translate the natural-language graph question into canonical SIRC. "
        "Use only the supplied schema and return SIRC text only."
    ),
    1: (
        "Compile the canonical SIRC intermediate representation into one "
        "executable read-only Neo4j Cypher query. Use only the supplied schema "
        "and return Cypher text only."
    ),
}


def make_task_input(
    task_id: int,
    *,
    schema: Any,
    question: str = "",
    natcypher: str = "",
    sirc: str = "",
) -> str:
    schema_text = format_schema(schema, question=question)
    ir_text = (sirc or natcypher).strip()
    if task_id == 0:
        return f"Question:\n{question.strip()}\n\nSchema:\n{schema_text.strip()}"
    if task_id == 1:
        return f"SIRC:\n{ir_text}\n\nSchema:\n{schema_text.strip()}"
    raise ValueError(f"Unknown task_id: {task_id}")


def load_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["task_id"]) not in TASK_NAMES:
                raise ValueError(f"{path}:{line_number}: invalid task_id")
            if record.get("schema_format") != SCHEMA_FORMAT_VERSION:
                raise ValueError(
                    f"{path}:{line_number}: expected schema_format="
                    f"{SCHEMA_FORMAT_VERSION!r}; regenerate data with python -m prepare"
                )
            for field in ("system", "input", "output"):
                if not str(record.get(field, "")).strip():
                    raise ValueError(f"{path}:{line_number}: empty {field}")
            records.append(record)
    if not records:
        raise ValueError(f"No training records found in {path}")
    return records


def render_prompt(tokenizer, record: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": record["system"]},
        {"role": "user", "content": record["input"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def tokenize_record(
    tokenizer, record: Dict[str, Any], max_length: int
) -> Dict[str, Any]:
    """Tokenize without silently dropping any prompt or completion tokens."""
    prompt_ids = tokenizer.encode(
        render_prompt(tokenizer, record), add_special_tokens=False
    )
    completion = str(record["output"]).rstrip() + "<|im_end|>\n"
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    total_length = len(prompt_ids) + len(completion_ids)
    if total_length > max_length:
        raise ValueError(
            "Training record exceeds token budget: "
            f"prompt={len(prompt_ids)}, completion={len(completion_ids)}, "
            f"total={total_length}, max_length={max_length}. "
            "Regenerate compact data or increase --max-length; truncation is disabled."
        )
    input_ids = prompt_ids + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids.copy(),
        "task_id": int(record["task_id"]),
    }


class SIRCMTLDataset(Dataset):
    def __init__(self, path: Path | str, tokenizer, max_length: int):
        self.records = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return tokenize_record(
            self.tokenizer, self.records[index], self.max_length
        )


class TaskAwareDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(
        self, features: Sequence[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(len(feature["input_ids"]) for feature in features)
        multiple = self.pad_to_multiple_of
        if multiple:
            max_length = ((max_length + multiple - 1) // multiple) * multiple
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        task_ids = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(
                feature["attention_mask"] + [0] * padding
            )
            batch["labels"].append(feature["labels"] + [-100] * padding)
            task_ids.append(int(feature["task_id"]))
        result = {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }
        result["task_id"] = torch.tensor(task_ids, dtype=torch.long)
        return result

