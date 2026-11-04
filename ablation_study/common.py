from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

ABLATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ABLATION_ROOT.parent
MTL_ROOT = REPO_ROOT / "MTL-LoRA"
for _root in (REPO_ROOT, MTL_ROOT):
    text = str(_root)
    if text not in sys.path:
        sys.path.insert(0, text)

from independent_two_lora.cypher_prefix_automaton import CypherPrefixAutomaton
from natcypher.train_ir_lora import resolve_checkpoint
from src.utils import wrap_model

DEFAULT_BASE_MODEL = REPO_ROOT / "models/Qwen__Qwen3.6-35B-A3B"
DEFAULT_DATA = REPO_ROOT / "natcypher_data/compact_v03_text2cypher"
DEFAULT_DEEPSPEED = MTL_ROOT / "config/ds3_natcypher.json"
DEFAULT_TEST_SOURCES = DEFAULT_DATA / "test_sources.jsonl"
DEFAULT_STAGE1_DIR = REPO_ROOT / "outputs/ablation/wo_automaton/stage1"
DEFAULT_STAGE2_DIR = REPO_ROOT / "outputs/ablation/wo_automaton/stage2"
DEFAULT_WO_SIRC_DIR = REPO_ROOT / "outputs/ablation/wo_sirc"
DEFAULT_WO_BOTH_DIR = REPO_ROOT / "outputs/ablation/wo_both"
TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,"
    "in_proj_qkv,in_proj_a,in_proj_b,in_proj_z,out_proj"
)


def resolve_adapter_checkpoint(path: Path) -> Path:
    """Accept a run directory, a step directory, or a nested checkpoint folder."""
    path = Path(path)
    candidates = [path, path / "checkpoint"]
    if path.is_dir():
        steps = sorted(path.glob("adapter-checkpoints/step-*"))
        if steps:
            candidates.extend([steps[-1] / "checkpoint", steps[-1]])
    for candidate in candidates:
        if (candidate / "final_checkpoint.pt").is_file() and (
            candidate / "adapter_config.json"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        f"No trained adapter found under {path}. Run the matching train.py first."
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def clean_generation(text: str) -> str:
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    fence = re.fullmatch(r"```(?:cypher|natcypher|sirc|text)?\s*(.*?)```", text, re.DOTALL | re.I)
    return fence.group(1).strip() if fence else text


def sirc_text(source: dict[str, Any]) -> str:
    return str(
        source.get("predicted_sirc")
        or source.get("predicted_natcypher")
        or source.get("sirc")
        or source.get("natcypher")
        or ""
    ).strip()


def load_standard_lora(checkpoint: Path, base_model: Path | None = None):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    weights_path, config_path = resolve_checkpoint(checkpoint)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("type") != "lora":
        raise ValueError(f"Expected a standard LoRA adapter, got {config.get('type')!r}")
    model_path = base_model or Path(config.get("base_model") or DEFAULT_BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    wrap_model(
        model,
        target_layer=config["target_modules"],
        adapter_config={
            "type": "lora",
            "r": int(config["r"]),
            "lora_alpha": int(config["lora_alpha"]),
            "lora_dropout": float(config["lora_dropout"]),
        },
        module_prefix=config.get("module_prefix"),
    )
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    missing_lora = [key for key in incompatible.missing_keys if "lora" in key]
    if incompatible.unexpected_keys or missing_lora:
        raise RuntimeError(
            "Adapter mismatch: "
            f"missing_lora={missing_lora[:5]}, unexpected={incompatible.unexpected_keys[:5]}"
        )
    model.eval()
    return model, tokenizer


def release_model(model, tokenizer) -> None:
    del model
    del tokenizer
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def render_chat(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.no_grad()
def generate_unconstrained(
    model,
    tokenizer,
    prompt: str,
    *,
    source_id: Any = "<unknown>",
    max_input_tokens: int = 1024,
    max_new_tokens: int = 512,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    prompt_length = int(encoded["input_ids"].shape[1])
    if prompt_length > max_input_tokens:
        raise ValueError(
            f"Prompt for {source_id!r} requires {prompt_length} tokens "
            f"(limit {max_input_tokens}); truncation is disabled"
        )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=tokenizer.pad_token_id,
    )
    raw = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
    return clean_generation(raw)


def _decoded_candidate(tokenizer, generated: Sequence[int], token_id: int) -> str:
    return tokenizer.decode(
        [*generated, token_id],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


@torch.no_grad()
def generate_constrained_cypher(
    model,
    tokenizer,
    prompt: str,
    schema: str,
    *,
    source_id: Any = "<unknown>",
    max_input_tokens: int = 1024,
    max_new_tokens: int = 256,
    max_output_characters: int = 1024,
    candidate_limit: int = 4096,
) -> tuple[str, dict[str, Any]]:
    """Greedy decode, keeping only tokens that stay valid under the prefix automaton."""
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    prompt_length = int(encoded["input_ids"].shape[1])
    if prompt_length > max_input_tokens:
        raise ValueError(
            f"Constrained prompt for {source_id!r} requires "
            f"{prompt_length} tokens (limit {max_input_tokens})"
        )
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    automaton = CypherPrefixAutomaton.from_schema(schema)
    generated: list[int] = []
    past_key_values = None
    interventions = 0
    rejected_tokens = 0
    max_rank_checked = 0
    terminated = False
    stop_reason = "max_new_tokens"
    end_ids = {
        token_id
        for token_id in (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
        if token_id is not None and token_id >= 0
    }
    special_ids = set(tokenizer.all_special_ids) - end_ids

    for _ in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        outputs = model(**model_inputs, return_dict=True)
        logits = outputs.logits[0, -1]
        past_key_values = outputs.past_key_values
        ranked = torch.argsort(logits, descending=True)
        limit = len(ranked) if candidate_limit <= 0 else min(candidate_limit, len(ranked))
        selected_id = None
        selected_state = None
        selected_rank = None

        for rank, candidate_tensor in enumerate(ranked[:limit], start=1):
            candidate_id = int(candidate_tensor.item())
            if candidate_id in special_ids:
                continue
            if candidate_id in end_ids:
                if automaton.accepts():
                    selected_id = candidate_id
                    selected_state = automaton
                    selected_rank = rank
                    terminated = True
                    stop_reason = "eos"
                    break
                rejected_tokens += 1
                continue
            candidate_text = _decoded_candidate(tokenizer, generated, candidate_id)
            if not candidate_text:
                rejected_tokens += 1
                continue
            candidate_state = CypherPrefixAutomaton(automaton.schema).advance(
                candidate_text
            )
            if candidate_state.valid:
                selected_id = candidate_id
                selected_state = candidate_state
                selected_rank = rank
                break
            rejected_tokens += 1

        if selected_id is None:
            stop_reason = "no_viable_token_within_candidate_limit"
            break
        max_rank_checked = max(max_rank_checked, int(selected_rank or 1))
        if selected_rank and selected_rank > 1:
            interventions += 1
        if selected_id in end_ids:
            break
        generated.append(selected_id)
        automaton = selected_state or automaton
        if len(automaton.text) >= max_output_characters:
            stop_reason = "max_output_characters"
            break
        next_token = torch.tensor([[selected_id]], dtype=input_ids.dtype, device=device)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device),
            ),
            dim=1,
        )

    text = tokenizer.decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    accepted = automaton.accepts()
    telemetry = {
        "accepted": accepted,
        "termination": stop_reason,
        "terminated_with_eos": terminated,
        "generated_tokens": len(generated),
        "intervention_steps": interventions,
        "rejected_token_candidates": rejected_tokens,
        "max_rank_checked": max_rank_checked,
        "candidate_limit": candidate_limit,
        "max_output_characters": max_output_characters,
        "acceptance_error": "" if accepted else automaton.acceptance_error(),
    }
    return text, telemetry


def add_lora_train_arguments(parser, *, output_dir: Path) -> None:
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_DATA / "train.jsonl")
    parser.add_argument("--dev-file", type=Path, default=DEFAULT_DATA / "dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--deepspeed", type=Path, default=DEFAULT_DEEPSPEED)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--eval-strategy", choices=("no", "epoch", "steps"), default="no"
    )
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--adapter-save-steps", type=int, default=50)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=TARGET_MODULES)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local-rank", "--local_rank", dest="local_rank", type=int, default=-1
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-overlength-dev", action="store_true")


def lora_train_argv(
    args,
    *,
    output_dir: Path | None = None,
    flavor: str = "ir",
) -> list[str]:
    destination = output_dir or args.output_dir
    argv = [
        "--base-model", str(args.base_model),
        "--train-file", str(args.train_file),
        "--dev-file", str(args.dev_file),
        "--output-dir", str(destination),
        "--deepspeed", str(args.deepspeed),
        "--max-length", str(args.max_length),
        "--epochs", str(args.epochs),
        "--max-steps", str(args.max_steps),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--warmup-ratio", str(args.warmup_ratio),
        "--logging-steps", str(args.logging_steps),
        "--eval-strategy", str(args.eval_strategy),
        "--adapter-save-steps", str(args.adapter_save_steps),
        "--rank", str(args.rank),
        "--alpha", str(args.alpha),
        "--dropout", str(args.dropout),
        "--target-modules", str(args.target_modules),
        "--seed", str(args.seed),
        "--local-rank", str(args.local_rank),
    ]
    if flavor != "direct":
        argv.extend(["--eval-steps", str(args.eval_steps)])
        if getattr(args, "allow_overlength_dev", False):
            argv.append("--allow-overlength-dev")
    if args.adapter_checkpoint:
        argv.extend(["--adapter-checkpoint", str(args.adapter_checkpoint)])
    if args.no_gradient_checkpointing:
        argv.append("--no-gradient-checkpointing")
    return argv


def execute_predictions(
    predictions: list[dict[str, Any]],
    *,
    source_path: Path,
    output_dir: Path,
    workers: int = 8,
    timeout: float = 90.0,
    retries: int = 3,
) -> dict[str, Any]:
    from natcypher.eval_execution import _execution_api, evaluate_predictions, write_json

    rows = [
        {**prediction, "example_index": prediction.get("example_index", index)}
        for index, prediction in enumerate(predictions)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_path = output_dir / "execution.csv"
    GoldCache, HttpExecutor, _, _, _ = _execution_api()
    executor = HttpExecutor(
        "https://demo.neo4jlabs.com:7473", timeout, retries, "query_v2"
    )
    try:
        gold_cache = GoldCache(executor)
        _, summary = evaluate_predictions(
            rows,
            execution_path,
            source_path=source_path,
            workers=workers,
            executor=executor,
            gold_cache=gold_cache,
            timeout=timeout,
            retries=retries,
            count_empty_match=False,
            order_sensitive=False,
        )
    finally:
        executor.close()
    summary["execution_csv"] = str(execution_path)
    write_json(output_dir / "summary.json", summary)
    return summary

