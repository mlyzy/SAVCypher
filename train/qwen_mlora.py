from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch import nn

from train.mtl_lora.adapter.mlora import mLoRALinear
from train.mtl_lora.utils import set_no_grad, wrap_model


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_a",
    "in_proj_b",
    "in_proj_z",
    "out_proj",
)
LANGUAGE_MODEL_PREFIX = "model.language_model.layers"


def iter_mlora_layers(model: nn.Module) -> Iterable[mLoRALinear]:
    for module in model.modules():
        if isinstance(module, mLoRALinear):
            yield module


def set_task_ids(model: nn.Module, task_ids: Optional[torch.Tensor]) -> None:
    layers = list(iter_mlora_layers(model))
    if not layers:
        raise RuntimeError("The model has no mLoRALinear layers")
    for layer in layers:
        layer.set_task_ids(task_ids)


class TaskAwareCausalLM(nn.Module):
    """Keep Hugging Face model internals unchanged while routing MTL task IDs."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self._mlora_adapter_config = getattr(
            base_model, "_mlora_adapter_config", None
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        task_id=None,
        lambda_index=None,
        **kwargs,
    ):
        selected_task = task_id if task_id is not None else lambda_index
        if selected_task is None:
            raise ValueError("task_id is required for MTL-LoRA forward")
        set_task_ids(self.base_model, selected_task)
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, *args, task_id=None, lambda_index=None, **kwargs):
        selected_task = task_id if task_id is not None else lambda_index
        if selected_task is None:
            raise ValueError("task_id is required for MTL-LoRA generation")
        set_task_ids(self.base_model, selected_task)
        return self.base_model.generate(*args, **kwargs)

    def enable_input_require_grads(self):
        return self.base_model.enable_input_require_grads()

    def gradient_checkpointing_enable(self, **kwargs):
        return self.base_model.gradient_checkpointing_enable(**kwargs)


def inject_mlora(
    model: nn.Module,
    *,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    module_prefix: str = LANGUAGE_MODEL_PREFIX,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    task_count: int = 2,
    b_experts: int = 3,
    temperature: float = 0.1,
) -> TaskAwareCausalLM:
    config: Dict[str, Any] = {
        "type": "mlora",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "lambda_num": task_count,
        "B_num": b_experts,
        "B_scale": temperature,
        "diagonal_format": False,
    }
    wrap_model(
        model,
        target_layer=target_modules,
        adapter_config=config,
        module_prefix=module_prefix,
    )
    set_no_grad(model, verbose=False)
    return TaskAwareCausalLM(model)


def load_adapter_checkpoint(
    model: nn.Module, checkpoint: Path | str
) -> torch.nn.modules.module._IncompatibleKeys:
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "final_checkpoint.pt"
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected adapter checkpoint keys: {result.unexpected_keys[:10]}"
        )
    return result


def read_adapter_config(checkpoint: Path | str) -> Dict[str, Any]:
    checkpoint = Path(checkpoint)
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    path = directory / "adapter_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing adapter metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

