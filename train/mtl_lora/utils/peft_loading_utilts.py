import os
import json
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..adapter import (
    DoRALinear,
    LoRALinear,
    MMOELoraLinear,
    MultiLoRALinear,
    mLoRALinear,
    mLoRAMergedLinear,
)
from .dist import get_global_rank

ADAPTER_MAPPING = {
    "mlora": mLoRALinear,
    "mlora_merged": mLoRAMergedLinear,
    "multilora": MultiLoRALinear,
    "moelora": MMOELoraLinear,
    "dora": DoRALinear,
    "lora": LoRALinear,
}


def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_lora_param_maybe_zero_3(named_params, valid_keys: List[str] = ["lora"]):
    to_return = {k: v for k, v in named_params if any([x in k for x in valid_keys])}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def save_pretrain(
    model,
    output_dir: str,
    prefix: List[str] = [
        "lora",
    ],
) -> None:
    # Every ZeRO-3 rank must enter GatheredParameters in the same order.
    # Only gating this operation on rank 0 deadlocks after other ranks exit.
    return_dict = get_lora_param_maybe_zero_3(
        model.named_parameters(), valid_keys=prefix
    )
    if get_global_rank() == 0:
        # Filtering before gathering avoids collecting every frozen 35B
        # parameter and exhausting host memory.
        output_dit = os.path.join(output_dir, "checkpoint")
        os.makedirs(output_dit, exist_ok=True)
        path = os.path.join(output_dit, f"final_checkpoint.pt")
        torch.save(return_dict, path)
        metadata = getattr(model, "_mlora_adapter_config", None)
        if metadata is not None:
            with open(
                os.path.join(output_dit, "adapter_config.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
        print(f"Model saved to {path}")


def set_no_grad(model, logger=None, verbose=True) -> None:
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_params += num_params
        if param.requires_grad:
            if verbose:
                print(name)
            trainable_params += num_params

    if logger is not None:
        logger.info(
            f"Trainable params: {trainable_params:,d}, All params: {all_params:,d}, trainable: {100 * trainable_params/all_params:.2f}"
        )
    else:
        print(
            f"Trainable params: {trainable_params:,d}, All params: {all_params:,d}, trainable: {100 * trainable_params/all_params:.2f}"
        )


def wrap_model(
    model: nn.Module,
    target_layer: Sequence[str],
    adapter_config: Dict[str, Any],
    module_prefix: Optional[str] = None,
):
    def _resolve_submodule(path):
        if hasattr(model, "get_submodule"):
            return model.get_submodule(path)
        current = model
        for part in path.split("."):
            current = getattr(current, part)
        return current

    def _get_submodule(key):
        parent = _resolve_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = _resolve_submodule(key)
        return parent, target, target_name

    config = dict(adapter_config)
    adapter_type = config.pop("type", "").lower()
    assert (
        adapter_type in ADAPTER_MAPPING.keys()
    ), f"Adapter {adapter_type} not supported"

    adapter_class = ADAPTER_MAPPING[adapter_type]
    adapter_impl = partial(adapter_class, **config)

    key_list = [key for key, _ in model.named_modules()]
    replaced = []

    for key in key_list:
        if module_prefix and not key.startswith(module_prefix):
            continue
        if not any(key == layer or key.endswith("." + layer) for layer in target_layer):
            continue
        target = _resolve_submodule(key)
        if not isinstance(target, nn.Linear):
            continue
        if isinstance(target, adapter_class):
            continue
        if hasattr(target.weight, "quant_state"):
            raise TypeError(
                f"Quantized module {key} is not supported by the custom MTL-LoRA layer"
            )
        if any(True for _ in target.children()):
            continue
        if key:
            parent, target, target_name = _get_submodule(key)

            adapter_module = adapter_impl(
                in_features=target.in_features,
                out_features=target.out_features,
                bias=target.bias is not None,
            )
            # Under ZeRO-3 the original parameter is already partitioned and
            # target.state_dict()["weight"] is a size-0 local shard. Reuse the
            # Parameter object instead of trying to copy that shard into a
            # newly allocated dense weight.
            adapter_module.weight = target.weight
            adapter_module.bias = target.bias
            for parameter_name, parameter in adapter_module.named_parameters(
                recurse=True
            ):
                if parameter_name.startswith("lora"):
                    parameter.data = parameter.data.to(
                        device=target.weight.device, dtype=target.weight.dtype
                    )
            setattr(parent, target_name, adapter_module)
            replaced.append(key)

    if not replaced:
        raise ValueError(
            "No nn.Linear modules matched target layers "
            f"{list(target_layer)} with prefix {module_prefix!r}"
        )
    model._mlora_replaced_modules = replaced
    model._mlora_adapter_config = {
        "type": adapter_type,
        **config,
        "target_modules": list(target_layer),
        "module_prefix": module_prefix,
        "replaced_modules": replaced,
    }

    return model
