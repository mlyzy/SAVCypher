from contextlib import nullcontext

import torch
import torch.nn as nn

try:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
except ImportError:  # Adapter unit tests and CPU inspection do not require DeepSpeed.
    deepspeed = None
    ZeroParamStatus = None


class EmptyContext:
    def __enter__(self):
        pass

    def __exit__(self, *exec):
        pass


def should_gather(param):
    if deepspeed is None:
        return nullcontext()
    if param is not None:
        should = (
            hasattr(param, "ds_id")
            and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        )
        return deepspeed.zero.GatheredParameters(
            param, modifier_rank=0, enabled=should
        )
    else:
        return EmptyContext()


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
        tunable_scaler: bool = False,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        if tunable_scaler:
            self.lora_scaler = nn.Parameter(self.weight.new_zeros(()))
        else:
            self.lora_scaler = None

    def compute_tunable_scale(self, requires_grad=False):
        if self.lora_scaler is None:
            return 1.0
        elif requires_grad:
            return torch.sigmoid(self.lora_scaler)
        else:
            return torch.sigmoid(self.lora_scaler.data).item()
