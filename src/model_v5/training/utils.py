import torch
from typing import Optional, Dict
import os


def compile_model(model, mode: str = "reduce-overhead", backend: str = "inductor"):

    try:
        compiled = torch.compile(model, mode=mode, backend=backend)
        print(
            f"[compile] Model compiled with mode={mode}, backend={backend}"
        )
        return compiled
    except Exception as e:
        print(f"[compile] Failed: {e}. Using uncompiled model.")
        return model


def get_optimal_dataloader_kwargs(
    num_cpu_cores: Optional[int] = None,
) -> Dict:

    if num_cpu_cores is None:
        num_cpu_cores = os.cpu_count() or 4

    return {
        "pin_memory": torch.cuda.is_available(),
        "num_workers": min(4, num_cpu_cores),
        "prefetch_factor": 2,
        "persistent_workers": True,
    }


class BatchSizeConfig:

    def __init__(
        self,
        target_batch: int = 128,
        micro_batch: int = 16,
        grad_accum_steps: Optional[int] = None,
    ):
        self.target_batch = target_batch
        self.micro_batch = micro_batch

        if grad_accum_steps is not None:
            self.grad_accum_steps = grad_accum_steps
        else:
            self.grad_accum_steps = max(1, target_batch // micro_batch)

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum_steps

    def __repr__(self) -> str:
        return (
            f"BatchSizeConfig(micro={self.micro_batch}, "
            f"accum={self.grad_accum_steps}, "
            f"effective={self.effective_batch})"
        )
