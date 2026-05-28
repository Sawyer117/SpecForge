import os

import torch
import torch.distributed as dist

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
    ):
        # TODO: For now, we only support cosine annealing warmup lr scheduler and AdamW optimizer
        # TODO: We should make these parameters configurable
        #   These magic numbers: weight_decay=0.0, max_grad_norm=0.5, total_steps=800k, warmup_steps=12k are copied from
        #   https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def step(self):
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )

        # ============================================================
        # DIAG (issue #9): clip_grad_norm_ on FSDP-sharded params sees
        # only the LOCAL shard's L2 norm, not the true global norm.
        # Set SPECFORGE_DIAG_CLIP_NORM=1 to print per-rank local vs
        # global norm + the clip factor each version would apply.
        # Default off => zero overhead, zero behavior change.
        # ============================================================
        if int(os.environ.get("SPECFORGE_DIAG_CLIP_NORM", "0")):
            with torch.no_grad():
                grads = [p.grad for p in self.fp32_params if p.grad is not None]
                if grads:
                    local_sq = torch.zeros(
                        (), device=grads[0].device, dtype=torch.float32
                    )
                    for g in grads:
                        local_sq = local_sq + (g.detach().float() ** 2).sum()
                    global_sq = local_sq.clone()
                    world = (
                        dist.get_world_size()
                        if dist.is_available() and dist.is_initialized()
                        else 1
                    )
                    if world > 1:
                        dist.all_reduce(global_sq, op=dist.ReduceOp.SUM)
                    rank = dist.get_rank() if world > 1 else 0
                    local_norm = local_sq.sqrt().item()
                    global_norm = global_sq.sqrt().item()
                    ratio = local_norm / max(global_norm, 1e-12)
                    local_clip = min(
                        1.0, self.max_grad_norm / max(local_norm, 1e-12)
                    )
                    global_clip = min(
                        1.0, self.max_grad_norm / max(global_norm, 1e-12)
                    )
                    fires_buggy = (
                        "BUGGY_FIRES" if local_clip < 1.0 else "no_local_clip"
                    )
                    fires_true = (
                        "TRUE_FIRES" if global_clip < 1.0 else "no_global_clip"
                    )
                    print(
                        f"[CLIP_DIAG r{rank}/{world}] "
                        f"local={local_norm:.4f} global={global_norm:.4f} "
                        f"ratio={ratio:.4f} | "
                        f"local_clip={local_clip:.4f} global_clip={global_clip:.4f} "
                        f"({fires_buggy}, {fires_true})",
                        flush=True,
                    )

        torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
