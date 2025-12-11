import torch

from src.base.registries import OptimizerRegistry


OptimizerRegistry.registry["adam"] = torch.optim.Adam
OptimizerRegistry.registry["adamw"] = torch.optim.AdamW
OptimizerRegistry.registry["cosine_lr"] = (
    torch.optim.lr_scheduler.CosineAnnealingLR
)
