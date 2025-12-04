import torch.nn as nn

from src.base.registries import LossRegistry


LossRegistry.registry["mse"] = nn.MSELoss
