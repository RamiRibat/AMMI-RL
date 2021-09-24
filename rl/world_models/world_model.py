# TODO: world_model
import random
import numpy as np
import torch as T
from torch.distributions.normal import Normal
nn = T.nn

from .mlp import MLPNet


LOG_SIGMA_MAX = 2
LOG_SIGMA_MIN = -20



class WorldModel(nn.Module):
    def __init__(self) -> None:
        pass