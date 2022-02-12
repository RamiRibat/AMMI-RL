import time
import wandb

import numpy as np
import torch as T
import torch.nn.functional as F

from mbrl.mbrl import MBRL
from mfrl.sac import SAC



class MBPO(MBRL, SAC):
    """
    Algorithm: Model-Based Policy Optimization (Dyna-style, Model-Based)

        1: Initialize policy πφ, predictive model pθ, environment dataset Denv, model dataset Dmodel
        2: for N epochs do
        3:      Train model pθ on Denv via maximum likelihood
        4:      for E steps do
        5:          Take action in environment according to πφ; add to Denv
        6:          for M model rollouts do
        7:              Sample st uniformly from Denv
        8:              Perform k-step model rollout starting from st using policy πφ; add to Dmodel
        9:          for G gradient updates do
        10:             Update policy parameters on model data: φ ← φ − λπ ˆ∇φ Jπ(φ, Dmodel)

    """
    def __init__(self) -> None:
        pass


    def _build(self):
        pass


    def _set_sac(self):
        pass


    def _seed_fake_world(self):
        pass

    
    def learn(self):
        pass


    def set_rollout_length(self):
        pass


    def rollout_world_model(self):
        pass


    def sac_batch(self):
        pass