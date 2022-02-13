''' Source: MBPO/mbpo/models/fake_env.py '''

# Imports
import numpy as np
import torch as th

# import pdb

class FakeWorld:

    def __init__(self, dyn_models, static_fns, env_name, train_env, config):
        print('Initialize Fake Environment!')
        self.dyn_models = dyn_models
        self.static_fns = static_fns
        self.env_name = env_name
        self.train_env = train_env
        self.config = config


    def step(self, Os, As, deterministic=False): ###
        device = self.config['Experiment']['device']
        # assert len(Os.shape) == len(As.shape) ###
        if len(Os.shape) != len(As.shape) or len(Os.shape) == 1: # not a batch
            Os = Os[None]
            As = As[None]

        Os = th.as_tensor(Os, dtype=th.float32).to(device)
        As = th.as_tensor(As, dtype=th.float32).to(device)

        # Predictions
        sample_type = self.config['Model']['Sample_type']
        with th.no_grad():
            Os_next, Rs, Means, STDs = self.dyn_models(Os, As, deterministic, sample_type)

        # Terminations
        if self.env_name[:4] == 'pddm':
            Ds = np.zeros([Os.shape[0], 1], dtype=bool)
            _, D = self.train_env.get_reward(Os.detach().cpu().numpy(),
                                             As.detach().cpu().numpy())
            Ds[:,0] = D[:]
        else:
            Ds = self.static_fns.termination_fn(Os.detach().cpu().numpy(),
                                                As.detach().cpu().numpy(),
                                                Os_next.detach().cpu().numpy())

        INFOs = {'mean': Means.detach().cpu().numpy(), 'std': STDs.detach().cpu().numpy()}
        # INFOs = None

        return Os_next, Rs, Ds, INFOs ###



    def step_model(self, Os, As, m, deterministic=False): ###
        device = self.config['Experiment']['device']
        # assert len(Os.shape) == len(As.shape) ###
        if len(Os.shape) != len(As.shape) or len(Os.shape) == 1: # not a batch
            Os = Os[None]
            As = As[None]

        Os = th.as_tensor(Os, dtype=th.float32).to(device)
        As = th.as_tensor(As, dtype=th.float32).to(device)

        # Predictions
        sample_type = m
        with th.no_grad():
            Os_next, Rs, Means, STDs = self.dyn_models(Os, As, deterministic, sample_type)

        # Terminations
        if self.env_name[:4] == 'pddm':
            Ds = np.zeros([Os.shape[0], 1], dtype=bool)
            _, D = self.train_env.get_reward(Os.detach().cpu().numpy(),
                                             As.detach().cpu().numpy())
            Ds[:,0] = D[:]
        else:
            Ds = self.static_fns.termination_fn(Os.detach().cpu().numpy(),
                                                As.detach().cpu().numpy(),
                                                Os_next.detach().cpu().numpy())

        INFOs = {'mean': Means.detach().cpu().numpy(), 'std': STDs.detach().cpu().numpy()}
        # INFOs = None

        return Os_next, Rs, Ds, INFOs ###


    def step_np(self, Os, As, deterministic=False):
        Os_next, Rs, Ds, INFOs = self.step(Os, As, deterministic)
        Rs = Rs.detach().cpu().numpy()
        Os_next = Os_next.detach().cpu().numpy()
        return Os_next, Rs, Ds, INFOs


    def train(self, env_buffer, batch_size):
        Jmodel, mEpochs = self.dyn_models.trainModels(env_buffer, batch_size)
        return Jmodel, mEpochs

    def close(self):
        pass
