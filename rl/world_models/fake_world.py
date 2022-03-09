''' Source: MBPO/mbpo/models/fake_env.py '''

import numpy as np
import torch as T

# T.multiprocessing.set_sharing_strategy('file_system')





class FakeWorld:

    def __init__(self, world_model, static_fns, env_name, train_env, configs, device):
        # print('init FakeWorld!')
        self.world_model = world_model
        self.static_fns = static_fns
        self.env_name = env_name
        self.train_env = train_env
        self.configs = configs
        self._device_ = device


    def step(self, Os, As, deterministic=False): ###
        device = self._device_
        # assert len(Os.shape) == len(As.shape) ###
        if len(Os.shape) != len(As.shape) or len(Os.shape) == 1: # not a batch
            Os = Os[None]
            As = As[None]

        Os = T.as_tensor(Os, dtype=T.float32).to(device)
        As = T.as_tensor(As, dtype=T.float32).to(device)

        # Predictions
        sample_type = self.configs['world_model']['sample_type']
        with T.no_grad():
            # Os_next, Rs, MEANs, SIGMAs = self.world_model(Os, As, deterministic, sample_type)
            Os_next, _, MEANs, SIGMAs = self.world_model(Os, As, deterministic, sample_type)

        # Terminations
        if self.env_name[:4] == 'pddm':
            Ds = np.zeros([Os.shape[0], 1], dtype=bool)
            _, D = self.train_env.get_reward(Os.detach().cpu().numpy(),
                                             As.detach().cpu().numpy())
            Ds[:,0] = D[:]
        else:
            Rs = self.static_fns.reward_fn(Os.detach().cpu().numpy(), As.detach().cpu().numpy())

            Ds = self.static_fns.termination_fn(Os.detach().cpu().numpy(),
                                                As.detach().cpu().numpy(),
                                                Os_next.detach().cpu().numpy())

        INFOs = {'mu': MEANs.detach().cpu().numpy(), 'sigma': SIGMAs.detach().cpu().numpy()}
        # INFOs = None

        return Os_next, Rs, Ds, INFOs ###



    def step_model(self, Os, As, m, deterministic=False): ###
        device = self._device_
        # assert len(Os.shape) == len(As.shape) ###
        if len(Os.shape) != len(As.shape) or len(Os.shape) == 1: # not a batch
            Os = Os[None]
            As = As[None]

        Os = T.as_tensor(Os, dtype=T.float32).to(device)
        As = T.as_tensor(As, dtype=T.float32).to(device)

        # Predictions
        sample_type = m
        with T.no_grad():
            Os_next, Rs, MEANs, SIGMAs = self.world_model(Os, As, deterministic, sample_type)

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

        INFOs = {'mu': Means.detach().cpu().numpy(), 'sigma': SIGMA.detach().cpu().numpy()}
        # INFOs = None

        return Os_next, Rs, Ds, INFOs ###


    def step_np(self, Os, As, deterministic=False):
        Os_next, Rs, Ds, INFOs = self.step(Os, As, deterministic)
        # Rs = Rs.detach().cpu().numpy()
        Os_next = Os_next.detach().cpu().numpy()
        return Os_next, Rs, Ds, INFOs


    def train(self, data_module):
        data_module.update_dataset()
        # JTrainLog, JValLog, LossTest, WMLogs = self.world_model.train_WM(data_module)
        JTrainLog, JValLog, LossTest = self.world_model.train_WM(data_module)
        return JTrainLog, JValLog, LossTest#, WMLogs


    def close(self):
        pass
