''' Source: MBPO/mbpo/models/fake_env.py '''

import numpy as np
import torch as T

# T.multiprocessing.set_sharing_strategy('file_system')


class FakeWorld:
    """
    source: https://github.com/Xingyu-Lin/mbpo_pytorch/predict_env.py
    """
    def __init__(self, model, env_name="Hopper-v2", model_type='pytorch'):
        self.model = model
        self.env_name = env_name
        self.model_type = model_type

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if env_name == "Hopper-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif 'walker_' in env_name:
            torso_height =  next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            done = done[:, None]
            return done

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        # ## [ num_networks, batch_size ]
        # log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))
        # ## [ batch_size ]
        # prob = np.exp(log_prob).sum(0)
        # ## [ batch_size ]
        # log_prob = np.log(prob)
        # stds = np.std(means, 0).mean(-1)

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * T.log(2 * T.tensor(np.pi)) + T.log(variances).sum(-1) + (T.pow(x - means, 2) / variances).sum(-1))
        ## [ batch_size ]
        prob = T.exp(log_prob).sum(0)
        ## [ batch_size ]
        log_prob = T.log(prob)
        stds = T.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        # print('obs: ', obs)
        # print('act: ', act)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        # inputs = np.concatenate((obs, act), axis=-1)
        inputs = T.cat((obs, act), axis=-1)

        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)

        ensemble_model_means[:, :, 1:] += obs
        # ensemble_model_stds = np.sqrt(ensemble_model_vars)
        ensemble_model_stds = T.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            size = ensemble_model_means.shape
            # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            ensemble_samples = ensemble_model_means + T.normal(0, 1, size=size) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
            model_idxes = T.as_tensor(model_idxes)
        # else:
        #     model_idxes = self.model.random_inds(batch_size)

        # batch_idxes = np.arange(0, batch_size)
        batch_idxes = T.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info

    def train_fake_world(self, buffer):
        # Get all samples from environment
        # data = buffer.return_all_np_stack()
        data = buffer.return_all_stack()
        state, action, reward, next_state, done = data.values()
        # print('state: ', state)
        delta_state = next_state - state
        print('reward: ', reward.shape)
        print('delta_state: ', delta_state.shape)

        # inputs = np.concatenate((state, action), axis=-1)
        inputs = T.cat((state, action), axis=-1)
        # print('FakeWorld: inputs', inputs.shape)
        # labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)
        labels = T.cat(( T.reshape( reward, (reward.shape[0], -1) ), delta_state ), axis=-1)
        # print('inputs: ', inputs)
        # print('labels: ', labels)

        holdout_mse_mean = self.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

        return holdout_mse_mean







class FakeWorldOld:

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
