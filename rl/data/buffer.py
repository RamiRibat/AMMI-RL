"""
Inspired by:

    1. SpinningUp OpenAI
    2. CleanRL

"""

import random
# import scipy.signal
import numpy as np
import torch as T
# from torch.utils.data import random_split, DataLoader
# from torch.utils.data.dataset import IterableDataset

# T.multiprocessing.set_sharing_strategy('file_system')


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


import scipy.signal

def discount_cumsum(x, discount): # source: https://github.com/openai/spinningup/
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    # return scipy.signal.lfilter( [1], [1, float(-discount)], x[::-1], axis=0 )[::-1]
    y = scipy.signal.lfilter( [1], [1, float(-discount)], T.flip(x, [0]), axis=0 )
    return T.flip(T.tensor(y), [0])







class TrajBuffer:
    """
    A simple buffer for storing trajectories
    """

    def __init__(self, obs_dim, act_dim, horizon, num_traj, max_size, seed, device='cpu', gamma=0.995, gae_lambda=0.99):
        # print('Initialize Trajectory Buffer')
        self.obs_buf = T.zeros((num_traj, horizon, obs_dim), dtype=T.float32)
        self.pre_act_buf = T.zeros((num_traj, horizon, act_dim), dtype=T.float32)
        self.act_buf = T.zeros((num_traj, horizon, act_dim), dtype=T.float32)
        self.rew_buf = T.zeros((num_traj, horizon+1, 1), dtype=T.float32)
        self.obs_next_buf = T.zeros((num_traj, horizon, obs_dim), dtype=T.float32)
        self.ter_buf = T.zeros((num_traj, horizon, 1), dtype=T.float32)
        self.ret_buf = T.zeros((num_traj, horizon, 1), dtype=T.float32)
        self.val_buf = T.zeros((num_traj, horizon+1, 1), dtype=T.float32)
        self.adv_buf = T.zeros((num_traj, horizon, 1), dtype=T.float32)
        self.log_pi_buf = T.zeros((num_traj, horizon, 1), dtype=T.float32)

        self.ter_idx = T.zeros((num_traj, 1), dtype=T.float32)
        self.ter_ret = T.zeros((num_traj, 1), dtype=T.float32)

        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.horizon, self.num_traj, self.max_size = horizon, num_traj, max_size
        self.ptr, self.last_traj = 0, 0
        self.last_z = 0
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.normz_adv = True


    def total_size(self):
        last_idx = min(self.last_traj+1, self.num_traj)
        return int(self.ter_idx[:last_idx].sum())


    def average_horizon(self):
        last_idx = min(self.last_traj+1, self.num_traj)
        total_size = self.ter_idx[:last_idx].sum()
        live_traj = T.count_nonzero(self.ter_idx[:last_idx])
        # print('live_traj: ', )
        if live_traj > 0:
            return int(total_size//live_traj)
        else:
            return 0


    def average_return(self):
        last_idx = min(self.last_traj+1, self.num_traj)
        total_size = self.ter_ret[:last_idx].sum()
        live_traj = T.count_nonzero(self.ter_ret[:last_idx])
        # print('live_traj: ', )
        if live_traj > 0:
            return int(total_size//live_traj)
        else:
            return 0


    def normalize(self, x):
        return (x - x.mean()) / (x.std() + 1e-8)


    def clean_buffer(self):
        # print('Clean Buffer: Max size reached!'+(' ')*50)
        if self.last_traj == self.num_traj-1:
            ptr = self.ptr
        else:
            ptr = self.last_z
        a = ptr
        while self.total_size() >= self.max_size:
            # print(f'Reduce buffer size: ptr={ptr}')
            self.ter_idx[ptr] = 0
            ptr += 1
            if ptr == self.num_traj:
                ptr = 0
        self.last_z = z = ptr
        # print(f'Reduce buffer size: a={a}-->z={z} | ptr={self.ptr} | last_traj={self.last_traj} | size={self.total_size()}')


    def batch_data(self, recent=False):
        full_size = self.total_size()

        self.obs_batch = T.zeros((full_size, self.obs_dim), dtype=T.float32)
        self.pre_act_batch = T.zeros((full_size, self.act_dim), dtype=T.float32)
        self.act_batch = T.zeros((full_size, self.act_dim), dtype=T.float32)
        self.rew_batch = T.zeros((full_size, 1), dtype=T.float32)
        self.obs_next_batch = T.zeros((full_size, self.obs_dim), dtype=T.float32)
        self.ter_batch = T.zeros((full_size, 1), dtype=T.float32)
        self.ret_batch = T.zeros((full_size, 1), dtype=T.float32)
        self.val_batch = T.zeros((full_size, 1), dtype=T.float32)
        self.adv_batch = T.zeros((full_size, 1), dtype=T.float32)
        self.log_pi_batch = T.zeros((full_size, 1), dtype=T.float32)

        i = 0
        # print('ter_idx: ', self.ter_idx)
        # print('ter_batch: ', self.ter_batch.shape)
        for traj in range(self.last_traj):
        # for traj in range(self.last_traj+1):
            # print('traj: ', traj)
            j = int(self.ter_idx[traj])
            # print('j: ', j)
            self.obs_batch[i:i+j] = self.obs_buf[traj, :j, :]
            self.pre_act_batch[i:i+j] = self.pre_act_buf[traj, :j, :]
            self.act_batch[i:i+j] = self.act_buf[traj, :j, :]
            self.rew_batch[i:i+j] = self.rew_buf[traj, :j, :]
            self.obs_next_batch[i:i+j] = self.obs_next_buf[traj, :j, :]
            self.ter_batch[i:i+j] = self.ter_buf[traj, :j, :]
            self.ret_batch[i:i+j] = self.ret_buf[traj, :j, :]
            self.val_batch[i:i+j] = self.val_buf[traj, :j, :]
            self.adv_batch[i:i+j] = self.adv_buf[traj, :j, :]
            self.log_pi_batch[i:i+j] = self.log_pi_buf[traj, :j, :]

            i = i+j

        if recent:
            # print('recent: ', recent)
            self.obs_batch = self.obs_batch[-recent:]
            self.pre_act_batch = self.pre_act_batch[-recent:]
            self.act_batch = self.act_batch[-recent:]
            self.rew_batch = self.rew_batch[-recent:]
            self.obs_next_batch = self.obs_next_batch[-recent:]
            self.ter_batch = self.ter_batch[-recent:]
            self.ret_batch = self.ret_batch[-recent:]
            self.val_batch = self.val_batch[-recent:]
            self.adv_batch = self.adv_batch[-recent:]
            self.log_pi_batch = self.log_pi_batch[-recent:]


    def store_transition(self, o, pre_a, a, r, o_next, d, v, log_pi, e):
        assert self.total_size() < self.max_size

        self.obs_buf[ self.ptr, e-1, : ] = T.Tensor(o)
        self.pre_act_buf[ self.ptr, e-1, : ] = T.Tensor(pre_a)
        self.act_buf[ self.ptr, e-1, : ] = T.Tensor(a)
        self.rew_buf[ self.ptr, e-1, : ] = T.tensor(r)
        self.obs_next_buf[ self.ptr, e-1, : ] = T.Tensor(o_next)
        # self.ter_buf[ self.ptr, e-1, : ] = T.Tensor([d])
        self.val_buf[ self.ptr, e-1, : ] = T.Tensor(v)
        self.log_pi_buf[ self.ptr, e-1, : ] = T.Tensor(log_pi)


    def traj_tail(self, next_done, next_value, e): # Source: CleanRL
        # print('ptr: ', self.ptr)
        next_done = T.Tensor([next_done])
        if self.gae_lambda: # GAE-lambda
            lastgaelam = 0
            for t in reversed(range(e)):
                if t == e-1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - self.ter_buf[ self.ptr, t + 1, : ]
                    next_values = self.val_buf[ self.ptr, t + 1, : ]
                delta = self.rew_buf[ self.ptr, t, : ] + self.gamma * next_values * next_nonterminal - self.val_buf[ self.ptr, t, : ]
                self.adv_buf[ self.ptr, t, : ] = lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
            self.ret_buf[self.ptr] = self.adv_buf[self.ptr] + self.val_buf[self.ptr]

        self.ter_idx[self.ptr] = e

        # if self.total_size() > self.max_size:
        #     self.ptr = 0
        #     print('new buffer!')
        # else:
        self.ptr +=1
        # print(f'new trajectory [{self.ptr}]')


    def store(self, o, pre_a, a, r, o_next, v, log_pi, e):
        if self.total_size() >= self.max_size:
            self.clean_buffer()
        self.obs_buf[ self.ptr, e-1, : ] = T.Tensor(o)
        self.pre_act_buf[ self.ptr, e-1, : ] = T.Tensor(pre_a)
        self.act_buf[ self.ptr, e-1, : ] = T.Tensor(a)
        self.rew_buf[ self.ptr, e-1, : ] = T.tensor(r)
        self.obs_next_buf[ self.ptr, e-1, : ] = T.Tensor(o_next)
        # self.ter_buf[ self.ptr, e-1, : ] = T.Tensor([d])
        self.val_buf[ self.ptr, e-1, : ] = T.Tensor(v)
        self.log_pi_buf[ self.ptr, e-1, : ] = T.Tensor(log_pi)

    def storeii(self, o, pre_a, a, r, o_next, d, v, log_pi, e):
        if self.total_size() >= self.max_size:
            self.clean_buffer()
        self.obs_buf[ self.ptr, e-1, : ] = T.Tensor(o)
        self.pre_act_buf[ self.ptr, e-1, : ] = T.Tensor(pre_a)
        self.act_buf[ self.ptr, e-1, : ] = T.Tensor(a)
        self.rew_buf[ self.ptr, e-1, : ] = T.tensor(r)
        self.obs_next_buf[ self.ptr, e-1, : ] = T.Tensor(o_next)
        self.ter_buf[ self.ptr, e-1, : ] = T.Tensor([d])
        self.val_buf[ self.ptr, e-1, : ] = T.Tensor(v)
        self.log_pi_buf[ self.ptr, e-1, : ] = T.Tensor(log_pi)


    def finish_path(self, e, v):
        # print(f'\n[ finish_path ] e={e} | ptr={self.ptr} | size={self.total_size()}')
        self.rew_buf[ self.ptr, e, : ] = T.Tensor(v)
        self.val_buf[ self.ptr, e, : ] = T.Tensor(v)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = self.rew_buf[ self.ptr, :e, : ] + self.gamma * self.val_buf[ self.ptr, 1:e+1, : ] - self.val_buf[ self.ptr, :e, : ]
        self.adv_buf[ self.ptr, :e, : ] = discount_cumsum(deltas, self.gamma * self.gae_lambda)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[ self.ptr, :e, : ] = discount_cumsum(self.rew_buf[ self.ptr, :e+1, : ], self.gamma)[:e, :]
        self.ter_idx[self.ptr] = e
        # self.ter_ret[self.ptr] = self.ret_buf[ self.ptr, e, : ]
        # print(f'\n[ finish_path ] e={e} | ptr={self.ptr} | size={self.total_size()}')
        if self.last_traj < self.num_traj-1:
            self.ptr +=1
            self.last_traj +=1
        elif self.ptr < self.num_traj-1:
            self.ptr +=1
        else:
            self.ptr = 0


    def store_batch(self, O, pre_A, A, R, O_next, V, Log_Pi, e):
        if self.total_size() >= self.max_size:
            self.clean_buffer()

        batch_size = len(O)

        self.obs_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(O)
        self.pre_act_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(pre_A)
        self.act_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(A)
        self.rew_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.tensor(R)
        self.obs_next_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(O_next)
        # self.ter_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor([d])
        self.val_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(V)
        self.log_pi_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(Log_Pi)


    def finish_path_batch(self, E, V):
        # print('finish_path_batch!')
        # print(f'\n[ finish_path_batch ] ptr={self.ptr} | size={self.total_size()}')
        batch_size = len(E)
        for i, e in enumerate(E):
            e = int(e)
            self.rew_buf[ self.ptr+i, e, : ] = T.Tensor(V[i])
            self.val_buf[ self.ptr+i, e, : ] = T.Tensor(V[i])
            deltas = self.rew_buf[ self.ptr+i, :e, : ] + self.gamma * self.val_buf[ self.ptr+i, 1:e+1, : ] - self.val_buf[ self.ptr+i, :e, : ]
            self.adv_buf[ self.ptr+i, :e, : ] = discount_cumsum(deltas, self.gamma * self.gae_lambda)
            self.ret_buf[ self.ptr+i, :e, : ] = discount_cumsum(self.rew_buf[ self.ptr+i, :e+1, : ], self.gamma)[:e, :]

        self.ter_idx[self.ptr:self.ptr+batch_size] = E

        if self.last_traj < self.num_traj-1:
            self.ptr +=batch_size
            self.last_traj +=batch_size
        elif self.ptr < self.num_traj-1:
            self.ptr +=batch_size
        else:
            self.ptr = 0


    def store_transition_batch(self, O, A, R, D, V, log_Pi, e):
        assert self.total_size() < self.max_size

        batch_size = len(O[:,0])

        self.obs_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(O)
        self.act_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(A)
        self.rew_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.tensor(R.reshape(-1,1))
        self.ter_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.tensor(D, dtype=T.bool)
        self.val_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(V)
        self.log_pi_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(log_Pi)

        # self.ptr +=batch_size


    def traj_tail_batch(self, next_done, next_value, e): # Source: CleanRL
        # print('ptr: ', self.ptr)
        # next_done = T.Tensor([next_done])
        # if self.gae_lambda: # GAE-lambda
        #     lastgaelam = 0
        #     for t in reversed(range(e)):
        #         if t == e - 1:
        #             next_nonterminal = 1.0 - next_done
        #             next_values = next_value
        #         else:
        #             next_nonterminal = 1.0 - self.ter_buf[self.ptr][t + 1]
        #             next_values = self.val_buf[self.ptr][t + 1]
        #         delta = self.rew_buf[self.ptr][t] + self.gamma * next_values * next_nonterminal - self.val_buf[self.ptr][t]
        #         self.adv_buf[self.ptr][t] = lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
        #     self.ret_buf[self.ptr] = self.adv_buf[self.ptr] + self.val_buf[self.ptr]

        self.ter_idx[self.ptr] = e
        if self.total_size() > self.max_size:
            self.ptr = 0
        else:
            self.ptr +=batch_size


    def sample_batch(self, batch_size=64, recent=False, device=False):
        # if self.total_size() >= self.max_size:
        #     self.clean_buffer()
        # assert self.ptr == self.max_size
        # device = self.device
        batch_size = min(batch_size, self.total_size())
        if recent:
            idxs = np.random.randint(0, batch_size, size=batch_size)
            recent = batch_size
        else:
            idxs = np.random.randint(0, self.total_size(), size=batch_size) # old

        self.batch_data(recent)

        # Adv normalization
        if self.normz_adv:
            self.adv_batch[idxs] = self.normalize(self.adv_batch[idxs])

        batch = dict(observations=self.obs_batch[idxs], # 1
        			 pre_actions=self.pre_act_batch[idxs], # 2.1
        			 actions=self.act_batch[idxs], # 2.2
                     observations_next=self.obs_next_batch[idxs], # 3
        			 rewards=self.rew_batch[idxs], # 4
                     terminals=self.ter_batch[idxs], # 5
        			 returns=self.ret_batch[idxs], # 6
                     values=self.val_batch[idxs], # 7
        			 advantages=self.adv_batch[idxs], # 8
        			 log_pis=self.log_pi_batch[idxs] # 9
                     )
        if device:
            return {k: v.to(device) for k,v in batch.items()} # 7
        else:
            return {k: v            for k,v in batch.items()}


    def sample_batch_for_reply(self, batch_size=64, recent=False, device=False):
        # assert self.ptr == self.max_size
        # device = self.device
        batch_size = min(batch_size, self.total_size())
        idxs = np.random.randint(0, self.total_size(), size=batch_size)

        self.batch_data(recent)

        # Adv normalization
        if self.normz_adv:
            self.adv_batch[idxs] = self.normalize(self.adv_batch[idxs])

        batch = dict(observations=self.obs_batch[idxs],
        			 actions=self.act_batch[idxs],
        			 rewards=self.rew_batch[idxs],
                     observations_next=self.obs_next_batch[idxs],
                     terminals=self.ter_batch[idxs]
                     )
        if device:
            return {k: v.to(device) for k,v in batch.items()} # 7
        else:
            return {k: v            for k,v in batch.items()}


    def update_init_obs(self):
        # print('update_init_obs!')
        old_size = len(self.init_obs)
        if self.ptr > old_size:
            start = old_size
            add_size = self.ptr - old_size
        else:
            start = 0
            add_size = self.ptr
        # print(f'ptr={self.ptr} | add_size={add_size}')
        new_init_obs = T.zeros((old_size+add_size, self.obs_dim), dtype=T.float32)
        new_init_obs[:old_size] = self.init_obs

        i = old_size
        for traj in range(start, start+add_size+1):
            j = int(self.ter_idx[traj])
            if j > 0:
                new_init_obs[i, :] = self.obs_buf[traj, 0, :]
                i +=1
        self.init_obs = new_init_obs


    def sample_init_obs_batch(self, batch_size=200, device=False):
        if not hasattr(self, 'init_obs'):
            last_idx = min(self.ptr+1, self.num_traj)
            live_traj = T.count_nonzero(self.ter_idx[:last_idx])
            self.init_obs = T.zeros((live_traj, self.obs_dim), dtype=T.float32)
            i = 0
            for traj in range(self.last_traj+1):
                j = int(self.ter_idx[traj])
                if j > 0:
                    self.init_obs[i, :] = self.obs_buf[traj, 0, :]
                    i +=1
        elif (self.ptr > len(self.init_obs)) or (self.ptr < self.last_traj):
            self.update_init_obs()

        batch_size = min(batch_size, len(self.init_obs))
        idxs = np.random.randint(0, len(self.init_obs), size=batch_size)

        if device:
            return self.init_obs[idxs].to(device)
        else:
            return self.init_obs[idxs]


    def sample_inds(self, inds, device=False):
        assert self.ptr == self.max_size

        # Adv normalization
        if self.normz_adv:
            self.adv_buf[inds] = self.normalize(self.adv_buf[inds])

        batch = dict(observations=self.obs_buf[inds],
        			 actions=self.act_buf[inds],
        			 returns=self.ret_buf[inds],
                     values=self.val_buf[inds],
        			 advantages=self.adv_buf[inds],
        			 log_pis=self.log_pi_buf[inds])
        if device:
            return {k: v.to(device) for k,v in batch.items()}
        else:
            return {k: v            for k,v in batch.items()}


    def return_all(self, device=False):
        assert self.ptr == self.max_size
        self.reset()

        # Adv normalization
        if self.normz_adv:
            self.adv_buf = self.normalize(self.adv_buf)

        buffer = dict(observations=self.obs_buf,
                      actions=self.act_buf,
                      returns=self.ret_buf,
                      values=self.val_buf,
                      advantages=self.adv_buf,
                      log_pis=self.log_pi_buf)

        if device:
            return {k: v.to(device) for k,v in buffer.items()}
        else:
            return {k: v            for k,v in buffer.items()}


    def return_all_np(self):
        assert self.ptr == self.max_size
        self.reset()

        # Adv normalization
        if self.normz_adv:
            self.adv_buf = self.normalize(self.adv_buf)

        buffer = dict(observations=self.obs_buf,
                      actions=self.act_buf,
                      returns=self.ret_buf,
                      values=self.val_buf,
                      advantages=self.adv_buf,
                      log_pis=self.log_pi_buf)

        return {k: v.numpy() for k, v in buffer.items()}


    def data_for_WM(self, recent=False, device=False):
        idxs = np.random.randint(0, self.total_size(), size=self.total_size())
        self.batch_data(recent)
        buffer = dict(observations=self.obs_batch[idxs],
        			  actions=self.act_batch[idxs],
                      rewards=self.rew_batch[idxs],
                      observations_next=self.obs_next_batch[idxs],
                      terminals=self.ter_batch[idxs])
        if device:
            return {k: v.to(device) for k,v in buffer.items()}
        else:
            return {k: v            for k,v in buffer.items()}


    def data_for_WM_stack(self):
        data = self.data_for_WM(recent=False)
        return {k: T.stack([v])[0] for k, v in data.items()}


    def reset(self):
        # print('Reset Buffer!')
        self.ter_idx = T.zeros((self.num_traj, 1), dtype=T.float32)
        self.ptr, self.last_traj = 0, 0
        self.last_z = 0




class TrajBufferB:
    """
    A simple buffer for storing trajectories
    """

    def __init__(self, obs_dim, act_dim, horizon, num_traj, max_size, seed, device='cpu', gamma=0.995, gae_lambda=0.99):
        print('Initialize Trajectory Buffer')
        self.obs_buf = T.zeros((num_traj, horizon, obs_dim), dtype=T.float32)
        self.pre_act_buf = T.zeros((num_traj, horizon, act_dim), dtype=T.float32)
        self.act_buf = T.zeros((num_traj, horizon, act_dim), dtype=T.float32)
        self.rew_buf = T.zeros((num_traj, horizon+1, 1), dtype=T.float32)
        self.obs_next_buf = T.zeros((num_traj, horizon, obs_dim), dtype=T.float32)
        self.ter_buf = T.zeros((num_traj, horizon, 1), dtype=T.float32)
        self.ret_buf = T.zeros((num_traj, horizon, 1), dtype=T.float32)
        self.val_buf = T.zeros((num_traj, horizon+1, 1), dtype=T.float32)
        self.adv_buf = T.zeros((num_traj, horizon, 1), dtype=T.float32)
        self.log_pi_buf = T.zeros((num_traj, horizon, 1), dtype=T.float32)

        self.ter_idx = T.zeros((num_traj, 1), dtype=T.float32)
        self.ter_ret = T.zeros((num_traj, 1), dtype=T.float32)

        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.horizon, self.num_traj, self.max_size = horizon, num_traj, max_size
        self.ptr, self.last_traj = 0, 0
        self.last_z = 0
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.normz_adv = True


    def total_size(self):
        last_idx = min(self.last_traj+1, self.num_traj)
        return int(self.ter_idx[:last_idx].sum())


    def average_horizon(self):
        last_idx = min(self.last_traj+1, self.num_traj)
        total_size = self.ter_idx[:last_idx].sum()
        live_traj = T.count_nonzero(self.ter_idx[:last_idx])
        # print('live_traj: ', )
        if live_traj > 0:
            return int(total_size//live_traj)
        else:
            return 0


    def average_return(self):
        last_idx = min(self.last_traj+1, self.num_traj)
        total_size = self.ter_ret[:last_idx].sum()
        live_traj = T.count_nonzero(self.ter_ret[:last_idx])
        # print('live_traj: ', )
        if live_traj > 0:
            return int(total_size//live_traj)
        else:
            return 0


    def normalize(self, x):
        return (x - x.mean()) / (x.std() + 1e-8)


    def clean_buffer(self):
        # print('Clean Buffer: Max size reached!'+(' ')*50)
        if self.last_traj == self.num_traj-1:
            ptr = self.ptr
        else:
            ptr = self.last_z
        a = ptr
        while self.total_size() >= self.max_size:
            # print(f'Reduce buffer size: ptr={ptr}')
            self.ter_idx[ptr] = 0
            ptr += 1
            if ptr == self.num_traj:
                ptr = 0
        self.last_z = z = ptr
        # print(f'Reduce buffer size: a={a}-->z={z} | ptr={self.ptr} | last_traj={self.last_traj} | size={self.total_size()}')


    def batch_data(self, recent=False):
        full_size = self.total_size()

        self.obs_batch = T.zeros((full_size, self.obs_dim), dtype=T.float32)
        self.act_batch = T.zeros((full_size, self.act_dim), dtype=T.float32)
        self.rew_batch = T.zeros((full_size, 1), dtype=T.float32)
        self.obs_next_batch = T.zeros((full_size, self.obs_dim), dtype=T.float32)
        self.ter_batch = T.zeros((full_size, 1), dtype=T.float32)
        self.ret_batch = T.zeros((full_size, 1), dtype=T.float32)
        self.val_batch = T.zeros((full_size, 1), dtype=T.float32)
        self.adv_batch = T.zeros((full_size, 1), dtype=T.float32)
        self.log_pi_batch = T.zeros((full_size, 1), dtype=T.float32)

        i = 0
        for traj in range(self.last_traj+1):
            j = int(self.ter_idx[traj])
            self.obs_batch[i:i+j] = self.obs_buf[traj, :j, :]
            self.act_batch[i:i+j] = self.act_buf[traj, :j, :]
            self.rew_batch[i:i+j] = self.rew_buf[traj, :j, :]
            self.obs_next_batch[i:i+j] = self.obs_next_buf[traj, :j, :]
            self.ter_batch[i:i+j] = self.ter_buf[traj, :j, :]
            self.ret_batch[i:i+j] = self.ret_buf[traj, :j, :]
            self.val_batch[i:i+j] = self.val_buf[traj, :j, :]
            self.adv_batch[i:i+j] = self.adv_buf[traj, :j, :]
            self.log_pi_batch[i:i+j] = self.log_pi_buf[traj, :j, :]

            i = i+j

        if recent:
            self.obs_batch = self.obs_batch[-recent:]
            self.act_batch = self.act_batch[-recent:]
            self.rew_batch = self.rew_batch[-recent:]
            self.obs_next_batch = self.obs_next_batch[-recent:]
            self.ter_batch = self.ter_batch[-recent:]
            self.ret_batch = self.ret_batch[-recent:]
            self.val_batch = self.val_batch[-recent:]
            self.adv_batch = self.adv_batch[-recent:]
            self.log_pi_batch = self.log_pi_batch[-recent:]


    def store_transition(self, o, a, r, o_next, d, v, log_pi, e):
        assert self.total_size() < self.max_size

        self.obs_buf[ self.ptr, e-1, : ] = T.Tensor(o)
        self.act_buf[ self.ptr, e-1, : ] = T.Tensor(a)
        self.rew_buf[ self.ptr, e-1, : ] = T.tensor(r)
        self.obs_next_buf[ self.ptr, e-1, : ] = T.Tensor(o_next)
        # self.ter_buf[ self.ptr, e-1, : ] = T.Tensor([d])
        self.val_buf[ self.ptr, e-1, : ] = T.Tensor(v)
        self.log_pi_buf[ self.ptr, e-1, : ] = T.Tensor(log_pi)


    def traj_tail(self, next_done, next_value, e): # Source: CleanRL
        # print('ptr: ', self.ptr)
        next_done = T.Tensor([next_done])
        if self.gae_lambda: # GAE-lambda
            lastgaelam = 0
            for t in reversed(range(e)):
                if t == e-1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - self.ter_buf[ self.ptr, t + 1, : ]
                    next_values = self.val_buf[ self.ptr, t + 1, : ]
                delta = self.rew_buf[ self.ptr, t, : ] + self.gamma * next_values * next_nonterminal - self.val_buf[ self.ptr, t, : ]
                self.adv_buf[ self.ptr, t, : ] = lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
            self.ret_buf[self.ptr] = self.adv_buf[self.ptr] + self.val_buf[self.ptr]

        self.ter_idx[self.ptr] = e

        # if self.total_size() > self.max_size:
        #     self.ptr = 0
        #     print('new buffer!')
        # else:
        self.ptr +=1
        # print(f'new trajectory [{self.ptr}]')


    def store(self, o, a, r, o_next, v, log_pi, e):
        if self.total_size() >= self.max_size:
            self.clean_buffer()
        self.obs_buf[ self.ptr, e-1, : ] = T.Tensor(o)
        self.pre_act_buf[ self.ptr, e-1, : ] = T.Tensor(a)
        self.act_buf[ self.ptr, e-1, : ] = T.Tensor(a)
        self.rew_buf[ self.ptr, e-1, : ] = T.tensor(r)
        self.obs_next_buf[ self.ptr, e-1, : ] = T.Tensor(o_next)
        # self.ter_buf[ self.ptr, e-1, : ] = T.Tensor([d])
        self.val_buf[ self.ptr, e-1, : ] = T.Tensor(v)
        self.log_pi_buf[ self.ptr, e-1, : ] = T.Tensor(log_pi)


    def finish_path(self, e, v):
        # print(f'\n[ finish_path ] e={e} | ptr={self.ptr} | size={self.total_size()}')
        self.rew_buf[ self.ptr, e, : ] = T.Tensor(v)
        self.val_buf[ self.ptr, e, : ] = T.Tensor(v)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = self.rew_buf[ self.ptr, :e, : ] + self.gamma * self.val_buf[ self.ptr, 1:e+1, : ] - self.val_buf[ self.ptr, :e, : ]
        self.adv_buf[ self.ptr, :e, : ] = discount_cumsum(deltas, self.gamma * self.gae_lambda)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[ self.ptr, :e, : ] = discount_cumsum(self.rew_buf[ self.ptr, :e+1, : ], self.gamma)[:e, :]
        self.ter_idx[self.ptr] = e
        # self.ter_ret[self.ptr] = self.ret_buf[ self.ptr, e, : ]
        # print(f'\n[ finish_path ] e={e} | ptr={self.ptr} | size={self.total_size()}')
        if self.last_traj < self.num_traj-1:
            self.ptr +=1
            self.last_traj +=1
        elif self.ptr < self.num_traj-1:
            self.ptr +=1
        else:
            self.ptr = 0


    def store_batch(self, O, A, R, O_next, V, Log_Pi, e):
        # print('store_batch!')
        if self.total_size() >= self.max_size:
            self.clean_buffer()

        batch_size = len(O)

        self.obs_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(O)
        self.act_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(A)
        self.rew_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.tensor(R)
        self.obs_next_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(O_next)
        # self.ter_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor([d])
        self.val_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(V)
        self.log_pi_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(Log_Pi)


    def finish_path_batch(self, E, V):
        # print('finish_path_batch!')
        # print(f'\n[ finish_path_batch ] ptr={self.ptr} | size={self.total_size()}')
        batch_size = len(E)
        for i, e in enumerate(E):
            e = int(e)
            # print(f'ptr={self.ptr} | i={i} | e={e} | V={V[i]}')
            self.rew_buf[ self.ptr+i, e, : ] = T.Tensor(V[i])
            self.val_buf[ self.ptr+i, e, : ] = T.Tensor(V[i])
            deltas = self.rew_buf[ self.ptr+i, :e, : ] + self.gamma * self.val_buf[ self.ptr+i, 1:e+1, : ] - self.val_buf[ self.ptr+i, :e, : ]
            self.adv_buf[ self.ptr+i, :e, : ] = discount_cumsum(deltas, self.gamma * self.gae_lambda)
            self.ret_buf[ self.ptr+i, :e, : ] = discount_cumsum(self.rew_buf[ self.ptr+i, :e+1, : ], self.gamma)[:e, :]

        self.ter_idx[self.ptr:self.ptr+batch_size] = E

        # A = T.cat([E,self.adv_buf[self.ptr:self.ptr+batch_size, :int(E.max()), :],self.ret_buf[self.ptr:self.ptr+batch_size, :int(E.max()), :]], axis=1)

        # print(f'\n[ finish_path ] e={e} | ptr={self.ptr} | size={self.total_size()}')
        if self.last_traj < self.num_traj-1:
            self.ptr +=batch_size
            self.last_traj +=batch_size
        elif self.ptr < self.num_traj-1:
            self.ptr +=batch_size
        else:
            self.ptr = 0


    def store_transition_batch(self, O, A, R, D, V, log_Pi, e):
        assert self.total_size() < self.max_size

        batch_size = len(O[:,0])

        self.obs_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(O)
        self.act_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(A)
        self.rew_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.tensor(R.reshape(-1,1))
        self.ter_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.tensor(D, dtype=T.bool)
        self.val_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(V)
        self.log_pi_buf[ self.ptr:self.ptr+batch_size, e-1, : ] = T.Tensor(log_Pi)

        # self.ptr +=batch_size


    def traj_tail_batch(self, next_done, next_value, e): # Source: CleanRL
        # print('ptr: ', self.ptr)
        # next_done = T.Tensor([next_done])
        # if self.gae_lambda: # GAE-lambda
        #     lastgaelam = 0
        #     for t in reversed(range(e)):
        #         if t == e - 1:
        #             next_nonterminal = 1.0 - next_done
        #             next_values = next_value
        #         else:
        #             next_nonterminal = 1.0 - self.ter_buf[self.ptr][t + 1]
        #             next_values = self.val_buf[self.ptr][t + 1]
        #         delta = self.rew_buf[self.ptr][t] + self.gamma * next_values * next_nonterminal - self.val_buf[self.ptr][t]
        #         self.adv_buf[self.ptr][t] = lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
        #     self.ret_buf[self.ptr] = self.adv_buf[self.ptr] + self.val_buf[self.ptr]

        self.ter_idx[self.ptr] = e
        if self.total_size() > self.max_size:
            self.ptr = 0
        else:
            self.ptr +=batch_size


    def sample_batch(self, batch_size=64, recent=False, device=False):
        # if self.total_size() >= self.max_size:
        #     self.clean_buffer()
        # assert self.ptr == self.max_size
        # device = self.device
        batch_size = min(batch_size, self.total_size())
        if recent:
            idxs = np.random.randint(0, batch_size, size=batch_size)
        else:
            idxs = np.random.randint(0, self.total_size(), size=batch_size) # old

        self.batch_data(recent)

        # Adv normalization
        if self.normz_adv:
            self.adv_batch[idxs] = self.normalize(self.adv_batch[idxs])

        batch = dict(observations=self.obs_batch[idxs], # 1
        			 actions=self.act_batch[idxs], # 2
                     observations_next=self.obs_next_batch[idxs], # 3
        			 rewards=self.rew_batch[idxs], # 4
                     terminals=self.ter_batch[idxs], # 5
        			 returns=self.ret_batch[idxs], # 6
                     values=self.val_batch[idxs], # 7
        			 advantages=self.adv_batch[idxs], # 8
        			 log_pis=self.log_pi_batch[idxs] # 9
                     )
        if device:
            return {k: v.to(device) for k,v in batch.items()} # 7
        else:
            return {k: v            for k,v in batch.items()}


    def sample_batch_for_reply(self, batch_size=64, recent=False, device=False):
        # assert self.ptr == self.max_size
        # device = self.device
        batch_size = min(batch_size, self.total_size())
        idxs = np.random.randint(0, self.total_size(), size=batch_size)

        self.batch_data(recent)

        # Adv normalization
        if self.normz_adv:
            self.adv_batch[idxs] = self.normalize(self.adv_batch[idxs])

        batch = dict(observations=self.obs_batch[idxs],
        			 actions=self.act_batch[idxs],
        			 rewards=self.rew_batch[idxs],
                     observations_next=self.obs_next_batch[idxs],
                     terminals=self.ter_batch[idxs]
                     )
        if device:
            return {k: v.to(device) for k,v in batch.items()} # 7
        else:
            return {k: v            for k,v in batch.items()}


    def update_init_obs(self):
        # print('update_init_obs!')
        old_size = len(self.init_obs)
        if self.ptr > old_size:
            start = old_size
            add_size = self.ptr - old_size
        else:
            start = 0
            add_size = self.ptr
        # print(f'ptr={self.ptr} | add_size={add_size}')
        new_init_obs = T.zeros((old_size+add_size, self.obs_dim), dtype=T.float32)
        new_init_obs[:old_size] = self.init_obs

        i = old_size
        for traj in range(start, start+add_size+1):
            j = int(self.ter_idx[traj])
            if j > 0:
                new_init_obs[i, :] = self.obs_buf[traj, 0, :]
                i +=1
        self.init_obs = new_init_obs


    def sample_init_obs_batch(self, batch_size=200, device=False):
        if not hasattr(self, 'init_obs'):
            last_idx = min(self.ptr+1, self.num_traj)
            live_traj = T.count_nonzero(self.ter_idx[:last_idx])
            self.init_obs = T.zeros((live_traj, self.obs_dim), dtype=T.float32)
            i = 0
            for traj in range(self.last_traj+1):
                j = int(self.ter_idx[traj])
                if j > 0:
                    self.init_obs[i, :] = self.obs_buf[traj, 0, :]
                    i +=1
        elif (self.ptr > len(self.init_obs)) or (self.ptr < self.last_traj):
            self.update_init_obs()

        batch_size = min(batch_size, len(self.init_obs))
        idxs = np.random.randint(0, len(self.init_obs), size=batch_size)

        if device:
            return self.init_obs[idxs].to(device)
        else:
            return self.init_obs[idxs]


    def sample_inds(self, inds, device=False):
        assert self.ptr == self.max_size

        # Adv normalization
        if self.normz_adv:
            self.adv_buf[inds] = self.normalize(self.adv_buf[inds])

        batch = dict(observations=self.obs_buf[inds],
        			 actions=self.act_buf[inds],
        			 returns=self.ret_buf[inds],
                     values=self.val_buf[inds],
        			 advantages=self.adv_buf[inds],
        			 log_pis=self.log_pi_buf[inds])
        if device:
            return {k: v.to(device) for k,v in batch.items()}
        else:
            return {k: v            for k,v in batch.items()}


    def return_all(self, device=False):
        assert self.ptr == self.max_size
        self.reset()

        # Adv normalization
        if self.normz_adv:
            self.adv_buf = self.normalize(self.adv_buf)

        buffer = dict(observations=self.obs_buf,
                      actions=self.act_buf,
                      returns=self.ret_buf,
                      values=self.val_buf,
                      advantages=self.adv_buf,
                      log_pis=self.log_pi_buf)

        if device:
            return {k: v.to(device) for k,v in buffer.items()}
        else:
            return {k: v            for k,v in buffer.items()}


    def return_all_np(self):
        assert self.ptr == self.max_size
        self.reset()

        # Adv normalization
        if self.normz_adv:
            self.adv_buf = self.normalize(self.adv_buf)

        buffer = dict(observations=self.obs_buf,
                      actions=self.act_buf,
                      returns=self.ret_buf,
                      values=self.val_buf,
                      advantages=self.adv_buf,
                      log_pis=self.log_pi_buf)

        return {k: v.numpy() for k, v in buffer.items()}


    def data_for_WM(self, recent=False, device=False):
        idxs = np.random.randint(0, self.total_size(), size=self.total_size())
        self.batch_data(recent)
        buffer = dict(observations=self.obs_batch[idxs],
        			  actions=self.act_batch[idxs],
                      rewards=self.rew_batch[idxs],
                      observations_next=self.obs_next_batch[idxs],
                      terminals=self.ter_batch[idxs])
        if device:
            return {k: v.to(device) for k,v in buffer.items()}
        else:
            return {k: v            for k,v in buffer.items()}


    def data_for_WM_stack(self):
        data = self.data_for_WM(recent=False)
        return {k: T.stack([v])[0] for k, v in data.items()}


    def reset(self):
        # print('Reset Buffer!')
        self.ter_idx = T.zeros((self.num_traj, 1), dtype=T.float32)
        self.ptr, self.last_traj = 0, 0
        self.last_z = 0







class ReplayBuffer:
    """
    FIFO Replay buffer for off-policy data:
        __init__: initialize: empty matrices for storing traj's, poniter, size, max_size
        store_transition: D ← D ∪ {(st, at, rt, st+1)}
        sample_batch: B ~ D(|B|); |B|: batch_size
    """

    def __init__(self, obs_dim, act_dim, size, seed, device):
        print('Initialize ReplayBuffer')
        self.obs_buf = T.zeros((size, obs_dim), dtype=T.float32)
        self.act_buf = T.zeros((size, act_dim), dtype=T.float32)
        self.rew_buf = T.zeros((size, 1), dtype=T.float32)
        self.obs_next_buf = T.zeros((size, obs_dim), dtype=T.float32)
        self.ter_buf = T.zeros((size, 1), dtype=T.float32)

        self.ptr, self.size, self.max_size = 0, 0, size


    def total_size(self):
        return self.size


    def store_transition(self, o, a, r, o_next, d):
        self.obs_buf[self.ptr] = T.Tensor(o)
        self.act_buf[self.ptr] = T.Tensor(a)
        self.rew_buf[self.ptr] = T.tensor(r)
        self.obs_next_buf[self.ptr] = T.Tensor(o_next)
        self.ter_buf[self.ptr] = T.tensor([d], dtype=T.bool)

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)


    def override_batch(self, O, A, R, O_next, D, batch_size):
        available_size = self.max_size - self.ptr # 84
        self.obs_buf[self.ptr:self.ptr+available_size] = T.Tensor(O[:available_size,:])
        self.act_buf[self.ptr:self.ptr+available_size] = T.Tensor(A[:available_size,:])
        self.rew_buf[self.ptr:self.ptr+available_size] = T.tensor(R[:available_size].reshape(-1,1))
        self.obs_next_buf[self.ptr:self.ptr+available_size] = T.Tensor(O_next[:available_size,:])
        self.ter_buf[self.ptr:self.ptr+available_size] = T.tensor(D[:available_size])
        self.ptr = (self.ptr+available_size) % self.max_size # 0
        remain_size = batch_size - available_size # 316
        # print('ptr=', self.ptr)

        if self.ptr+remain_size > self.max_size:
            self.override_batch(O[available_size:,:],
                                A[available_size:,:],
                                R[available_size:],
                                O_next[available_size:,:],
                                D[available_size:],
                                remain_size)
        else:
            self.obs_buf[self.ptr:self.ptr+remain_size] = T.Tensor(O[available_size:,:])
            self.act_buf[self.ptr:self.ptr+remain_size] = T.Tensor(A[available_size:,:])
            self.rew_buf[self.ptr:self.ptr+remain_size] = T.tensor(R[available_size:].reshape(-1,1))
            self.obs_next_buf[self.ptr:self.ptr+remain_size] = T.Tensor(O_next[available_size:,:])
            self.ter_buf[self.ptr:self.ptr+remain_size] = T.tensor(D[available_size:])
            self.ptr = (self.ptr+remain_size) % self.max_size # 316


    def store_batch(self, O, A, R, O_next, D):
    	batch_size = len(O[:,0])
    	# print(f"store_batch, size={batch_size}, ptr={self.ptr}, max_size={self.max_size}")

    	if self.ptr+batch_size > self.max_size:
            self.override_batch(O, A, R, O_next, D, batch_size)
    	else:
    		self.obs_buf[self.ptr:self.ptr+batch_size] = T.Tensor(O)
    		self.act_buf[self.ptr:self.ptr+batch_size] = T.Tensor(A)
    		self.rew_buf[self.ptr:self.ptr+batch_size] = T.tensor(R.reshape(-1,1))
    		self.obs_next_buf[self.ptr:self.ptr+batch_size] = T.Tensor(O_next)
    		self.ter_buf[self.ptr:self.ptr+batch_size] = T.tensor(D, dtype=T.bool)
    		self.ptr = (self.ptr+batch_size) % self.max_size

    		self.size = min(self.size+batch_size, self.max_size)
        # print('ptr=', self.ptr)


    def sample_batch(self, batch_size=32, device=False):
        # device = self.device
        idxs = np.random.randint(0, self.size, size=batch_size)
        # print('Index:	', idxs[0: 5])
        batch = dict(observations=self.obs_buf[idxs],
        			 actions=self.act_buf[idxs],
        			 rewards=self.rew_buf[idxs],
        			 observations_next=self.obs_next_buf[idxs],
        			 terminals=self.ter_buf[idxs])
        if device:
            return {k: v.to(device) for k,v in batch.items()}
        else:
            return {k: v            for k,v in batch.items()}


    def sample_batch_np(self, batch_size=32):
        # device = self.device
        idxs = np.random.randint(0, self.size, size=batch_size)
        # print('Index:	', idxs[0: 5])
        batch = dict(observations=self.obs_buf[idxs],
        			actions=self.act_buf[idxs],
        			rewards=self.rew_buf[idxs],
        			observations_next=self.obs_next_buf[idxs],
        			terminals=self.ter_buf[idxs])
        return {k: v.numpy() for k,v in batch.items()}


    def get_recent_data(self, batch_size=32, device=False):
        # device = self.device
        batch = dict(observations=self.obs_buf[-batch_size:],
        			actions=self.act_buf[-batch_size:],
        			rewards=self.rew_buf[-batch_size:],
        			observations_next=self.obs_next_buf[-batch_size:],
        			terminals=self.ter_buf[-batch_size:])
        if device:
            return {k: v.to(device) for k,v in batch.items()}
        else:
            return {k: v            for k,v in batch.items()}


    def data_for_WM_all(self, device=False):
        # device = self.device
        idxs = np.random.randint(0, self.size, size=self.size)
        buffer = dict(observations=self.obs_buf[idxs],
        			actions=self.act_buf[idxs],
        			rewards=self.rew_buf[idxs],
        			observations_next=self.obs_next_buf[idxs],
        			terminals=self.ter_buf[idxs])
        if device:
            return {k: v.to(device) for k,v in buffer.items()}
        else:
            return {k: v            for k,v in buffer.items()}


    def data_for_WM_np(self):
    	# device = self.device
    	idxs = np.random.randint(0, self.size, size=self.size)
    	buffer = dict(observations=self.obs_buf[idxs],
                      actions=self.act_buf[idxs],
                      rewards=self.rew_buf[idxs],
                      observations_next=self.obs_next_buf[idxs],
                      terminals=self.ter_buf[idxs])
    	return {k: v.numpy() for k, v in buffer.items()}


    def data_for_WM_stack(self):
        data = self.data_for_WM_all()
        return {k: T.stack([v])[0] for k, v in data.items()}


    def data_for_WM_stack_np(self):
        data = self.return_all_np()
        return {k: np.stack(v) for k, v in data.items()}
