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


# def discount_cumsum(x, discount): # source: https://github.com/openai/spinningup/
#     """
#     magic from rllab for computing discounted cumulative sums of vectors.
#     input:
#         vector x,
#         [x0,
#          x1,
#          x2]
#     output:
#         [x0 + discount * x1 + discount^2 * x2,
#          x1 + discount * x2,
#          x2]
#     """
#     return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]







class TrajBuffer:
    """
    A simple buffer for storing trajectories
    """

    def __init__(self, obs_dim, act_dim, size, seed, device='cpu', gamma=0.99, gae_lambda=0.95):
        self.obs_buf = T.zeros((size, obs_dim), dtype=T.float32)
        self.act_buf = T.zeros((size, act_dim), dtype=T.float32)
        self.rew_buf = T.zeros((size, 1), dtype=T.float32)
        self.ter_buf = T.zeros((size, 1), dtype=T.float32)
        self.ret_buf = T.zeros((size, 1), dtype=T.float32)
        self.val_buf = T.zeros((size, 1), dtype=T.float32)
        self.adv_buf = T.zeros((size, 1), dtype=T.float32)
        self.log_pi_buf = T.zeros((size, 1), dtype=T.float32)

        self.ptr, self.max_size = 0, size
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.normz_adv = True


    def normalize(self, x):
        return (x - x.mean()) / (x.std() + 1e-8)


    def store_transition(self, o, a, r, d, v, log_pi):
        assert self.ptr < self.max_size

        self.obs_buf[self.ptr] = T.Tensor(o)
        self.act_buf[self.ptr] = T.Tensor(a)
        self.rew_buf[self.ptr] = T.tensor(r)
        self.ter_buf[self.ptr] = T.Tensor([d])
        self.val_buf[self.ptr] = T.Tensor(v)
        self.log_pi_buf[self.ptr] = T.Tensor(log_pi)

        self.ptr +=1


    def store_batch(self, O, A, R, D, V, log_Pi):
        batch_size = len(O[:,0])
        # print('store_batch/ptr: ', self.ptr)
        # print('store_batch/buffer_size: ', self.max_size)
        # print('store_batch/batch_size: ', batch_size)

        self.obs_buf[self.ptr:self.ptr+batch_size] = T.Tensor(O)
        self.act_buf[self.ptr:self.ptr+batch_size] = T.Tensor(A)
        self.rew_buf[self.ptr:self.ptr+batch_size] = T.tensor(R.reshape(-1,1))
        self.ter_buf[self.ptr:self.ptr+batch_size] = T.tensor(D, dtype=T.bool)
        self.val_buf[self.ptr:self.ptr+batch_size] = T.Tensor(V)
        self.log_pi_buf[self.ptr:self.ptr+batch_size] = T.Tensor(log_Pi)

        self.ptr +=batch_size


    def traj_tail(self, next_done, next_value): # Source: CleanRL
        # print('ptr: ', self.ptr)
        next_done = T.Tensor([next_done])
        if self.gae_lambda: # GAE-lambda
            lastgaelam = 0
            # for t in reversed(range(self.max_size)):
            for t in reversed(range(self.ptr)):
                if t == self.max_size - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - self.ter_buf[t + 1]
                    next_values = self.val_buf[t + 1]
                delta = self.rew_buf[t] + self.gamma * next_values * next_nonterminal - self.val_buf[t]
                self.adv_buf[t] = lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
            self.ret_buf = self.adv_buf + self.val_buf
        # else:
        #     for t in reversed(range(self.max_size)):
        #         if t == self.max_size - 1:
        #             next_nonterminal = 1.0 - next_done
        #             next_return = next_value
        #         else:
        #             next_nonterminal = 1.0 - self.ter_buf[t + 1]
        #             next_return = self.ret_buf[t + 1]
        #         self.ret_buf[t] = self.rew_buf[t] + self.gamma * next_nonterminal * next_return
        #     aself.adv_buf = returns - values


    def sample_batch(self, batch_size=64, device=False):
        # assert self.ptr == self.max_size
        # device = self.device
        inds = np.random.randint(0, self.max_size, size=batch_size)
        # inds = np.random.randint(0, self.max_size, size=batch_size)

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


    def reset(self):
        self.ptr = 0






class ReplayBuffer:
    """
    FIFO Replay buffer for off-policy data:
        __init__: initialize: empty matrices for storing traj's, poniter, size, max_size
        store_transition: D ← D ∪ {(st, at, rt, st+1)}
        sample_batch: B ~ D(|B|); |B|: batch_size
    """

    def __init__(self, obs_dim, act_dim, size, seed, device):
        self.obs_buf = T.zeros((size, obs_dim), dtype=T.float32)
        self.act_buf = T.zeros((size, act_dim), dtype=T.float32)
        self.rew_buf = T.zeros((size, 1), dtype=T.float32)
        self.obs_next_buf = T.zeros((size, obs_dim), dtype=T.float32)
        self.ter_buf = T.zeros((size, 1), dtype=T.float32)

        self.ptr, self.size, self.max_size = 0, 0, size


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


    def return_all(self, device=False):
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


    def return_all_np(self):
    	# device = self.device
    	idxs = np.random.randint(0, self.size, size=self.size)
    	buffer = dict(observations=self.obs_buf[idxs],
                      actions=self.act_buf[idxs],
                      rewards=self.rew_buf[idxs],
                      observations_next=self.obs_next_buf[idxs],
                      terminals=self.ter_buf[idxs])
    	return {k: v.numpy() for k, v in buffer.items()}


    def return_all_stack(self):
        data = self.return_all()
        return {k: T.stack([v])[0] for k, v in data.items()}


    def return_all_stack_np(self):
        data = self.return_all_np()
        return {k: np.stack(v) for k, v in data.items()}












class ReplayBufferNP:
    """
    FIFO Replay buffer for off-policy data:
        __init__: initialize: empty matrices for storing traj's, poniter, size, max_size
        store_transition: D ← D ∪ {(st, at, rt, st+1)}
        sample_batch: B ~ D(|B|); |B|: batch_size
    """

    def __init__(self, obs_dim, act_dim, size, seed, device):

        # np.random.seed(seed)
        # T.manual_seed(seed)

        # self.name = None
        # self.device = device

        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)
        self.obs_next_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.ter_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size


    def store_transition(self, obs, act, rew, obs_next, ter):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs_next_buf[self.ptr] = obs_next
        self.ter_buf[self.ptr] = ter

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)


    def override_batch(self, O, A, R, O_next, D, batch_size):
        available_size = self.max_size - self.ptr # 84
        self.obs_buf[self.ptr:self.ptr+available_size] = O[:available_size,:]
        self.act_buf[self.ptr:self.ptr+available_size] = A[:available_size,:]
        self.rew_buf[self.ptr:self.ptr+available_size] = R[:available_size].reshape(-1,1)
        self.obs_next_buf[self.ptr:self.ptr+available_size] = O_next[:available_size,:]
        self.ter_buf[self.ptr:self.ptr+available_size] = D[:available_size]
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
            self.obs_buf[self.ptr:self.ptr+remain_size] = O[available_size:,:]
            self.act_buf[self.ptr:self.ptr+remain_size] = A[available_size:,:]
            self.rew_buf[self.ptr:self.ptr+remain_size] = R[available_size:].reshape(-1,1)
            self.obs_next_buf[self.ptr:self.ptr+remain_size] = O_next[available_size:,:]
            self.ter_buf[self.ptr:self.ptr+remain_size] = D[available_size:]
            self.ptr = (self.ptr+remain_size) % self.max_size # 316


    def store_batch(self, O, A, R, O_next, D):
    	batch_size = len(O[:,0])
    	# print(f"store_batch, size={batch_size}, ptr={self.ptr}, max_size={self.max_size}")

    	if self.ptr+batch_size > self.max_size:
            self.override_batch(O, A, R, O_next, D, batch_size)
    	else:
    		self.obs_buf[self.ptr:self.ptr+batch_size] = O
    		self.act_buf[self.ptr:self.ptr+batch_size] = A
    		self.rew_buf[self.ptr:self.ptr+batch_size] = R.reshape(-1,1)
    		self.obs_next_buf[self.ptr:self.ptr+batch_size] = O_next
    		self.ter_buf[self.ptr:self.ptr+batch_size] = D
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
            return {k: T.as_tensor(v, dtype=T.float32).to(device) for k,v in batch.items()}
        else:
            return {k: T.as_tensor(v, dtype=T.float32) for k,v in batch.items()}


    def sample_batch_np(self, batch_size=32):
        # device = self.device
        idxs = np.random.randint(0, self.size, size=batch_size)
        # print('Index:	', idxs[0: 5])
        batch = dict(observations=self.obs_buf[idxs],
        			actions=self.act_buf[idxs],
        			rewards=self.rew_buf[idxs],
        			observations_next=self.obs_next_buf[idxs],
        			terminals=self.ter_buf[idxs])
        return {k: v for k,v in batch.items()}


    def get_recent_data(self, batch_size=32, device=False):
        # device = self.device
        batch = dict(observations=self.obs_buf[-batch_size:],
        			actions=self.act_buf[-batch_size:],
        			rewards=self.rew_buf[-batch_size:],
        			observations_next=self.obs_next_buf[-batch_size:],
        			terminals=self.ter_buf[-batch_size:])
        if device:
            return {k: T.as_tensor(v, dtype=T.float32).to(device) for k,v in batch.items()}
        else:
            return {k: T.as_tensor(v, dtype=T.float32) for k,v in batch.items()}


    def return_all(self, device=False):
        # device = self.device
        idxs = np.random.randint(0, self.size, size=self.size)
        buffer = dict(observations=self.obs_buf[idxs],
        			actions=self.act_buf[idxs],
        			rewards=self.rew_buf[idxs],
        			observations_next=self.obs_next_buf[idxs],
        			terminals=self.ter_buf[idxs])
        if device:
            return {k: T.as_tensor(v, dtype=T.float32).to(device) for k,v in buffer.items()}
        else:
            return {k: T.as_tensor(v, dtype=T.float32) for k,v in buffer.items()}


    def return_all_np(self):
    	# device = self.device
    	idxs = np.random.randint(0, self.size, size=self.size)
    	buffer = dict(observations=self.obs_buf[idxs],
                      actions=self.act_buf[idxs],
                      rewards=self.rew_buf[idxs],
                      observations_next=self.obs_next_buf[idxs],
                      terminals=self.ter_buf[idxs])
    	return {k: v for k, v in buffer.items()}


    def return_all_np_stack(self):
        data = self.return_all_np()
        return {k: np.stack(v) for k, v in data.items()}
        # state, action, reward, next_state, done = np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done)
        # return state, action, reward, next_state, done






















class DataBuffer:
    """
    A simple FIFO experience replay buffer for an agent.
    """

    def __init__(self, obs_dim, act_dim, size, seed, device):

        # np.random.seed(seed)
        # T.manual_seed(seed)

        # self.name = None
        self.device = device

        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)
        self.obs_next_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.ter_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size


    # def calc_normz(self):
    #
    #     return obs_bias, obs_scale, act_bias, act_scale, out_bias, out_scal
    #     pass


    def store_transition(self, obs, act, rew, obs_next, ter):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs_next_buf[self.ptr] = obs_next
        self.ter_buf[self.ptr] = ter

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)


    def override_batch(self, O, A, R, O_next, D, batch_size):
        available_size = self.max_size - self.ptr # 84
        self.obs_buf[self.ptr:self.ptr+available_size] = O[:available_size,:]
        self.act_buf[self.ptr:self.ptr+available_size] = A[:available_size,:]
        self.rew_buf[self.ptr:self.ptr+available_size] = R[:available_size].reshape(-1,1)
        self.obs_next_buf[self.ptr:self.ptr+available_size] = O_next[:available_size,:]
        self.ter_buf[self.ptr:self.ptr+available_size] = D[:available_size]
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
            self.obs_buf[self.ptr:self.ptr+remain_size] = O[available_size:,:]
            self.act_buf[self.ptr:self.ptr+remain_size] = A[available_size:,:]
            self.rew_buf[self.ptr:self.ptr+remain_size] = R[available_size:].reshape(-1,1)
            self.obs_next_buf[self.ptr:self.ptr+remain_size] = O_next[available_size:,:]
            self.ter_buf[self.ptr:self.ptr+remain_size] = D[available_size:]
            self.ptr = (self.ptr+remain_size) % self.max_size # 316


    def store_batch(self, O, A, R, O_next, D):
    	batch_size = len(O[:,0])
    	# print(f"store_batch, size={batch_size}, ptr={self.ptr}, max_size={self.max_size}")

    	if self.ptr+batch_size > self.max_size:
            self.override_batch(O, A, R, O_next, D, batch_size)
    	else:
    		self.obs_buf[self.ptr:self.ptr+batch_size] = O
    		self.act_buf[self.ptr:self.ptr+batch_size] = A
    		self.rew_buf[self.ptr:self.ptr+batch_size] = R.reshape(-1,1)
    		self.obs_next_buf[self.ptr:self.ptr+batch_size] = O_next
    		self.ter_buf[self.ptr:self.ptr+batch_size] = D
    		self.ptr = (self.ptr+batch_size) % self.max_size

    		self.size = min(self.size+batch_size, self.max_size)
        # print('ptr=', self.ptr)


    def sample_batch(self, batch_size=32):
    	device = self.device
    	idxs = np.random.randint(0, self.size, size=batch_size)
    	# print('Index:	', idxs[0: 5])
    	batch = dict(observations=self.obs_buf[idxs],
    				actions=self.act_buf[idxs],
    				rewards=self.rew_buf[idxs],
    				observations_next=self.obs_next_buf[idxs],
    				terminals=self.ter_buf[idxs])
    	return {k: T.as_tensor(v, dtype=T.float32).to(device) for k,v in batch.items()}


    def get_recent_data(self, batch_size=32):
    	device = self.device
    	batch = dict(observations=self.obs_buf[-batch_size:],
    				actions=self.act_buf[-batch_size:],
    				rewards=self.rew_buf[-batch_size:],
    				observations_next=self.obs_next_buf[-batch_size:],
    				terminals=self.ter_buf[-batch_size:])
    	return {k: T.as_tensor(v, dtype=T.float32).to(device) for k,v in batch.items()}


    def return_all(self):
    	device = self.device
    	idxs = np.random.randint(0, self.size, size=self.size)
    	buffer = dict(observations=self.obs_buf[idxs],
    				actions=self.act_buf[idxs],
    				rewards=self.rew_buf[idxs],
    				observations_next=self.obs_next_buf[idxs],
    				terminals=self.ter_buf[idxs])
    	return {k: T.as_tensor(v, dtype=T.float32).to(device) for k, v in buffer.items()}


    def return_all_np(self):
    	device = self.device
    	idxs = np.random.randint(0, self.size, size=self.size)
    	buffer = dict(observations=self.obs_buf[idxs],
                      actions=self.act_buf[idxs],
                      rewards=self.rew_buf[idxs],
                      observations_next=self.obs_next_buf[idxs],
                      terminals=self.ter_buf[idxs])
    	return {k: v for k, v in buffer.items()}
