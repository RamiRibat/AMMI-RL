import random
import numpy as np
import torch as T
# from torch.utils.data import random_split, DataLoader
# from torch.utils.data.dataset import IterableDataset

# T.multiprocessing.set_sharing_strategy('file_system')



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)



class ReplayBufferOld: # Done !
    """
    FIFO Replay buffer for off-policy data:
        __init__: initialize: empty matrices for storing traj's, poniter, size, max_size
        store_transition: D ← D ∪ {(st, at, rt, st+1)}
        sample_batch: B ~ D(|B|); |B|: batch_size
    """
    def __init__(self, obs_dim, act_dim, max_size, seed, device):
        # print('Initialize ReplayBuffer!')
        # if seed:
        #     random.seed(seed), np.random.seed(seed), T.manual_seed(seed)
        self.device = device

        self.observation_buffer = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros((max_size, act_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((max_size, 1), dtype=np.float32)
        self.observation_next_buffer = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.terminal_buffer = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, max_size


    def store_transition(self, o, a, r, o_next, d):
        self.observation_buffer[self.ptr] = o
        self.action_buffer[self.ptr] = a
        self.reward_buffer[self.ptr] = r
        self.observation_next_buffer[self.ptr] = o_next
        self.terminal_buffer[self.ptr] = d

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)


    def sample_batch(self, batch_size=32):
        inx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(observations = self.observation_buffer[inx],
                     actions = self.action_buffer[inx],
                     rewards = self.reward_buffer[inx],
                     observations_next = self.observation_next_buffer[inx],
                     terminals = self.terminal_buffer[inx])
        return {k: T.tensor(v, dtype=T.float32).to(self.device) for k, v in batch.items()}





class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for an agent.
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
            return {k: T.as_tensor(v, dtype=T.float32).to(device) for k,v in batch.items()}
        else:
            return {k: T.as_tensor(v, dtype=T.float32) for k,v in batch.items()}


    def return_all_np(self):
    	# device = self.device
    	idxs = np.random.randint(0, self.size, size=self.size)
    	buffer = dict(observations=self.obs_buf[idxs],
                      actions=self.act_buf[idxs],
                      rewards=self.rew_buf[idxs],
                      observations_next=self.obs_next_buf[idxs],
                      terminals=self.ter_buf[idxs])
    	return {k: v for k, v in buffer.items()}









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
