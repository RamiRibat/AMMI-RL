import random
import numpy as np
import torch as T
# from torch.utils.data import random_split, DataLoader
# from torch.utils.data.dataset import IterableDataset





class ReplayBuffer: # Done !
    """
    FIFO Replay buffer for off-policy data:
        __init__: initialize: empty matrices for storing traj's, poniter, size, max_size
        store_transition: D ← D ∪ {(st, at, rt, st+1)}
        sample_batch: B ~ D(|B|); |B|: batch_size
    """
    def __init__(self, obs_dim, act_dim, max_size, seed, device):
        # print('Initialize ReplayBuffer!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)
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
