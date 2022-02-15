# import random
from typing import Type, List, Tuple, Optional

# import numpy as np
import torch as T
from torch.utils.data import random_split, DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
T.multiprocessing.set_sharing_strategy('file_system')

from pytorch_lightning import LightningDataModule




class RLDataset_iter(IterableDataset): # Take env_buffer, return all samples in a Dataset

    def __init__(self, env_buffer):
        super(RLDataset, self).__init__()
        # print('init RLDataset!')
        self.buffer = env_buffer

    def __len__(self):
        return self.buffer.size

    def __iter__(self):
        data = self.buffer.return_all()

        Os = data['observations']
        As = data['actions']
        Rs = data['rewards']
        Os_next = data['observations_next']
        Ds = data['terminals']

        for i in range(len(Ds)):
        	yield Os[i], As[i], Rs[i], Os_next[i], Ds[i]



class RLDataset(Dataset): # Take env_buffer, return all samples in a Dataset

    def __init__(self, env_buffer):
        super(RLDataset, self).__init__()
        # print('init RLDataset!')
        self.buffer = env_buffer

    def __len__(self):
        return self.buffer.size

    def __getitem__(self, idx):
        # if T.is_tensor(idx):
        #     idx = idx.tolist()
        buffer = self.buffer.return_all()

        data = buffer

        Os = data['observations'][idx]
        As = data['actions'][idx]
        Rs = data['rewards'][idx]
        Os_next = data['observations_next'][idx]
        Ds = data['terminals'][idx]

        return Os, As, Rs, Os_next, Ds


class RLDataModule(LightningDataModule):

    def __init__(self, data_buffer, configs):
        super(RLDataModule, self).__init__()
        # print('init RLDataModule!')
        self.data_buffer = data_buffer
        self.configs = configs

    def setup(self, stage: Optional[str] = None):
        val_ratio = self.configs['model_val_ratio']
        rl_dataset = RLDataset(self.data_buffer) # Take env_buffer & sample all data

        if stage == "fit" or stage is None:
            # print('stage == fit')
            val = int(val_ratio*len(rl_dataset))
            train = len(rl_dataset) - val
            self.train_set, self.val_set = random_split(rl_dataset, [train, val])
            # self.train_set = rl_dataset

    def train_dataloader(self):
        # print('train_dataloader')
        batch_size = self.configs['model_batch_size']
        train_loader = DataLoader(dataset=self.train_set,
        						  batch_size=batch_size,
            					  shuffle=False,
        						  num_workers=4, # Calls X RLDataset.__iter__() times
        						  # pin_memory=True
        						  )
        return train_loader

    def val_dataloader(self):
        # print('val_dataloader')
        batch_size = self.configs['model_batch_size']
        val_loader = DataLoader(dataset=self.val_set,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4, # Calls X RLDataset.__iter__() times
                                # pin_memory=True
        						  )
        return val_loader
