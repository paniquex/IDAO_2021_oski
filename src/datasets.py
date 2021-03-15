from typing import List, Dict, Optional, Union

from abc import abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset


class TrainDataset(Dataset):
    def __init__(self, data, embeds, target_col):
        super(TrainDataset, self).__init__()
        self.embeds = embeds
        self.data = data
        self.target_col = target_col
        
    def __getitem__(self, i):
        return {'x': self.embeds[self.data.iloc[i]['id']],
                'target': self.data.iloc[i][self.target_col]}
    
    def __len__(self):
        return len(self.data)

    
class ValidDataset(Dataset):
    def __init__(self, data, embeds, target_col):
        super(ValidDataset, self).__init__()
        self.embeds = embeds
        self.data = data
        self.target_col = target_col
        
    def __getitem__(self, i):
        return {'x': self.embeds[self.data.iloc[i]['id']],
                'target': self.data.iloc[i][self.target_col],
                'id': self.data.iloc[i]['id']}
    
    def __len__(self):
        return len(self.data)

    
class TestDataset(Dataset):
    def __init__(self, embeds, n_fold):
        super(TestDataset, self).__init__()
        self.embeds = embeds
        self.n_fold = n_fold
        self.keys = [x for x in embeds.keys()]
        
    def __getitem__(self, i):
        tid = self.keys[i]
        return {'x': self.embeds[tid][self.n_fold], 'id': tid}
    
    def __len__(self):
        return len(self.keys)
    
