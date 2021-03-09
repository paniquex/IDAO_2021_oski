import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader

from efficientnet_pytorch import EfficientNet

from .utils import pil_loader


class CParticle(nn.Module):
    
    def __init__(self, embed_dim):
        super(CParticle, self).__init__()
        
        self.embed_dim = embed_dim
        self.layers = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(),
            nn.Linear(4 * self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 2),
            nn.Softmax(dim=1)
        )
        
        
    def forward(self, x):
        z = self.layers(x)
        return z
    
class RParticle(nn.Module):
    """
    Regression Layer.
    """
    
    def __init__(self, embed_dim):
        super(CParticle, self).__init__()
        
        self.embed_dim = embed_dim
        self.layers = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(),
            nn.Linear(4 * self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
        
        
    def forward(self, x):
        z = self.layers(x)
        return z
    

def train_loop(model, optim_class, criterion):
    pass