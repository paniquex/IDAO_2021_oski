from collections import defaultdict

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader


from .utils import init_layer, init_bn


class CNet(nn.Module):
    
    def __init__(self, config):
        super(CNet, self).__init__()
        self.config = config
        self.embed_dim = config['embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.use_dropout = config['use_dropout']
        self.drop_rate = config['drop_rate']
        self.use_bn = config['use_bn']
        self.act = config['act_name']
        self.act_params = config['act_params']
        self.hid_blocks = config['n_hid_blocks']
        
        layers = []
        layer_names = []
        layers.append(nn.Linear(self.embed_dim, self.hidden_dim))
        layers.append(self._get_act(self.act))
        for i in range(self.hid_blocks):
            if use_bn:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self._get_act(self.act))
            if i == self.hid_blocks - 1 and use_dropout:
                layers.append(nn.Dropout(self.drop_rate))
                
        layers.append(self.Linear(self.hidden_dim, 2))
        layers.append(nn.Softmax(dim=1))
        self.layers = nn.ModuleList(layers)
        self.layers.apply(self._init_layers)
        
        
    def _get_act(self, act_name, act_params):
        if name == 'relu':
            return nn.ReLU(**act_params)
        elif name == 'gelu':
            return nn.GELU(**act_params)
        elif name == 'leakyrelu':
            return nn.LeakyReLU(**act_params)
        
    def _init_layers(self, layer):
        if isinstance(m, nn.BatchNorm1d):
            init_bn(layer)
        else:
            init_layer(layer)    
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class RNet(nn.Module):
    
    def __init__(self, config):
        super(RNet, self).__init__()
        self.config = config
        self.embed_dim = config['embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.use_dropout = config['use_dropout']
        self.drop_rate = config['drop_rate']
        self.use_bn = config['use_bn']
        self.act = config['act_name']
        self.act_params = config['act_params']
        self.hid_blocks = config['n_hid_blocks']
        
        layers = []
        layer_names = []
        layers.append(nn.Linear(self.embed_dim, self.hidden_dim))
        layers.append(self._get_act(self.act))
        for i in range(self.hid_blocks):
            if use_bn:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self._get_act(self.act))
            if i == self.hid_blocks - 1 and use_dropout:
                layers.append(nn.Dropout(self.drop_rate))
                
        layers.append(self.Linear(self.hidden_dim, 1))
        self.layers = nn.ModuleList(layers)
        self.layers.apply(self._init_layers)
        
        
    def _get_act(self, act_name, act_params):
        if act_name == 'relu':
            return nn.ReLU(**act_params)
        elif act_name == 'gelu':
            return nn.GELU(**act_params)
        elif act_name == 'leakyrelu':
            return nn.LeakyReLU(**act_params)
        
    def _init_layers(self, layer):
        if isinstance(m, nn.BatchNorm1d):
            init_bn(layer)
        else:
            init_layer(layer)    
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    