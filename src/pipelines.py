from typing import List, Dict, Optional, Union

import os
import sys

from abc import abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset

from ranger import Ranger

from .nets import RNet, CNet
from .datasets import TrainDataset, TestDataset, ValidDataset


_OPTIMIZERS = {
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'Ranger': Ranger,
    'SGD': torch.optim.SGD,
}

_SCHEDULERS = {
    'LambdaLR': torch.optim.lr_scheduler.LambdaLR,
    'MultiplicativeLR': torch.optim.lr_scheduler.MultiplicativeLR,
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
    'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
    'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'CyclicLR': torch.optim.lr_scheduler.CyclicLR,
    'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR,
    'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'none': None
}

class Pipeline:
    """Modeling pipeline"""
    
    def __init__(self, config: dict):
        self.config = config
        self._is_fitted = False
        self._models = []
    
    @abstractmethod
    def fit(self, embeds_train, embed_folds):
        pass  
            
    def predict(self, embeds_test):
        pass
    
    @property
    def is_fitted(self):
        return self._is_fitted

    
class NNRegPipeline(Pipeline):
    
    def __init__(self, config: dict):
        super().__init__(config)
    
    def fit(self, embeds_train, embed_folds):
        n_folds = self.config['general_params']['n_folds']
        for i in range(n_folds):
            to_train = embed_folds[embed_folds.folds != i]
            to_valid = embed_folds[embed_folds.folds == i]

            model = RNet(self.config['reg_pipeline_params']['model_params'])
            self._models.append(model)

            opt_name = self.config['reg_pipeline_params']['optimizer']
            opt_params = self.config['reg_pipeline_params']['optimizer_params']
            optimizer = _OPTIMIZERS[opt_name](model.parameters(), **opt_params)

            schd_name = self.config['reg_pipeline_params']['scheduler']
            schd_params = self.config['reg_pipeline_params']['scheduler_params']
            scheduler = _SCHEDULERS[schd_name]
            if scheduler is not None:
                scheduler = scheduler(optimizer, **schd_params)

            criterion = nn.L1Loss()
            train_loader = DataLoader(TrainDataset(to_train, embeds_train,
                self.config['reg_pipeline_params']['target_col']),
                batch_size=self.config['reg_pipeline_params']['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(ValidDataset(to_valid, embeds_train,
                self.config['reg_pipeline_params']['target_col']),
                batch_size=1,
                shuffle=False
            )
            device = self.config['general_params']['device']
            model = model.to(device)
            n_epochs = self.config['reg_pipeline_params']['n_epochs']


            for epoch in range(n_epochs):
                model.train()
                train_loss = 0.0

                for batch_idx, bdata in enumerate(train_loader):
                    data, target = bdata['x'], bdata['target']
                    optimizer.zero_grad()
                    data = data.to(device)
                    target = data.to(device)
                    output = model(data)

                    loss = criterion(loss, target)

                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss)
                
                scheduler.step()

                model.eval()
                val_loss = 0.0
                for batch_idx, bdata in enumerate(val_loader):
                    data, target, vid = bdata['x'], bdata['target'], bdata['id']
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    loss = criterion(loss, target)
                    val_loss += float(loss)
                    
        val_loss = 0.0
        preds = {}
        for i in range(n_folds):
            to_valid = embed_folds[embed_folds.folds == i]
            val_loader = DataLoader(ValidDataset(to_valid, embeds_train,
                self.config['reg_pipeline_params']['target_col']),
                batch_size=1,
                shuffle=False
            )
            model = self._models[i]
            model.eval()
            criterion = nn.L1Loss()
            
            for batch_idx, bdata in enumerate(val_loader):
                data, target, vid = bdata['x'], bdata['target'], bdata['id']
                target = target.to(device)
                output = model(data)
                loss = criterion(loss, target)
                test_loss += float(loss)
                preds.update({str(vid): float(output)})
        
        return pd.DataFrame(data=preds)

    
    def predict(self, embeds_test):
        test_dataset = TestDataset(embeds_test)
        pred = []
        for i in len(self._models):
            model = self._models[i]
            model.eval()
            model.to('cpu')
            test_loader = DataLoader(TestDataset(embeds_test),
                                     batch_size=1,
                                     shuffle=False)
            cur_pred = {}
            for i, data in enumerate(test_loader):
                x, tid = data['x'], data['id']
                cur_pred[tid] = model(x)
        
        return self.aggregate(pd.DataFrame(data=pred))
    
    def aggregate(self, data):
        data = data.copy()
        data['reg'] = data.mean(axis=1)
        
        return data
    
    
class CBRegPipeline(Pipeline):
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, embeds_train, embed_folds):
        n_folds = self.config['general_params']['n_folds']
        for i in range(n_folds):
            to_train = embed_folds[embed_folds.folds != i]
            to_valid = embed_folds[embed_folds.folds == i]
            