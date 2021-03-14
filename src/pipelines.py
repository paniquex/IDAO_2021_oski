from typing import List, Dict, Optional, Union

import os
import sys
import csv

from abc import abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
import random

from catboost import Pool, CatBoostClassifier, CatBoostRegressor

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
    'CosineAnnealingWarmRestarts': \
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
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
    
    @abstractmethod
    def predict(self, embeds_test):
        pass
    
    @property
    def is_fitted(self):
        return self._is_fitted

    
class NNRegPipeline(Pipeline):
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.n_folds = config['general_params']['n_folds']
        self.target_col = config['reg_pipeline_params']['target_col']
        self.batch_size = config['reg_pipeline_params']['batch_size']
        self.device = config['general_params']['device']
        self.opt_name = config['reg_pipeline_params']['optimizer']
        self.opt_params = config['reg_pipeline_params']['optimizer_params']
        self.schd_name = config['reg_pipeline_params']['scheduler']
        self.schd_params = config['reg_pipeline_params']['scheduler_params']
        self.n_epochs = config['reg_pipeline_params']['n_epochs']
        
    def fit(self, embeds_train, embed_folds):
        for i in range(self.n_folds):
            model = RNet(self.model_params)
            self._models.append(model)
            
            optimizer = _OPTIMIZERS[self.opt_name](
                model.parameters(), **self.opt_params)
            
            scheduler = _SCHEDULERS[schd_name]
            if scheduler is not None:
                scheduler = scheduler(optimizer, **self.schd_params)
            
            criterion = nn.L1Loss()
            
            to_train = embed_folds[embed_folds.folds != i]
            to_valid = embed_folds[embed_folds.folds == i]
            train_loader = DataLoader(
                TrainDataset(to_train, embeds_train, self.target_col),
                batch_size=self.batch_size,shuffle=True)
            val_loader = DataLoader(
                ValidDataset(to_valid, embeds_train, self.target_col),
                batch_size=1, shuffle=False)
            
            model = model.to(self.device)
            for epoch in range(self.n_epochs):
                model.train()
                train_loss = 0.0
                for batch_idx, bdata in enumerate(train_loader):
                    data, target = bdata['x'], bdata['target']
                    optimizer.zero_grad()
                    data = data.to(self.device)
                    target = data.to(self.device)
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
                    data = data.to(self.device)
                    target = target.to(self.device)
                    output = model(data)
                    loss = criterion(loss, target)
                    val_loss += float(loss)

        val_loss = 0.0
        preds = {}
        for i in range(n_folds):
            to_valid = embed_folds[embed_folds.folds == i]
            val_loader = DataLoader(
                ValidDataset(to_valid, embeds_train, self.target_col),
                batch_size=1, shuffle=False)
            model = self._models[i]
            model.eval()
            criterion = nn.L1Loss()
            
            for batch_idx, bdata in enumerate(val_loader):
                data, target, vid = bdata['x'], bdata['target'], bdata['id']
                target = target.to(self.device)
                output = model(data)
                loss = criterion(loss, target)
                test_loss += float(loss)
                preds.update({str(vid): float(output)})
        
        return pd.DataFrame(data=preds)

    
                                       
    def predict(self, embeds_test):
        preds = []
        keys = [x for x in embeds_test.keys()]
        values = [x for x in embeds_test.values()]
        values = np.asarray(values)
        for i in range(len(self._models)):
            model = self._models[i]
            model.eval()
            model.to('cpu')
            x = torch.from_numpy(values[..., i])              
            pred = model(x)
            preds.append(pred.numpy())
                                       
        test_df = self.aggregate(pd.DataFrame(data=preds))
        test_df['id'] = keys
                                       
        return test_df
    
    def aggregate(self, data):
        data = data.copy()
        data['reg'] = data.mean(axis=1)
        
        return data
    
    
class CBRegPipeline(Pipeline):
    def __init__(self, config: dict):
        super().__init__(config)
        self.cb_params = self.config['reg_pipeline_params']['cb_params']
        self.model_params = self.cb_params['model_params']
        self.pool_params = self.cb_params['pool_params']
        self.cache_dir = self.pool_params['cache_dir']
        self.target_col = config['reg_pipeline']['target_col']
        self.n_folds = config['general_params']['n_folds']
                                       
    def fit(self, embeds_train, embed_folds):
        
        ans_valid = embed_folds.copy()
        ans_valid = ans[['id']]
        ans_valid[self.target_col] = 0
        for i in range(n_folds):
            to_train = embed_folds[embed_folds.folds != i]
            to_valid = embed_folds[embed_folds.folds == i]
            target_col = 
            model = CatBoostRegressor(**self.model_params)
            trn_ids, val_ids = [], []
            trn_X, val_X = [], []
            trn_y, val_y = [], []
            
            for d in to_train.iterrows():
                eid = d[1]['id']
                trn_ids.append(eid)
                trn_X.append(embeds_train[eid])
                trn_y.append(d[1][self.target_col])
            for d in to_valid.iterrows():
                eid = d[1]['id']
                val_ids.append(d[1]['id'])
                val_X.append(embeds_train[eid])
                val_y.append(d[1][self.target_col])
            
            train_path = os.path.join(self.cache_dir, f'train_embed_text_{i}.tsv')
            val_path = os.path.join(self.cache_dir, f'valid_embed_text_{i}.tsv')
            cd_path = os.path.join(self.cache_dir, 'pool_text.cd')

            with open(train_path, 'w') as f:
                writer = csv.writer(f, delimiter='\t', quotechar='"')
                for row in zip(to_train[:
                    writer.writerow(('0', ';'.join(map(str, row))))
        
            with open(cd_path, 'w') as f:
                f.write(
                    '0\tLabel\n'\
                    '1\tNumVector'
                )

            train_pool = None
            val_pool = None
            model.fit(train_pool, eval_set=[val_pool])
            val_preds = model.predict(val_pool)
            self._models.append(model)
            
            ans_valid.loc[to_valid.index, self.target_col] = val_preds
            
        return ans_valid
    
    def predict(self, embeds_test):
        keys = [x for x in embeds_test.keys()]
        values = [x for x in embeds_test.values()]
        values = np.asarray(values)
        preds = []
        for i in range(len(self._models)):
            model = self._models[i]
            pool_path = os.path.join(self.cache_dir, f'test_embed_text_{i}.tsv')
            cd_path = os.path.join(self.cache_dir, 'pool_text.cd')
            with open(pool_path, 'w') as f:
                writer = csv.writer(f, delimiter='\t', quotechar='"')
                for row in values[..., i]:
                    writer.writerow(('0', ';'.join(map(str, row))))
        
            with open(cd_path, 'w') as f:
                f.write(
                    '0\tLabel\n'\
                    '1\tNumVector'
                )
            test_pool = Pool(pool_path, column_descriptor=cd_path)
            pred = model.predict(test_pool)
            preds.append(pred)
        test_df = self.aggregate(pd.DataFrame(data=preds))
        test_df['id'] = keys
        return test_df
        
    def aggregate(self, data):
        data = data.copy()
        data[self.target_col] = data.mean(axis=1)
        
        return data
    