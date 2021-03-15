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
        self.reg_params = config['reg_pipeline_params']
        self.target_col = self.reg_params['target_col']
        self.nn_model = self.reg_params['nn_model']
        self.model_params = self.nn_model['model_params']
        self.device = self.nn_model['device']
        self.batch_size = self.nn_model['batch_size']
        self.opt_name = self.nn_model['optimizer']
        self.opt_params = self.nn_model['optimizer_params']
        self.schd_name = self.nn_model['scheduler']
        self.schd_params = self.nn_model['scheduler_params']
        self.n_epochs = self.nn_model['n_epochs']
        self.agg_params = self.reg_params['aggregation_params']
        self.agg_type = self.agg_params['type']
        
    def fit(self, embeds_train, embed_folds):
        for i in range(self.n_folds):
            model = RNet(self.model_params)
            self._models.append(model)
            
            optimizer = _OPTIMIZERS[self.opt_name](
                model.parameters(), **self.opt_params)
            
            scheduler = _SCHEDULERS[self.schd_name]
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
                train_elems = 0
                for batch_idx, bdata in enumerate(train_loader):
                    data, target = bdata['x'][0], bdata['target']
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    cur_elem = len(bdata)
                    train_elems += cur_elem
                    train_loss += float(loss) * cur_elem
                
                train_loss /= train_elems
                    
                scheduler.step()
                model.eval()
                valid_loss = 0.0
                valid_elems = 0
                for batch_idx, bdata in enumerate(val_loader):
                    data, target = bdata['x'][0], bdata['target']
                    data = data.to(self.device)
                    target = target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    cur_elem = len(bdata)
                    valid_elems += cur_elem
                    valid_loss += float(loss) * cur_elem
                    
                valid_loss /= valid_elems
                
                print(f'epoch: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}')

        val_loss = 0.0
        valid_elems = 0
        preds = []
        for i in range(self.n_folds):
            to_valid = embed_folds[embed_folds.folds == i]
            val_loader = DataLoader(
                ValidDataset(to_valid, embeds_train, self.target_col),
                batch_size=1, shuffle=False)
            model = self._models[i]
            model.to('cpu')
            model.eval()
            criterion = nn.L1Loss()
            
            for batch_idx, bdata in enumerate(val_loader):
                data, target, vid = bdata['x'][0], bdata['target'], bdata['id'][0]
                output = model(data)
                loss = criterion(output, target)
                cur_elem = len(bdata)
                valid_elems += cur_elem
                valid_loss += float(loss) * cur_elem
                preds.append({'id': str(vid), self.target_col: float(output)})
                
        valid_loss /= valid_elems
        print(f'last epoch, train_loss: {train_loss}, valid_loss: {valid_loss}')
        
        return pd.DataFrame(data=preds)

    
                                       
    def predict(self, embeds_test):
        preds = []
        for i in range(self.n_folds):
            test_loader = DataLoader(
                TestDataset(embeds_test, i),
                batch_size=1, shuffle=False)
            model = self._models[i]
            model.eval()
            model.to('cpu')
            preds_fold = []
            for bdata in test_loader:
                data, tid = bdata['x'], bdata['id'][0]
                output = model(data)
                preds_fold.append({'id': tid, f'fold_{i}': float(output)})
            preds_fold = pd.DataFrame(data=preds_fold)
            preds.append(preds_fold)
        
        pred = preds[0]
        for i in range(1, self.n_folds):
            pred = pred.merge(preds[i], on='id')
            
        return self.aggregate(pred)
    
    def aggregate(self, data):
        if self.agg_type == 'mean':
            data = data.copy()
            fold_names = [f'fold_{i}' for i in range(self.n_folds)]
            data[self.target_col] = data[fold_names].mean(axis=1)
        if self.agg_params['save_fold_cols']:
            return data
        return data[['id', self.target_col]]


    
    
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
                for y, row in zip(trn_y, trn_X):
                    writer.writerow((str(y), ';'.join(map(str, row))))
        
            
            with open(val_path, 'w') as f:
                writer = csv.writer(f, delimiter='\t', quotechar='"')
                for row in zip(val_y, val_X):
                    writer.writerow((str(y), ';'.join(map(str, row))))
            
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

    
class NNClsPipeline(Pipeline):
    def __init__(self, config):
        super(NNClsPipeline, self).__init__(config)
        

class CBClsPipeline(Pipeline):
    def __init__(self, config):
        super(NNClsPipeline, self).__init__(config)