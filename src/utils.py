from typing import List, Optional, Union, Dict

import os
import sys

import yaml
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image


def pil_loader(path: str) -> Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
    
def parse_folder(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Parse files in directory.
    
    Args:
        path: Folder with train files.
        columns: List of column names.
        
    Returns:
        Dataframe with parsed columns.
        
    """
    file_names = os.listdir(path)
    data = []
    for name in file_names:
        sname, ext = os.path.splitext(name)
        data.append([name, os.path.join(path, name)] + sname.split('_'))
        
    return pd.DataFrame(data=data, columns=columns)


def parse_er(path: str) -> pd.DataFrame:
    """Parse ER files from directory.
    
    Args:
        path: Folder with train ER.
        
    Returns:
        ER DataFrame.
        
    """
    map_renaming = {0: 'name_with_ext', 1: 'full_path',
                    2: 'id', 8: 'particle', 9: 'keV'}
    
    data = parse_folder(os.path.join(path, 'ER'))
    data = data.rename(map_renaming, axis=1)
    
    return data[list(map_renaming.values())]

def parse_nr(path: str) -> pd.DataFrame:
    """Parse NR files from directory.
    
    Args:
        path: Folder with train NR.
    
    Returns:
        NR DataFrame.
    
    """
    map_renaming = {0: 'name_with_ext', 1: 'full_path',
                    2: 'id', 9: 'particle', 10: 'keV'}
    
    data = parse_folder(os.path.join(path, 'NR'))
    data = data.rename(map_renaming, axis=1)
    
    return data[list(map_renaming.values())]


def parse_train(path: str, encode_class=True, class_name='class') -> pd.DataFrame:
    """Parse train dataset.
    
    Args:
        path: Folder with train. With 'ER' and 'NR' subdirs.
        
    Returns:
        Parsed files in table.
    """
    assert 'ER' in os.listdir(path)
    assert 'NR' in os.listdir(path)
    
    er = parse_er(path)
    nr = parse_nr(path)
    data = pd.concat([er, nr], ignore_index=True)
    
    if encode_class:
        data[class_name] = data['particle'].map({'ER': 0, 'NR': 1})
        
    return data


def parse_test(path: str, public=False):
    """Parse test.
    
    Args:
        path: Path to test.
        public: Flag if public or private.
        
    Returns:
        DataFrame with filenames and ids.
        
    """
     
    if public:
        dir_path = os.path.join(path, 'public_test')
    else:
        dir_path = os.path.join(path, 'private_test')
    data = parse_folder(dir_path, columns=[
        'name_with_ext', 'file_path', 'file_name'])
    
    return data

def seed_fix(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)
    

