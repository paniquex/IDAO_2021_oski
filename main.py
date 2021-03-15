"""
Just a script with pipeline.

1) read data and transform embeddings to it's folds
2) for every fold train its model
3) predict by every model a prediction to coressponding fold
4) aggregate predictions

TODO: add logging

zapusk:
# python3 main.py --config config/regression.yml

"""
import argparse
import yaml
import logging

import numpy as np
import pandas as pd

from src.utils import seed_fix
from src.pipelines import NNRegPipeline, NNClsPipeline, \
    CBRegPipeline, CBRegPipeline


def main(config_path: str):
    """Pipeline.
    
    Args:
        config_path: Path to config.
    
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    gp = config['general_params']
    
    task = gp['task']
    assert task in ['reg', 'cls']
    
    seed_fix(gp['random_seed'])
    
    embeds_train = np.load(gp['train_embeds_path'], allow_pickle=True)
    embeds_train = embeds_train.tolist()
    embeds_test = np.load(gp['test_embeds_path'], allow_pickle=True)
    embeds_test = embeds_test.tolist()
    
    
    embed_folds = pd.read_csv(gp['train_folds_path'])
    
    if task == 'reg':
        if gp['model_type'] == 'nn':
            pipeline = NNRegPipeline(config)
        elif gp['model_type'] == 'cb':
            pipeline = CBRegPipeline(config)
    else:
        if gp['model_type'] == 'nn':
            pipeline = NNClsPipeline(config)
        elif gp['model_type'] == 'cb':
            pipeline = CBClsPipeline(config)
        
    train_preds = pipeline.fit(embeds_train, embed_folds)
    test_preds = pipeline.predict(embeds_test)
    
    if gp['pred_train']:
        train_preds.to_csv(gp['pred_train_path'], index=False)
    test_preds.to_csv(gp['pred_test_path'], index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config path',
                        default='../config/main.yml', type=str)
    args = parser.parse_args()
    cpath = args.config
    main(cpath)
    