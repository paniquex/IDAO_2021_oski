"""
Final pipeline for IDAO 2021 Qualifiers.
"""


from yaml import safe_load
from argparse import ArgumentParser
from utils import parse_dataset, widen_valid_dataset
import os


def main(config_path):
    with open(config_path, 'r') as f:
        config = safe_load(f)
    
    print(config)
    
    paths = config['paths']
    
    data_root = paths['data_path']
    train_file = paths['train_file']
    val_file = paths['val_file']
    train_dir = paths['train_dir']
    public_test_dir = paths['public_test_dir']
    private_test_dir = paths['private_test_dir']
    test_file = paths['test_file']
    
    parse_dataset(data_root, train_file,
                  val_file, test_file,
                  train_dir, public_test_dir,
                  private_test_dir)
    
    gen_params = config['data_generation']
    aug_rounds = gen_params['aug_rounds']
    cutout_params = gen_params['cutout_params']
    widen_file = paths['widen_file']
    widen_valid_dir = paths['widen_valid_dir']
    
    widen_valid_dataset(data_root, val_file,
                        widen_valid_dir, widen_file,
                        aug_rounds, cutout_params)
                  


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default=os.path.join("..", "config", "main.yml"),
                        help='Main config path')
    args = parser.parse_args()
    config_path = args.config
    main(config_path)
    
