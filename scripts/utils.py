import os
import sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

import albumentations
from albumentations import CoarseDropout
import cv2

from tqdm import tqdm


def parse_dataset(data_path=os.path.join("..", "data"),
                  train_file='train.csv',
                  val_file='val_private.csv',
                  test_file='test.csv',
                  train_dir='train',
                  public_test_dir='public_test',
                  private_test_dir='private_test'):
    """
    Creates train, test and valid csv
    """
    train_dir_path = os.path.join(data_path, train_dir)
    train_nr_files = os.listdir(os.path.join(train_dir_path, 'NR'))
    train_er_files = os.listdir(os.path.join(train_dir_path, 'ER'))
    train_files = train_nr_files + train_er_files
    train_csv = pd.DataFrame([file.split("_")[6:8] if len(file.split("_")) > 18 \
                              else file.split("_")[5:7] for file in train_files])
    train_csv['file_path'] = train_files
    train_csv.loc[train_csv[0] == 'NR', 'file_path'] = train_dir_path + '/NR/' + train_csv['file_path']
    train_csv.loc[train_csv[0] == 'ER', 'file_path'] = train_dir_path + '/ER/' + train_csv['file_path']
    train_csv['target'] = train_csv[0] + train_csv[1]
    train_csv[1] = train_csv[1].astype(int)

    le = LabelEncoder()
    train_csv['target'] = le.fit_transform(train_csv['target'])
    train_csv.to_csv(os.path.join(data_path, train_file))

    mask_NR = (train_csv[0] == 'NR') & ((train_csv[1] == 1) | (train_csv[1] == 6) | (train_csv[1] == 20))
    mask_ER = (train_csv[0] == 'ER') & ((train_csv[1] == 3) | (train_csv[1] == 10) | (train_csv[1] == 30))
    val_private = train_csv[~(mask_NR | mask_ER)]

    train_csv[~(mask_NR | mask_ER)].to_csv(
        os.path.join(data_path, val_file))

    public_test = os.listdir(os.path.join(data_path, public_test_dir))
    test_csv = pd.DataFrame({'file_path': public_test,
                             'type': 'public'})
    private_test = os.listdir(os.path.join(data_path, private_test_dir))
    private_csv = pd.DataFrame({'file_path': private_test,
                                'type': 'private'})
    test_csv = test_csv.append(private_csv).reset_index()

    test_csv.loc[test_csv.type == 'public', 'file_path'] = data_path + "/" + public_test_dir + "/" + test_csv['file_path']
    test_csv.loc[test_csv.type == 'private', 'file_path'] = data_path + "/" + private_test_dir + "/" + test_csv['file_path']
    test_csv.to_csv(os.path.join(data_path, test_file))

def widen_valid_dataset(data_path=os.path.join("..", "data"),
                        val_file='val_private.csv',
                        save_dir='valid',
                        widen_file='widen_val_private.csv',
                        aug_rounds=2000,
                        cutout_params=None):
    """
    Expands dataset by using augmentations (cut-off).
    """
    if cutout_params is None:
        cutout_pararms = {'max_holes': 60, 'min_holes': 20,
                          'max_height': 40, 'max_width': 40,
                          'min_height': 20, 'min_width': 40,
                          'p': 1.0}

    val_path = os.path.join(data_path, val_file)
    val_private = pd.read_csv(val_path, index_col=0)
    os.makedirs(os.path.join(data_path, save_dir), exist_ok=True)

    aug = albumentations.Compose([
        CoarseDropout(**cutout_params)])

    new_private = []
    for data in val_private.iterrows():
        img = cv2.imread(data[1].file_path)
        name = data[1].file_path.split('/')[-1]
        new_private.append(data[1])
        for i in tqdm(range(aug_rounds), desc=name[:20], ncols=50):
            aug_img = aug(image=img)
            aug_data = data[1].copy()
            aug_data['file_path'] = os.path.join(data_path, save_dir, f'aug_{i}_{name}')
            new_private.append(aug_data)
            cv2.imwrite(aug_data['file_path'], aug_img['image'])
    new_private = pd.DataFrame(new_private)
    save_path = os.path.join(data_path, widen_file)
    new_private.reset_index(drop=True).to_csv(save_path)
