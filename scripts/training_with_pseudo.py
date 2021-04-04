USE_PRIVATE_AUGMENTED = True

import sys


sys.path.append("../src")

from collections import defaultdict
import os
import random

import numpy as np
import pandas as pd
import yaml
import shutil
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import albumentations
from albumentations import *
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from torch.utils.data import DataLoader
from torch import nn
import torch

from transformers import get_linear_schedule_with_warmup

from datasets import SimpleDataset
from models import Wrapper, MixUp
from pipeline_utils import training, pseudolabeling
from models import ENCODER_PARAMS


PATH_TO_CFG = "../config/main.yml"
with open(PATH_TO_CFG, "r") as file:
    config = yaml.load(file)

DATA_ROOT = config["paths"]["data_path"]

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

fix_seed(config["seed"])
device_ids = [str(id) for id in config["device_ids"]]
ids = ",".join(device_ids)
ids = '0'
DEVICE = torch.device(f"cuda:{ids}")

train = pd.read_csv(os.path.join(DATA_ROOT, config["paths"]["train_file"]), index_col=0)

val_private = pd.read_csv(os.path.join(DATA_ROOT, config["paths"]["val_file"]), index_col=0)

if USE_PRIVATE_AUGMENTED:
    augmented_private = pd.read_csv(os.path.join(DATA_ROOT, config["paths"]["widen_file"]), index_col=0)
    mask = ~augmented_private["file_path"].str.contains("train/")
    augmented_private = augmented_private[mask].reset_index()

test = pd.read_csv(os.path.join(DATA_ROOT, config["paths"]["test_file"]), index_col=0)

def focal_loss(input, target, focus=2.0, raw=False):

    if raw:
        input = torch.sigmoid(input)

    eps = 1e-7

    prob_true = input * target + (1 - input) * (1 - target)
    prob_true = torch.clamp(prob_true, eps, 1-eps)
    modulating_factor = (1.0 - prob_true).pow(focus)

    return (-modulating_factor * prob_true.log()).mean()



class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type="cosface", eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)
#         print(x.shape)
        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L), wf

MODELS_NAMES = ["resnest14d_1e-4_joint_BCE_L1_with_private_augmented", "resnest50d_4s2x40d_1e-3_joint_BCE_L1_with_private_augmented_old_center_crop=150", 
                "resnest50d_4s2x40d_1e-4_joint_BCE_L1_with_private_augmented", "tf_efficientnet_b0_ns_1e-4_joint_BCE_L1_with_private_augmented", 
                "tf_efficientnet_b3_ns_1e-4_joint_BCE_L1_with_private_augmented"]

CONFIG_ROOT = "../config/"
PREFIX = "../"

for model_name in MODELS_NAMES:
    PATH_TO_CFG = os.path.join(CONFIG_ROOT, model_name + "_config.yaml")
    with open(PATH_TO_CFG, "r") as file:
        model_config = yaml.load(file)
    le = LabelEncoder() 

    mask_NR = (train["0"] == "NR") & ((train["1"] == 1) | (train["1"] == 6) | (train["1"] == 20))
    mask_ER = (train["0"] == "ER") & ((train["1"] == 3) | (train["1"] == 10) | (train["1"] == 30))
    train = train[mask_NR | mask_ER]
    if USE_PRIVATE_AUGMENTED:
        train = train.append(augmented_private)
    train.index = pd.RangeIndex(0, len(train.index))
    if model_config["general"]["task_type"] == "regression":
        train["target"] = train["1"]
        val_private["target"] = val_private["1"]
    elif model_config["general"]["task_type"] == "classification":
        train["target"] = le.fit_transform(train["target"])
    #     val_private["target"] = le.fit_transform(val_private)
    elif model_config["general"]["task_type"] == "joint":
        train["target_regression"] = train["1"]
        train["target_classification"] = le.fit_transform(train["0"])
        train["target"] = train["target_regression"].astype(str) + "_" + train["target_classification"].astype(str)
        
        val_private["target_regression"] = val_private["1"]
        val_private["target_classification"] = le.fit_transform(val_private["0"])


    kfold = StratifiedKFold(n_splits=model_config["training"]["n_folds"], shuffle=True,
                            random_state=config["seed"])
    for fold, (t_idx, v_idx) in enumerate(kfold.split(train, train["target"])):
        train.loc[v_idx, "kfold"] = fold


        
    train.to_csv(os.path.join(DATA_ROOT, "train", "train_folds.csv"))

    transforms_train = albumentations.Compose([
        ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        CenterCrop(*model_config["preprocessing"]["center_crop_size"]),
        Resize(*model_config["preprocessing"]["img_size"]),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])

    transforms_val = albumentations.Compose([
        CenterCrop(*model_config["preprocessing"]["center_crop_size"]),
        Resize(*model_config["preprocessing"]["img_size"]),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])
    EPOCHS = model_config["training"]["n_epochs"]


    criterion_aam = None
    if model_config["general"]["task_type"] == "classification":
        if model_config["training"]["loss"]["clf"] == "FOCAL":
            criterion = focal_loss
        elif model_config["training"]["loss"]["clf"] == "AAM":
            criterion = "AAM"
            criterion_aam = AngularPenaltySMLoss
        elif model_config["training"]["loss"]["clf"] == "BCE":
            criterion = nn.BCELoss()
    elif model_config["general"]["task_type"] == "regression":
        if model_config["training"]["loss"]["reg"] == "L1":
            criterion = nn.L1Loss()
        elif model_config["training"]["loss"]["reg"] == "L2":
            criterion = nn.MSELoss()
    elif model_config["training"]["loss"] == "L1":
        criterion = nn.L1Loss()
    elif model_config["general"]["task_type"] == "joint":
        criterion = {}
        if model_config["training"]["loss"]["clf"] == "FOCAL":
            criterion["clf"] = focal_loss
        elif model_config["training"]["loss"]["clf"] == "AAM":
            criterion["clf"] = "AAM"
            criterion_aam = AngularPenaltySMLoss
        elif model_config["training"]["loss"]["clf"] == "BCE":
            criterion["clf"] = nn.BCELoss()
        if model_config["training"]["loss"]["reg"] == "L1":
            criterion["reg"] = nn.L1Loss()
        elif model_config["training"]["loss"]["reg"] == "L2":
            criterion["reg"] = nn.MSELoss()

    try:
        shutil.rmtree(os.path.join(PREFIX, model_config["general"]["out_path"]))
    except:
        pass

    try:
        os.mkdir(os.path.join(PREFIX, model_config["general"]["out_path"]))
    except:
        pass


    if model_config["general"]["task_type"] == "regression":
        model_config["general"]["classes_num"] = 1
    elif model_config["general"]["task_type"] == "joint":
        model_config["general"]["classes_num"] = 2
        
        
    samples2preds_all = {}
    samples2trues_all = {}

    models = []
    for i in range(model_config["training"]["n_folds"]):
        model_name = model_config["general"]["model_name"]
        model = None
        model = ENCODER_PARAMS[model_name]["init_op"]()
        model = Wrapper(model, feat_module=None, classes_num=model_config["general"]["classes_num"],
                        model_name=model_name,
                        spec_augmenter=None, 
                        mixup_module=None,
                        task_type=model_config["general"]["task_type"],
                        activation_func=model_config["training"]["activation_func"],
                        criterion_aam=criterion_aam)
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config["training"]["lr"])
        train_dataset = SimpleDataset(df=train[train["kfold"] != fold], mode="train",
                                      transform=transforms_train, classes_num=model_config["general"]["classes_num"],
                                      task_type=model_config["general"]["task_type"])

        val_dataset = SimpleDataset(df=train[train["kfold"] == fold], mode="val",
                                    transform=transforms_val, classes_num=model_config["general"]["classes_num"],
                                    task_type=model_config["general"]["task_type"])
        val_private_dataset = SimpleDataset(df=val_private, mode="val",
                                            transform=transforms_val, classes_num=model_config["general"]["classes_num"],
                                            task_type=model_config["general"]["task_type"])
        
        train_dataloader = DataLoader(train_dataset,
                                      **model_config["training"]["dataloader"])
        val_dataloader = DataLoader(val_dataset,
                                    **model_config["validation"]["dataloader"])
        val_private_dataloader = DataLoader(val_private_dataset,
                                            **model_config["validation"]["dataloader"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=(model_config["training"]["n_epochs"] - model_config["training"]["n_epochs_flat"])  * len(train_dataloader),
                                                          eta_min=1e-8)    
        samples2preds, samples2trues, model = training(EPOCHS=EPOCHS, model=model,
                                                train_dataloader=train_dataloader, 
                                                val_dataloaders_dct={"val_dataloader": val_dataloader,
                                                                      "val_private_dataloader": val_private_dataloader},
                                                DEVICE=DEVICE, criterion=criterion,
                                                optimizer=optimizer, scheduler=scheduler,
                                                config=model_config, fold=i,
                                                task_type=model_config["general"]["task_type"], CONFIG_PATH=PATH_TO_CFG, prefix=PREFIX)
        models.append(model)
        samples2preds_all.update(samples2preds)
        samples2trues_all.update(samples2trues)

    samples2preds_all = {}
    samples2trues_all = {}
    LR = model_config["training"]["lr"]
    flag_LR = True

    for j in range(model_config["pseudo"]["iter"]):
        with torch.no_grad():
            train, test = pseudolabeling(models, train, test, model_config, DEVICE, transforms_val)
            private = (train["type"].values == "private").sum()
            public = (train["type"].values == "public").sum()
            print("Pseudo labeling epoch", j)
            print("Private ratio", private / (public + private)) 
            print("Public ratio", public / (public + private))
            if flag_LR:
                LR *= model_config["pseudo"]["lr_coef"]
                flag_LR = False
        
        for i in range(model_config["training"]["n_folds"]):
            model = models[i]
            model.to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            train_dataset = SimpleDataset(df=train[train["kfold"] != fold], mode="train",
                                          transform=transforms_train, classes_num=model_config["general"]["classes_num"],
                                          task_type=model_config["general"]["task_type"])

            val_dataset = SimpleDataset(df=train[train["kfold"] == fold], mode="val",
                                        transform=transforms_val, classes_num=model_config["general"]["classes_num"],
                                        task_type=model_config["general"]["task_type"])
            val_private_dataset = SimpleDataset(df=val_private, mode="val",
                                                transform=transforms_val, classes_num=model_config["general"]["classes_num"],
                                                task_type=model_config["general"]["task_type"])
            
            train_dataloader = DataLoader(train_dataset,
                                          **model_config["training"]["dataloader"])
            val_dataloader = DataLoader(val_dataset,
                                        **model_config["validation"]["dataloader"])
            val_private_dataloader = DataLoader(val_private_dataset,
                                                **model_config["validation"]["dataloader"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=(model_config["pseudo"]["n_epochs"] - model_config["pseudo"]["n_epochs_flat"])  * len(train_dataloader),
                                                              eta_min=1e-8)    
            samples2preds, samples2trues, model = training(EPOCHS=model_config["pseudo"]["n_epochs"], model=model,
                                                    train_dataloader=train_dataloader, 
                                                    val_dataloaders_dct={"val_dataloader": val_dataloader,
                                                                          "val_private_dataloader": val_private_dataloader},
                                                    DEVICE=DEVICE, criterion=criterion,
                                                    optimizer=optimizer, scheduler=scheduler,
                                                    config=model_config, fold=i, pseudo_iter=j+1,
                                                    task_type=model_config["general"]["task_type"], CONFIG_PATH=PATH_TO_CFG)
            models[i] = model
            samples2preds_all.update(samples2preds)
            samples2trues_all.update(samples2trues)

