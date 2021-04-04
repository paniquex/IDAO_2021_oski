import sys
sys.path.append("../src")


from collections import defaultdict
import os
import random

import numpy as np
import pandas as pd
import yaml
import shutil
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch import nn
import torch



from transformers import get_linear_schedule_with_warmup

from datasets import SimpleDataset
from models import Wrapper
from pipeline_utils import evaluate_test
from models import ENCODER_PARAMS

import albumentations
from albumentations import *
from albumentations.pytorch import ToTensorV2


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
DEVICE = torch.device(f"cuda:{ids}")

MODELS_NAMES = ["resnest14d_1e-4_joint_BCE_L1_with_private_augmented", "resnest50d_4s2x40d_1e-3_joint_BCE_L1_with_private_augmented_old_center_crop=150", 
                "resnest50d_4s2x40d_1e-4_joint_BCE_L1_with_private_augmented", "tf_efficientnet_b0_ns_1e-4_joint_BCE_L1_with_private_augmented", 
                "tf_efficientnet_b3_ns_1e-4_joint_BCE_L1_with_private_augmented"]
CONFIG_ROOT = "../config/"
PREFIX = "../"

for MODEL_NAME in MODELS_NAMES:
    PATH_TO_CFG = os.path.join(CONFIG_ROOT, MODEL_NAME + "_config.yaml")
    with open(PATH_TO_CFG, "r") as file:
        model_config = yaml.load(file)
    
    transforms_test = albumentations.Compose([
        CenterCrop(*model_config["preprocessing"]["center_crop_size"]),
        Resize(*model_config["preprocessing"]["img_size"]),
        Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225],
         ),
        ToTensorV2()
    ])
    path = DATA_ROOT # directory with public_test and private_test directories
    
    file_names_public = [x for x in os.listdir(os.path.join(path, "public_test")) if ".png" in x]
    test_csv = pd.DataFrame({"file_path": file_names_public, "type": "public"})
    file_names_private = [x for x in os.listdir(os.path.join(path, "private_test")) if ".png" in x]
    test_csv = test_csv.append(pd.DataFrame({"file_path": file_names_private, "type": "private"})).reset_index()

    test_csv.loc[test_csv["type"] == "public", "file_path"] = str(os.path.join(path, "public_test")) + "/" + test_csv["file_path"]
    test_csv.loc[test_csv["type"] == "private", "file_path"] = str(os.path.join(path, "private_test")) + "/" + test_csv["file_path"]

    test = test_csv

    test_dataset = SimpleDataset(df=test, mode="test", transform=transforms_test)

    test_dataloader = DataLoader(test_dataset, **model_config["testing"]["dataloader"])
    
    if model_config["general"]["task_type"] == "regression":
        model_config["general"]["classes_num"] = 1
        
    model_name = model_config["general"]["model_name"]
    model = ENCODER_PARAMS[model_name]["init_op"]()


    if model_config["general"]["task_type"] == "regression":
        model_config["general"]["classes_num"] = 1
    elif model_config["general"]["task_type"] == "joint":
        model_config["general"]["classes_num"] = 2

    if model_config["training"]["loss"] == "AAM":
        criterion_aam = AngularPenaltySMLoss
    else:
        criterion_aam = None
    model = Wrapper(model, feat_module=None, classes_num=model_config["general"]["classes_num"],
                        model_name=model_name,
                    spec_augmenter=None, 
                    mixup_module=None,
                    task_type=model_config["general"]["task_type"],
                    activation_func=model_config["training"]["activation_func"],
                    criterion_aam=criterion_aam)
    model.to(DEVICE);
    
    try:
        os.mkdir(os.path.join(PREFIX, model_config["general"]["out_path"]))
    except:
        pass

    if model_config["general"]["task_type"] == "joint":
        sample2preds = None
    else:
        sample2preds = None

    names = [name for name in os.listdir(os.path.join(PREFIX, model_config["general"]["out_path"])) if name.find("best_model_fold") != -1]
    with torch.no_grad():
        for name in names:
            model.load_state_dict(torch.load(os.path.join(PREFIX, model_config["general"]["out_path"], name),
                                        map_location=torch.device(DEVICE))['model_state_dict'])
            if model_config["general"]["task_type"] == "joint":
                if sample2preds is None:
                        sample2preds = evaluate_test(model=model, dataloader=test_dataloader,
                                                          DEVICE=DEVICE, config=model_config)
                else:
                    sample2preds_new = evaluate_test(model=model, dataloader=test_dataloader,
                                                          DEVICE=DEVICE, config=model_config)
                    for key in ["clf", "reg"]:

                            for sample in sample2preds[key]:
                                sample2preds[key][sample] += sample2preds_new[key][sample]
            else:
                if sample2preds is None:
                    sample2preds = evaluate_test(model=model, dataloader=test_dataloader,
                                  DEVICE=DEVICE, config=model_config)
                else:
                    sample2preds_new = evaluate_test(model=model, dataloader=test_dataloader,
                                  DEVICE=DEVICE, config=model_config)
                    for sample in sample2preds:
                        sample2preds[sample] += sample2preds_new[sample]

    preds = pd.read_csv(os.path.join(DATA_ROOT, "track1_predictions_example.csv"))

    if model_config["general"]["task_type"] == "joint":
        for key in ["clf", "reg"]:
            for sample in sample2preds[key]:
                sample_short = sample.split("/")[-1][:-4]

                if key == "clf":
                    preds.loc[preds["id"] == sample_short, "classification_predictions"] = np.argmax(np.bincount(np.argmin(sample2preds[key][sample], axis=1)))                
                elif key == "reg":
                    preds.loc[preds["id"] == sample_short, "regression_predictions"] = np.mean(sample2preds[key][sample])
    else:
        for sample in sample2preds:

            sample_short = sample.split("/")[-1][:-4]
            if model_config["general"]["task_type"] == "regression":
                preds.loc[preds["id"] == sample_short, "regression_predictions"] = np.mean(sample2preds[sample])
            elif model_config["general"]["task_type"] == "classification":
                preds.loc[preds["id"] == sample_short, "classification_predictions"] = np.argmax(np.bincount(sample2preds[sample]))       
    
    preds.to_csv(os.path.join(PREFIX, model_config["general"]["out_path"], 
                              f"predictions_{model_config['general']['task_type']}_{model_config['general']['model_name']}.csv"), index=False)
