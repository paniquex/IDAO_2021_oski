import sys
from paths import PATH_APPEND
sys.path.append(PATH_APPEND)

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, mean_absolute_error
# from metrics import calculate_per_class_roc_auc
from collections import defaultdict
import torch
import os
from shutil import copyfile
from datasets import SimpleDataset
from torch.utils.data import DataLoader


def training(EPOCHS, model, train_dataloader,
             val_dataloaders_dct, DEVICE, criterion,
             optimizer, config, scheduler=None,
             fold=0, pseudo_iter=0, task_type="classification", CONFIG_PATH="/media/paniquex/samsung_2tb/IDAO_2021_oski/config/config.yaml"):
    if fold == 0:
        copyfile(CONFIG_PATH,
                 f"{config['general']['out_path']}config.yaml")
    tta_steps = 0
    best_scores = {}
    model_names = {}
    best_max = {}
    best_mean = {}
    for val_dataloader_name in val_dataloaders_dct:
        if task_type == "classification":
            best_scores[val_dataloader_name] = 0
            best_max[val_dataloader_name] = 0
            best_mean[val_dataloader_name] = 0
        elif task_type == "regression":
            best_scores[val_dataloader_name] = -10000
            best_max[val_dataloader_name] = -10000
            best_mean[val_dataloader_name] = -10000
        elif task_type == "joint":
            best_scores[val_dataloader_name] = -10000
            best_max[val_dataloader_name] = -10000
            best_mean[val_dataloader_name] = -10000
        model_names[val_dataloader_name] = []

    train_df = defaultdict(list)
    val_df = {}
    for val_dataloader_name in val_dataloaders_dct:
        val_df[val_dataloader_name] = defaultdict(list)

    if task_type == "joint":
        samples2trues_all = {"clf": {}, "reg": {}}
        samples2preds_all = {"clf": {}, "reg": {}}
    else:
        samples2trues_all = {}
        samples2preds_all = {}

    early_stopping_counter = 0
    early_stopping_criterion = config["training"]["early_stopping_criterion"]
    for epoch in range(0, EPOCHS + 1):
        losses, trues, preds = [], [], []
        t = tqdm(train_dataloader)
        model.train()
        for batch in t:
            if task_type == "joint":
                img_, class_ = batch["img"].to(DEVICE), {"clf": batch["label"]["clf"].to(DEVICE),
                                                         "reg": batch["label"]["reg"].to(DEVICE)}
            else:
                img_, class_ = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            output_dict = model(img_)  # class_) # predictions, mixuped classes
            if criterion == "AAM":
                preds = output_dict["preds"]
                if config["general"]["use_additional_loss_for_aam"]:
                    loss = 0.5 * output_dict["loss"] + torch.nn.BCELoss()(pred.float(), class_.float())
                else:
                    loss = output_dict["loss"]
            else:
                preds = output_dict["preds"]
                trues = {"clf": [], "reg": []}
                if task_type == "joint":
                    preds_clf = preds["clf"]
                    preds_reg = preds["reg"]
                    loss = criterion["clf"](preds_clf.float(), class_["clf"]) + criterion["reg"](preds_reg.float(),
                                                                                                 class_["reg"])
                else:
                    loss = criterion(preds.float(), class_.float())
            try:
                losses.append(loss.item())
            except:
                print("ERROR in loss list appending")

            if task_type == "joint":
                preds["clf"] = preds["clf"].detach().cpu().numpy()
                preds["reg"] = preds["reg"].detach().cpu().numpy()
                trues["clf"] = class_["clf"].detach().cpu().numpy()
                trues["reg"] = class_["reg"].detach().cpu().numpy()
            else:
                preds = pred.cpu().numpy()
                trues = class_.cpu().numpy()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            t.set_description(f"{np.mean(losses):.4f}")
            if epoch >= config["training"]["n_epochs_flat"]:
                if scheduler is not None:
                    try:
                        scheduler.step()
                    except:
                        scheduler.step(-np.max(best_max, best_mean))
        if task_type == "joint":
            for key in ["clf", "reg"]:
                trues[key] = np.vstack(trues[key])
                preds[key] = np.vstack(preds[key])
        else:
            trues = np.vstack(trues)
            preds = np.vstack(preds)

        loss = np.hstack(losses).mean()

        if task_type == "classification":
            preds_argmax = np.argmax(preds, axis=1)
            trues_argmax = np.argmax(trues, axis=1)
            score_f1 = f1_score(trues_argmax, preds_argmax, average='micro')
            prec = precision_score(trues_argmax, preds_argmax, average='micro')
            rec = recall_score(trues_argmax, preds_argmax, average='micro')
            roc_auc = roc_auc_score(trues_argmax, preds, multi_class="ovr")
        elif task_type == "regression":
            mae = mean_absolute_error(trues, preds)
        elif task_type == "joint":
            preds_argmax = np.argmax(preds["clf"], axis=1)
            trues_argmax = np.argmax(trues["clf"], axis=1)
            score_f1 = f1_score(trues_argmax, preds_argmax, average='micro')
            prec = precision_score(trues_argmax, preds_argmax, average='micro')
            rec = recall_score(trues_argmax, preds_argmax, average='micro')
            roc_auc = roc_auc_score(trues["clf"][:, 0], preds["clf"][:, 0])
            mae = mean_absolute_error(trues["reg"], preds["reg"])
            comp_metric = 1000 * (roc_auc - mae)

        if task_type == "classification":
            scores_dict = {"loss": loss,
                           "f1_score": score_f1,
                           "prec": prec,
                           "rec": rec,
                           "roc_auc": roc_auc}
        elif task_type == "regression":
            scores_dict = {"loss": loss, "mae": mae}
        elif task_type == "joint":
            scores_dict = {"loss": loss,
                           "f1_score": score_f1,
                           "prec": prec,
                           "rec": rec,
                           "roc_auc": roc_auc}
            scores_dict.update({"mae": mae, "comp_metric": comp_metric})
        info_dict = {"epoch": epoch}
        info_dict.update(scores_dict)
        log_results(train_df, "train", info_dict=info_dict, trues=trues,
                    preds_dict={"preds": preds},
                    out_path=config["general"]["out_path"])
        pd.DataFrame(train_df).to_csv(f"{config['general']['out_path']}{config['general']['model_name']}_train.csv",
                                      index=None)
        flag_early_stopping = False
        if epoch % config["training"]["logging_freq"] == 0:
            for val_dataloader_name in val_dataloaders_dct:
                scores_dict, trues, preds_dict, samples2preds, samples2trues = evaluate(model=model,
                                                                                        dataloader=val_dataloaders_dct[
                                                                                            val_dataloader_name],
                                                                                        criterion=criterion,
                                                                                        DEVICE=DEVICE,
                                                                                        tta_steps=tta_steps,
                                                                                        config=config,
                                                                                        task_type=task_type)
                if task_type == "joint":
                    for key in ["clf", "reg"]:
                        samples2preds_all[key].update(samples2preds[key])
                        samples2trues_all[key].update(samples2trues[key])
                else:
                    samples2preds_all.update(samples2preds)
                    samples2trues_all.update(samples2trues)

                info_dict = {"epoch": epoch}
                info_dict.update(scores_dict)
                log_results(val_df[val_dataloader_name], "val", info_dict=info_dict,
                            out_path=config["general"]["out_path"],
                            trues=trues, preds_dict=preds_dict)
                if not flag_early_stopping:
                    early_stopping_counter += 1
                    print(f"EARLY STOPPING COUNTER: {early_stopping_counter}/{early_stopping_criterion}")
                    flag_early_stopping = True

                if config["general"]["early_stopping_by"] == "roc_auc":
                    best_max[val_dataloader_name] = scores_dict["roc_auc"]
                    best_mean[val_dataloader_name] = scores_dict["roc_auc"]
                elif config["general"]["early_stopping_by"] == "f1":
                    best_max[val_dataloader_name] = scores_dict["f1_score"]
                    best_mean[val_dataloader_name] = scores_dict["f1_score"]
                elif config["general"]["early_stopping_by"] == "mae":
                    best_max[val_dataloader_name] = -scores_dict["mae"]
                    best_mean[val_dataloader_name] = -scores_dict["mae"]
                elif config["general"]["early_stopping_by"] == "comp_metric":
                    best_max[val_dataloader_name] = scores_dict["comp_metric"]
                    best_mean[val_dataloader_name] = scores_dict["comp_metric"]
                if max(best_max[val_dataloader_name], best_mean[val_dataloader_name]) > best_scores[
                    val_dataloader_name]:
                    early_stopping_counter = 0
                    optim_params = optimizer.state_dict()
                    model_params = model.state_dict()
                    all_params = {'model_state_dict': model_params, 'optimizer_state_dict': optim_params}
                    best_scores[val_dataloader_name] = max(best_max[val_dataloader_name],
                                                           best_mean[val_dataloader_name])
                    torch.save(all_params,
                               f"{config['general']['out_path']}{config['general']['model_name']}_score={best_scores[val_dataloader_name]:.5f}")
                    model_names[val_dataloader_name].append(
                        f"{config['general']['out_path']}{config['general']['model_name']}_score={best_scores[val_dataloader_name]:.5f}")
                pd.DataFrame(val_df[val_dataloader_name]).to_csv(
                    f"{config['general']['out_path']}{config['general']['model_name']}_{val_dataloader_name}_{pseudo_iter}_{fold}.csv",
                    index=None)

        if early_stopping_counter > early_stopping_criterion:
            break
    for val_dataloader_name in val_dataloaders_dct:
        best_model_name = model_names[val_dataloader_name][-1]
        print(model_names[val_dataloader_name])
        print(best_model_name)
        os.rename(best_model_name,
                  f"{config['general']['out_path']}best_model_fold{fold}_score={best_scores[val_dataloader_name]:.5f}.pth")
        for model_name in model_names[val_dataloader_name][:-1]:
            if model_name != best_model_name:
                os.remove(model_name)
    return samples2preds_all, samples2trues_all, model


def evaluate(model, dataloader, DEVICE,
             criterion=None, config=None, tta_steps=0, task_type="classification"):
    t = tqdm(dataloader)
    if task_type == "joint":
        samples2preds = {"clf": {}, "reg": {}}
        samples2trues = {"clf": {}, "reg": {}}
    else:
        samples2preds = {}
        samples2trues = {}
    with torch.no_grad():
        model.eval()
        losses = []
        if tta_steps == 0:
            print("Predict test without augmentations")
        for batch in t:
            output_dict = model(batch["img"].to(DEVICE))
            if task_type == "joint":
                class_ = {"clf": batch["label"]["clf"].to(DEVICE).float(),
                          "reg": batch["label"]["reg"].to(DEVICE).float()}
            else:
                class_ = batch["label"].to(DEVICE).float()
            if output_dict["label"] is not None:
                class_ = output_dict["label"]
            if criterion is not None:
                if criterion == "AAM":
                    preds = output_dict["preds"]
                    loss = output_dict["loss"]
                else:
                    preds = output_dict["preds"]
                    if task_type == "joint":
                        preds_clf = preds["clf"]
                        preds_reg = preds["reg"]
                        loss = criterion["clf"](preds_clf.float(), class_["clf"]) + criterion["reg"](preds_reg.float(),
                                                                                                     class_["reg"])
                    else:
                        loss = criterion(preds.float(), class_.float())
                try:
                    losses.append(loss.item())
                except:
                    print("ERROR in appending")
            if task_type == "joint":
                trues = {}
                preds["clf"] = preds["clf"].cpu().numpy()
                preds["reg"] = preds["reg"].cpu().numpy()
                trues["clf"] = class_["clf"].cpu().numpy()
                trues["reg"] = class_["reg"].cpu().numpy()
            else:
                preds = preds.cpu().numpy()
                trues = class_.cpu().numpy()
            if task_type == "joint":
                for key in ["clf", "reg"]:
                    for sample, true, pred in zip(batch['sample'], trues[key], preds[key]):
                        if sample not in samples2trues:
                            samples2trues[key][sample] = [true]
                        else:
                            samples2trues[key][sample].append(true)

                        if sample not in samples2preds:
                            samples2preds[key][sample] = [pred]
                        else:
                            samples2preds[key][sample].append(pred)

            else:
                for sample, true, pred in zip(batch['sample'], trues, preds):
                    if sample not in samples2trues:
                        samples2trues[sample] = [true]
                    else:
                        samples2trues[sample].append(true)
                  
                    if sample not in samples2preds:
                        samples2preds[sample] = [pred]
                    else:
                        samples2preds[sample].append(pred)
        if task_type == "joint":
            trues = {"clf": [], "reg": []}
            preds = {"clf": [], "reg": []}
            for key in ["clf", "reg"]:
                for sample in samples2preds[key]:
                    pred = np.vstack(samples2preds[key][sample])
                    true = np.vstack(samples2trues[key][sample])
                    preds[key].append(pred)
                    trues[key].append(true)
                preds[key] = np.vstack(preds[key])
                trues[key] = np.vstack(trues[key])
        else:
            trues = []
            preds = []
            for sample in samples2preds:
                pred = np.vstack(samples2preds[sample])
                true = np.vstack(samples2trues[sample])
                preds.append(pred)
                trues.append(true)
            preds = np.vstack(preds)
            trues = np.vstack(trues)
        if task_type == "classification":
            preds_argmax = np.argmax(preds, axis=1)
            trues_argmax = np.argmax(trues, axis=1)
            score_f1 = f1_score(trues_argmax, preds_argmax, average='micro')
            prec = precision_score(trues_argmax, preds_argmax, average='micro')
            rec = recall_score(trues_argmax, preds_argmax, average='micro')
            roc_auc = roc_auc_score(trues_argmax, preds, multi_class="ovr")
        elif task_type == "regression":
            mae = mean_absolute_error(trues, preds)
        elif task_type == "joint":
            preds_argmax = np.argmax(preds["clf"], axis=1)
            trues_argmax = np.argmax(trues["clf"], axis=1)
            score_f1 = f1_score(trues_argmax, preds_argmax, average='micro')
            prec = precision_score(trues_argmax, preds_argmax, average='micro')
            rec = recall_score(trues_argmax, preds_argmax, average='micro')
            roc_auc = roc_auc_score(trues["clf"][:, 0], preds["clf"][:, 0])
            mae = mean_absolute_error(trues["reg"], preds["reg"])
            comp_metric = 1000 * (roc_auc - mae)

        if task_type == "classification":
            scores_dict = {"loss": loss,
                           "f1_score": score_f1,
                           "prec": prec,
                           "rec": rec,
                           "roc_auc": roc_auc}
        elif task_type == "regression":
            scores_dict = {"loss": loss, "mae": mae}
        elif task_type == "joint":
            scores_dict = {"loss": loss,
                           "f1_score": score_f1,
                           "prec": prec,
                           "rec": rec,
                           "roc_auc": roc_auc}
            scores_dict.update({"mae": mae, "comp_metric": comp_metric})
        preds_dict = {"preds": preds}
        return scores_dict, trues, preds_dict, samples2preds, samples2trues


def log_results(df, prefix, info_dict, trues, preds_dict, out_path):
    for info in info_dict:
        df[f"{prefix}_{info}"].append(info_dict[info])

    np.save(f"{out_path}{prefix}_true_{info_dict['epoch']}.npy",
            trues)
    for pred_name in preds_dict:
        np.save(f"{out_path}{prefix}_pred_{pred_name}_{info_dict['epoch']}.npy",
                preds_dict[pred_name])


def evaluate_test(model, dataloader, DEVICE, config):
    t = tqdm(dataloader)
    model.eval()
    samples2preds = {}
    for batch in t:
        img_ = batch["img"].to(DEVICE)
        output_dict = model(img_)
        if config["general"]["task_type"] == "regression":
            preds = output_dict["preds"].cpu().numpy()
        elif config["general"]["task_type"] == "classification":
            preds = torch.argmax(output_dict["preds"], dim=1).cpu().numpy()
        assert len(batch['sample']) == preds.shape[0]
        for sample, pred in zip(batch['sample'], preds):
            if len(pred.shape) == 2:
                samples2preds[sample] = pred
            else:
                if sample not in samples2preds:
                    samples2preds[sample] = [pred]
                else:
                    samples2preds[sample].append(pred)
    return samples2preds


import torch
import torch.nn.functional as F
import torch.nn as nn


@torch.jit.script
def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return mish(input)


def convert_relu_to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            convert_relu_to_Mish(child)
                  
                  
def pseudolabeling(models, Train, Test, config, DEVICE, transforms_val):
    threshold = config["pseudo"]["threshold"]
    clf = torch.zeros((len(Test), 2))
    reg = torch.zeros(len(Test))

    task_type = config["general"]["task_type"]
    batch_size = config["testing"]["dataloader"]["batch_size"]
    for model in models:
        model.eval()

        test_dataset = SimpleDataset(df=Test, mode="test", classes_num=config["general"]["classes_num"],
                                        task_type=config["general"]["task_type"], transform=transforms_val)

        test_dataloader = DataLoader(test_dataset,
                                        **config["testing"]["dataloader"])

        for i, batch in enumerate(test_dataloader):
            output_dict = model(batch["img"].to(DEVICE))
            preds = output_dict["preds"]
            assert task_type == "joint"
            preds["clf"] = preds["clf"].cpu().numpy()
            preds["reg"] = preds["reg"].cpu().numpy()

            clf[i * batch_size : (i+1) * batch_size, :] += preds["clf"]
            reg[i * batch_size : (i+1) * batch_size] += preds["reg"]

        model.train()

    clf /= len(models)
    reg /= len(models)
    max_prob = clf.max(axis=1)[0]

    pseudo_X = Test.iloc[np.where(max_prob > threshold)[0]].copy()
    pseudo_clf = torch.max(clf[np.where(max_prob > threshold)[0]], axis=1)[1]
    pseudo_reg = reg[np.where(max_prob > threshold)[0]]

    if config["pseudo"]["reg_postprocessing"]:
        energy_values = Train["1"].unique()
        mapper = {i: energy_values[i] for i in range(len(energy_values))}
        pseudo_reg = np.vectorize(mapper.get)(np.argmin(abs(pseudo_reg - energy_values), axis=1))

    pseudo_X["target_regression"] = pseudo_reg
    pseudo_X["1"] = pseudo_reg
    pseudo_X["target_classification"] = pseudo_clf
    pseudo_X["0"] =  pseudo_X.apply(lambda x: "ER" if x["target_classification"] == 0 else "NR", axis=1)
    pseudo_X["target"] = pseudo_X.apply(lambda x: str(int(x["target_regression"])) + "_" + str(x["target_classification"]),axis=1)
    pseudo_X["kfold"] = config["training"]["n_folds"]
    new_Test = Test.iloc[np.where(max_prob <= threshold)[0]]

    new_Train = pd.concat([Train, pseudo_X],axis=0)

    return new_Train, new_Test
