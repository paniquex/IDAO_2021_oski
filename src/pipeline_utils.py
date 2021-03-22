import sys
sys.path.append("/media/paniquex/samsung_2tb/IDAO_2021_oski/src")
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, mean_absolute_error
# from metrics import calculate_per_class_roc_auc
from collections import defaultdict
import torch
import os
from shutil import copyfile


def training(EPOCHS, model, train_dataloader,
             val_dataloader, DEVICE, criterion,
             optimizer, config, scheduler=None,
             fold=0, task_type="classification"):
    if fold == 0:
        copyfile("/media/paniquex/samsung_2tb/IDAO_2021_oski/config/config.yaml", f"{config['general']['out_path']}config.yaml")
    tta_steps = 0
    if task_type == "classification":
        best_scores = {'val': 0}
    elif task_type == "regression":
        best_scores = {'val': -10000}
    model_names = []

    train_df = defaultdict(list)
    val_df = defaultdict(list)

    samples2trues_all = {}
    samples2preds_all = {}

    early_stopping_counter = 0
    early_stopping_criterion = config["training"]["early_stopping_criterion"]
    for epoch in range(0, EPOCHS + 1):
        losses, trues, preds = [], [], []
        t = tqdm(train_dataloader)
        model.train()
        for batch in t:
            img_, class_ = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            output_dict = model(img_) # class_) # predictions, mixuped classes
            if criterion == "AAM":
                pred = output_dict["preds"]
                if config["general"]["use_additional_loss_for_aam"]:
                    loss = 0.5 * output_dict["loss"] + torch.nn.BCELoss()(pred.float(), class_.float())
                else:
                    loss = output_dict["loss"]
            else:
                pred = output_dict["preds"]
                loss = criterion(pred.float(), class_.float())
            try:
                losses.append(loss.item())
            except:
                print("ERROR in loss list appending")

            preds.append(pred.cpu().detach().numpy())
            trues.append(class_.cpu().detach().numpy())

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

        trues = np.vstack(trues)
        preds = np.vstack(preds)

        loss = np.hstack(losses).mean()
        if task_type == "classification":
            trues_argmax = np.argmax(trues, axis=1)
            preds_argmax = np.argmax(preds, axis=1)
            score_f1 = f1_score(trues_argmax, preds_argmax, average='micro')
            prec = precision_score(trues_argmax, preds_argmax, average='micro')
            rec = recall_score(trues_argmax, preds_argmax, average='micro')
            # roc_auc = roc_auc_score(trues_argmax, preds, multi_class="ovr",)# labels=np.arange(config["general"]["classes_num"]))
            roc_auc = 0
        elif task_type == "regression":
            mae = mean_absolute_error(trues, preds)
        if task_type == "classification":
            info_dict = {"epoch": epoch,
                         "loss": loss,
                         "f1_score": score_f1,
                         "prec": prec,
                         "rec": rec,
                         "roc_auc": roc_auc}
        elif task_type == "regression":
            info_dict = {"epoch": epoch,
                         "loss": loss,
                         "mae": mae}
        log_results(train_df, "train", info_dict=info_dict, trues=trues,
                    preds_dict={"preds": preds},
                    out_path=config["general"]["out_path"])
        pd.DataFrame(train_df).to_csv(f"{config['general']['out_path']}{config['general']['model_name']}_train.csv",
                                      index=None)

        if epoch % config["training"]["logging_freq"] == 0:

            scores_dict, trues, preds_dict, samples2preds, samples2trues = evaluate(model=model, dataloader=val_dataloader,
                                                                                    criterion=criterion, DEVICE=DEVICE,
                                                                                    tta_steps=tta_steps, config=config,
                                                                                    task_type=task_type)
            samples2preds_all.update(samples2preds)
            samples2trues_all.update(samples2trues)

            info_dict = {"epoch": epoch}
            info_dict.update(scores_dict)
            log_results(val_df, "val", info_dict=info_dict, out_path=config["general"]["out_path"],
                        trues=trues, preds_dict=preds_dict)
            early_stopping_counter += 1
            print(f"EARLY STOPPING COUNTER: {early_stopping_counter}/{early_stopping_criterion}")
            if config["general"]["early_stopping_by"] == "roc_auc":
                best_max = scores_dict["roc_auc"]
                best_mean = scores_dict["roc_auc"]
            elif config["general"]["early_stopping_by"] == "f1":
                best_max = scores_dict["f1_score"]
                best_mean = scores_dict["f1_score"]
            elif config["general"]["early_stopping_by"] == "mae":
                best_max = -scores_dict["mae"]
                best_mean = -scores_dict["mae"]
            if max(best_max, best_mean) > best_scores['val']:
                early_stopping_counter = 0
                optim_params = optimizer.state_dict()
                model_params = model.state_dict()
                all_params = {'model_state_dict': model_params, 'optimizer_state_dict': optim_params}
                best_scores['val'] = max(best_max, best_mean)
                torch.save(all_params, f"{config['general']['out_path']}{config['general']['model_name']}_score={best_scores['val']:.5f}")
                model_names.append(f"{config['general']['out_path']}{config['general']['model_name']}_score={best_scores['val']:.5f}")
            pd.DataFrame(val_df).to_csv(f"{config['general']['out_path']}{config['general']['model_name']}_val_{fold}.csv", index=None)

            if early_stopping_counter > early_stopping_criterion:
                break
    best_model_name = model_names[-1]
    print(model_names)
    print(best_model_name)
    os.rename(best_model_name, f"{config['general']['out_path']}best_model_fold{fold}_score={best_scores['val']:.5f}.pth")
    for model_name in model_names[:-1]:
        if model_name != best_model_name:
            os.remove(model_name)
    return samples2preds_all, samples2trues_all


def evaluate(model, dataloader, DEVICE,
             criterion=None, config=None, tta_steps=0, task_type="classification"):
    t = tqdm(dataloader)
    samples2preds = {}
    samples2trues = {}
    with torch.no_grad():
        model.eval()
        losses = []
        if tta_steps == 0:
            print("Predict test without augmentations")
        for batch in t:
            output_dict = model(batch["img"].to(DEVICE))
            class_ = batch["label"].to(DEVICE).float()
            if output_dict["label"] is not None:
                class_ = output_dict["label"]
            if criterion is not None:
                if criterion == "AAM":
                    preds = output_dict["preds"]
                    loss = output_dict["loss"]
                else:
                    preds = output_dict["preds"]
                    loss = criterion(preds.float(), class_.float())
                try:
                    losses.append(loss.item())
                except:
                    print("ERROR in appending")
            preds = preds.cpu().numpy()
            trues = class_.cpu().numpy()
            for sample, true, pred in zip(batch['sample'], trues, preds):
                if sample not in samples2trues:
                    samples2trues[sample] = [true]
                else:
                    samples2trues[sample].append(true)

                if sample not in samples2preds:
                    samples2preds[sample] = [pred]
                else:
                    samples2preds[sample].append(pred)

        trues = []
        preds = []
        for sample in samples2preds:
            pred = np.vstack(samples2preds[sample])
            true = np.vstack(samples2trues[sample])
            preds.append(pred)  # [:24] to exclude silence class
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

        if task_type == "classification":
            scores_dict = {"loss": loss,
                           "f1_score": score_f1,
                           "prec": prec,
                           "rec": rec,
                           "roc_auc": roc_auc}
        elif task_type == "regression":
            scores_dict = {"loss": loss, "mae": mae}
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