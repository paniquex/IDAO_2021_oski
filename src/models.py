import sys
from paths import PATH_APPEND
sys.path.append(PATH_APPEND)

import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from functools import partial

from pipeline_utils import convert_relu_to_Mish
import torch.nn.functional as F
import timm
from timm.models.efficientnet import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, tf_efficientnet_b6_ns, tf_efficientnet_b7_ns,\
    tf_efficientnet_b0_ns, tf_efficientnet_l2_ns, tf_efficientnet_lite0
from timm.models.nfnet import nf_resnet50, nf_regnet_b0, nf_ecaresnet50


class Wrapper(nn.Module):
    def __init__(self, model, feat_module, classes_num, model_name,
                 spec_augmenter, task_type="classification", activation_func="ReLU",
                 mixup_module=None, criterion_aam=None, test=False):
        super().__init__()
        self.model = model
        if activation_func == "Mish":
            convert_relu_to_Mish(model)
        self.feat_module = feat_module
        self.mixup_module = mixup_module
        self.spec_augmenter = spec_augmenter
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        # self.max_pool = AdaptiveMaxPool2d((1, 1))
        self.dropout = Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)
        if task_type == "joint":
            self.fc_clf = Linear(ENCODER_PARAMS[model_name]['features'], classes_num)
            self.fc_reg = Linear(ENCODER_PARAMS[model_name]['features'], 1)
        else:
            self.fc = Linear(ENCODER_PARAMS[model_name]['features'], classes_num)
        self.test = test
        if criterion_aam is not None:
            self.criterion_aam = criterion_aam(in_features=2048 * 8, out_features=classes_num)
            # self.fc_aam = nn.Linear(16384, classes_num)
        else:
            self.criterion_aam = criterion_aam

        self.task_type = task_type

    def forward(self, imgs, labels=None, return_embeddings=False):
        # with torch.no_grad():
        #     if self.mixup_module and labels is not None:
        #         imgs, labels = self.mixup_module(imgs, labels)
        # x = self.feat_module(imgs)
        
        # if labels is not None:
        #     if self.spec_augmenter:
        #         x = self.spec_augmenter(x)

        x = self.model.forward_features(imgs)
        if self.criterion_aam is not None:
            x = self.stat_pool(x)
            x = torch.flatten(x, start_dim=1, end_dim=2)
            if labels is None:
                x = self.criterion_aam.fc(x)
                loss = torch.tensor([0])
            else:
                loss, x = self.criterion_aam(x, torch.argmax(labels, axis=1).long())
            x = self.softmax(x)
            output_dict = {
                'preds': x,
                'loss': loss,
                'label': labels
            }
            return output_dict
        else:
            x = self.avg_pool(x).flatten(1)
            if return_embeddings:
                output_dict = {"embeddings": x}
            else:
                output_dict = {}
            x = self.dropout(x)
            if self.task_type == "joint":
                x_clf = self.softmax(self.fc_clf(x))
                x_reg = self.fc_reg(x).view(-1)
                x = {"clf": x_clf, "reg": x_reg}
            else:
                x = self.fc(x)
                if self.task_type == "classification":
                    x = self.softmax(x)
                elif self.task_type == "regression":
                    x = x.view(-1)
            output_dict.update({
                'preds': x,
                'label': labels
            })
            return output_dict


class MixUp(nn.Module):
    def __init__(self, prob=0.3, alpha=0.4, mixup_mode="basic"):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
        self.mixup_mode = mixup_mode
        
    def forward(self, imgs, labels):
        inds = np.arange(imgs.shape[0])
        new_inds = inds.copy()
        np.random.shuffle(new_inds)
        aug_count = int(inds[inds != new_inds].shape[0] * self.prob)
        to_augment = np.random.choice(inds[inds != new_inds], aug_count, replace=False)
        betas = torch.tensor(np.random.beta(self.alpha, self.alpha, size=aug_count),
                             dtype=torch.float).unsqueeze(1).to(imgs.device)
        # new_inds = torch.tensor(new_inds)
        # to_augment = torch.tensor(to_augment)
        imgs[to_augment] = betas * imgs[to_augment] + (1 - betas) * imgs[new_inds][to_augment]
        if self.mixup_mode == "basic":
            labels[to_augment] = betas * labels[to_augment] + (1 - betas) * labels[new_inds][to_augment]
        elif self.mixup_mode == "or":
            labels[to_augment] = torch.clamp_max(labels[to_augment] + labels[new_inds][to_augment], max=1.)
        return imgs, labels


ENCODER_PARAMS = {
    "nf_resnet50": {
            "features": 2048,
            "init_op": partial(nf_resnet50, pretrained=True, in_chans=3)
        },
    "nf_regnet_b0": {
                "features": 960,
                "init_op": partial(nf_regnet_b0, pretrained=True, in_chans=3)
            },
    "nf_ecaresnet50": {
                    "features": 2048,
                    "init_op": partial(nf_ecaresnet50, pretrained=True, in_chans=3)
                },
    "resnest14d": {
        "features": 2048,
        "init_op": partial(timm.models.resnest14d, pretrained=True, in_chans=3)
    },
    "resnest26d": {
        "features": 2048,
        "init_op": partial(timm.models.resnest26d, pretrained=True, in_chans=3)
    },
    "resnest50d": {
        "features": 2048,
        "init_op": partial(timm.models.resnest50d, pretrained=True, in_chans=3)
    },
    "resnest50d_4s2x40d": {
        "features": 2048,
        "init_op": partial(timm.models.resnest50d_4s2x40d, pretrained=True, in_chans=3)
    },
    "resnest101e": {
        "features": 2048,
        "init_op": partial(timm.models.resnest101e, pretrained=True, in_chans=3)
    },
"resnest200e": {
        "features": 2048,
        "init_op": partial(timm.models.resnest200e, pretrained=True, in_chans=3)
    },
"resnest269e": {
        "features": 2048,
        "init_op": partial(timm.models.resnest269e, pretrained=True, in_chans=3)
    },

    "densenet201": {
        "features": 1920,
        "init_op": partial(timm.models.densenet201, pretrained=True, in_chans=3)
    },
    "dpn92": {
        "features": 2688,
        "init_op": partial(timm.models.dpn92, pretrained=True, in_chans=3)
    },
    "dpn131": {
        "features": 2688,
        "init_op": partial(timm.models.dpn131, pretrained=True, in_chans=3)
    },

        "tf_efficientnet_b0_ns": {
            "features": 1280,
            "init_op": partial(tf_efficientnet_b0_ns, pretrained=True, drop_path_rate=0.2, in_chans=3)
        },
"tf_efficientnet_lite0": {
            "features": 1280,
            "init_op": partial(tf_efficientnet_lite0, pretrained=True, drop_path_rate=0.2, in_chans=3)
        },
        "tf_efficientnet_b3_ns": {
            "features": 1536,
            "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2, in_chans=3)
        },
        "tf_efficientnet_b2_ns": {
            "features": 1408,
            "init_op": partial(tf_efficientnet_b2_ns, pretrained=True, drop_path_rate=0.2, in_chans=3)
        },
        "tf_efficientnet_b4_ns": {
            "features": 1792,
            "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.2, in_chans=3)
        },
        "tf_efficientnet_b5_ns": {
            "features": 2048,
            "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2, in_chans=3)
        },
        "tf_efficientnet_b6_ns": {
            "features": 2304,
            "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2, in_chans=3)
        },
        "tf_efficientnet_l2_ns": {
                    "features": 2304,
                    "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2, in_chans=3)
                },
}
