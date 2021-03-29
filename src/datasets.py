import sys
from paths import PATH_APPEND
sys.path.append(PATH_APPEND)

from torch.utils.data import Dataset
import torch
import cv2
import os


class SimpleDataset(Dataset):
    def __init__(self, df, mode, target_cols="target", transform=None, classes_num=12, task_type="classification"):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.samples = self.df["file_path"].values
        if mode != "test":
            if task_type == "joint":
                self.labels_clf = df[target_cols + "_classification"].values[:, None] == list(range(classes_num))
                self.labels_reg = df[target_cols + "_regression"].values

            else:
                self.labels = df[target_cols].values
                if task_type == "classification":
                    self.labels = self.labels[:, None] == list(range(classes_num))
        self.task_type = task_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        img = cv2.imread(row.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']
        if self.mode == 'test':
            return {"sample": self.samples[index], "img": img[0].unsqueeze(0)}
        else:
            if self.task_type == "joint":
                label = {"clf": torch.tensor(self.labels_clf[index]).float(),
                         "reg": torch.tensor(self.labels_reg[index]).float()}
            else:
                label = torch.tensor(self.labels[index]).float()

            return {"sample": self.samples[index], "img": img[0].unsqueeze(0), "label": label}


class EmbGenerationDataset(Dataset):
    def __init__(self, df, fold, paths_list, transform=None):
        self.df = df.reset_index(drop=True)
        print(len(df.file_path.values))
        self.file_names = []
        for path in paths_list:
            fnames = os.listdir(path)
            fnames = [os.path.join(path, fname) for fname in fnames]
            if ("NR" in path) or ("ER" in path):
                fnames = [fname for fname in fnames if (fname in self.df[df["kfold"] == fold].file_path.values) or (fname not in self.df.file_path.values)]
                print("CHECK1", len(fnames))
            self.file_names.extend(fnames)
        print(len(self.file_names))
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):

        img = cv2.imread(self.file_names[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']
        return {"sample": self.file_names[index], "img": img}
