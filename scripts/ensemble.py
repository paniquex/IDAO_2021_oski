import pandas as pd
import numpy as np
import os
import yaml


PATH_TO_CFG = "../config/main.yml"
with open(PATH_TO_CFG, "r") as file:
    config = yaml.load(file)

DATA_ROOT = config["paths"]["data_path"]

PREFIX = "../"
dirs = [x for x in os.listdir(os.path.join(PREFIX, "experiments")) if x.find("augmented") != -1]

csv_list = []

for x in dirs:
    name = os.listdir(os.path.join(PREFIX, "experiments", x))
    name = [n for n in name if (n.find("predictions") != -1) & (n.find("postprocessed") == -1)][0]
    csv_list.append(pd.read_csv(os.path.join(PREFIX, "experiments", x, name)))
    
file_names_public = [x for x in os.listdir(os.path.join(DATA_ROOT, "public_test")) if ".png" in x] #+ os.listdir(os.path.join(path, "private_test"))
test_csv = pd.DataFrame({"file_path": file_names_public, "type": "public"})
file_names_private = [x for x in os.listdir(os.path.join(DATA_ROOT, "private_test")) if ".png" in x]#+ os.listdir(os.path.join(path, "private_test"))
test_csv = test_csv.append(pd.DataFrame({"file_path": file_names_private, "type": "private"})).reset_index()
test_csv["file_path"] = test_csv["file_path"].str.replace(".png", "")

test_csv["id"] = test_csv["file_path"]
test_csv.pop("index")
test_csv.pop("file_path");

for i in range(len(csv_list)):
    csv_list[i] = csv_list[i].merge(test_csv, on="id")

ensembled_csv = csv_list[0].copy()

for i in range(1, len(csv_list)):
    ensembled_csv["regression_predictions"] += csv_list[i]["regression_predictions"]
    ensembled_csv["classification_predictions"] += csv_list[i]["classification_predictions"]

    
ensembled_csv["regression_predictions"] /= len(csv_list)
ensembled_csv["classification_predictions"] = ensembled_csv["classification_predictions"] > (len(csv_list) / 2)
ensembled_csv["classification_predictions"] = ensembled_csv["classification_predictions"].astype(int)

public_energies_1 = np.array([3, 10, 30])
public_energies_0 = np.array([1, 6, 20])

mapper_public_0 = {i: public_energies_0[i] for i in range(len(public_energies_0))}
mapper_public_1 = {i: public_energies_1[i] for i in range(len(public_energies_1))}


mask_public_0 = (ensembled_csv["type"] == "public") & (ensembled_csv["classification_predictions"] == 0)
mask_public_1 = (ensembled_csv["type"] == "public") & (ensembled_csv["classification_predictions"] == 1)

mask_private_0 = (ensembled_csv["type"] == "private") & (ensembled_csv["classification_predictions"] == 0)
mask_private_1 = (ensembled_csv["type"] == "private") & (ensembled_csv["classification_predictions"] == 1)


ensembled_csv.loc[mask_public_0, "regression_predictions"] = np.vectorize(mapper_public_0.get)(np.argmin(np.abs(ensembled_csv[mask_public_0]["regression_predictions"].values[:, None] - public_energies_0),
                                                axis=1))
ensembled_csv.loc[mask_public_1, "regression_predictions"] = np.vectorize(mapper_public_1.get)(np.argmin(np.abs(ensembled_csv[mask_public_1]["regression_predictions"].values[:, None] - public_energies_1),
                                                axis=1))
ensembled_csv.loc[mask_private_0, "regression_predictions"] = np.vectorize(mapper_public_1.get)(np.argmin(np.abs(ensembled_csv[mask_private_0]["regression_predictions"].values[:, None] - public_energies_1),
                                                axis=1))
ensembled_csv.loc[mask_private_1, "regression_predictions"] = np.vectorize(mapper_public_0.get)(np.argmin(np.abs(ensembled_csv[mask_private_1]["regression_predictions"].values[:, None] - public_energies_0),
                                                axis=1))

ensembled_csv.drop(columns=["type"]).to_csv(os.path.join(PREFIX, config["paths"]["predictions_path"],
                                                         "ensemble_post_smart.csv"), index=False)

ensembled_csv
