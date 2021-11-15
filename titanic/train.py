import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import os.path as osp
from argparse import ArgumentParser, Namespace
from utils import validate_path
from pytorch_tabnet.callbacks import Callback
import wandb
from base_model import model as base_model_params
from typing import Dict, Any

RANDOM_STATE = hash("fuck yeah!") % (2 ** 32 - 1)
print(RANDOM_STATE)


def main(args: Namespace):
    train_path = osp.join(args.input_dir, "train.csv")
    train_df = pd.read_csv(train_path, index_col="PassengerId")
    y = train_df["survived"].values
    X = train_df.drop(["survived"], axis=1).values
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mean_aucs = []
    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        valid_auc = model._callback_container.callbacks[-1].best_loss
        mean_aucs.append(valid_auc)
    mean_auc = np.mean(np.array(mean_aucs))
    return mean_auc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-in", "--input-dir", type=validate_path, default="data/processed"
    )
    parser.add_argument("-out", "--output-dir", type=validate_path, default=".")
    parser.add_argument("-w", "--wandb-project", type=str, default="titanic")
    arg_groups: Dict[str, Any] = dict()

    for param_group in base_model_params:
        arg_groups[param_group] = parser.add_argument_group(param_group)
        for param in base_model_params[param_group]:
            param_kwargs = param.copy()
            name = param_kwargs["name"]
            del param_kwargs["name"]
            arg_groups[param_group].add_argument(f"--{name}", **param_kwargs)
    args = parser.parse_args()
    print(args)
    # main(args)
