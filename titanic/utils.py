import os.path as osp
import pandas as pd
from typing import NamedTuple


class TrainTestDataFrames(NamedTuple):
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def read_data(train_path: str, test_path: str) -> TrainTestDataFrames:
    train_df: pd.DataFrame = pd.read_csv(train_path, index_col="PassengerId")
    test_df: pd.DataFrame = pd.read_csv(test_path, index_col="PassengerId")
    dfs = TrainTestDataFrames(train_df, test_df)
    return dfs


def validate_path(path: str) -> str:
    if not osp.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    else:
        return path
