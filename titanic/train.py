import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import os.path as osp
from argparse import ArgumentParser, Namespace
from utils import validate_path, read_data

RANDOM_STATE = hash("fuck yeah!")


def main(args: Namespace):
    train_path = osp.join(args.input_dir, "train.csv")
    test_path = osp.join(args.input_dir, "test.csv")
    train_df, test_df = read_data(train_path, test_path)
    print(train_df)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-in", "--input-dir", type=validate_path, default="data/processed"
    )
    parser.add_argument("-out", "--output-dir", type=validate_path, default=".")
    args = parser.parse_args()
    main(args)
