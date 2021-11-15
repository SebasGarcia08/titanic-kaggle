# External libraries
from utils import validate_path, read_data
import os.path as osp
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# built-in libraries
from argparse import ArgumentParser, Namespace
from typing import Dict, List, NamedTuple, Optional, Tuple
from dataclasses import dataclass
from abc import ABC


class TrainTestDataFrames(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame


class FeatureCreator(ABC):
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


@dataclass
class FamilyNumberCreator(FeatureCreator):
    col: str = "familysz"

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.col] = df["sibsp"] + df["parch"] + 1
        return df


def binnarize(x: int, bins: Dict[str, Tuple[int, int]]) -> Optional[str]:
    for k, v in bins.items():
        if v[0] <= x < v[1]:
            return k


@dataclass
class FamilyCategoryCreator(FeatureCreator):
    col_2_bin: str
    binnerized_col: str
    bins: Dict[str, Tuple[float, float]]

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, self.binnerized_col] = df.loc[:, self.col_2_bin].apply(
            lambda x: binnarize(x, self.bins)
        )
        return df


@dataclass
class Featurizer(object):
    featurizers: List[FeatureCreator]

    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        for f in self.featurizers:
            df = f.create(df)
        return df


def lowercase_cols(df: pd.DataFrame) -> pd.DataFrame:
    columns: List[str] = df.columns
    rename_mapper: Dict[str, str] = dict((col, col.lower()) for col in columns)
    return df.rename(columns=rename_mapper)


def featurize(df: pd.DataFrame) -> pd.DataFrame:

    age_mapper: Dict[str, Tuple[float, float]] = {
        "child": (0, 13),
        "teen": (13, 22),
        "young": (22, 30),
        "adult": (30, 50),
        "old": (50, np.inf),
    }

    family_mapper: Dict[str, Tuple[float, float]] = {
        "zero": (0, 1),
        "medium": (1, 4),
        "large": (4, np.inf),
    }

    # Apply features
    featurerizers: List[FeatureCreator] = [
        FamilyNumberCreator(col="family_number"),
        FamilyCategoryCreator(
            col_2_bin="family_number", binnerized_col="familysz", bins=family_mapper
        ),
        FamilyCategoryCreator(
            col_2_bin="age", binnerized_col="age_group", bins=age_mapper
        ),
    ]
    featurizer = Featurizer(featurizers=featurerizers)
    df = featurizer.featurize(df)
    return df


def handle_missing_values_continuous(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    imputer = KNNImputer(weights="distance")

    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    train_df = imputer.fit_transform(train_df)
    test_df = imputer.transform(test_df)

    train_df = scaler.inverse_transform(train_df)
    test_df = scaler.inverse_transform(test_df)

    return train_df, test_df


def label_encode_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for col in train_df.columns:
        lb = LabelEncoder()
        train_df.loc[:, col] = lb.fit_transform(train_df[col]).astype("int")
        test_df.loc[:, col] = lb.transform(test_df[col]).astype("int")
    return train_df, test_df


def handle_missing_values_categorical(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="most_frequent")
    train_df = imputer.fit_transform(train_df)
    test_df = imputer.transform(test_df)
    return train_df, test_df


def main(args: Namespace):
    print("Reading data...")
    train_path = osp.join(args.input_dir, "train.csv")
    test_path = osp.join(args.input_dir, "test.csv")
    train_df, test_df = read_data(train_path, test_path)

    train_df = lowercase_cols(train_df)
    test_df = lowercase_cols(test_df)

    handled = handle_missing_values_continuous(
        train_df=train_df[args.cont_features], test_df=test_df[args.cont_features],
    )
    train_df[args.cont_features], test_df[args.cont_features] = handled

    handled = handle_missing_values_categorical(
        train_df=train_df[args.cat_features], test_df=test_df[args.cat_features],
    )
    train_df[args.cat_features], test_df[args.cat_features] = handled
    train_df = featurize(train_df)
    test_df = featurize(test_df)

    categortical_features = args.cat_features + ["age_group", "familysz"]
    features = args.cont_features + categortical_features
    res = label_encode_features(
        train_df[categortical_features], test_df[categortical_features]
    )
    train_df.loc[:, categortical_features], test_df.loc[:, categortical_features] = res

    ## DELETE DUPLICATES
    non_duplicated_idxs = train_df[features].drop_duplicates().index
    duplicated_idxs = train_df[train_df.duplicated(subset=features)].index
    print(f"Deleting {duplicated_idxs} ids because are duplicated")

    train_df = train_df.loc[non_duplicated_idxs, features + ["survived"]]
    test_df = test_df[features]

    train_df.to_csv(osp.join(args.output_dir, "train.csv"))
    test_df.to_csv(osp.join(args.output_dir, "test.csv"))

    print(train_df, train_df.isna().sum())
    print()
    print(test_df, test_df.isna().sum())


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-in",
        "--input-dir",
        type=validate_path,
        default="data/raw/",
        help="input directory",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=validate_path,
        default="data/processed/",
        help="output directory",
    )

    parser.add_argument(
        "-catf",
        "--cat-features",
        nargs="+",
        help="Categorical features to use",
        default=["pclass", "sex", "embarked"],
    )
    parser.add_argument(
        "-contf",
        "--cont-features",
        nargs="+",
        help="Categorical features to use",
        default=["age", "fare", "sibsp", "parch"],
    )

    args = parser.parse_args()
    main(args)
