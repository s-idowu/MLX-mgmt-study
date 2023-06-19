from datetime import date
import pandas as pd

from config import Config

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
test_df = pd.read_csv(str(Config.DATASET_PATH / "test.csv"))


def extract_features(df):
    return df[["RM", "LSTAT"]]


train_features = extract_features(train_df)
test_features = extract_features(test_df)

train_features.to_csv(str(Config.FEATURES_PATH / "train_features.csv"), index=None)
test_features.to_csv(str(Config.FEATURES_PATH / "test_features.csv"), index=None)

train_df.Target.to_csv(
    str(Config.FEATURES_PATH / "train_labels.csv"), index=None
)
test_df.Target.to_csv(
    str(Config.FEATURES_PATH / "test_labels.csv"), index=None
)
