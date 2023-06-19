import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yaml

from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

params = yaml.safe_load(open('params.yaml'))['train']

X_train = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))


model = LinearRegression(normalize=True)
#model = RandomForestRegressor()

model = model.fit(X_train, y_train.to_numpy().ravel())

pickle.dump(model, open(str(Config.MODELS_PATH / "model.pickle"), "wb"))
