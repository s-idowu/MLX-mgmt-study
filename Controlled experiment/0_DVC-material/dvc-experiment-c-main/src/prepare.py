from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

from config import Config


Config.ORIGINAL_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)


params = yaml.safe_load(open('params.yaml'))['prepare']

data = load_boston()
df = pd.DataFrame(data=data['data'], columns = data['feature_names'])
df['Target'] = data.target

df.to_csv(str(Config.ORIGINAL_DATASET_FILE_PATH / "boston.csv"), sep = ',', index = False)


df_train, df_test = train_test_split( df, test_size=params["split"], random_state=params["seed"] )

df_train.to_csv(str(Config.DATASET_PATH / "train.csv"), index=None)
df_test.to_csv(str(Config.DATASET_PATH / "test.csv"), index=None)
