from pathlib import Path

class Config:
    ASSETS_PATH = Path("./data")
    ORIGINAL_DATASET_FILE_PATH = ASSETS_PATH / "raw"
    DATASET_PATH = ASSETS_PATH / "prepared"
    FEATURES_PATH = ASSETS_PATH / "features"
    MODELS_PATH =  Path("./model")
    METRICS_FILE_PATH = Path("./metrics")
