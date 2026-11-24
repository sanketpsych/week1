import pandas as pd

from churn_prediction.config import config
from churn_prediction.logger import logger


def load_data():

    logger.info("Loading dataset")

    df = pd.read_csv(config.DATA_PATH)

    # 1. Dataset shape
    print("Dataset shape:", df.shape)

    # 2. Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # 3. Duplicate records
    print("\nDuplicate records:", df.duplicated().sum())

    # 4. Target distribution
    print("\nTarget distribution:")
    print(df[config.TARGET_COLUMN].value_counts())

    # 5. Leakage columns
    print("\nLeakage columns present:")
    print([
        col for col in config.LEAKAGE_COLUMNS
        if col in df.columns
    ])

    logger.info("Dataset loaded successfully")

    return df