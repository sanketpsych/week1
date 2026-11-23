import pandas as pd

from churn_prediction.config import config
from churn_prediction.logger import logger


def load_data():
    "load the dataset"
    df = pd.read_csv(
        config.DATA_PATH)
    logger.info(f"dataset loaded successfully.Shape:{df.shape}")
  
    
    return df
