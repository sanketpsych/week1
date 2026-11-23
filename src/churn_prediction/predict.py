import joblib
import pandas as pd

from churn_prediction.config import config
from churn_prediction.exception import PreditionError
from churn_prediction.logger import logger


# function to load the model
def load_model ():
    try:
        logger.info("loading the model")

        model = joblib.load(
            config.MODEL_PATH
        )
        logger.info("model added successfully")

        return model
    except Exception as e:
        raise PreditionError(
            f"failed to load model: {e}"
        )
# function to predict the model
def predict(model , customer_data):
    # arguments : model - trained model
    # customer_data - preprocessed data
    # returns the prediction
    try:
        prediction = model.predict(
            customer_data
        )
        return prediction
    except Exception as e:
        raise PreditionError(
            f"prediction failed{e}"
        )
  