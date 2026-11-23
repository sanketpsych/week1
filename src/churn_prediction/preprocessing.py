
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_prediction.config import config
from churn_prediction.exception import PreprocessingError
from churn_prediction.logger import logger


def preprocess_data(df: pd.DataFrame):

    try:
        

        logger.info("Starting preprocessing")
        if "customerID" in df.columns:
            #drop the column that doesnt improve the model decision
            df = df.drop(columns=["customerID"])
        df[config.TARGET_COLUMN] = (
            #target variable encoding
            df[config.TARGET_COLUMN]
            .map(
                {
                    "Yes": 1,
                    "No": 0
                }
            )
        )
        df = df.fillna(0)
        #to convert categorical columns into a numerical binary columns
        df = pd.get_dummies(
            df,
            drop_first=True
        )

    
        X = df.drop(
            columns=[config.TARGET_COLUMN]
        )

     

        y = df[
            config.TARGET_COLUMN
        ]

  

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE
            )
        )

        logger.info(
            "Preprocessing completed successfully"
        )

        logger.info(
            f"Training rows: {len(X_train)}"
        )

        logger.info(
            f"Testing rows: {len(X_test)}"
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

    except Exception as e:  # noqa: BLE001

        raise PreprocessingError(
            f"Preprocessing failed: {e}"
        )