"""
train.py

This file is responsible for:
1. Creating the ML pipeline
2. Training the model
3. Saving the trained model
"""

import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from churn_prediction.config import config
from churn_prediction.logger import logger
from churn_prediction.exception import TraingError


def train_model(X_train, y_train):
    """
    Train Logistic Regression pipeline.
    """

    try:

        logger.info(
            "Training model"
        )

        # Pipeline:
        # Step 1 -> Scale features
        # Step 2 -> Train Logistic Regression

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=config.RANDOM_STATE,
                    class_weight="balanced"
                )
            )
        ])

        # Train complete pipeline
        model.fit(
            X_train,
            y_train
        )

        logger.info(
            "Model training completed"
        )

        # Save pipeline
        joblib.dump(
            model,
            config.MODEL_PATH
        )

        logger.info(
            f"Model saved at {config.MODEL_PATH}"
        )

        return model

    except Exception as e:

        raise TraingError(
            f"Training failed: {e}"
        )