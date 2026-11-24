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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score,recall_score)


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
        #grid search hyper parameter tuning
        param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__solver": [
        "liblinear",
        "lbfgs"
    ]
}
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring="f1",
            n_jobs=1
        )
        logger.info("starting gridsearchcv")
        # Train complete pipeline
        grid_search.fit(
            X_train,
            y_train
        )
        logger.info("gridSearchcv completed")
        model = grid_search.best_estimator_

        #training prediction
        y_train_prob = model.predict_proba(
            X_train
        )[: , 1]

        y_train_pred = (
                y_train_prob >= config.THRESHOLD
        ).astype(int)



        #training metrix
        train_accuracy = accuracy_score(
            y_train,
            y_train_pred
        )
        train_recall = recall_score(
            y_train,
            y_train_pred
        )
        logger.info(
            f"train accuracy: {train_accuracy:.4f}"
        )
        logger.info(
            f"train recall : {train_recall:.4f}"
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