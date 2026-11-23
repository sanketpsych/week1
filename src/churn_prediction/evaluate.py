import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from churn_prediction.config import config
from churn_prediction.exception import EvaluationError
from churn_prediction.logger import logger


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model using a custom threshold.
    """

    try:

        logger.info(
            "Starting model evaluation..."
        )

        logger.info(
            f"Using threshold: {config.THRESHOLD}"
        )

        # Generate prediction probabilities
        y_prob = model.predict_proba(X_test)[:, 1]

        # Apply threshold
        y_pred = (
            y_prob >= config.THRESHOLD
        ).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        # Store metrics
        metrics = {
            "threshold": config.THRESHOLD,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        # Save metrics
        with open(
            config.METRICS_PATH,
            "w"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        logger.info(
            "Metrics saved successfully"
        )

        # Save predictions
        predictions_df = pd.DataFrame(
            {
                "Actual": y_test,
                "Predicted": y_pred,
                "Probability": y_prob
            }
        )

        predictions_df.to_csv(
            config.PREDICTIONS_PATH,
            index=False
        )

        logger.info(
            "Predictions saved successfully"
        )

        # Classification Report
        report = classification_report(
            y_test,
            y_pred
        )

        with open(
            config.REPORT_PATH,
            "w"
        ) as file:

            file.write(report)

        logger.info(
            "Classification report saved"
        )

        # Confusion Matrix
        cm = confusion_matrix(
            y_test,
            y_pred
        )

        plt.figure(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(
            f"Confusion Matrix (Threshold={config.THRESHOLD})"
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        plt.savefig(
            config.CONFUSION_MATRIX_PATH
        )

        plt.close()

        logger.info(
            "Confusion matrix saved"
        )

        logger.info(
            f"Accuracy: {accuracy:.4f}"
        )

        logger.info(
            f"Precision: {precision:.4f}"
        )

        logger.info(
            f"Recall: {recall:.4f}"
        )

        logger.info(
            f"F1 Score: {f1:.4f}"
        )

        return metrics

    except Exception as e:

        raise EvaluationError(
            f"Evaluation failed: {e}"
        )