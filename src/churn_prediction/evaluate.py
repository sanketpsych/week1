

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
 
    try:
        logger.info(
            "Starting model evaluation..."
        )

        # Generate predictions
       
        y_pred = model.predict(X_test)

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
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        # Save metrics.json
 

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


        # Save predictions.csv


        predictions_df = pd.DataFrame(
            {
                "Actual": y_test,
                "Predicted": y_pred
            }
        )

        predictions_df.to_csv(
            config.PREDICTIONS_PATH,
            index=False
        )

        logger.info(
            "Predictions saved successfully"
        )

       
        # Classification report
     
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

   
        # Confusion matrix


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
            "Confusion Matrix"
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

        return metrics

    except Exception as e:

        raise EvaluationError(
            f"Evaluation failed: {e}"
        )