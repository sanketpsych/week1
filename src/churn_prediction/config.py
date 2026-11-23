from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Dataset
    DATA_PATH: str = "data/customer_churn.csv"
    TARGET_COLUMN: str = "Churn"
    ID_COLUMN: str = "customerID"

    #threshold 
    THRESHOLD : float = 0.30



    # Columns that leak the outcome 
    LEAKAGE_COLUMNS: tuple = (
        "Churn Reason",
        "Churn Score",
        "Churn Label",
        "Churn Category",
        "Customer Status",
    )
    DROP_DUPLICATES: bool = False

    # Train-Test Split
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # Model
    MODEL_PATH: str = "models/churn_model.joblib"

    # Results
    METRICS_PATH: str = "results/metrics.json"
    PREDICTIONS_PATH: str = "results/predictions.csv"
    REPORT_PATH: str = "results/classification_report.txt"
    CONFUSION_MATRIX_PATH: str = "results/confusion_matrix.png"


config = Config()