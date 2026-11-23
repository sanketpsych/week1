import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent / "src")
)

from churn_prediction.data_loader import load_data
from churn_prediction.preprocessing import preprocess_data
from churn_prediction.train import train_model
from churn_prediction.evaluate import evaluate_model
from churn_prediction.logger import logger

def main():
    logger.info(
        "prediction started"
    )
    # load the dataset
    df = load_data()
    #preprocess dataset
    X_train , X_test, Y_train , y_test =(
        preprocess_data(df)
    )
    # train model
    model = train_model(
        X_train , Y_train
    )
    # evaluate model 
    metrics = evaluate_model(
        model , X_test , y_test
    )
    logger.info(
        f"Final Metrics : {metrics}"
    )
    logger.info(
        "Pipeline completed successfully"
    )

if __name__ == "__main__":
   main()