"""
test_preprocessing.py

Tests for preprocessing.py
"""

from churn_prediction.data_loader import load_data
from churn_prediction.preprocessing import preprocess_data


def test_preprocess_data():
    """
    Verify preprocessing pipeline.
    """

    df = load_data()

    X_train, X_test, y_train, y_test = (
        preprocess_data(df)
    )

    # Training data must not be empty
    assert len(X_train) > 0

    # Testing data must not be empty
    assert len(X_test) > 0

    # Target values must exist
    assert len(y_train) > 0

    assert len(y_test) > 0