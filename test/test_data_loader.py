import pandas as pd

from churn_prediction.data_loader import load_data


def test_load_data():
    df = load_data()

    assert isinstance(
        df , pd.DataFrame
    )

    assert not df.empty