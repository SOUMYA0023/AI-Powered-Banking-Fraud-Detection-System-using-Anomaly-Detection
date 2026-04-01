import pandas as pd
import pytest

from core.preprocessing import add_scaled_amount, get_feature_columns, prepare_single_transaction


class DummyScaler:
    def transform(self, X):
        return [[0.5]]


def test_add_scaled_amount_column():
    df = pd.DataFrame({"Amount": [10.0, 20.0], "Time": [1, 2], "Class": [0, 1], **{f"V{i}": [0.1, 0.2] for i in range(1, 29)}})
    out, _ = add_scaled_amount(df)
    assert "Amount_scaled" in out.columns


def test_prepare_single_transaction_missing_amount_raises():
    feature_cols = get_feature_columns()
    payload = {"Time": 0.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    with pytest.raises(ValueError):
        prepare_single_transaction(payload, feature_cols, DummyScaler())
