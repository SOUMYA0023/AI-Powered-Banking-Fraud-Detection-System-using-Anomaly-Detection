import pytest

from fraud_detection import predict_transaction


def test_predict_transaction_basic_shape():
    payload = {"Time": 0.0, "Amount": 120.0}
    payload.update({f"V{i}": 0.0 for i in range(1, 29)})
    res = predict_transaction(payload, model="ensemble", threshold=0.5)
    assert "prediction" in res
    assert "fraud_probability" in res
    assert 0.0 <= res["fraud_probability"] <= 1.0


def test_missing_fields_raises():
    with pytest.raises(Exception):
        predict_transaction({"Time": 0.0, "Amount": 10.0}, model="ensemble")
