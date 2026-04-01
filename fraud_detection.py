"""Backward-compatible inference entrypoint."""

from core.inference import InferenceEngine

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def predict_transaction(transaction_data, model="ensemble", threshold=None):
    engine = _get_engine()
    return engine.predict(transaction_data, model=model, threshold=threshold)


if __name__ == "__main__":
    sample = {"Time": 0, "Amount": 100.0}
    sample.update({f"V{i}": 0.0 for i in range(1, 29)})
    print(predict_transaction(sample))
