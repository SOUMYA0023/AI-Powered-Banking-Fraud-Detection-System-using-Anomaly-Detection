"""
AI-Powered Banking Fraud Detection Engine
=========================================
Detects fraudulent transactions using Isolation Forest and Random Forest models.
"""

import os
import numpy as np
import pandas as pd
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Model files
ISOLATION_FOREST_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
RANDOM_FOREST_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "amount_scaler.pkl")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


def _load_models():
    """Load trained models and scaler. Returns (iso_forest, rf_model, scaler, feature_cols)."""
    if not os.path.exists(ISOLATION_FOREST_PATH):
        raise FileNotFoundError(
            "Models not found. Please run the notebook first: notebooks/fraud_analysis.ipynb"
        )
    iso_forest = joblib.load(ISOLATION_FOREST_PATH)
    rf_model = joblib.load(RANDOM_FOREST_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return iso_forest, rf_model, scaler, feature_cols


def _prepare_transaction(transaction_data: dict, feature_cols: list, scaler) -> np.ndarray:
    """
    Convert transaction dict/row into feature vector for prediction.

    transaction_data can be:
    - dict with keys: Time, Amount, V1...V28 (optionally Amount_scaled)
    - pandas Series or single row DataFrame
    """
    if isinstance(transaction_data, pd.DataFrame):
        row = transaction_data.iloc[0]
    elif isinstance(transaction_data, pd.Series):
        row = transaction_data
    else:
        row = pd.Series(transaction_data)

    # Build feature vector in correct order (DataFrame for sklearn feature names)
    features = {}
    for col in feature_cols:
        if col in row:
            features[col] = [row[col]]
        elif col == "Amount_scaled" and "Amount" in row:
            amount = float(row["Amount"])
            features[col] = [scaler.transform([[amount]])[0][0]]
        else:
            raise ValueError(f"Missing feature: {col}")

    return pd.DataFrame(features)


def predict_transaction(transaction_data, model="ensemble"):
    """
    Predict whether a transaction is fraudulent.

    Parameters
    ----------
    transaction_data : dict, pd.Series, or pd.DataFrame
        Must contain: Time, Amount, V1, V2, ..., V28
    model : str, optional
        "isolation_forest", "random_forest", or "ensemble" (default)
        Ensemble uses both models and averages the decision.

    Returns
    -------
    dict
        {
            "prediction": "Fraud" | "Legitimate",
            "fraud_probability": float (0 to 1),
            "is_fraud": bool,
            "model_used": str
        }
    """
    iso_forest, rf_model, scaler, feature_cols = _load_models()
    X = _prepare_transaction(transaction_data, feature_cols, scaler)

    result = {"prediction": "Legitimate", "fraud_probability": 0.0, "is_fraud": False, "model_used": model}

    if model == "isolation_forest":
        pred = iso_forest.predict(X)[0]
        scores = -iso_forest.score_samples(X)
        prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        prob = min(max(float(prob[0]), 0), 1)
        result["fraud_probability"] = prob
        result["is_fraud"] = pred == -1
        result["prediction"] = "Fraud" if result["is_fraud"] else "Legitimate"

    elif model == "random_forest":
        prob = float(rf_model.predict_proba(X)[0, 1])
        result["fraud_probability"] = prob
        result["is_fraud"] = rf_model.predict(X)[0] == 1
        result["prediction"] = "Fraud" if result["is_fraud"] else "Legitimate"

    else:  # ensemble
        # Isolation Forest
        iso_pred = iso_forest.predict(X)[0]
        iso_scores = -iso_forest.score_samples(X)
        iso_prob = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
        iso_prob = float(np.clip(iso_prob[0], 0, 1))

        # Random Forest
        rf_prob = float(rf_model.predict_proba(X)[0, 1])

        # Average probability
        prob = (iso_prob + rf_prob) / 2
        result["fraud_probability"] = prob
        result["is_fraud"] = prob >= 0.5
        result["prediction"] = "Fraud" if result["is_fraud"] else "Legitimate"
        result["model_used"] = "ensemble"

    return result


# --- Example Usage ---
if __name__ == "__main__":
    # Sample transaction (values from typical Kaggle credit card dataset)
    sample = {
        "Time": 0,
        "Amount": 100.0,
        "V1": -1.3598, "V2": -0.07278, "V3": 2.5363, "V4": 1.3782, "V5": -0.3383,
        "V6": 0.4624, "V7": 0.2396, "V8": 0.0987, "V9": 0.3640, "V10": -0.0183,
        "V11": 0.2778, "V12": -0.1105, "V13": 0.0670, "V14": 0.1285, "V15": -0.1891,
        "V16": 0.1336, "V17": -0.0211, "V18": 0.0403, "V19": 0.2550, "V20": -0.0700,
        "V21": -0.1260, "V22": -0.0450, "V23": 0.0700, "V24": 0.0500, "V25": -0.0590,
        "V26": -0.0650, "V27": 0.0900, "V28": -0.0500,
    }

    try:
        out = predict_transaction(sample)
        print("Example prediction:")
        print(f"  Prediction: {out['prediction']}")
        print(f"  Fraud Probability: {out['fraud_probability']:.4f}")
        print(f"  Model: {out['model_used']}")
    except FileNotFoundError as e:
        print(e)
        print("Run the notebook to train models first.")
