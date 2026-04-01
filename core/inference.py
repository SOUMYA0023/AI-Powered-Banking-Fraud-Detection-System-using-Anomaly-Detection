import os
from typing import Dict, Any

import joblib
import numpy as np

from core.preprocessing import prepare_single_transaction
from core.explainability import permutation_explanation, shap_explanation_if_available, explanation_text
from core.logger import get_logger


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")

logger = get_logger("inference")


class InferenceEngine:
    def __init__(self, models_dir: str = MODELS_DIR, artifacts_dir: str = ARTIFACTS_DIR):
        self.models_dir = models_dir
        self.artifacts_dir = artifacts_dir
        self._load()

    def _load(self):
        try:
            self.iso = joblib.load(os.path.join(self.models_dir, "isolation_forest.pkl"))
            self.rf = joblib.load(os.path.join(self.models_dir, "random_forest.pkl"))
            self.scaler = joblib.load(os.path.join(self.models_dir, "amount_scaler.pkl"))
            self.feature_cols = joblib.load(os.path.join(self.models_dir, "feature_columns.pkl"))

            self.threshold = 0.5
            threshold_path = os.path.join(self.artifacts_dir, "optimal_threshold.txt")
            if os.path.exists(threshold_path):
                self.threshold = float(open(threshold_path, "r", encoding="utf-8").read().strip())

            self.weights = {"iso": 0.3, "rf": 0.7, "xgb": 0.0}
            weights_path = os.path.join(self.artifacts_dir, "ensemble_weights.pkl")
            if os.path.exists(weights_path):
                self.weights = joblib.load(weights_path)

            self.xgb = None
            xgb_path = os.path.join(self.models_dir, "xgboost.pkl")
            if os.path.exists(xgb_path):
                self.xgb = joblib.load(xgb_path)

            logger.info("Models loaded successfully")
        except Exception as exc:
            logger.exception("Model loading failed")
            raise RuntimeError(f"Model loading failed: {exc}") from exc

    def _iso_probability(self, x_row):
        score = -self.iso.score_samples(x_row)
        # map score to [0,1] with logistic transform
        p = 1.0 / (1.0 + np.exp(-score))
        return float(np.clip(p[0], 0.0, 1.0))

    def predict(self, transaction: Dict[str, Any], model: str = "ensemble", threshold: float = None) -> Dict[str, Any]:
        thr = self.threshold if threshold is None else float(threshold)
        x_row = prepare_single_transaction(transaction, self.feature_cols, self.scaler)

        rf_prob = float(self.rf.predict_proba(x_row)[0, 1])
        iso_prob = self._iso_probability(x_row)
        xgb_prob = None
        if self.xgb is not None:
            xgb_prob = float(self.xgb.predict_proba(x_row)[0, 1])

        if model == "random_forest":
            prob = rf_prob
            model_used = "random_forest"
        elif model == "isolation_forest":
            prob = iso_prob
            model_used = "isolation_forest"
        elif model == "xgboost":
            if xgb_prob is None:
                raise ValueError("xgboost model not available. Train with XGBoost enabled.")
            prob = xgb_prob
            model_used = "xgboost"
        else:
            # weighted ensemble
            prob = self.weights.get("rf", 0.7) * rf_prob + self.weights.get("iso", 0.3) * iso_prob
            if xgb_prob is not None:
                prob += self.weights.get("xgb", 0.0) * xgb_prob
            model_used = "ensemble"

        is_fraud = prob >= thr
        label = "Fraud" if is_fraud else "Legitimate"

        contribs = shap_explanation_if_available(self.rf, x_row, top_k=5)
        if contribs is None:
            contribs = permutation_explanation(self.rf, x_row, top_k=5)

        result = {
            "prediction": label,
            "is_fraud": bool(is_fraud),
            "fraud_probability": float(prob),
            "threshold": float(thr),
            "model_used": model_used,
            "model_probs": {
                "isolation_forest": iso_prob,
                "random_forest": rf_prob,
                "xgboost": xgb_prob,
            },
            "top_features": contribs,
            "explanation": explanation_text(contribs),
        }
        logger.info("Prediction | model=%s prob=%.4f thr=%.4f label=%s", model_used, prob, thr, label)
        return result
