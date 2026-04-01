from typing import Dict, List

import numpy as np
import pandas as pd


def permutation_explanation(model, x_row: pd.DataFrame, top_k: int = 5) -> List[Dict[str, float]]:
    """Simple local permutation-based contribution estimate."""
    base = float(model.predict_proba(x_row)[0, 1]) if hasattr(model, "predict_proba") else 0.5
    out = []
    for col in x_row.columns:
        perturbed = x_row.copy()
        perturbed[col] = 0.0
        p = float(model.predict_proba(perturbed)[0, 1]) if hasattr(model, "predict_proba") else base
        out.append({"feature": col, "impact": base - p})
    out = sorted(out, key=lambda d: abs(d["impact"]), reverse=True)[:top_k]
    return out


def shap_explanation_if_available(model, x_row: pd.DataFrame, top_k: int = 5):
    try:
        import shap
    except ImportError:
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_row)
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        feats = []
        for f, v in zip(x_row.columns, values):
            feats.append({"feature": f, "impact": float(v)})
        feats = sorted(feats, key=lambda d: abs(d["impact"]), reverse=True)[:top_k]
        return feats
    except Exception:
        return None


def explanation_text(contribs: List[Dict[str, float]]) -> str:
    if not contribs:
        return "No feature-level explanation available."
    top = contribs[:3]
    phrases = []
    for c in top:
        direction = "increased" if c["impact"] >= 0 else "reduced"
        phrases.append(f"{c['feature']} {direction} fraud risk")
    return "; ".join(phrases) + "."
