import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve, auc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fraud_detection import predict_transaction  # noqa: E402


st.set_page_config(page_title="Fraud Detection Pro", page_icon="🛡️", layout="wide")


def load_metrics():
    path = os.path.join(ROOT, "artifacts", "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_artifact_array(name):
    path = os.path.join(ROOT, "artifacts", name)
    if os.path.exists(path):
        return np.load(path)
    return None


st.title("🛡️ AI-Powered Banking Fraud Detection (Production Upgrade)")

with st.sidebar:
    st.header("Controls")
    model = st.selectbox("Model", ["ensemble", "random_forest", "isolation_forest", "xgboost"])
    threshold = st.slider("Fraud Threshold", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
    amount = st.number_input("Amount", min_value=0.0, value=100.0)
    time_val = st.number_input("Time", min_value=0.0, value=0.0)

    st.subheader("Features (V1-V28)")
    features = {}
    for i in range(1, 29):
        features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.01)

    run = st.button("Predict")

if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")
    if run:
        tx = {"Time": time_val, "Amount": amount, **features}
        try:
            result = predict_transaction(tx, model=model, threshold=threshold)
            color = "red" if result["is_fraud"] else "green"
            st.markdown(f"### :{color}[{result['prediction']}]")
            st.metric("Fraud Probability", f"{result['fraud_probability']*100:.2f}%")
            st.write("Explanation:", result["explanation"])
            st.write("Top feature contributions")
            st.dataframe(pd.DataFrame(result["top_features"]))

            st.session_state.history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": result["model_used"],
                    "probability": result["fraud_probability"],
                    "threshold": result["threshold"],
                    "prediction": result["prediction"],
                }
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with col2:
    st.subheader("Risk Gauge")
    if run and "result" in locals():
        prob = result["fraud_probability"] * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 30], "color": "#b7f7d4"},
                    {"range": [30, 70], "color": "#ffeaa7"},
                    {"range": [70, 100], "color": "#fab1a0"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": threshold * 100},
            },
            title={"text": "Fraud Risk %"},
        ))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

metrics = load_metrics()
if metrics:
    st.subheader("Model Comparison")
    rows = []
    for k in ["isolation_forest", "random_forest_calibrated", "xgboost", "ensemble"]:
        if k in metrics:
            row = {"model": k}
            row.update({m: metrics[k].get(m, None) for m in ["precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy"]})
            rows.append(row)
    cmp_df = pd.DataFrame(rows)
    st.dataframe(cmp_df, use_container_width=True)
    if not cmp_df.empty:
        fig_cmp = px.bar(cmp_df, x="model", y=["f1", "pr_auc", "roc_auc"], barmode="group", title="Model Metrics")
        st.plotly_chart(fig_cmp, use_container_width=True)

    y_test = load_artifact_array("y_test.npy")
    ens_prob = load_artifact_array("ensemble_prob.npy")
    if y_test is not None and ens_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, ens_prob)
        roc_auc = auc(fpr, tpr)
        p, r, _ = precision_recall_curve(y_test, ens_prob)

        c1, c2 = st.columns(2)
        with c1:
            fig_roc = px.line(x=fpr, y=tpr, title=f"ROC Curve (AUC={roc_auc:.3f})")
            fig_roc.update_xaxes(title="FPR")
            fig_roc.update_yaxes(title="TPR")
            st.plotly_chart(fig_roc, use_container_width=True)
        with c2:
            fig_pr = px.line(x=r, y=p, title="Precision-Recall Curve")
            fig_pr.update_xaxes(title="Recall")
            fig_pr.update_yaxes(title="Precision")
            st.plotly_chart(fig_pr, use_container_width=True)

st.subheader("Transaction History")
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history).iloc[::-1]
    st.dataframe(hist_df, use_container_width=True)
else:
    st.info("No predictions yet.")
