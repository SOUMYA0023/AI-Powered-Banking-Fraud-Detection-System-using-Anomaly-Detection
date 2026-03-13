"""
AI-Powered Banking Fraud Detection - Interactive Streamlit Dashboard
====================================================================
"""

import sys
import os

# Add project root for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Custom fraud detection engine
from fraud_detection import predict_transaction

# Page config
st.set_page_config(
    page_title="Banking Fraud Detection | AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        padding: 2rem;
    }
    
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #e0e0e0;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #00d4aa;
        margin-bottom: 2rem;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #16213e 0%, #0f0f23 100%);
    }
    
    .sidebar .sidebar-content {
        background: transparent;
    }
    
    div[data-testid="stSidebar"] label {
        color: #b0b0b0 !important;
    }
    
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .fraud-result {
        background: linear-gradient(135deg, #ff4757 0%, #c0392b 100%);
        color: white;
        border: 2px solid #ff6b7a;
    }
    
    .legit-result {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: white;
        border: 2px solid #55efc4;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00d4aa, #00b894) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #00b894, #00d4aa) !important;
        box-shadow: 0 4px 15px rgba(0,212,170,0.4);
    }
</style>
""", unsafe_allow_html=True)


def load_sample_dataset():
    """Load a sample of the dataset for visualization (if exists)."""
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "creditcard.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path, nrows=5000)
        return df
    return None


def main():
    st.title("🛡️ AI-Powered Banking Fraud Detection System")
    st.markdown("*Detect suspicious transactions in real-time using Machine Learning*")
    st.markdown("---")

    # Sidebar inputs
    with st.sidebar:
        st.header("⚙️ Transaction Input")
        st.markdown("Enter transaction details or use sample values.")

        amount = st.number_input("Amount ($)", min_value=0.01, max_value=50000.0, value=100.0, step=1.0)
        time_val = st.number_input("Time (seconds)", min_value=0, max_value=172800, value=0, step=100)

        st.subheader("Feature Values (V1–V28)")
        st.caption("Anonymized PCA components. Use defaults for demo.")

        v_values = {}
        for i in range(1, 29):
            v_values[f"V{i}"] = st.number_input(
                f"V{i}", value=0.0, step=0.01,
                key=f"v{i}",
                help=f"PCA component V{i}"
            )

        model_choice = st.selectbox(
            "Model",
            ["ensemble", "isolation_forest", "random_forest"],
            format_func=lambda x: {"ensemble": "Ensemble (Best)", "isolation_forest": "Isolation Forest", "random_forest": "Random Forest"}[x]
        )

        predict_btn = st.button("🔍 Predict Fraud", use_container_width=True)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Prediction Result")
        if predict_btn:
            transaction = {
                "Time": time_val,
                "Amount": amount,
                **{f"V{i}": v_values[f"V{i}"] for i in range(1, 29)}
            }
            try:
                result = predict_transaction(transaction, model=model_choice)
                prob = result["fraud_probability"]
                is_fraud = result["is_fraud"]

                css_class = "fraud-result" if is_fraud else "legit-result"
                label = "🚨 FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE TRANSACTION"
                st.markdown(f'<div class="result-box {css_class}">{label}</div>', unsafe_allow_html=True)
                st.metric("Fraud Probability", f"{prob*100:.2f}%")
                st.caption(f"Model used: {result['model_used']}")
            except FileNotFoundError as e:
                st.error("Models not found. Please run the Jupyter notebook first to train models.")
                st.code("jupyter notebook notebooks/fraud_analysis.ipynb")
            except Exception as e:
                st.error(f"Error: {str(e)}")

        st.subheader("How it works")
        st.markdown("""
        - **Isolation Forest**: Identifies anomalies by isolating observations.
        - **Random Forest**: Supervised classifier trained on fraud labels.
        - **Ensemble**: Combines both for robust predictions.
        """)

    with col2:
        st.subheader("Fraud Probability Gauge")
        if predict_btn:
            try:
                result = predict_transaction({
                    "Time": time_val, "Amount": amount,
                    **{f"V{i}": v_values[f"V{i}"] for i in range(1, 29)}
                }, model=model_choice)
                prob = result["fraud_probability"]

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Fraud Risk %"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#00d4aa"},
                        "steps": [
                            {"range": [0, 30], "color": "rgba(0,212,170,0.3)"},
                            {"range": [30, 70], "color": "rgba(255,193,7,0.3)"},
                            {"range": [70, 100], "color": "rgba(255,71,87,0.4)"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "value": 50,
                        },
                    },
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e0e0e0"},
                    height=280,
                )
                st.plotly_chart(fig, use_container_width=True)
            except FileNotFoundError:
                st.info("Train models first to see the gauge.")

    st.markdown("---")
    st.subheader("Transaction Pattern Analytics")

    df = load_sample_dataset()
    if df is not None:
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(
                df, x="Amount", color="Class",
                color_discrete_map={0: "#00d4aa", 1: "#ff4757"},
                title="Amount Distribution by Class",
                nbins=40,
            )
            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e0e0e0"},
                legend={"title": "Class", "bgcolor": "rgba(0,0,0,0)"},
            )
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fraud_counts = df["Class"].value_counts()
            fig2 = px.pie(
                values=fraud_counts.values,
                names=["Legitimate", "Fraud"],
                color_discrete_sequence=["#00d4aa", "#ff4757"],
                title="Class Distribution",
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e0e0e0"},
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Place `creditcard.csv` in the dataset/ folder to view transaction analytics.")

    st.markdown("---")
    st.caption("AI-Powered Banking Fraud Detection • B.Tech CS Portfolio Project")


if __name__ == "__main__":
    main()
