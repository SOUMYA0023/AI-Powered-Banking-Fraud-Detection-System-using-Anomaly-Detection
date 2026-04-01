# AI-Powered Banking Fraud Detection System - Project Master Guide

## 1) Project Overview

### Problem Statement
Financial institutions process millions of transactions daily, and only a tiny fraction are fraudulent. Traditional rule-based systems miss novel fraud patterns and generate high false positives. This project builds an AI-driven fraud detection system that identifies suspicious transactions using anomaly detection + supervised learning.

### Why Fraud Detection Matters in Banking
- Direct financial loss from fraudulent transactions
- Customer trust and retention impact
- Regulatory and compliance pressure
- Operational cost due to manual review workload
- Need for low-latency, high-precision risk scoring in production

### System Objective
Design a robust ML system that:
- Detects fraud with high recall and usable precision
- Produces calibrated, interpretable fraud probabilities
- Supports both UI-based and API-based inference
- Is modular, testable, and deployment-ready

---

## 2) System Architecture

### End-to-End Flow
Data -> Preprocessing -> Model Training -> Inference Engine -> Streamlit UI / FastAPI

### Component Roles
- **Isolation Forest**: unsupervised anomaly detector; flags unusual behavior even without labels.
- **Random Forest (Calibrated)**: supervised model that learns fraud/non-fraud boundary from labeled data.
- **Ensemble Layer**: weighted combination of model signals to improve robustness and reduce single-model bias.

### High-Level Runtime Path
1. Transaction input is validated.
2. `Amount` is scaled using trained scaler.
3. Features are aligned to training schema.
4. Selected model (or ensemble) predicts fraud probability.
5. Threshold logic converts probability into `Fraud` / `Legitimate`.
6. Explainability layer returns top contributing features.
7. Output served to UI/API and logged.

---

## 3) Dataset Explanation

### Source
Kaggle: Credit Card Fraud Detection dataset  
`https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`

### Feature Schema
- `Time`: seconds elapsed between this transaction and first transaction
- `Amount`: transaction amount
- `V1` to `V28`: PCA-transformed anonymized features
- `Class`: target label (`0` legit, `1` fraud)

### Why the Dataset Is Imbalanced
Fraud is rare by nature; legitimate behavior dominates transaction flow. In this dataset, fraud class is a tiny minority.

### Core Challenges in Fraud Datasets
- Severe class imbalance
- Concept drift (fraud patterns evolve)
- High false positive risk
- Need for precision-recall optimization (not just accuracy)
- Explainability requirement in regulated environments

---

## 4) Data Preprocessing

### Missing Values
Pipeline checks schema and nulls; raises explicit errors if required columns are missing.

### Feature Scaling
`Amount` is scaled using `StandardScaler` and stored as `Amount_scaled`.

### Why Scaling Matters
- Stabilizes distance/partition behavior
- Improves consistency between training and inference
- Prevents amount magnitude from dominating splits/anomaly scores

### Train-Test Strategy
- Stratified split keeps fraud ratio consistent across train/test
- Reproducible random seed
- Optional SMOTE applied to training data only

---

## 5) Exploratory Data Analysis (EDA)

### A) Fraud vs Non-Fraud Distribution
- **Shows**: class counts and imbalance severity
- **Why important**: explains why naive accuracy is misleading
- **Insight**: strong imbalance requires threshold tuning and imbalance handling

### B) Amount Distribution
- **Shows**: transaction amount spread (often right-skewed)
- **Why important**: identifies tails/outliers and fraud amount behavior
- **Insight**: fraud patterns can exist in unusual amount bands but not exclusively high amounts

### C) Correlation Heatmap
- **Shows**: relationships among selected features and target
- **Why important**: detects informative components and redundancy
- **Insight**: some PCA features correlate strongly with fraud risk signal

### D) Fraud Over Time
- **Shows**: temporal concentration of fraudulent events
- **Why important**: useful for operational alerting windows
- **Insight**: fraud may cluster over certain time ranges, enabling adaptive monitoring

---

## 6) Machine Learning Models

### Isolation Forest
**What**: tree-based anomaly detector isolating rare points quickly.  
**Why used**: catches anomalous behavior without relying fully on labels.  
**Strengths**:
- Works in unsupervised mode
- Good for unknown/novel fraud patterns
  
**Limitations**:
- Raw scores are not directly calibrated probabilities
- Can over-flag if threshold is naive

### Random Forest
**What**: ensemble of decision trees for supervised classification.  
**Why used here**: strong baseline for tabular fraud data.  
**Strengths**:
- Handles nonlinear interactions
- Robust to noise
- Feature importance support

**Limitations**:
- Uncalibrated probabilities by default
- May under-detect minority class without balancing

### Ensemble Model
**Why combine**:
- Isolation Forest contributes anomaly sensitivity
- Random Forest contributes learned class discrimination

**How final score is made**:
`ensemble_prob = w_iso * iso_prob + w_rf * rf_prob (+ w_xgb * xgb_prob if available)`

Final label:
`Fraud if ensemble_prob >= threshold else Legitimate`

---

## 7) Model Evaluation

### Metrics Used
- **Accuracy**: overall correctness
- **Precision**: out of predicted fraud, how many are true fraud
- **Recall**: out of real fraud, how many were caught
- **F1**: harmonic mean of precision and recall
- **ROC-AUC**: ranking quality across thresholds
- **PR-AUC**: precision-recall area, critical for imbalanced fraud tasks
- **Balanced Accuracy**: average recall across classes

### Why Accuracy Is Misleading
In highly imbalanced data, a model can predict almost all transactions as legit and still get very high accuracy. Fraud systems must prioritize recall/precision trade-off, not raw accuracy.

---

## 8) Inference System

### `predict_transaction()` Step-by-Step
1. Load model artifacts (models, scaler, feature schema, threshold, weights).
2. Validate input fields.
3. Scale `Amount` and build ordered feature frame.
4. Compute per-model probabilities.
5. Apply selected model or ensemble logic.
6. Apply threshold to classify.
7. Generate explanation (SHAP or permutation fallback).
8. Return structured response.

### Example Input
```json
{
  "Time": 1000,
  "Amount": 250.5,
  "V1": 0.1,
  "V2": -0.2,
  "...": "...",
  "V28": 0.0
}
```

### Example Output
```json
{
  "prediction": "Fraud",
  "is_fraud": true,
  "fraud_probability": 0.83,
  "threshold": 0.72,
  "model_used": "ensemble",
  "top_features": [
    {"feature": "V14", "impact": 0.21},
    {"feature": "Amount_scaled", "impact": 0.09}
  ],
  "explanation": "V14 increased fraud risk; Amount_scaled increased fraud risk."
}
```

---

## 9) Frontend (Streamlit Dashboard)

### UI Flow
1. User selects model and threshold in sidebar.
2. User enters transaction fields.
3. User clicks predict.
4. UI renders prediction, probability gauge, explanation, and history.

### User Interaction Loop
Input -> Predict -> Inspect result -> Adjust threshold/model -> Re-run

### Main UI Sections
- Sidebar: model selection + threshold + features
- Prediction panel: label/probability/explanation
- Gauge panel: real-time risk view
- Model comparison panel: metric table and bar chart
- ROC/PR panel: evaluation curve views
- History panel: inference audit trail for the session

---

## 10) API Layer

### Endpoints
- `GET /health` -> service heartbeat
- `POST /predict` -> fraud scoring endpoint

### Request Format (`/predict`)
- `Time`
- `Amount`
- `features` object containing `V1`..`V28`
- optional `model`, optional `threshold`

### Response Format
- prediction label
- probability
- threshold used
- per-model probabilities
- feature-level explanation

### Real-World Usage
- Core scoring microservice in transaction pipeline
- Can be called by payment gateway/risk orchestration service
- Supports integration with monitoring, alerting, and case management

---

## 11) Explainable AI

### Method
- Preferred: SHAP (when available)
- Fallback: local permutation contribution approximation

### What User Gets
- Top contributing features per prediction
- Human-readable explanation sentence

### Why Important
- Analyst trust and triage speed
- Regulatory explainability expectations
- Easier threshold/policy tuning

---

## 12) Edge Cases and Error Handling

1. **Missing dataset**  
   - Loader raises explicit `FileNotFoundError`.
2. **Missing model artifacts**  
   - Inference engine returns clear initialization failure.
3. **Invalid input payload**  
   - Required fields validated; friendly error returned in UI/API.
4. **Missing features (V1..V28)**  
   - Validation error lists missing fields.
5. **Dependency issues (e.g., SHAP/XGBoost/OpenMP)**  
   - Graceful fallback (permutation explainability, no-xgboost path).
6. **Threshold outside bounds**  
   - UI/API constrain to `[0,1]`.
7. **Corrupt model file**  
   - Handled with model loading try/except and logs.

---

## 13) Limitations

- If using synthetic data instead of Kaggle full data, metrics are less meaningful.
- Isolation Forest score-to-probability mapping is heuristic.
- Probability calibration improves interpretability but is still data-distribution dependent.
- Fixed threshold may degrade under drift; periodic recalibration required.

---

## 14) Future Improvements

- Real-time streaming detection (Kafka/Flink)
- Feature store + online/offline consistency checks
- Drift detection + scheduled retraining pipeline
- Better ensemble/meta-learner stacking
- Cost-sensitive optimization (business-aware thresholding)
- Secure model serving with auth, rate limits, and model versioning

---

## 15) Viva Questions and Answers (25+)

1. **Why is fraud detection difficult?**  
   Fraud is rare, patterns evolve quickly, and false positives are costly.

2. **Why not rely on accuracy?**  
   Accuracy can be high even when fraud recall is poor in imbalanced data.

3. **What is anomaly detection?**  
   Detecting unusual points that deviate from normal behavior patterns.

4. **Why use Isolation Forest here?**  
   It catches suspicious outliers without depending entirely on labels.

5. **Why also use Random Forest?**  
   It learns labeled fraud patterns and improves class-discriminative power.

6. **Why combine models in an ensemble?**  
   To reduce single-model weaknesses and improve robustness.

7. **What is class imbalance?**  
   One class (fraud) has far fewer samples than the other.

8. **How does SMOTE help?**  
   It synthesizes minority examples to improve model exposure to fraud patterns.

9. **Why calibrate probabilities?**  
   Raw model scores may be overconfident; calibration improves probability reliability.

10. **What does precision mean in this project?**  
    Share of flagged transactions that are truly fraud.

11. **What does recall mean in this project?**  
    Share of actual fraud transactions detected by the model.

12. **When would you prefer higher recall?**  
    When missing fraud is more costly than reviewing extra alerts.

13. **What is PR-AUC and why important?**  
    Area under precision-recall curve; highly informative for imbalanced tasks.

14. **What is ROC-AUC?**  
    Threshold-independent separability measure between classes.

15. **How is threshold selected?**  
    From PR curve, choosing the value that maximizes F1 (or target recall policy).

16. **What if dataset becomes balanced?**  
    Accuracy becomes more meaningful; PR-AUC still useful but less critical.

17. **How does `predict_transaction()` ensure consistency?**  
    Uses saved scaler and exact feature order from training artifacts.

18. **How do you explain model decisions?**  
    SHAP values (or permutation fallback) identify top feature contributions.

19. **What is concept drift?**  
    Data distribution changes over time, reducing model performance.

20. **How to handle drift in production?**  
    Monitor drift metrics and retrain/recalibrate periodically.

21. **Why expose API and UI both?**  
    API for system integration, UI for analysts/demos and manual review.

22. **What happens if XGBoost is unavailable?**  
    System continues with RF + Isolation Forest; ensemble reweights accordingly.

23. **Why keep logs for predictions?**  
    For auditability, debugging, and monitoring in production.

24. **How do you test reliability?**  
    Unit tests for preprocessing and inference, including edge-case validation.

25. **What are next production steps for fintech-grade deployment?**  
    Auth, observability, model registry, CI/CD, canary rollout, and governance controls.

26. **Why not only use deep learning?**  
    Tree ensembles are strong for tabular data, easier to explain, and faster to operationalize.

27. **What business trade-off does threshold control?**  
    Fraud catch rate vs false-alert volume (risk vs operations cost).

---

## 16) Key Talking Points

### 60-Second Explanation
This project is a production-oriented fraud detection system for banking transactions. It uses a calibrated Random Forest plus Isolation Forest anomaly detection, combines them in a weighted ensemble, and applies threshold tuning using precision-recall analysis. It provides interpretable outputs with feature-level explanations, supports both a Streamlit dashboard and FastAPI endpoint, and includes logging, tests, and Docker deployment readiness.

### 2-Minute Explanation
The system starts with the Kaggle credit-card fraud dataset, validates schema, scales amount, and handles imbalance with controlled SMOTE and class-weighted learning. We train Isolation Forest to capture outliers and calibrated Random Forest for supervised fraud classification. Their outputs are combined via weighted ensemble, and we select an operational threshold from the PR curve to optimize fraud detection trade-offs.  
For inference, `predict_transaction()` validates payloads, applies the trained scaler and feature ordering, computes per-model probabilities, applies thresholding, and returns both label and calibrated probability. We add explainability using SHAP (or permutation fallback), making each prediction interpretable.  
Operationally, we expose the model through FastAPI (`/predict`, `/health`) and a Streamlit dashboard with model comparison, threshold control, ROC/PR visuals, and transaction history. Tests, logs, and Docker support make the project production-ready and interview-strong.

### Recruiter-Friendly Pitch
Built a fintech-style fraud detection platform with calibrated ML, explainable predictions, API + dashboard interfaces, threshold-optimized risk scoring, and production engineering practices (testing, logging, modular architecture, Docker).

