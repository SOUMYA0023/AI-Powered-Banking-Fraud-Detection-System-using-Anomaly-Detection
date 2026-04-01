import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    required = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df


def add_scaled_amount(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    out = df.copy()
    scaler = StandardScaler()
    out["Amount_scaled"] = scaler.fit_transform(out[["Amount"]])
    return out, scaler


def get_feature_columns() -> List[str]:
    return [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time"]


def prepare_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }


def maybe_apply_smote(X_train, y_train, random_state: int = 42, sampling_strategy: float = 0.1):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return X_train, y_train, "smote_unavailable"

    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res, "smote_applied"


def prepare_single_transaction(transaction: dict, feature_cols: List[str], scaler: StandardScaler) -> pd.DataFrame:
    if transaction is None:
        raise ValueError("transaction payload cannot be None")

    row = dict(transaction)
    if "Amount_scaled" not in row:
        if "Amount" not in row:
            raise ValueError("Missing Amount for scaling")
        amt_df = pd.DataFrame([[float(row["Amount"])]], columns=["Amount"])
        row["Amount_scaled"] = float(scaler.transform(amt_df)[0][0])

    missing = [c for c in feature_cols if c not in row]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    clean = {c: float(row[c]) for c in feature_cols}
    return pd.DataFrame([clean], columns=feature_cols)
