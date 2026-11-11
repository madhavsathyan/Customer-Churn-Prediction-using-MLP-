"""
preprocessing.py
~~~~~~~~~~~~~~~~
Shared preprocessing pipeline for training (train.py), the Jupyter notebook,
and the Streamlit app (app.py).

Single source of truth: preprocess_input() converts raw Telco-format feature
values into the scaled numpy array expected by the MLP.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Raw categorical feature options (used by the app UI)
# ---------------------------------------------------------------------------
YES_NO          = ["No", "Yes"]
GENDER_OPTIONS  = ["Female", "Male"]
MULTI_LINES     = ["No", "Yes", "No phone service"]
INTERNET_SVC    = ["DSL", "Fiber optic", "No"]
ADDON_OPTIONS   = ["No", "Yes", "No internet service"]
CONTRACT_OPTIONS= ["Month-to-month", "One year", "Two year"]
PAYMENT_OPTIONS = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]

# ---------------------------------------------------------------------------
# Exact ordered feature list after one-hot encoding (must match training)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to a raw Telco CSV dataframe.
    Works for both single-row inference and batch CSV uploads.

    Steps:
      1. Drop customerID if present.
      2. Fix TotalCharges blank → 0.0.
      3. Binary-map gender, Partner, Dependents, PhoneService, PaperlessBilling.
      4. One-hot encode multi-level categoricals (drop_first=True, dtype=int).
      5. Reindex to FEATURE_COLUMNS (fills zeros for any missing OHE columns).

    Returns a DataFrame with columns matching FEATURE_COLUMNS.
    """
    df = df.copy()

    # Drop ID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Drop target if present (batch upload may include it)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Binary encodings
    binary_map = {
        "gender":          {"Female": 0, "Male": 1},
        "Partner":         {"No": 0, "Yes": 1},
        "Dependents":      {"No": 0, "Yes": 1},
        "PhoneService":    {"No": 0, "Yes": 1},
        "PaperlessBilling":{"No": 0, "Yes": 1},
        "SeniorCitizen":   {0: 0, 1: 1, "0": 0, "1": 1},
    }
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # One-hot encode multi-level categoricals
    ohe_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    # Only encode columns that exist in df
    ohe_present = [c for c in ohe_cols if c in df.columns]
    if ohe_present:
        df = pd.get_dummies(df, columns=ohe_present, drop_first=True, dtype=int)

    # Align to exact expected column order, fill missing OHE columns with 0
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df


def preprocess_input(raw_dict: dict, scaler) -> np.ndarray:
    """
    Convert a single customer's raw feature dictionary to a scaled numpy array.

    Parameters
    ----------
    raw_dict : dict
        Keys are raw Telco column names (e.g. "gender": "Female",
        "Contract": "Month-to-month", "tenure": 12, ...).
    scaler : sklearn StandardScaler
        Fitted scaler (loaded from artifacts/scaler.pkl).

    Returns
    -------
    np.ndarray of shape (1, 30), ready to pass to model.predict().
    """
    df_single = pd.DataFrame([raw_dict])
    df_cleaned = clean_dataframe(df_single)
    # Pass as DataFrame with feature names to silence sklearn warning
    X_scaled = scaler.transform(df_cleaned)
    return X_scaled


def preprocess_batch(df_raw: pd.DataFrame, scaler) -> np.ndarray:
    """
    Convert a batch CSV DataFrame to a scaled numpy array.

    Parameters
    ----------
    df_raw : pd.DataFrame  — raw Telco-format data (customerID/Churn optional).
    scaler : sklearn StandardScaler

    Returns
    -------
    np.ndarray of shape (N, 30)
    """
    df_cleaned = clean_dataframe(df_raw)
    X_scaled = scaler.transform(df_cleaned)
    return X_scaled


def get_risk_label(prob: float, low_thresh: float = 0.30, high_thresh: float = 0.50) -> str:
    """Return 'High', 'Medium', or 'Low' churn risk label."""
    if prob >= high_thresh:
        return "High"
    elif prob >= low_thresh:
        return "Medium"
    return "Low"


def get_retention_suggestions(raw_dict: dict) -> list[str]:
    """
    Return 3–5 plain-language retention suggestions based on input features.
    """
    suggestions = []
    contract    = raw_dict.get("Contract", "Month-to-month")
    internet    = raw_dict.get("InternetService", "No")
    payment     = raw_dict.get("PaymentMethod", "")
    tenure      = int(raw_dict.get("tenure", 0))
    security    = raw_dict.get("OnlineSecurity", "No")
    tech        = raw_dict.get("TechSupport", "No")
    monthly     = float(raw_dict.get("MonthlyCharges", 0))
    streaming_tv  = raw_dict.get("StreamingTV", "No")
    streaming_mv  = raw_dict.get("StreamingMovies", "No")

    if contract == "Month-to-month":
        suggestions.append(
            "📋 **Switch to Annual Plan:** Offer a 10–15% discount on a 1-year contract. "
            "Month-to-month subscribers are 3× more likely to churn than annual subscribers."
        )
    if tenure <= 12:
        suggestions.append(
            "🤝 **Dedicated Onboarding:** Assign a customer success manager for the first 90 days. "
            "Proactive check-ins in months 1–6 significantly reduce early-lifecycle churn."
        )
    if internet == "Fiber optic" and monthly > 70:
        suggestions.append(
            "⚡ **Fiber Service Review:** Conduct a satisfaction survey on internet speed and reliability. "
            "Consider offering a temporary speed upgrade or a bill credit to retain high-value fiber subscribers."
        )
    if "Electronic check" in payment:
        suggestions.append(
            "💳 **Auto-Pay Enrollment:** Offer a \\$5/month discount for switching to automatic bank debit "
            "or credit card billing. Manual payment creates monthly friction that increases defection intent."
        )
    if security == "No" and internet != "No":
        suggestions.append(
            "🔒 **Free Security Trial:** Offer a 3-month free trial of Online Security. "
            "Add-on subscribers have significantly lower churn rates."
        )
    if tech == "No" and internet != "No":
        suggestions.append(
            "🛠️ **Tech Support Upgrade:** Add Tech Support at a reduced bundle rate. "
            "Subscribers with active support contracts report higher satisfaction and stay longer."
        )
    if streaming_tv == "No" and streaming_mv == "No" and internet != "No":
        suggestions.append(
            "🎬 **Streaming Bundle Offer:** Bundle StreamingTV + StreamingMovies at a discounted rate. "
            "Entertainment add-ons increase perceived value and switching costs."
        )

    # Always cap at 4 most relevant suggestions
    return suggestions[:4] if suggestions else [
        "📞 **Proactive Outreach:** Schedule a customer satisfaction call to identify unmet needs.",
        "🎁 **Loyalty Reward:** Offer a loyalty bonus (e.g. free month or speed upgrade) for continued subscription.",
    ]
