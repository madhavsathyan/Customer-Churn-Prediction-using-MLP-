"""
app.py
~~~~~~
Streamlit multi-tab dashboard for the Telco Customer Churn MLP.

Tabs
----
  1. Predict Churn       — Single-customer scoring with gauge + suggestions
  2. Model Performance   — Metrics, confusion matrix, ROC/PR curves, threshold slider
  3. Training Insights   — Loss & accuracy curves, overfitting guide
  4. Data Explorer       — EDA charts, feature importance
  5. Batch Prediction    — CSV upload → scored results → download
  6. About / Concepts    — Architecture, math, glossary

Run:
    streamlit run app.py
"""

import os
import io
import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS_DIR = "artifacts"
DATA_PATH     = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ---------------------------------------------------------------------------
# Color palette (consistent across all charts)
# ---------------------------------------------------------------------------
CLR_PRIMARY   = "#2b5c8f"
CLR_SECONDARY = "#d95f02"
CLR_TERTIARY  = "#1a9e77"
CLR_DANGER    = "#e31a1c"
CLR_WARN      = "#f4a261"
CLR_OK        = "#2ca02c"
PLOTLY_TEMPLATE = "plotly_white"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _artifacts_ok() -> bool:
    required = [
        "churn_mlp.keras", "scaler.pkl", "feature_columns.json",
        "metrics.json", "history.csv", "test_predictions.csv",
        "baseline_metrics.json", "feature_importance.csv",
    ]
    return all(os.path.exists(os.path.join(ARTIFACTS_DIR, f)) for f in required)


def _missing_artifacts_msg():
    st.error(
        "⚠️  **Artifact files not found.**\n\n"
        "Please run the training script first:\n"
        "```\n"
        "python train.py\n"
        "```\n"
        "This will generate all required model and data files in the `/artifacts` folder."
    )


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, "churn_mlp.keras"))


@st.cache_resource(show_spinner="Loading scaler…")
def load_scaler():
    import joblib
    return joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))


@st.cache_data(show_spinner=False)
def load_metrics():
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_baseline_metrics():
    with open(os.path.join(ARTIFACTS_DIR, "baseline_metrics.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_history():
    return pd.read_csv(os.path.join(ARTIFACTS_DIR, "history.csv"))


@st.cache_data(show_spinner=False)
def load_test_predictions():
    return pd.read_csv(os.path.join(ARTIFACTS_DIR, "test_predictions.csv"))


@st.cache_data(show_spinner=False)
def load_feature_importance():
    return pd.read_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"))


@st.cache_data(show_spinner=False)
def load_raw_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        return df
    return None


# ---------------------------------------------------------------------------
# Common plot helpers
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: str = "", color: str = CLR_PRIMARY):
    st.markdown(
        f"""
        <div style="background:{color}10;border-left:4px solid {color};
                    padding:12px 16px;border-radius:6px;margin-bottom:8px;">
          <div style="font-size:0.78rem;color:#555;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.05em;">{label}</div>
          <div style="font-size:1.6rem;font-weight:700;color:{color};">{value}</div>
          {'<div style="font-size:0.78rem;color:#888;">'+delta+'</div>' if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )
    preds = (np.array(y_prob) >= threshold).astype(int)
    acc   = accuracy_score(y_true, preds)
    prec  = precision_score(y_true, preds, zero_division=0)
    rec   = recall_score(y_true, preds, zero_division=0)
    f1    = f1_score(y_true, preds, zero_division=0)
    cm    = confusion_matrix(y_true, preds)
    return acc, prec, rec, f1, cm


def churn_gauge(probability: float) -> go.Figure:
    """Plotly gauge chart for churn probability."""
    if probability >= 0.50:
        bar_color = CLR_DANGER
        label_text = "High Risk"
    elif probability >= 0.30:
        bar_color = CLR_WARN
        label_text = "Medium Risk"
    else:
        bar_color = CLR_OK
        label_text = "Low Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": bar_color}},
        delta={"reference": 50, "increasing": {"color": CLR_DANGER},
               "decreasing": {"color": CLR_OK}},
        title={"text": f"Churn Probability<br><span style='font-size:1.1em;color:{bar_color}'>{label_text}</span>",
               "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#555", "ticksuffix": "%"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#ddd",
            "steps": [
                {"range": [0,  30], "color": "#d4edda"},
                {"range": [30, 50], "color": "#fff3cd"},
                {"range": [50, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"},
    )
    return fig


# ---------------------------------------------------------------------------
# Preset customer profiles
# ---------------------------------------------------------------------------

LIKELY_CHURN = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 79.85, "TotalCharges": 160.60,
}

LIKELY_STAY = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 79.50, "TotalCharges": 4890.0,
}


# ===========================================================================
# TAB 1: PREDICT CHURN
# ===========================================================================

def tab_predict(model, scaler, metrics):
    from preprocessing import preprocess_input, get_risk_label, get_retention_suggestions

    st.header("🔮 Predict Customer Churn")
    st.caption(
        "Fill in a customer's details and click **Predict** to get the churn probability, "
        "risk level, and personalised retention suggestions."
    )

    # Preset buttons
    col_p1, col_p2, col_p3 = st.columns([1, 1, 4])
    load_churn = col_p1.button("⚡ Likely to Churn", use_container_width=True)
    load_stay  = col_p2.button("🛡️ Likely to Stay",  use_container_width=True)

    # Decide defaults
    if load_churn:
        defaults = LIKELY_CHURN
    elif load_stay:
        defaults = LIKELY_STAY
    else:
        defaults = st.session_state.get("form_defaults", LIKELY_STAY)

    st.session_state["form_defaults"] = defaults

    # --------------- Input form ---------------
    with st.form("predict_form"):
        st.subheader("Customer Profile")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Female", "Male"],
                                  index=["Female", "Male"].index(defaults["gender"]))
            senior = st.selectbox("Senior Citizen", [0, 1],
                                  index=defaults["SeniorCitizen"])
            partner = st.selectbox("Partner", ["No", "Yes"],
                                   index=["No", "Yes"].index(defaults["Partner"]))
            dependents = st.selectbox("Dependents", ["No", "Yes"],
                                      index=["No", "Yes"].index(defaults["Dependents"]))
            tenure = st.slider("Tenure (months)", 0, 72, int(defaults["tenure"]))

        with c2:
            st.markdown("**Services**")
            phone_svc = st.selectbox("Phone Service", ["No", "Yes"],
                                     index=["No", "Yes"].index(defaults["PhoneService"]))
            multi_lines = st.selectbox("Multiple Lines",
                                       ["No", "Yes", "No phone service"],
                                       index=["No", "Yes", "No phone service"].index(defaults["MultipleLines"]))
            internet = st.selectbox("Internet Service",
                                    ["DSL", "Fiber optic", "No"],
                                    index=["DSL", "Fiber optic", "No"].index(defaults["InternetService"]))
            online_sec = st.selectbox("Online Security",
                                      ["No", "Yes", "No internet service"],
                                      index=["No", "Yes", "No internet service"].index(defaults["OnlineSecurity"]))
            online_bak = st.selectbox("Online Backup",
                                      ["No", "Yes", "No internet service"],
                                      index=["No", "Yes", "No internet service"].index(defaults["OnlineBackup"]))

        with c3:
            st.markdown("**Add-ons & Billing**")
            dev_prot = st.selectbox("Device Protection",
                                    ["No", "Yes", "No internet service"],
                                    index=["No", "Yes", "No internet service"].index(defaults["DeviceProtection"]))
            tech_sup = st.selectbox("Tech Support",
                                    ["No", "Yes", "No internet service"],
                                    index=["No", "Yes", "No internet service"].index(defaults["TechSupport"]))
            stream_tv = st.selectbox("Streaming TV",
                                     ["No", "Yes", "No internet service"],
                                     index=["No", "Yes", "No internet service"].index(defaults["StreamingTV"]))
            stream_mv = st.selectbox("Streaming Movies",
                                     ["No", "Yes", "No internet service"],
                                     index=["No", "Yes", "No internet service"].index(defaults["StreamingMovies"]))
            contract  = st.selectbox("Contract",
                                     ["Month-to-month", "One year", "Two year"],
                                     index=["Month-to-month", "One year", "Two year"].index(defaults["Contract"]))
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"],
                                     index=["No", "Yes"].index(defaults["PaperlessBilling"]))
            payment   = st.selectbox("Payment Method",
                                     ["Bank transfer (automatic)",
                                      "Credit card (automatic)",
                                      "Electronic check",
                                      "Mailed check"],
                                     index=["Bank transfer (automatic)",
                                            "Credit card (automatic)",
                                            "Electronic check",
                                            "Mailed check"].index(defaults["PaymentMethod"]))

        st.markdown("**Charges**")
        ch1, ch2 = st.columns(2)
        monthly_charges = ch1.number_input("Monthly Charges ($)", 0.0, 200.0,
                                           float(defaults["MonthlyCharges"]), 0.5)
        total_charges   = ch2.number_input("Total Charges ($)", 0.0, 10000.0,
                                           float(defaults["TotalCharges"]), 1.0)

        submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True,
                                          type="primary")

    # --------------- Prediction ---------------
    if submitted:
        raw = {
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_svc,
            "MultipleLines": multi_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bak,
            "DeviceProtection": dev_prot, "TechSupport": tech_sup,
            "StreamingTV": stream_tv, "StreamingMovies": stream_mv,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        # Validate tenure=0 & TotalCharges
        if tenure == 0 and total_charges > 0:
            st.warning("⚠️ Tenure is 0 months — Total Charges will be treated as $0.00 "
                       "(new subscriber, no bills yet).")
            raw["TotalCharges"] = 0.0

        try:
            X_input = preprocess_input(raw, scaler)
            prob = float(model.predict(X_input, verbose=0).ravel()[0])
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

        risk = get_risk_label(prob)
        risk_color = {
            "High": CLR_DANGER, "Medium": CLR_WARN, "Low": CLR_OK
        }[risk]

        # Layout: gauge | suggestions
        g_col, s_col = st.columns([1, 1])

        with g_col:
            st.plotly_chart(churn_gauge(prob), use_container_width=True)

            # Key figures
            opt_thresh = metrics.get("best_cost_threshold", 0.5)
            st.info(
                f"**Risk Level:** :{risk_color}[{risk} Risk]  \n"
                f"**Churn Probability:** `{prob*100:.1f}%`  \n"
                f"**Tuned Cost Threshold:** `{opt_thresh:.2f}` "
                f"({'⚠️ CHURN ALERT' if prob >= opt_thresh else '✅ LIKELY TO STAY'})",
                icon="📊",
            )

        with s_col:
            st.subheader("💡 Retention Recommendations")
            suggestions = get_retention_suggestions(raw)
            for sug in suggestions:
                st.markdown(f"> {sug}")

        # Feature summary table
        with st.expander("🔍 Input Summary (Encoded Features)"):
            from preprocessing import clean_dataframe
            df_input = pd.DataFrame([raw])
            df_cleaned = clean_dataframe(df_input)
            st.dataframe(df_cleaned.T.rename(columns={0: "Value"}),
                         use_container_width=True)


# ===========================================================================
# TAB 2: MODEL PERFORMANCE
# ===========================================================================

def tab_performance(metrics, baseline_metrics, test_preds_df):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    st.header("📊 Model Performance")

    y_true  = test_preds_df["y_true"].values
    y_prob  = test_preds_df["y_prob"].values

    # --- Metric Cards ---
    st.subheader("Test Set Metrics (Default Threshold = 0.50)")
    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card_data = [
        ("Accuracy",   f"{metrics['accuracy']:.1%}",   f"LR: {baseline_metrics['accuracy']:.1%}",   CLR_PRIMARY),
        ("Precision",  f"{metrics['precision']:.1%}",  f"LR: {baseline_metrics['precision']:.1%}",  CLR_SECONDARY),
        ("Recall",     f"{metrics['recall']:.1%}",     f"LR: {baseline_metrics['recall']:.1%}",     CLR_TERTIARY),
        ("F1-Score",   f"{metrics['f1_score']:.3f}",   f"LR: {baseline_metrics['f1_score']:.3f}",   CLR_PRIMARY),
        ("ROC-AUC",    f"{metrics['roc_auc']:.3f}",    f"LR: {baseline_metrics['roc_auc']:.3f}",    CLR_OK),
    ]
    for col, (label, val, delta, color) in zip([c1,c2,c3,c4,c5], metric_card_data):
        with col:
            metric_card(label, val, delta, color)

    st.markdown("---")

    # --- Interactive Threshold Slider ---
    st.subheader("🎚️ Interactive Threshold Tuning")
    st.caption(
        "Move the slider to explore the precision–recall trade-off. "
        "**Lower threshold → higher recall (catch more churners, more false alarms).** "
        "**Higher threshold → higher precision (fewer false alarms, miss more churners).**"
    )

    threshold = st.slider(
        "Classification Threshold", 0.05, 0.95,
        float(metrics.get("default_threshold", 0.5)), 0.01,
        help="Default = 0.50 | Cost-optimal = " + str(metrics.get("best_cost_threshold", 0.14)),
    )

    acc, prec, rec, f1, cm = compute_metrics_at_threshold(y_true, y_prob, threshold)

    t1, t2, t3, t4 = st.columns(4)
    with t1: metric_card("Accuracy",  f"{acc:.1%}",  color=CLR_PRIMARY)
    with t2: metric_card("Precision", f"{prec:.1%}", color=CLR_SECONDARY)
    with t3: metric_card("Recall",    f"{rec:.1%}",  color=CLR_TERTIARY)
    with t4: metric_card("F1-Score",  f"{f1:.3f}",   color=CLR_PRIMARY)

    # Confusion matrix heatmap
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        cm_labels = [["TN", "FP"], ["FN", "TP"]]
        cm_vals   = [[tn, fp], [fn, tp]]
        cm_text   = [[f"{v}<br><span style='font-size:0.7em'>{lbl}</span>"
                      for v, lbl in zip(row_v, row_l)]
                     for row_v, row_l in zip(cm_vals, cm_labels)]

        fig_cm = go.Figure(go.Heatmap(
            z=cm_vals,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            text=cm_text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            hovertemplate="Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(
            title=f"Confusion Matrix (threshold = {threshold:.2f})",
            height=320, margin=dict(l=10, r=10, t=50, b=10),
            template=PLOTLY_TEMPLATE,
        )
        fig_cm.update_xaxes(side="top")

        st.plotly_chart(fig_cm, use_container_width=True)

        with st.expander("📌 Business Cost Note"):
            cost_fn, cost_fp, cost_tp = 1000, 50, 50
            cost_total = fn * cost_fn + fp * cost_fp + tp * cost_tp
            cost_per   = cost_total / len(y_true)
            st.markdown(
                f"At **τ = {threshold:.2f}**: "
                f"FN × \$1,000 + FP × \$50 + TP × \$50 "
                f"= **\${cost_total:,.0f}** total  |  **\${cost_per:.2f} / customer**  \n"
                f"> Cost-optimal threshold **τ = {metrics['best_cost_threshold']:.2f}** "
                f"minimises the expected cost per customer."
            )

    st.markdown("---")

    # --- ROC & PR Curves ---
    st.subheader("ROC & Precision-Recall Curves")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    auc_val  = roc_auc_score(y_true, y_prob)
    pr_auc   = average_precision_score(y_true, y_prob)

    fig_roc_pr = make_subplots(rows=1, cols=2,
                               subplot_titles=["ROC Curve", "Precision-Recall Curve"])

    fig_roc_pr.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"MLP (AUC={auc_val:.3f})",
        line=dict(color=CLR_PRIMARY, width=2.5),
    ), row=1, col=1)
    fig_roc_pr.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="#aaa", dash="dash"), showlegend=False,
    ), row=1, col=1)

    fig_roc_pr.add_trace(go.Scatter(
        x=rec_arr, y=prec_arr, mode="lines", name=f"MLP (AP={pr_auc:.3f})",
        line=dict(color=CLR_TERTIARY, width=2.5),
    ), row=1, col=2)
    prevalence = y_true.mean()
    fig_roc_pr.add_hline(y=prevalence, line_dash="dash", line_color="#aaa",
                         annotation_text=f"Prevalence {prevalence:.2f}", row=1, col=2)

    fig_roc_pr.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig_roc_pr.update_yaxes(title_text="True Positive Rate",  row=1, col=1)
    fig_roc_pr.update_xaxes(title_text="Recall",    row=1, col=2)
    fig_roc_pr.update_yaxes(title_text="Precision", row=1, col=2)
    fig_roc_pr.update_layout(height=380, template=PLOTLY_TEMPLATE,
                              margin=dict(l=10, r=10, t=50, b=10),
                              legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig_roc_pr, use_container_width=True)

    # --- Comparison Table ---
    st.markdown("---")
    st.subheader("🏆 MLP vs Logistic Regression Benchmark")
    comp_df = pd.DataFrame({
        "Metric":               ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
        "Logistic Regression":  [f"{baseline_metrics[k]:.4f}" for k in
                                 ["accuracy","precision","recall","f1_score","roc_auc","pr_auc"]],
        "MLP (This Model)":     [f"{metrics[k]:.4f}" for k in
                                 ["accuracy","precision","recall","f1_score","roc_auc","pr_auc"]],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 3: TRAINING INSIGHTS
# ===========================================================================

def tab_training(hist_df, metrics):
    st.header("🏋️ Training Insights")

    best_ep = metrics.get("best_epoch", int(hist_df["val_loss"].idxmin()) + 1)

    # Loss curves
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=hist_df["epoch"], y=hist_df["loss"],
        mode="lines+markers", name="Train Loss",
        line=dict(color=CLR_PRIMARY, width=2), marker=dict(size=4),
    ))
    fig_loss.add_trace(go.Scatter(
        x=hist_df["epoch"], y=hist_df["val_loss"],
        mode="lines+markers", name="Validation Loss",
        line=dict(color=CLR_SECONDARY, width=2), marker=dict(size=4),
    ))
    fig_loss.add_vline(
        x=best_ep, line_dash="dash", line_color=CLR_OK,
        annotation_text=f"Best epoch ({best_ep})",
        annotation_position="top right",
    )
    fig_loss.update_layout(
        title="Training vs Validation Loss (Binary Cross-Entropy)",
        xaxis_title="Epoch", yaxis_title="Loss",
        height=360, template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    # Accuracy curves
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=hist_df["epoch"], y=hist_df["accuracy"],
        mode="lines+markers", name="Train Accuracy",
        line=dict(color=CLR_PRIMARY, width=2), marker=dict(size=4),
    ))
    fig_acc.add_trace(go.Scatter(
        x=hist_df["epoch"], y=hist_df["val_accuracy"],
        mode="lines+markers", name="Validation Accuracy",
        line=dict(color=CLR_TERTIARY, width=2), marker=dict(size=4),
    ))
    fig_acc.add_vline(
        x=best_ep, line_dash="dash", line_color=CLR_DANGER,
        annotation_text=f"Best epoch ({best_ep})",
        annotation_position="top right",
    )
    fig_acc.update_layout(
        title="Training vs Validation Accuracy",
        xaxis_title="Epoch", yaxis_title="Accuracy",
        height=360, template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # AUC curve if available
    if "auc" in hist_df.columns and "val_auc" in hist_df.columns:
        fig_auc = go.Figure()
        fig_auc.add_trace(go.Scatter(
            x=hist_df["epoch"], y=hist_df["auc"],
            mode="lines", name="Train AUC",
            line=dict(color=CLR_PRIMARY, width=2),
        ))
        fig_auc.add_trace(go.Scatter(
            x=hist_df["epoch"], y=hist_df["val_auc"],
            mode="lines", name="Validation AUC",
            line=dict(color=CLR_SECONDARY, width=2),
        ))
        fig_auc.update_layout(
            title="Training vs Validation ROC-AUC",
            xaxis_title="Epoch", yaxis_title="ROC-AUC",
            height=320, template=PLOTLY_TEMPLATE,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    # Learning rate schedule
    if "learning_rate" in hist_df.columns:
        fig_lr = go.Figure(go.Scatter(
            x=hist_df["epoch"], y=hist_df["learning_rate"],
            mode="lines+markers", name="Learning Rate",
            line=dict(color=CLR_WARN, width=2), marker=dict(size=4),
        ))
        fig_lr.update_layout(
            title="Learning Rate Schedule (ReduceLROnPlateau)",
            xaxis_title="Epoch", yaxis_title="Learning Rate",
            height=270, template=PLOTLY_TEMPLATE,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_lr, use_container_width=True)

    # Interpretation guide
    st.markdown("---")
    st.subheader("📖 How to Read the Learning Curves")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**✅ Well-Generalising (This Model)**\n\n"
                   "Train and validation losses decrease together and converge to "
                   "similar values. Dropout + EarlyStopping prevented overfitting.")
    with col2:
        st.warning("**⚠️ Overfitting**\n\n"
                   "Train loss keeps falling while validation loss rises. "
                   "The model memorises the training data. "
                   "Fix: more dropout, L2 regularisation, or less data.")
    with col3:
        st.error("**❌ Underfitting**\n\n"
                 "Both train and validation losses remain high. "
                 "The model lacks capacity. "
                 "Fix: more neurons/layers, fewer regularisation constraints, "
                 "more epochs.")


# ===========================================================================
# TAB 4: DATA EXPLORER
# ===========================================================================

def tab_explorer(df_raw, feat_imp_df):
    st.header("🔭 Data Explorer")

    if df_raw is None:
        st.warning(f"Dataset file `{DATA_PATH}` not found. Place it in the project root.")
        return

    # Churn distribution
    st.subheader("1. Target Distribution")
    churn_counts = df_raw["Churn"].value_counts()
    fig_dist = go.Figure(go.Bar(
        x=churn_counts.index, y=churn_counts.values,
        marker_color=[CLR_PRIMARY, CLR_SECONDARY],
        text=[f"{v:,} ({v/len(df_raw)*100:.1f}%)" for v in churn_counts.values],
        textposition="outside",
    ))
    fig_dist.update_layout(
        title="Churn Class Distribution",
        xaxis_title="Churn", yaxis_title="Count",
        height=320, template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Churn Rate by Categorical Features")

    exp_cols = ["Contract", "InternetService", "PaymentMethod"]
    ecols = st.columns(len(exp_cols))

    for col, feature in zip(ecols, exp_cols):
        churn_by = df_raw.groupby(feature)["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        churn_by.columns = [feature, "Churn Rate (%)"]
        churn_by = churn_by.sort_values("Churn Rate (%)", ascending=False)

        fig = px.bar(
            churn_by, x=feature, y="Churn Rate (%)",
            color="Churn Rate (%)", color_continuous_scale="RdYlGn_r",
            title=f"Churn by {feature}", text_auto=".1f",
            height=320, template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=5, r=5, t=50, b=5))
        col.plotly_chart(fig, use_container_width=True)

    # Tenure distribution by churn
    st.markdown("---")
    st.subheader("3. Tenure Distribution by Churn Status")
    fig_tenure = px.histogram(
        df_raw, x="tenure", color="Churn", nbins=36,
        barmode="overlay", opacity=0.7,
        color_discrete_map={"No": CLR_PRIMARY, "Yes": CLR_SECONDARY},
        title="Tenure Distribution: Churned vs Retained",
        labels={"tenure": "Tenure (months)"},
        height=350, template=PLOTLY_TEMPLATE,
    )
    fig_tenure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_tenure, use_container_width=True)

    # Monthly charges violin
    fig_violin = px.violin(
        df_raw, x="Churn", y="MonthlyCharges", color="Churn",
        box=True, points=False,
        color_discrete_map={"No": CLR_PRIMARY, "Yes": CLR_SECONDARY},
        title="Monthly Charges Distribution by Churn Status",
        height=350, template=PLOTLY_TEMPLATE,
    )
    fig_violin.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_violin, use_container_width=True)

    # Correlation heatmap on numeric features
    st.markdown("---")
    st.subheader("4. Feature Correlation Heatmap")
    num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    df_num = df_raw[num_cols + ["Churn"]].copy()
    df_num["Churn_num"] = (df_num["Churn"] == "Yes").astype(int)
    df_num = df_num.drop(columns=["Churn"])
    corr = df_num.corr(numeric_only=True).round(3)

    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
    ))
    fig_corr.update_layout(
        title="Correlation Heatmap (Numeric Features + Churn)",
        height=380, template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.subheader("5. Permutation Feature Importance (MLP)")
    st.caption(
        "Mean drop in Test ROC-AUC when each feature is permuted (5 shuffles). "
        "Larger drop → feature is more important for the model's predictions."
    )
    top_n = st.slider("Show top N features", 5, 30, 15)
    fi_top = feat_imp_df.head(top_n).iloc[::-1]

    fig_fi = go.Figure(go.Bar(
        x=fi_top["importance"], y=fi_top["feature"],
        orientation="h",
        error_x=dict(type="data", array=fi_top["std_dev"].tolist(), visible=True),
        marker_color=CLR_PRIMARY,
    ))
    fig_fi.update_layout(
        title=f"Top {top_n} Features by Permutation Importance",
        xaxis_title="Mean Drop in ROC-AUC",
        yaxis_title="Feature",
        height=max(350, top_n * 25 + 80),
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ===========================================================================
# TAB 5: BATCH PREDICTION
# ===========================================================================

def tab_batch(model, scaler):
    from preprocessing import preprocess_batch, get_risk_label, FEATURE_COLUMNS

    st.header("📂 Batch Prediction")
    st.caption(
        "Upload a CSV file in the same format as the IBM Telco Customer Churn dataset. "
        "The app will score every row and return a downloadable results CSV."
    )

    # Sample download
    sample_path = "sample_data/sample_batch.csv"
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button(
                "⬇️ Download Sample CSV (20 rows)",
                f, file_name="sample_batch.csv", mime="text/csv",
            )

    uploaded = st.file_uploader(
        "Upload CSV for batch prediction",
        type=["csv"],
        help="Must contain the same columns as the original Telco dataset. "
             "customerID and Churn columns are optional.",
    )

    if uploaded is None:
        st.info("👆 Upload a CSV to get started, or download the sample above.")
        return

    try:
        df_up = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    # --- Validation ---
    REQUIRED_COLS = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]
    missing_cols = [c for c in REQUIRED_COLS if c not in df_up.columns]
    if missing_cols:
        st.error(
            f"**Missing required columns:** `{missing_cols}`\n\n"
            "Please ensure your CSV has all the standard Telco dataset columns."
        )
        return

    if len(df_up) == 0:
        st.error("Uploaded CSV contains no rows.")
        return

    st.success(f"✅ File validated — **{len(df_up)} rows** loaded.")
    st.dataframe(df_up.head(5), use_container_width=True)

    if st.button("🚀 Run Batch Predictions", type="primary"):
        with st.spinner(f"Scoring {len(df_up)} customers…"):
            try:
                X_batch = preprocess_batch(df_up, scaler)
                probs = model.predict(X_batch, verbose=0).ravel()
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        df_results = df_up.copy()
        df_results["churn_probability"] = probs.round(4)
        df_results["risk_label"] = [get_risk_label(p) for p in probs]
        df_results["predicted_churn"] = (probs >= 0.5).astype(int)

        # Summary stats
        st.markdown("---")
        st.subheader("📋 Prediction Results")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Customers", len(df_results))
        s2.metric("High Risk", int((df_results["risk_label"] == "High").sum()))
        s3.metric("Medium Risk", int((df_results["risk_label"] == "Medium").sum()))
        s4.metric("Low Risk", int((df_results["risk_label"] == "Low").sum()))

        # Risk distribution chart
        risk_counts = df_results["risk_label"].value_counts().reindex(
            ["High", "Medium", "Low"], fill_value=0
        )
        fig_risk = go.Figure(go.Bar(
            x=risk_counts.index, y=risk_counts.values,
            marker_color=[CLR_DANGER, CLR_WARN, CLR_OK],
            text=risk_counts.values, textposition="outside",
        ))
        fig_risk.update_layout(
            title="Risk Distribution in Uploaded Batch",
            xaxis_title="Risk Level", yaxis_title="Count",
            height=300, template=PLOTLY_TEMPLATE,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        # Full results table (sorted by churn_probability descending)
        display_cols = (
            (["customerID"] if "customerID" in df_results.columns else []) +
            ["churn_probability", "risk_label", "predicted_churn",
             "tenure", "Contract", "MonthlyCharges"]
        )
        display_cols = [c for c in display_cols if c in df_results.columns]
        st.dataframe(
            df_results[display_cols].sort_values("churn_probability", ascending=False),
            use_container_width=True,
        )

        # Download button
        csv_bytes = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Results CSV",
            csv_bytes,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

        # If ground truth available
        if "Churn" in df_up.columns:
            y_true_batch = (df_up["Churn"] == "Yes").astype(int).values
            from sklearn.metrics import roc_auc_score
            try:
                auc_batch = roc_auc_score(y_true_batch, probs)
                st.success(f"🎯 Batch ROC-AUC (ground truth detected): **{auc_batch:.4f}**")
            except Exception:
                pass


# ===========================================================================
# TAB 6: ABOUT / CONCEPTS
# ===========================================================================

def tab_about():
    st.header("📚 About This Demo & Key Concepts")

    st.info(
        "⚠️ **Academic Demo Disclaimer:** This application is built for educational "
        "purposes as part of a machine learning project. It uses the IBM Telco Customer "
        "Churn dataset and a Multilayer Perceptron trained on historical data. "
        "Predictions should NOT be used as sole basis for real business decisions.",
        icon="🎓",
    )

    st.markdown("---")
    st.subheader("🏛️ Neural Network Architecture")
    st.code(
        """
Input Layer  (d = 30 preprocessed features)
       │
       ▼
Dense Layer 1  (64 units, ReLU activation)   — Params: 30×64 + 64 = 1,984
       │
       ▼
Dropout Layer 1  (p = 0.30)                  — No learnable params
       │
       ▼
Dense Layer 2  (32 units, ReLU activation)   — Params: 64×32 + 32 = 2,080
       │
       ▼
Dropout Layer 2  (p = 0.30)                  — No learnable params
       │
       ▼
Output Layer  (1 unit, Sigmoid activation)   — Params: 32×1 + 1  = 33
       │
       ▼
  P(Churn | X) ∈ [0, 1]                      ← Total trainable: 4,097
        """,
        language="text",
    )

    # Concept cards
    concepts = [
        ("🟢 ReLU Activation",
         "$$f(z) = \\max(0, z), \\quad f'(z) = \\mathbf{1}[z > 0]$$",
         "Used in hidden layers. Maintains a constant gradient of 1 for positive inputs, "
         "eliminating the vanishing gradient problem that plagues Sigmoid/Tanh in deep networks. "
         "Creates sparse, efficient representations."),
        ("🔵 Sigmoid Activation",
         "$$\\sigma(z) = \\frac{1}{1+e^{-z}}, \\quad \\sigma'(z) = \\sigma(z)(1-\\sigma(z))$$",
         "Used only at the output layer for binary classification. Squashes any logit value "
         "into (0, 1), enabling interpretation as the posterior probability P(Churn | X)."),
        ("📉 Binary Cross-Entropy Loss",
         "$$\\mathcal{L} = -\\frac{1}{N}\\sum_{i=1}^{N}\\left[y_i\\log\\hat{y}_i + (1-y_i)\\log(1-\\hat{y}_i)\\right]$$",
         "The objective function. Measures the dissimilarity between the true binary labels "
         "and the predicted probabilities. Minimised via gradient descent."),
        ("⬅️ Backpropagation",
         "$$\\delta^{[L]} = \\frac{1}{B}(\\hat{y}-y), \\quad \\frac{\\partial \\mathcal{L}}{\\partial W^{[l]}} = (A^{[l-1]})^T \\delta^{[l]}$$",
         "Efficient reverse-mode automatic differentiation. Computes gradients of the loss "
         "with respect to all weights using the multivariable chain rule, propagating errors "
         "backwards layer by layer."),
        ("⚡ Adam Optimizer",
         "$$\\theta_{t+1} = \\theta_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t}+\\epsilon}\\hat{m}_t$$",
         "Combines Momentum (smooths gradient direction) and RMSprop (scales by gradient "
         "variance). Adapts learning rates per-parameter, converging faster than vanilla SGD "
         "on ill-conditioned loss landscapes."),
        ("🔀 Dropout Regularisation",
         "$$\\tilde{A}^{[l]} = A^{[l]} \\odot M^{[l]} \\cdot \\frac{1}{1-p}, \\quad M_{ij} \\sim \\text{Bernoulli}(1-p)$$",
         "Randomly zeroes p=30% of neuron outputs each training step. Prevents fragile "
         "co-adaptations, acts as an ensemble of 2^N thinned subnetworks. At inference, "
         "dropout is turned off (inverted scaling ensures correct expectations)."),
    ]

    for i in range(0, len(concepts), 2):
        c1, c2 = st.columns(2)
        for col, idx in zip([c1, c2], [i, i+1]):
            if idx < len(concepts):
                name, formula, description = concepts[idx]
                with col:
                    st.markdown(f"### {name}")
                    st.latex(formula.replace("$$", "").strip())
                    st.markdown(description)
                    st.markdown("")

    # Preprocessing note
    st.markdown("---")
    st.subheader("🔧 Preprocessing Pipeline")
    st.markdown(
        """
        All transformations are defined in `preprocessing.py` and shared across
        `train.py`, `churn_mlp.ipynb`, and this Streamlit app to guarantee
        **zero divergence between training and inference**:

        | Step | Detail |
        |------|--------|
        | **TotalCharges fix** | Blank strings → `0.0` (new subscribers, `tenure = 0`) |
        | **Binary encoding** | gender, Partner, Dependents, PhoneService, PaperlessBilling → 0/1 |
        | **One-hot encoding** | 10 multi-level columns with `drop_first=True` (avoids multicollinearity) |
        | **StandardScaler** | z = (x − μ_train) / σ_train — fitted only on training split |
        | **Class weights** | w_1 ≈ 1.88, w_0 ≈ 0.68 (penalises missing churners 2.76× more) |
        """
    )

    st.markdown("---")
    st.subheader("📁 Project File Structure")
    st.code(
        """
ai ml 2/
├── churn_mlp.ipynb                   # Full academic notebook with executed outputs
├── preprocessing.py                  # Shared preprocessing pipeline (single source of truth)
├── train.py                          # Standalone training + artifact export script
├── app.py                            # This Streamlit dashboard
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── artifacts/
│   ├── churn_mlp.keras               # Trained Keras model
│   ├── scaler.pkl                    # Fitted StandardScaler
│   ├── feature_columns.json          # Exact ordered column list
│   ├── metrics.json                  # Test metrics + thresholds
│   ├── history.csv                   # Per-epoch training history
│   ├── test_predictions.csv          # y_true + y_prob on test split
│   ├── baseline_metrics.json         # Logistic Regression benchmark
│   └── feature_importance.csv        # Permutation importance
├── sample_data/
│   └── sample_batch.csv              # 20-row demo for Batch Prediction tab
└── figures/                          # 300 DPI plots exported by notebook
        """,
        language="text",
    )

    st.markdown("---")
    st.subheader("🚀 Quick Start")
    st.code(
        """
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model & export all artifacts
python train.py

# 3. Launch the dashboard
streamlit run app.py
        """,
        language="bash",
    )


# ===========================================================================
# SIDEBAR
# ===========================================================================

def render_sidebar(metrics):
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Keras_logo.svg/120px-Keras_logo.svg.png",
                 width=60)
        st.title("📡 Telco Churn\nPredictor")
        st.caption("Academic ML Demo — MLP on IBM Telco Dataset")
        st.markdown("---")

        if metrics:
            st.markdown("**📊 Model Snapshot**")
            st.metric("ROC-AUC",  f"{metrics['roc_auc']:.3f}")
            st.metric("Recall",   f"{metrics['recall']:.1%}")
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
            st.metric("Best Epoch", metrics.get("best_epoch", "—"))
            st.markdown("---")

        st.markdown(
            "**Dataset:** [IBM Telco Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  \n"
            "**Model:** MLP (4,097 params)  \n"
            "**Framework:** TensorFlow/Keras  "
        )
        st.markdown("---")
        st.caption("⚠️ This is an academic demonstration only.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    # --- Artifact check ---
    if not _artifacts_ok():
        render_sidebar(None)
        _missing_artifacts_msg()
        return

    # --- Load all resources ---
    model      = load_model()
    scaler     = load_scaler()
    metrics    = load_metrics()
    base_m     = load_baseline_metrics()
    hist_df    = load_history()
    test_preds = load_test_predictions()
    feat_imp   = load_feature_importance()
    df_raw     = load_raw_data()

    # --- Sidebar ---
    render_sidebar(metrics)

    # --- Tabs ---
    tabs = st.tabs([
        "🔮 Predict Churn",
        "📊 Model Performance",
        "🏋️ Training Insights",
        "🔭 Data Explorer",
        "📂 Batch Prediction",
        "📚 About / Concepts",
    ])

    with tabs[0]:
        tab_predict(model, scaler, metrics)
    with tabs[1]:
        tab_performance(metrics, base_m, test_preds)
    with tabs[2]:
        tab_training(hist_df, metrics)
    with tabs[3]:
        tab_explorer(df_raw, feat_imp)
    with tabs[4]:
        tab_batch(model, scaler)
    with tabs[5]:
        tab_about()


if __name__ == "__main__":
    main()

