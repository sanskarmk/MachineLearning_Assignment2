"""
Streamlit Web Application – Phishing Website Detection
========================================================
ML Assignment 2 | BITS Pilani M.Tech AIML

This app loads pre-trained model .pkl files at startup (trained offline via
train_all_models_offline.py) and uses them to predict on uploaded test data.
No re-training happens inside the app.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "model")
COMPARISON_CSV = os.path.join(BASE_DIR, "model_comparison_results.csv")
TEST_DATA_CSV = os.path.join(BASE_DIR, "test_data.csv")

# ═══════════════════════════════════════════════════════════════
# MODEL REGISTRY  (key shown in dropdown → pkl filename)
# ═══════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "Logistic Regression": {"file": "logistic_regression.pkl", "needs_scaling": True},
    "Decision Tree":       {"file": "decision_tree.pkl",       "needs_scaling": False},
    "KNN":                 {"file": "knn.pkl",                 "needs_scaling": True},
    "Naive Bayes":         {"file": "naive_bayes.pkl",         "needs_scaling": False},
    "Random Forest":       {"file": "random_forest.pkl",       "needs_scaling": False},
    "XGBoost":             {"file": "xgboost.pkl",             "needs_scaling": False},
}


# ═══════════════════════════════════════════════════════════════
# LOAD MODELS + SCALER AT STARTUP  (cached so it runs only once)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    """Load every .pkl model and the scaler from the model/ directory."""
    loaded = {}
    for display_name, spec in MODEL_REGISTRY.items():
        pkl_path = os.path.join(MODELS_DIR, spec["file"])
        try:
            loaded[display_name] = joblib.load(pkl_path)
        except Exception as e:
            loaded[display_name] = None
            st.warning(f"Could not load {display_name}: {e}")
    # Scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    scaler = None
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        pass
    return loaded, scaler


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def parse_uploaded_csv(uploaded_file):
    """Read uploaded CSV, drop 'url' if present, return (X DataFrame, y array)."""
    df = pd.read_csv(uploaded_file)
    if "url" in df.columns:
        df = df.drop(columns=["url"])
    X = df.drop(columns=["status"])
    y = df["status"]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.map({"legitimate": 0, "phishing": 1})
    return X, y.values


def compute_metrics(y_true, y_pred, y_prob):
    """Compute the 6 required evaluation metrics."""
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "F1":        round(f1_score(y_true, y_pred), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }


def predict_with_model(model, X, needs_scaling, scaler):
    """Run prediction using a pre-trained model. Scale features if needed."""
    X_input = scaler.transform(X) if (needs_scaling and scaler is not None) else X
    y_pred = model.predict(X_input)
    y_prob = model.predict_proba(X_input)[:, 1]
    return y_pred, y_prob


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Phishing Detection – ML Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none !important;}
    section[data-testid="stSidebar"] button[kind="header"] {display: none !important;}
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {display: none !important;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }
    .element-container {margin-bottom: 0.25rem !important;}
    div[data-testid="stVerticalBlock"] > div {gap: 0.3rem !important;}
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3a3a5c;
        border-radius: 10px;
        padding: 10px 14px;
        text-align: center;
    }
    div[data-testid="stMetric"] label {
        color: #a0a0c0 !important;
        font-size: 0.75rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        color: #00d4ff !important;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        min-width: 280px !important;
        max-width: 320px !important;
        transform: none !important;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
    }
    .stDataFrame {font-size: 0.8rem !important;}
    .main-title {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 0.5rem;
    }
    .section-hdr {
        font-size: 0.95rem;
        font-weight: 700;
        color: #c0c0e0;
        border-left: 3px solid #7b2ff7;
        padding-left: 8px;
        margin: 0.5rem 0 0.3rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7b2ff7, #00d4ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.45rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# LOAD MODELS AT STARTUP
# ═══════════════════════════════════════════════════════════════
loaded_models, scaler = load_all_models()
n_loaded = sum(1 for v in loaded_models.values() if v is not None)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="main-title">🛡️ Phishing Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">ML Assignment 2 — BITS Pilani</p>', unsafe_allow_html=True)
    st.divider()

    # Download test_data.csv button
    if os.path.exists(TEST_DATA_CSV):
        with open(TEST_DATA_CSV, "rb") as f:
            st.download_button(
                "⬇  Download test_data.csv",
                f,
                "test_data.csv",
                "text/csv",
                use_container_width=True,
            )

    # File uploader
    uploaded_dataset = st.file_uploader("📂 Upload test CSV", type=["csv"])

    # Model selector
    selected_algorithm = st.selectbox(
        "🤖 Algorithm",
        list(MODEL_REGISTRY.keys()),
    )

    run_clicked = st.button("🚀 Evaluate Model", use_container_width=True)

    st.divider()
    st.caption("Sanskar Maheshkumar Khandelwal | 2025AA05332")
    st.caption("BITS Pilani - M.Tech AIML")


# ═══════════════════════════════════════════════════════════════
# MAIN AREA – HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown(
    '<p class="main-title" style="font-size:1.6rem;">📊 Model Evaluation Dashboard</p>',
    unsafe_allow_html=True,
)
if n_loaded == len(MODEL_REGISTRY):
    st.caption(f"✅ All {n_loaded} pre-trained models loaded from ./model/")
elif n_loaded > 0:
    st.caption(f"⚠️ {n_loaded}/{len(MODEL_REGISTRY)} models loaded from ./model/")
else:
    st.caption("❌ No pre-trained models found. Run train_all_models_offline.py first.")


# ═══════════════════════════════════════════════════════════════
# RENDER HELPERS
# ═══════════════════════════════════════════════════════════════
def render_single_model(metrics, y_true, y_pred, model_name):
    """Display metrics, confusion matrix, and classification report for one model."""
    st.markdown(f'<p class="section-hdr">Performance — {model_name}</p>', unsafe_allow_html=True)
    cols = st.columns(6)
    for col, (label, key) in zip(cols, [
        ("Accuracy", "Accuracy"), ("AUC", "AUC"), ("Precision", "Precision"),
        ("Recall", "Recall"), ("F1 Score", "F1"), ("MCC", "MCC"),
    ]):
        col.metric(label, f"{metrics[key]:.4f}")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown('<p class="section-hdr">Confusion Matrix</p>', unsafe_allow_html=True)
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=["Legit", "Phish"], columns=["Pred Legit", "Pred Phish"])
        st.dataframe(cm_df, use_container_width=True, height=120)

    with right_col:
        st.markdown('<p class="section-hdr">Classification Report</p>', unsafe_allow_html=True)
        cr = pd.DataFrame(
            classification_report(
                y_true, y_pred,
                target_names=["Legitimate", "Phishing"],
                output_dict=True,
            )
        ).transpose().round(4)
        st.dataframe(cr, use_container_width=True, height=180)


# ═══════════════════════════════════════════════════════════════
# EVALUATE SINGLE MODEL
# ═══════════════════════════════════════════════════════════════
if run_clicked:
    if uploaded_dataset is None:
        st.error("Please upload a CSV file before evaluating.")
    elif loaded_models.get(selected_algorithm) is None:
        st.error(f"Model '{selected_algorithm}' is not loaded. Train models first.")
    else:
        X_test, y_test = parse_uploaded_csv(uploaded_dataset)
        needs_scaling = MODEL_REGISTRY[selected_algorithm]["needs_scaling"]
        model = loaded_models[selected_algorithm]

        with st.spinner(f"Running predictions with {selected_algorithm}…"):
            y_pred, y_prob = predict_with_model(model, X_test, needs_scaling, scaler)
            metrics = compute_metrics(y_test, y_pred, y_prob)

        render_single_model(metrics, y_test, y_pred, selected_algorithm)


# ═══════════════════════════════════════════════════════════════
# COMPARISON TABLE (always visible, loaded from pre-computed CSV)
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="section-hdr">All Models Comparison</p>', unsafe_allow_html=True)

if os.path.exists(COMPARISON_CSV):
    cmp = pd.read_csv(COMPARISON_CSV)
    numeric_cols = [c for c in cmp.columns if c != "Model"]
    st.dataframe(
        cmp.style
            .format({c: "{:.4f}" for c in numeric_cols})
            .background_gradient(cmap="YlGn", subset=numeric_cols),
        use_container_width=True,
        height=260,
    )
else:
    st.info("No comparison data found. Run train_all_models_offline.py to generate it.")
