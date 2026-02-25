import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer KNN Classifier",
    page_icon="🧬",
    layout="wide",
)

# ─── Load Artifacts ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("knn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

# ─── Load Dataset ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("KNNAlgorithmDataset_csv.xls")
    df.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")
    df.dropna(inplace=True)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df

df = load_data()

# ─── Sidebar ────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/dna-helix.png", width=80)
st.sidebar.title("🧬 KNN Classifier")
st.sidebar.markdown("**Breast Cancer Detection**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔬 Predict", "📊 Model Metrics", "📁 Dataset"])

# ════════════════════════════════════════════════════════════════
# HOME PAGE
# ════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🧬 Breast Cancer Detection using KNN")
    st.markdown("### K-Nearest Neighbors Classification")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Model Accuracy", "95.61%", "+Best K=5")
    col2.metric("📊 Dataset Rows", f"{len(df)}", "Samples")
    col3.metric("📐 Features Used", f"{len(feature_names)}", "Numeric Features")

    st.markdown("---")
    st.markdown("""
    ### About this App
    This app uses the **Wisconsin Breast Cancer Dataset** to classify tumors as:
    - 🟢 **Benign (B)** — Non-cancerous
    - 🔴 **Malignant (M)** — Cancerous

    The **K-Nearest Neighbors (KNN)** algorithm is trained on 30 features extracted from 
    cell nuclei images. The best K value was found to be **5** using cross-validation.

    ### How to Use
    1. Go to **🔬 Predict** → Enter values manually or use random sample
    2. Go to **📊 Model Metrics** → View accuracy, confusion matrix, charts
    3. Go to **📁 Dataset** → Explore the raw data
    """)

    # Class distribution chart
    st.markdown("### 📊 Dataset Class Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    counts = df["diagnosis"].value_counts()
    ax.bar(["Benign (0)", "Malignant (1)"], counts.values, color=["#2ecc71", "#e74c3c"], edgecolor="white")
    ax.set_ylabel("Count")
    ax.set_title("Benign vs Malignant")
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════
# PREDICT PAGE
# ════════════════════════════════════════════════════════════════
elif page == "🔬 Predict":
    st.title("🔬 Tumor Prediction")
    st.markdown("Enter the feature values below or use a random sample from the dataset.")
    st.markdown("---")

    # Random sample button
    if st.button("🎲 Load Random Sample from Dataset"):
        sample = df.sample(1).iloc[0]
        st.session_state["sample"] = sample
        st.session_state["actual"] = "Malignant 🔴" if sample["diagnosis"] == 1 else "Benign 🟢"

    # Show actual label if sample loaded
    if "actual" in st.session_state:
        st.info(f"📌 Actual Label from Dataset: **{st.session_state['actual']}**")

    # Group features
    mean_features    = [f for f in feature_names if "mean" in f]
    se_features      = [f for f in feature_names if "_se" in f]
    worst_features   = [f for f in feature_names if "worst" in f]

    def get_default(fname):
        if "sample" in st.session_state:
            return float(st.session_state["sample"][fname])
        return float(df[fname].mean())

    input_data = {}

    with st.expander("📐 Mean Features", expanded=True):
        cols = st.columns(3)
        for i, feat in enumerate(mean_features):
            input_data[feat] = cols[i % 3].number_input(
                feat, value=get_default(feat), format="%.5f", key=feat
            )

    with st.expander("📏 Standard Error Features"):
        cols = st.columns(3)
        for i, feat in enumerate(se_features):
            input_data[feat] = cols[i % 3].number_input(
                feat, value=get_default(feat), format="%.5f", key=feat
            )

    with st.expander("⚠️ Worst Features"):
        cols = st.columns(3)
        for i, feat in enumerate(worst_features):
            input_data[feat] = cols[i % 3].number_input(
                feat, value=get_default(feat), format="%.5f", key=feat
            )

    st.markdown("---")
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        input_df = pd.DataFrame([[input_data[f] for f in feature_names]], columns=feature_names)
        scaled   = scaler.transform(input_df)
        pred     = model.predict(scaled)[0]
        prob     = model.predict_proba(scaled)[0]

        st.markdown("### 🎯 Prediction Result")
        if pred == 1:
            st.error(f"## 🔴 MALIGNANT (Cancerous)\n\nConfidence: **{prob[1]*100:.1f}%**")
        else:
            st.success(f"## 🟢 BENIGN (Non-Cancerous)\n\nConfidence: **{prob[0]*100:.1f}%**")

        col1, col2 = st.columns(2)
        col1.metric("Benign Probability",   f"{prob[0]*100:.1f}%")
        col2.metric("Malignant Probability", f"{prob[1]*100:.1f}%")

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["Benign", "Malignant"], [prob[0], prob[1]], color=["#2ecc71", "#e74c3c"])
        ax.set_xlim(0, 1)
        ax.set_title("Prediction Probabilities")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

# ════════════════════════════════════════════════════════════════
# MODEL METRICS PAGE
# ════════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.title("📊 Model Performance Metrics")
    st.markdown("---")

    X = df[feature_names]
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)
    y_pred = model.predict(X_test_sc)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"], output_dict=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Accuracy",  f"{acc*100:.2f}%")
    col2.metric("🎯 Precision (Benign)",    f"{report['Benign']['precision']*100:.1f}%")
    col3.metric("🔍 Recall (Malignant)",    f"{report['Malignant']['recall']*100:.1f}%")
    col4.metric("📐 F1 Score (weighted)",   f"{report['weighted avg']['f1-score']*100:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Confusion Matrix
    with col_a:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Benign", "Malignant"],
                    yticklabels=["Benign", "Malignant"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    # K vs Accuracy
    with col_b:
        st.markdown("#### K Value vs Accuracy")
        k_vals, accs = [], []
        for k in range(1, 21):
            from sklearn.neighbors import KNeighborsClassifier
            knn_temp = KNeighborsClassifier(n_neighbors=k)
            knn_temp.fit(X_train_sc, y_train)
            accs.append(accuracy_score(y_test, knn_temp.predict(X_test_sc)))
            k_vals.append(k)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(k_vals, accs, marker="o", color="#3498db", linewidth=2)
        ax2.axvline(x=5, color="red", linestyle="--", label="Best K=5")
        ax2.set_xlabel("K Value")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("K vs Accuracy")
        ax2.legend()
        ax2.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig2)

    # Feature importance (using variance)
    st.markdown("#### 🔑 Top 10 Features by Variance")
    feat_var = pd.Series(X.var().values, index=feature_names).nlargest(10)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    feat_var.sort_values().plot(kind="barh", ax=ax3, color="#9b59b6")
    ax3.set_title("Top 10 Features by Variance")
    ax3.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig3)

# ════════════════════════════════════════════════════════════════
# DATASET PAGE
# ════════════════════════════════════════════════════════════════
elif page == "📁 Dataset":
    st.title("📁 Dataset Explorer")
    st.markdown("---")

    display_df = df.copy()
    display_df["diagnosis"] = display_df["diagnosis"].map({1: "Malignant 🔴", 0: "Benign 🟢"})

    col1, col2 = st.columns(2)
    search = col1.text_input("🔍 Filter by diagnosis", placeholder="Type Benign or Malignant")
    show_n = col2.slider("Rows to display", 10, len(df), 20)

    if search:
        display_df = display_df[display_df["diagnosis"].str.contains(search, case=False)]

    st.dataframe(display_df.head(show_n), use_container_width=True)
    st.caption(f"Showing {min(show_n, len(display_df))} of {len(display_df)} rows")

    st.markdown("### 📈 Feature Distribution")
    feat_select = st.selectbox("Select Feature", feature_names)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, name in [(0, "#2ecc71", "Benign"), (1, "#e74c3c", "Malignant")]:
        ax.hist(df[df["diagnosis"] == label][feat_select], bins=30,
                alpha=0.6, color=color, label=name)
    ax.set_xlabel(feat_select)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {feat_select}")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
