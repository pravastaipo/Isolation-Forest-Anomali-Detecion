import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Training Isolation Forest",
    layout="centered"
)

st.title("Training Model IForest")


# ==================================================
# UPLOAD FILE
# ==================================================
uploaded_file = st.file_uploader(
    "📂 Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # ==================================================
    # LOAD DATA
    # ==================================================
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("📊 Informasi Dataset Awal")

    st.write(
        f"Jumlah Baris: {df_raw.shape[0]}"
    )

    st.write(
        f"Jumlah Kolom: {df_raw.shape[1]}"
    )

    with st.expander("👁️ Preview Dataset"):

        st.dataframe(df_raw.head(10))


    # ==================================================
    # FITUR DAN LABEL
    # ==================================================
    X_original = df_raw.drop(
        ['Time', 'Class'],
        axis=1
    )

    y_true = df_raw['Class']

    # ==================================================
    # SPLIT DATA
    # ==================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_original,
        y_true,
        test_size=0.2,
        random_state=22,
        stratify=y_true
    )

    st.subheader("📌 Pembagian Dataset")

    st.write(
        f"Data Training: {X_train.shape[0]} baris"
    )

    st.write(
        f"Data Testing: {X_test.shape[0]} baris"
    )

    # ==================================================
    # NORMALISASI
    # ==================================================
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # ==================================================
    # INPUT PARAMETER
    # ==================================================
    st.subheader("⚙️ Parameter Isolation Forest")

    n_estimators = st.selectbox(
        "n_estimators",
       [100, 200, 300, 400, 500]
    )

    contamination = st.selectbox(
        "contamination",
        [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 'auto']
    )

    # ==================================================
    # MODEL
    # ==================================================
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=22
    )

    # ==================================================
    # TRAINING MODEL
    # ==================================================
    model.fit(X_train_scaled)

    # ==================================================
    # PREDIKSI TRAINING
    # ==================================================
    anomaly_score = model.decision_function(
        X_train_scaled
    )

    y_pred = model.predict(X_train_scaled)

    y_pred_binary = [
        1 if x == -1 else 0
        for x in y_pred
    ]

    y_pred_label = [
        'Anomaly' if x == -1 else 'Normal'
        for x in y_pred
    ]

    # ==================================================
    # DATAFRAME HASIL
    # ==================================================
    df_train_result = X_train.copy()

    df_train_result['Actual_Class'] = y_train.values

    df_train_result['Prediction'] = y_pred_label

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================
    cm = confusion_matrix(
        y_train,
        y_pred_binary
    )

    st.subheader("📊 Confusion Matrix Training")

    fig2, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal', 'Fraud'],
        yticklabels=['Normal', 'Fraud']
    )

    ax.set_title("Confusion Matrix Training")

    st.pyplot(fig2)

    # ==================================================
    # EVALUASI ANOMALI
    # ==================================================
    precision = precision_score(
        y_train,
        y_pred_binary
    )

    recall = recall_score(
        y_train,
        y_pred_binary
    )

    f1 = f1_score(
        y_train,
        y_pred_binary
    )

    auc = roc_auc_score(
        y_train,
        -anomaly_score
    )

    # ==================================================
    # EVALUASI NORMAL
    # ==================================================
    precision_normal = precision_score(
        y_train,
        y_pred_binary,
        pos_label=0
    )

    recall_normal = recall_score(
        y_train,
        y_pred_binary,
        pos_label=0
    )

    f1_normal = f1_score(
        y_train,
        y_pred_binary,
        pos_label=0
    )

    # ==================================================
    # TAMPILKAN EVALUASI
    # ==================================================
    st.subheader("📈 Evaluasi Kelas Anomali")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Precision",
        f"{precision:.4f}"
    )

    col2.metric(
        "Recall",
        f"{recall:.4f}"
    )

    col3.metric(
        "F1-Score",
        f"{f1:.4f}"
    )

    col4.metric(
        "ROC-AUC",
        f"{auc:.4f}"
    )

    st.subheader("📈 Evaluasi Kelas Normal")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Precision Normal",
        f"{precision_normal:.4f}"
    )

    col2.metric(
        "Recall Normal",
        f"{recall_normal:.4f}"
    )

    col3.metric(
        "F1 Normal",
        f"{f1_normal:.4f}"
    )

else:

    st.info(
        "Silakan upload dataset terlebih dahulu."
    )