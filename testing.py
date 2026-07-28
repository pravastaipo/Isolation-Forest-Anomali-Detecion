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
    page_title="Testing IForest Bersih",
    layout="centered"
)

st.title("Testing Model Isolation Forest")

# ==================================================
# FUNGSI HAPUS DUPLIKAT
# ==================================================
def handle_duplicate(dataframe):

    duplicate_rows = dataframe.duplicated().sum()

    if duplicate_rows > 0:

        cleaned_df = dataframe.drop_duplicates()

        st.warning(
            f"⚠️ Ditemukan {duplicate_rows} data duplikat. Data telah dibersihkan."
        )

    else:

        cleaned_df = dataframe

        st.success(
            "✅ Tidak ditemukan data duplikat."
        )

    return cleaned_df, duplicate_rows

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

    st.subheader("📊 Informasi Dataset")

    st.write(
        f"Jumlah Baris: {df_raw.shape[0]}"
    )

    st.write(
        f"Jumlah Kolom: {df_raw.shape[1]}"
    )

    # ==================================================
    # HAPUS DUPLIKAT
    # ==================================================
    df_no_duplicates, duplicate_rows = handle_duplicate(df_raw)

    # ==================================================
    # FITUR DAN LABEL
    # ==================================================
    X_original = df_no_duplicates.drop(
        ['Time', 'Class'],
        axis=1
    )

    y_true = df_no_duplicates['Class']

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

    X_test_scaled = scaler.transform(X_test)

    # ==================================================
    # PARAMETER OPTIMAL
    # ==================================================
    model = IsolationForest(
        n_estimators=300,
        contamination=0.002,
        random_state=22
    )

    # ==================================================
    # TRAINING MODEL
    # ==================================================
    model.fit(X_train_scaled)

    # ==================================================
    # TESTING MODEL
    # ==================================================
    anomaly_score = model.decision_function(
        X_test_scaled
    )

    y_pred = model.predict(X_test_scaled)

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
    df_test_result = X_test.copy()

    df_test_result['Actual_Class'] = y_test.values

    df_test_result['Prediction'] = y_pred_label

    st.subheader("📄 Hasil Prediksi Testing")

    st.dataframe(
        df_test_result.head(100)
    )

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================
    cm = confusion_matrix(
        y_test,
        y_pred_binary
    )

    st.subheader("📊 Confusion Matrix Testing")

    fig2, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal', 'Fraud'],
        yticklabels=['Normal', 'Fraud']
    )

    ax.set_title("Confusion Matrix Testing")

    st.pyplot(fig2)

    # ==================================================
    # EVALUASI ANOMALI
    # ==================================================
    precision = precision_score(
        y_test,
        y_pred_binary
    )

    recall = recall_score(
        y_test,
        y_pred_binary
    )

    f1 = f1_score(
        y_test,
        y_pred_binary
    )

    auc = roc_auc_score(
        y_test,
        -anomaly_score
    )

    # ==================================================
    # EVALUASI NORMAL
    # ==================================================
    precision_normal = precision_score(
        y_test,
        y_pred_binary,
        pos_label=0
    )

    recall_normal = recall_score(
        y_test,
        y_pred_binary,
        pos_label=0
    )

    f1_normal = f1_score(
        y_test,
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


    # ====== Tombol Download ======
    csv = df_test_result.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Hasil CSV",
        data=csv,
        file_name="hasil_test_bersih.csv",
        mime="text/csv"
    )

else:

    st.info(
        "Silakan upload dataset terlebih dahulu."
    )