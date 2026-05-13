"""
IoT Intrusion Detection: Machine Learning Evaluation Pipeline
Author: Muhammed Suhaib Hussain, 33105107
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. COLAB SETUP: MOUNT DRIVE & UNZIP DATA ---
print("[INFO] Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

print("[INFO] Unzipping dataset...")
ZIP_PATH = "/content/drive/MyDrive/archive.zip"
DATA_DIR = "/content/edge_iiotset"
RESULTS_DIR = "/content/drive/MyDrive/dissertation_results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)
print("[INFO] Unzip complete!")

# --- 2. PIPELINE IMPORTS ---
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

CSV_PATH = os.path.join(DATA_DIR, "ML-EdgeIIoT-dataset.csv")

# --- 3. PIPELINE FUNCTIONS ---
def load_and_sample_data(csv_path, sample_size=100000):
    print("\n[INFO] Loading dataset...")
    df_big = pd.read_csv(csv_path, nrows=500000, low_memory=False)
    print(f"[INFO] Original Dataset Shape: {df_big.shape}")
    
    df_sample = df_big.sample(n=sample_size, random_state=42)
    print(f"[INFO] Working Sample Shape: {df_sample.shape}")
    
    return df_sample

def remove_data_leakage(df):
    print("[INFO] Removing leakage-prone features...")
    
    drop_cols = [
        "Attack_label", "Attack_type",
        "ip.src_host", "ip.dst_host",                  
        "http.file_data", "http.request.full_uri",     
        "http.request.uri.query", "tcp.payload", 
        "mqtt.topic", "mqtt.msg", "mqtt.msg_decoded_as", "mqtt.protoname",
        "http.response", "http.request.method",        
        "http.referer", "http.request.version", 
        "dns.qry.name", "dns.qry.name.len",
        "tcp.flags", "tcp.flags.ack", "tcp.connection.syn", 
        "tcp.connection.fin", "tcp.connection.rst", 
        "tcp.connection.synack", "tcp.srcport", "tcp.dstport", "udp.port",
        "frame.time"                                   
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["Attack_label"]

    X_numeric = X.select_dtypes(include=["int64", "float64"])
    print(f"[INFO] Features remaining after cleaning: {X_numeric.shape[1]}")
    return X_numeric, y

def evaluate_model(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n--- {name} Performance ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")

    return {
        "Model": name,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "FPR": fpr
    }

def main():
    df = load_and_sample_data(CSV_PATH)
    X, y = remove_data_leakage(df)
    feature_names = X.columns.tolist()

    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True
    )

    print("[INFO] Injecting noise...")
    X_train_noisy = X_train + np.random.normal(0, 0.01, X_train.shape)
    X_test_noisy = X_test + np.random.normal(0, 0.01, X_test.shape)

    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_noisy)
    X_test_scaled = scaler.transform(X_test_noisy)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1),
        "SVM": SVC(class_weight="balanced"),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=-1)
    }

    results = []
    trained_models = {}

    print("[INFO] Commencing model training...")
    for name, model in models.items():
        if name in ["Logistic Regression", "SVM"]:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train_noisy, y_train)
            preds = model.predict(X_test_noisy)
            
        trained_models[name] = model
        results.append(evaluate_model(name, y_test, preds))

        ConfusionMatrixDisplay.from_predictions(y_test, preds)
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(os.path.join(RESULTS_DIR, f"{name.replace(' ', '_').lower()}_cm.png"), bbox_inches="tight")
        plt.show()

    results_df = pd.DataFrame(results).sort_values(by="F1-score", ascending=False)
    results_csv_path = os.path.join(RESULTS_DIR, "model_results_expanded.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n[INFO] Results successfully saved to: {results_csv_path}")

    plot_df = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "FPR"]]
    plot_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Expanded Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "expanded_model_performance.png"), bbox_inches="tight")
    plt.show()

    print("[INFO] Extracting Feature Importances...")
    xgb_model = trained_models["XGBoost"]
    xgb_importances = pd.DataFrame({"Feature": feature_names, "Importance": xgb_model.feature_importances_}).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(xgb_importances["Feature"].head(10)[::-1], xgb_importances["Importance"].head(10)[::-1])
    plt.title("Top 10 Feature Importances - XGBoost")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "xgb_feature_importance.png"), bbox_inches="tight")
    plt.show()

    print("\n[SUCCESS] Pipeline complete! Graphs displayed and saved to your Google Drive.")

if __name__ == "__main__":
    main()