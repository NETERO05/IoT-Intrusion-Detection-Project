"""
IoT Intrusion Detection: Machine Learning Evaluation Pipeline
Author: Muhammed Suhaib Hussain
Student ID: 33105107
University of West London

Project:
Assessing Cybersecurity Risks in Internet of Things Networks Using Machine Learning

Description:
This code loads the Edge-IIoTset dataset, prepares the data, removes leakage-prone
columns, trains five machine learning models, and evaluates them using cybersecurity
metrics such as accuracy, precision, recall, F1-score and false positive rate.
"""

# -------------------------------
# 1. SETUP AND DATA EXTRACTION
# -------------------------------

print("Setting up the environment...")

!pip install xgboost -q

from google.colab import drive
drive.mount('/content/drive')

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ZIP_PATH = "/content/drive/MyDrive/archive.zip"
DATA_DIR = "/content/edge_iiotset"
RESULTS_DIR = "/content/drive/MyDrive/dissertation_results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

dataset_file = os.path.join(DATA_DIR, "ML-EdgeIIoT-dataset.csv")

if not os.path.exists(dataset_file):
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Dataset extracted successfully.")
else:
    print("Dataset already extracted.")


# -------------------------------
# 2. IMPORT MACHINE LEARNING TOOLS
# -------------------------------

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


# -------------------------------
# 3. LOAD DATA
# -------------------------------

def load_and_sample_data(csv_path, sample_size=100000):
    """
    Loads the Edge-IIoTset dataset and takes a 100,000-sample working subset.
    """

    print("\nLoading dataset.")
    df_big = pd.read_csv(csv_path, nrows=500000, low_memory=False)

    print("Original loaded shape:", df_big.shape)
    print("\nAttack label distribution before sampling:")
    print(df_big["Attack_label"].value_counts())

    df_sample = df_big.sample(n=sample_size, random_state=42)

    print(f"\nWorking sample created: {sample_size} rows")
    print("Sample attack label distribution:")
    print(df_sample["Attack_label"].value_counts())

    return df_sample


# -------------------------------
# 4. REMOVE LEAKAGE-PRONE COLUMNS
# -------------------------------

def remove_data_leakage(df):
    """
    Removes columns that could cause data leakage.

    Attack_label is removed from the input features, but it is still used as
    the target variable for supervised learning.
    """

    print("\nRemoving leakage-prone columns and keeping behavioural features.")

    drop_cols = [
        "Attack_label",
        "Attack_type",

        "ip.src_host",
        "ip.dst_host",

        "http.file_data",
        "http.request.full_uri",
        "http.request.uri.query",
        "tcp.payload",
        "mqtt.topic",
        "mqtt.msg",
        "mqtt.msg_decoded_as",
        "mqtt.protoname",

        "http.response",
        "http.request.method",
        "http.referer",
        "http.request.version",
        "dns.qry.name",
        "dns.qry.name.len",

        "tcp.flags",
        "tcp.flags.ack",
        "tcp.connection.syn",
        "tcp.connection.fin",
        "tcp.connection.rst",
        "tcp.connection.synack",
        "tcp.srcport",
        "tcp.dstport",
        "udp.port",

        "frame.time"
    ]

    y = df["Attack_label"]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    print("Feature shape before numeric filter:", X.shape)

    X_numeric = X.select_dtypes(include=["int64", "float64"])

    print("Feature shape after numeric filter:", X_numeric.shape)
    print("Remaining numerical features:")
    print(X_numeric.columns.tolist())

    return X_numeric, y


# -------------------------------
# 5. EVALUATE MODEL
# -------------------------------

def evaluate_model(name, y_true, y_pred):
    """
    Evaluates each model using accuracy, precision, recall, F1-score
    and false positive rate.
    """

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n---------- {name} Results ----------")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")

    return {
        "Model": name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "FPR": fpr
    }


# -------------------------------
# 6. MAIN PIPELINE
# -------------------------------

def main():
    df = load_and_sample_data(CSV_PATH)
    X, y = remove_data_leakage(df)
    feature_names = X.columns.tolist()

    print("\nTarget values:")
    print(y.value_counts())

    print("\nSplitting data into training and testing sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
        shuffle=True
    )

    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    np.random.seed(42)

    X_train_noisy = X_train + np.random.normal(0, 0.01, X_train.shape)
    X_test_noisy = X_test + np.random.normal(0, 0.01, X_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_noisy)
    X_test_scaled = scaler.transform(X_test_noisy)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        "SVM": SVC(
            class_weight="balanced"
        ),

        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    print("\nStarting model training...")

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if name in ["Logistic Regression", "SVM"]:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train_noisy, y_train)
            predictions = model.predict(X_test_noisy)

        trained_models[name] = model
        results.append(evaluate_model(name, y_test, predictions))

        ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(
            os.path.join(RESULTS_DIR, f"{name.replace(' ', '_').lower()}_confusion_matrix.png"),
            bbox_inches="tight"
        )
        plt.show()

    results_df = pd.DataFrame(results).sort_values(by="F1-score", ascending=False)

    print("\n---------- FINAL MODEL COMPARISON ----------")
    print(results_df)

    results_path = os.path.join(RESULTS_DIR, "model_performance_summary.csv")
    results_df.to_csv(results_path, index=False)

    print(f"\nResults saved to: {results_path}")

    plot_df = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "FPR"]]

    plot_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "model_performance_comparison.png"),
        bbox_inches="tight"
    )
    plt.show()

    print("\nGenerating Random Forest feature importance...")

    rf_model = trained_models["Random Forest"]

    rf_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Random Forest Features:")
    print(rf_importances.head(10))

    plt.figure(figsize=(10, 6))
    plt.barh(
        rf_importances["Feature"].head(10)[::-1],
        rf_importances["Importance"].head(10)[::-1]
    )
    plt.title("Top 10 Feature Importances - Random Forest")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "random_forest_feature_importance.png"),
        bbox_inches="tight"
    )
    plt.show()

    print("\nGenerating XGBoost feature importance...")

    xgb_model = trained_models["XGBoost"]

    xgb_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": xgb_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 10 XGBoost Features:")
    print(xgb_importances.head(10))

    plt.figure(figsize=(10, 6))
    plt.barh(
        xgb_importances["Feature"].head(10)[::-1],
        xgb_importances["Importance"].head(10)[::-1]
    )
    plt.title("Top 10 Feature Importances - XGBoost")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "xgboost_feature_importance.png"),
        bbox_inches="tight"
    )
    plt.show()

    print("\nPipeline complete. All results and charts have been saved to Google Drive.")


# -------------------------------
# 7. RUN THE PIPELINE
# -------------------------------

if __name__ == "__main__":
    main()