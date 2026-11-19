"""
Intelligent Fault Detection
---------------------------
A simple machine-learning project that detects and classifies faults in a
machine (e.g. an electric motor) from sensor readings.

Pipeline:
  1. Generate (simulated) sensor data  -> replace with your own CSV if you have one
  2. Pre-process (scale features)
  3. Train two models:
       a) Isolation Forest  -> unsupervised "is something wrong?" (anomaly detection)
       b) Random Forest     -> supervised "what kind of fault is it?" (classification)
  4. Evaluate and visualise results
  5. Predict on new live sensor readings

Install requirements:
    pip install numpy pandas scikit-learn matplotlib joblib

Run:
    python fault_detection.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

FEATURES = ["vibration", "temperature", "current", "voltage", "rpm"]
LABELS = {0: "Normal", 1: "Bearing Fault", 2: "Overheating", 3: "Overload"}


# ----------------------------------------------------------------------------
# 1. DATA GENERATION (simulated sensor data)
# ----------------------------------------------------------------------------
def generate_data(n_per_class: int = 500) -> pd.DataFrame:
    """Create synthetic sensor readings for each machine condition."""
    rows = []

    # (vibration mm/s, temperature C, current A, voltage V, rpm) -> mean values
    profiles = {
        0: dict(mean=[2.0, 55, 10, 230, 1500], std=[0.3, 3, 0.8, 3, 15]),   # Normal
        1: dict(mean=[6.5, 62, 11, 229, 1470], std=[1.0, 4, 1.0, 3, 25]),   # Bearing fault
        2: dict(mean=[2.8, 95, 12, 228, 1490], std=[0.5, 6, 1.0, 3, 20]),   # Overheating
        3: dict(mean=[3.5, 75, 18, 215, 1380], std=[0.6, 5, 1.5, 5, 40]),   # Overload
    }

    for label, p in profiles.items():
        data = rng.normal(p["mean"], p["std"], size=(n_per_class, len(FEATURES)))
        df = pd.DataFrame(data, columns=FEATURES)
        df["fault"] = label
        rows.append(df)

    df = pd.concat(rows, ignore_index=True)
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. TRAINING
# ----------------------------------------------------------------------------
def train_models(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df["fault"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # a) Anomaly detector: trained ONLY on normal data
    normal_train = X_train_s[y_train == 0]
    anomaly_model = IsolationForest(
        n_estimators=200, contamination=0.02, random_state=RANDOM_STATE
    )
    anomaly_model.fit(normal_train)

    # b) Fault classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_train_s, y_train)

    return scaler, anomaly_model, clf, X_test_s, y_test


# ----------------------------------------------------------------------------
# 3. EVALUATION
# ----------------------------------------------------------------------------
def evaluate(anomaly_model, clf, X_test_s, y_test):
    # Classifier
    y_pred = clf.predict(X_test_s)
    print("=" * 60)
    print("FAULT CLASSIFIER (Random Forest)")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=list(LABELS.values())))

    # Anomaly detector (1 = normal, -1 = anomaly)
    anomaly_pred = anomaly_model.predict(X_test_s)
    detected_fault = anomaly_pred == -1
    actual_fault = y_test != 0
    detection_rate = (detected_fault & actual_fault).sum() / actual_fault.sum()
    false_alarm = (detected_fault & ~actual_fault).sum() / (~actual_fault).sum()
    print("=" * 60)
    print("ANOMALY DETECTOR (Isolation Forest)")
    print("=" * 60)
    print(f"Fault detection rate : {detection_rate * 100:.2f}%")
    print(f"False alarm rate     : {false_alarm * 100:.2f}%\n")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=list(LABELS.values())).plot(
        cmap="Blues", xticks_rotation=30
    )
    plt.title("Fault Classification - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.close()

    # Feature importance plot
    importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
    importances.plot(kind="barh", color="teal")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print("Saved plots: confusion_matrix.png, feature_importance.png")


# ----------------------------------------------------------------------------
# 4. PREDICTION ON NEW READINGS
# ----------------------------------------------------------------------------
def predict_fault(reading: dict, scaler, anomaly_model, clf) -> str:
    """Predict the condition from a single sensor reading."""
    x = pd.DataFrame([reading])[FEATURES].values
    x_s = scaler.transform(x)

    is_anomaly = anomaly_model.predict(x_s)[0] == -1
    label = clf.predict(x_s)[0]
    confidence = clf.predict_proba(x_s)[0].max() * 100

    status = "ANOMALY DETECTED" if is_anomaly else "Looks normal"
    return f"{status} | Diagnosis: {LABELS[label]} ({confidence:.1f}% confidence)"


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # To use your own data instead, load a CSV with columns:
    #   vibration, temperature, current, voltage, rpm, fault
    # df = pd.read_csv("your_data.csv")
    df = generate_data()
    df.to_csv("sensor_data.csv", index=False)
    print(f"Dataset created: {df.shape[0]} samples\n")

    scaler, anomaly_model, clf, X_test_s, y_test = train_models(df)
    evaluate(anomaly_model, clf, X_test_s, y_test)

    # Save trained models for later use
    joblib.dump({"scaler": scaler, "anomaly": anomaly_model, "clf": clf}, "fault_model.joblib")
    print("Model saved to fault_model.joblib\n")

    # Demo: predict on new readings
    print("=" * 60)
    print("LIVE PREDICTION DEMO")
    print("=" * 60)
    samples = {
        "Healthy motor": dict(vibration=2.1, temperature=54, current=10.2, voltage=231, rpm=1498),
        "Worn bearing":  dict(vibration=7.0, temperature=63, current=11.0, voltage=229, rpm=1465),
        "Hot motor":     dict(vibration=2.9, temperature=98, current=12.1, voltage=227, rpm=1488),
        "Heavy load":    dict(vibration=3.6, temperature=76, current=18.5, voltage=214, rpm=1375),
    }
    for name, reading in samples.items():
        print(f"{name:14s}: {predict_fault(reading, scaler, anomaly_model, clf)}")
