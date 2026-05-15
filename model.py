import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURES = [
    "violation_count",
    "storage_overflow_pct",
    "months_since_inspection",
    "worker_complaints",
    "accident_history",
]

RISK_BINS   = [0, 40, 70, 100]
RISK_LABELS = ["Safe", "Moderate", "Critical"]

MODEL_PATH  = "data/risk_model.pkl"
SCALER_PATH = "data/scaler.pkl"


def compute_raw_score(row):
    """Deterministic weighted score used to create training labels."""
    score = 0
    score += min(row["violation_count"] * 8, 30)
    overflow = row["storage_overflow_pct"]
    score += min(max(overflow, 0) * 0.5, 25)
    score += min(row["months_since_inspection"] * 0.8, 20)
    score += min(row["worker_complaints"] * 3, 15)
    score += min(row["accident_history"] * 5, 20)
    noise  = np.random.normal(0, 3)
    return float(np.clip(score + noise, 0, 100))


def train(data_dir="data"):
    from data_generator import save_all
    csv_path = f"{data_dir}/factories.csv"
    if not os.path.exists(csv_path):
        save_all(data_dir)

    df = pd.read_csv(csv_path)

    np.random.seed(42)
    df["raw_score"] = df.apply(compute_raw_score, axis=1)
    df["risk_label"] = pd.cut(
        df["raw_score"], bins=RISK_BINS, labels=RISK_LABELS, include_lowest=True
    )

    X = df[FEATURES].fillna(0)
    y = df["risk_label"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("[model] Classification report:")
    print(classification_report(y_test, clf.predict(X_test)))

    os.makedirs(data_dir, exist_ok=True)
    with open(MODEL_PATH,  "wb") as f: pickle.dump(clf,    f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print(f"[model] Saved model → {MODEL_PATH}, scaler → {SCALER_PATH}")

    df.to_csv(f"{data_dir}/factories.csv", index=False)
    return clf, scaler


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        train()
    with open(MODEL_PATH,  "rb") as f: clf    = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    return clf, scaler


def predict_score(clf, scaler, feature_dict):
    """Return a 0-100 float risk score for a single factory's features."""
    X = pd.DataFrame([feature_dict])[FEATURES].fillna(0)
    X_scaled = scaler.transform(X)
    proba = clf.predict_proba(X_scaled)[0]
    classes = list(clf.classes_)

    weights = {"Safe": 20, "Moderate": 55, "Critical": 90}
    score = sum(proba[i] * weights[c] for i, c in enumerate(classes))
    return round(float(np.clip(score, 0, 100)), 1)


def score_to_label(score):
    if score <= 40:  return "Safe"
    if score <= 70:  return "Moderate"
    return "Critical"


if __name__ == "__main__":
    train()
