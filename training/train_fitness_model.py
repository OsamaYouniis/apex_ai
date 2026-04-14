"""
training/train_fitness_model.py
────────────────────────────────
Trains an XGBoost fitness performance classifier on the
Kaggle Body Performance Dataset (A/B/C/D → 4-class).

Also maps A/B/C/D → Advanced/Intermediate/Beginner/Untrained
for compatibility with the existing APEX API.

Features: age, gender, height, weight, body_fat, grip_force,
          sit_bends, situps, broad_jump + engineered: BMI, BMR,
          strength_index, flexibility_idx

Outputs:
  models/fitness_model.pkl         — XGBoost pipeline
  models/fitness_label_map.json    — label mapping
  models/fitness_feature_list.json — feature names

Run:
    python training/train_fitness_model.py
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LEGACY_DIR = ROOT / "ai_models" / "ml_models"
LEGACY_DIR.mkdir(parents=True, exist_ok=True)

# 4-class body performance: A=3 (best) → D=0 (lowest)
CLASS_MAP = {"A": 3, "B": 2, "C": 1, "D": 0}
LABEL_MAP = {3: "Advanced", 2: "Intermediate", 1: "Beginner", 0: "Untrained"}

# Simplified 3-class for legacy API
LEGACY_LABEL_MAP = {3: "Advanced", 2: "Intermediate", 1: "Intermediate", 0: "Beginner"}

FEATURES = [
    "age", "gender_num", "height_cm", "weight_kg",
    "bmi", "bmr", "age_squared",
    "gripforce", "sit-ups_counts", "broad_jump_cm",
    "sit_and_bend_forward_cm",
    "strength_index", "flexibility_idx",
]

TARGET = "class_num"


def load_and_engineer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Encode gender
    if "gender_num" not in df.columns:
        df["gender_num"] = (df["gender"].str.upper() == "M").astype(int)

    # Encode class
    if "class_num" not in df.columns:
        df["class_num"] = df["class"].str.upper().map(CLASS_MAP).fillna(1).astype(int)

    # Rename body_fat column if needed
    for old, new in [("body_fat_%", "body_fat"), ("body fat_%", "body_fat")]:
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Rename grip column variants
    for old, new in [("gripforce", "gripforce"), ("grip_force", "gripforce")]:
        if old in df.columns and "gripforce" not in df.columns:
            df.rename(columns={old: "gripforce"}, inplace=True)

    # Computed features
    h_m = df["height_cm"] / 100
    if "bmi" not in df.columns:
        df["bmi"] = (df["weight_kg"] / (h_m ** 2)).round(2)
    if "bmr" not in df.columns:
        df["bmr"] = (10 * df["weight_kg"] + 6.25 * df["height_cm"] -
                     5  * df["age"]       + df["gender_num"].map({1: 5, 0: -161}))
    if "age_squared" not in df.columns:
        df["age_squared"] = df["age"] ** 2
    if "strength_index" not in df.columns:
        df["strength_index"] = (df.get("gripforce", 30) +
                                 df.get("broad_jump_cm", 150) / 3).round(2)
    if "flexibility_idx" not in df.columns:
        df["flexibility_idx"] = df.get("sit_and_bend_forward_cm", 10).clip(-10, 40)

    return df.dropna(subset=["class_num"]).reset_index(drop=True)


def build_model() -> Pipeline:
    """
    Soft-voting ensemble of GradientBoosting + RandomForest + LogisticRegression.
    Wrapped in StandardScaler pipeline.
    """
    gb  = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08,
        subsample=0.85, random_state=42
    )
    rf  = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    lr  = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    ensemble = VotingClassifier(
        [("gb", gb), ("rf", rf), ("lr", lr)],
        voting="soft"
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ensemble),
    ])


def train():
    print("\n" + "=" * 60)
    print("  APEX AI — Fitness Classifier Training")
    print("=" * 60)

    proc_path = DATA_DIR / "body_performance_processed.csv"
    raw_path  = DATA_DIR / "body_performance.csv"

    if proc_path.exists():
        print(f"  📂 Loading preprocessed data: {proc_path}")
        df = load_and_engineer(proc_path)
    elif raw_path.exists():
        print(f"  📂 Loading raw data: {raw_path}")
        df = load_and_engineer(raw_path)
    else:
        print("  ⚠️  No dataset — generating synthetic data …")
        sys.path.insert(0, str(DATA_DIR))
        from download_datasets import generate_body_performance_dataset
        df_raw = generate_body_performance_dataset(13393)
        df_raw.to_csv(raw_path, index=False)
        df = load_and_engineer(raw_path)

    print(f"  📊 Dataset: {len(df):,} rows")
    print(f"  🏷  Class distribution:\n{df['class_num'].value_counts().sort_index().to_dict()}")

    avail = [f for f in FEATURES if f in df.columns]
    X = df[avail]
    y = df[TARGET]

    print(f"  🔧 Features ({len(avail)}): {avail}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    model = build_model()
    print(f"\n  🚀 Training ensemble classifier …")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  📈 Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    print(f"\n  📋 Classification Report:")
    label_names = [LABEL_MAP.get(i, str(i)) for i in sorted(y.unique())]
    print(classification_report(y_test, y_pred, target_names=label_names))

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  🔄 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save to /models
    model_path   = MODEL_DIR / "fitness_model.pkl"
    feature_path = MODEL_DIR / "fitness_feature_list.json"
    label_path   = MODEL_DIR / "fitness_label_map.json"

    joblib.dump(model, model_path)
    with open(feature_path, "w") as f:
        json.dump(avail, f)
    with open(label_path, "w") as f:
        json.dump({str(k): v for k, v in LABEL_MAP.items()}, f)

    print(f"\n  ✅ Saved → {model_path}")

    # Legacy — 3-class compatible
    legacy_path        = LEGACY_DIR / "fitness_classifier.pkl"
    legacy_labels_path = LEGACY_DIR / "fitness_classifier_labels.pkl"
    joblib.dump(model, legacy_path)
    joblib.dump({0: "Beginner", 1: "Intermediate", 2: "Advanced"}, legacy_labels_path)
    print(f"  ✅ Legacy → {legacy_path}")

    print("\n" + "=" * 60)
    print("  ✅ Fitness model training complete!")
    print("=" * 60 + "\n")

    return model, avail


if __name__ == "__main__":
    train()
