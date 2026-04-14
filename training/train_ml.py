"""
training/train_ml.py
─────────────────────
Trains three scikit-learn models on the synthetic fitness dataset:
  1. Linear Regression  → predict TDEE calories
  2. Random Forest Regressor → predict 30-day weight change
  3. Logistic Regression      → classify fitness level (0/1/2)

Also builds a simple content-based recommendation engine.

Run:
    cd apex-ai-project
    python training/train_ml.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score, classification_report
)
from sklearn.pipeline import Pipeline

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "ai_models" / "ml_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── LOAD DATASETS ────────────────────────────────────────────────────────────
def load_fitness():
    path = DATA_DIR / "fitness_profiles.csv"
    if not path.exists():
        print("⚠️  Dataset missing — generating now …")
        sys.path.insert(0, str(DATA_DIR))
        from generate_datasets import generate_fitness_profiles, generate_workout_history
        generate_fitness_profiles()
        generate_workout_history()
    return pd.read_csv(path)

def load_workouts():
    return pd.read_csv(DATA_DIR / "workout_history.csv")


# ─── 1. CALORIE PREDICTION (Linear Regression) ────────────────────────────────
def train_calorie_model(df: pd.DataFrame):
    FEATURES = ["age", "weight_kg", "height_cm", "activity_level", "gender"]
    TARGET   = "calories_tdee"

    X = df[FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LinearRegression())
    ])
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    print(f"\n📈 Calorie Model (Linear Regression)")
    print(f"   MAE : {mean_absolute_error(y_te, preds):.1f} kcal")
    print(f"   R²  : {r2_score(y_te, preds):.4f}")

    path = MODEL_DIR / "calorie_regression.pkl"
    joblib.dump(model, path)
    print(f"   ✅  Saved → {path}")
    return model


# ─── 2. WEIGHT CHANGE PREDICTION (Random Forest Regressor) ────────────────────
def train_weight_model(df: pd.DataFrame):
    FEATURES = ["age", "weight_kg", "height_cm", "activity_level", "gender",
                "calories_tdee"]
    TARGET   = "weight_change_30d"

    X = df[FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestRegressor(n_estimators=150, max_depth=8,
                                         random_state=42, n_jobs=-1))
    ])
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    print(f"\n⚖️  Weight Change Model (Random Forest Regressor)")
    print(f"   MAE : {mean_absolute_error(y_te, preds):.3f} kg")
    print(f"   R²  : {r2_score(y_te, preds):.4f}")

    path = MODEL_DIR / "weight_regression.pkl"
    joblib.dump(model, path)
    print(f"   ✅  Saved → {path}")
    return model


# ─── 3. FITNESS LEVEL CLASSIFIER (Logistic Regression + RF) ───────────────────
def train_fitness_classifier(df: pd.DataFrame):
    FEATURES = ["age", "weight_kg", "height_cm", "activity_level", "gender"]
    TARGET   = "fitness_level"
    LABELS   = ["Beginner", "Intermediate", "Advanced"]

    X = df[FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)

    # Primary: Logistic Regression
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=500, random_state=42))
    ])
    lr_model.fit(X_tr, y_tr)
    lr_acc = accuracy_score(y_te, lr_model.predict(X_te))

    # Secondary: Random Forest (usually better)
    rf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=150, random_state=42,
                                          n_jobs=-1))
    ])
    rf_model.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_te, rf_model.predict(X_te))

    best = rf_model if rf_acc >= lr_acc else lr_model
    best_name = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
    best_acc  = max(rf_acc, lr_acc)

    print(f"\n🏅 Fitness Classifier")
    print(f"   Logistic Regression accuracy : {lr_acc:.4f}")
    print(f"   Random Forest accuracy        : {rf_acc:.4f}")
    print(f"   Best model → {best_name}  ({best_acc:.4f})")
    print(classification_report(y_te, best.predict(X_te),
                                 target_names=LABELS, zero_division=0))

    path = MODEL_DIR / "fitness_classifier.pkl"
    joblib.dump(best, path)
    print(f"   ✅  Saved → {path}")

    # Save label map for the API
    label_map = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}
    joblib.dump(label_map, MODEL_DIR / "fitness_classifier_labels.pkl")
    return best


# ─── 4. RECOMMENDATION ENGINE (collaborative filtering lite) ──────────────────
def train_recommender(df_workouts: pd.DataFrame):
    """
    Content-based recommender:
    - For each fitness level compute the mean rating per exercise
    - At inference time, return top-K exercises for a given fitness level
    """
    rec = (
        df_workouts
        .groupby(["fitness_level", "exercise"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "avg_rating"})
    )

    # Also store difficulty for filtering
    diff = (
        df_workouts[["exercise", "difficulty", "muscle_group"]]
        .drop_duplicates()
    )
    rec = rec.merge(diff, on="exercise")
    rec_dict = {}
    for lvl in [0, 1, 2]:
        subset = (
            rec[rec["fitness_level"] == lvl]
            .sort_values("avg_rating", ascending=False)
            .head(10)
            .to_dict("records")
        )
        rec_dict[lvl] = subset

    path = MODEL_DIR / "recommender.pkl"
    joblib.dump(rec_dict, path)
    print(f"\n🔀 Recommender trained for 3 fitness levels")
    print(f"   ✅  Saved → {path}")
    return rec_dict


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  APEX AI — ML Model Training")
    print("=" * 60)

    df_fitness  = load_fitness()
    df_workouts = load_workouts()

    print(f"\nDataset loaded: {len(df_fitness)} fitness profiles, "
          f"{len(df_workouts)} workout logs")

    train_calorie_model(df_fitness)
    train_weight_model(df_fitness)
    train_fitness_classifier(df_fitness)
    train_recommender(df_workouts)

    print("\n" + "=" * 60)
    print("  ✅ All ML models trained and saved to ai_models/ml_models/")
    print("=" * 60)
