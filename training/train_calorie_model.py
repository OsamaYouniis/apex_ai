"""
training/train_calorie_model.py
────────────────────────────────
Trains an XGBoost + RandomForest ensemble calorie-burn prediction model
on real Kaggle data (Calories Burn Prediction dataset).

Features used (9):
  Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
  + engineered: BMI, BMR, HR_intensity, weight_x_duration

Outputs:
  models/calorie_model.pkl        — trained pipeline (scaler + XGBoost)
  models/calorie_feature_list.json — ordered feature names

Run:
    python training/train_calorie_model.py
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Also save to ai_models for backward-compat with existing backend
LEGACY_DIR = ROOT / "ai_models" / "ml_models"
LEGACY_DIR.mkdir(parents=True, exist_ok=True)


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

FEATURES = [
    "age", "height", "weight", "duration", "heart_rate", "body_temp",
    "gender_num", "bmi", "bmr", "hr_intensity", "weight_x_duration",
    "age_squared",
]

TARGET = "calories"


def load_and_engineer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Handle both raw (with Gender col) and preprocessed versions
    if "gender_num" not in df.columns:
        if "gender" in df.columns:
            df["gender_num"] = (df["gender"].str.lower() == "male").astype(int)
        else:
            df["gender_num"] = 1  # default

    # Compute derived features if missing
    h_m = df["height"] / 100
    if "bmi" not in df.columns:
        df["bmi"] = (df["weight"] / (h_m ** 2)).round(2)
    if "bmr" not in df.columns:
        df["bmr"] = (10 * df["weight"] + 6.25 * df["height"] -
                     5  * df["age"]   + df["gender_num"].map({1: 5, 0: -161}))
    if "age_squared" not in df.columns:
        df["age_squared"] = df["age"] ** 2
    if "weight_x_duration" not in df.columns:
        df["weight_x_duration"] = df["weight"] * df["duration"]
    if "hr_intensity" not in df.columns and "heart_rate" in df.columns:
        df["hr_intensity"] = (df["heart_rate"] / (220 - df["age"])).round(3)

    # Target column
    if "calories" not in df.columns and "calories_burned" in df.columns:
        df["calories"] = df["calories_burned"]

    # Drop outliers
    df = df[(df["calories"] > 5) & (df["calories"] < 1000)]
    df = df[(df["weight"] > 30)  & (df["weight"] < 200)]

    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


# ─── MODEL ARCHITECTURE ───────────────────────────────────────────────────────

def build_model() -> Pipeline:
    """
    Voting ensemble: GradientBoosting + RandomForest + Ridge
    Wrapped in a sklearn Pipeline with StandardScaler.
    """
    gb  = GradientBoostingRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.85, random_state=42
    )
    rf  = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    rdg = Ridge(alpha=10.0)

    ensemble = VotingRegressor([("gb", gb), ("rf", rf), ("ridge", rdg)])

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ensemble),
    ])


# ─── TRAIN & EVALUATE ─────────────────────────────────────────────────────────

def train():
    print("\n" + "=" * 60)
    print("  APEX AI — Calorie Model Training")
    print("=" * 60)

    # Load data (prefer preprocessed, fall back to raw, fall back to synthetic)
    proc_path = DATA_DIR / "calories_processed.csv"
    raw_path  = DATA_DIR / "calories_dataset.csv"

    if proc_path.exists():
        print(f"  📂 Loading preprocessed data: {proc_path}")
        df = load_and_engineer(proc_path)
    elif raw_path.exists():
        print(f"  📂 Loading raw data: {raw_path}")
        df = load_and_engineer(raw_path)
    else:
        print("  ⚠️  No dataset found — generating synthetic data …")
        sys.path.insert(0, str(DATA_DIR))
        from download_datasets import generate_calorie_dataset, preprocess_calorie_data
        df_raw = generate_calorie_dataset(15000)
        df     = load_and_engineer(proc_path) if proc_path.exists() else load_and_engineer(
            proc_path if (df_raw.to_csv(proc_path, index=False) or True) else raw_path
        )
        # simpler fallback
        df_raw = generate_calorie_dataset(15000)
        df_raw.to_csv(raw_path, index=False)
        df = load_and_engineer(raw_path)

    print(f"  📊 Dataset: {len(df):,} rows × {len(FEATURES)} features")

    # Available features (handle missing heart_rate/body_temp)
    avail = [f for f in FEATURES if f in df.columns]
    X = df[avail]
    y = df[TARGET]

    print(f"  🔧 Using features: {avail}")
    print(f"  🎯 Target range: {y.min():.0f} – {y.max():.0f} kcal")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Build and train
    model = build_model()
    print(f"\n  🚀 Training ensemble model …")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = mean_squared_error(y_test, y_pred, squared=False)
    r2     = r2_score(y_test, y_pred)

    print(f"\n  📈 Test Set Performance:")
    print(f"     MAE  : {mae:.2f} kcal")
    print(f"     RMSE : {rmse:.2f} kcal")
    print(f"     R²   : {r2:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
    print(f"\n  🔄 5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save to /models (new location)
    model_path   = MODEL_DIR / "calorie_model.pkl"
    feature_path = MODEL_DIR / "calorie_feature_list.json"
    joblib.dump(model, model_path)
    with open(feature_path, "w") as f:
        json.dump(avail, f)

    print(f"\n  ✅ Saved → {model_path}")
    print(f"  ✅ Saved → {feature_path}")

    # Also save to legacy location for backward-compat
    legacy_path = LEGACY_DIR / "calorie_regression.pkl"
    joblib.dump(model, legacy_path)
    print(f"  ✅ Legacy → {legacy_path}")

    # Sample prediction
    sample = X_test.iloc[0:1]
    pred   = model.predict(sample)[0]
    actual = y_test.iloc[0]
    print(f"\n  🔍 Sample: predicted={pred:.0f} kcal | actual={actual:.0f} kcal | "
          f"error={abs(pred-actual):.0f} kcal")

    print("\n" + "=" * 60)
    print("  ✅ Calorie model training complete!")
    print("=" * 60 + "\n")

    return model, avail


if __name__ == "__main__":
    train()
