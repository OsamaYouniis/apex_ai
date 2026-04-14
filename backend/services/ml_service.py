"""
backend/services/ml_service.py  (v5 — richer features)
────────────────────────────────────────────────────────
Loads trained scikit-learn models. Uses 9 features (was 5):
  age, weight_kg, height_cm, activity_level, gender,
  bmi, bmr, age_squared, weight_height  (+ calories_tdee for weight model)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import joblib
    _JOBLIB = True
except ImportError:
    _JOBLIB = False

ROOT      = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT / "ai_models" / "ml_models"


def _load(filename: str):
    p = MODEL_DIR / filename
    if not p.exists():
        print(f"⚠️  {filename} not found — run notebooks/ml_models.ipynb")
        return None
    return joblib.load(p) if _JOBLIB else None


# ── Load once at import ───────────────────────────────────────────────────────
calorie_model  = _load("calorie_regression.pkl")
weight_model   = _load("weight_regression.pkl")
fitness_clf    = _load("fitness_classifier.pkl")
fitness_labels = _load("fitness_classifier_labels.pkl") or {0:"Beginner",1:"Intermediate",2:"Advanced"}
recommender    = _load("recommender.pkl")


# ── Feature builder ────────────────────────────────────────────────────────────

def _build_features(age: int, weight_kg: float, height_cm: float,
                    activity_level: int, gender: int,
                    calories_tdee: Optional[float] = None) -> pd.DataFrame:
    """Build enriched feature dataframe for all models."""
    h_m = height_cm / 100
    bmi = weight_kg / (h_m ** 2)
    bmr = (10 * weight_kg + 6.25 * height_cm - 5 * age +
           (5 if gender == 1 else -161))

    base = {
        "age":           age,
        "weight_kg":     weight_kg,
        "height_cm":     height_cm,
        "activity_level": activity_level,
        "gender":        gender,
        "bmi":           bmi,
        "bmr":           bmr,
        "age_squared":   age ** 2,
        "weight_height": weight_kg / height_cm,
    }
    if calories_tdee is not None:
        base["calories_tdee"] = calories_tdee

    return pd.DataFrame([base])


# ── Prediction functions ───────────────────────────────────────────────────────

def predict_calories(age: int, weight_kg: float, height_cm: float,
                     activity_level: int, gender: int) -> float:
    if calorie_model is None:
        mult = {1:1.2, 2:1.375, 3:1.55, 4:1.725, 5:1.9}
        bmr  = 10*weight_kg + 6.25*height_cm - 5*age + (5 if gender==1 else -161)
        return round(bmr * mult.get(activity_level, 1.375))
    X = _build_features(age, weight_kg, height_cm, activity_level, gender)
    # calorie model was trained on the first 9 features
    try:
        return float(round(calorie_model.predict(X)[0]))
    except Exception:
        # Fallback if feature mismatch
        X_base = X[["age","weight_kg","height_cm","activity_level","gender"]]
        return float(round(calorie_model.predict(X_base)[0]))


def predict_weight_change(age: int, weight_kg: float, height_cm: float,
                           activity_level: int, gender: int,
                           calories_tdee: Optional[float] = None) -> float:
    if calories_tdee is None:
        calories_tdee = predict_calories(age, weight_kg, height_cm, activity_level, gender)
    if weight_model is None:
        return round((calories_tdee - 2000) / 7700 * 30, 2)
    X = _build_features(age, weight_kg, height_cm, activity_level, gender, calories_tdee)
    try:
        return float(round(weight_model.predict(X)[0], 2))
    except Exception:
        return round((calories_tdee - 2000) / 7700 * 30, 2)


def classify_fitness(age: int, weight_kg: float, height_cm: float,
                     activity_level: int, gender: int) -> dict:
    X = _build_features(age, weight_kg, height_cm, activity_level, gender)
    label_map = {0:"Beginner", 1:"Intermediate", 2:"Advanced"}

    if fitness_clf is None:
        score = activity_level * 20 + (10 if age < 35 else 0)
        lvl   = 0 if score < 30 else (1 if score < 60 else 2)
        probs = {0:0.0, 1:0.0, 2:0.0}; probs[lvl] = 1.0
    else:
        try:
            lvl = int(fitness_clf.predict(X)[0])
            if hasattr(fitness_clf[-1], "predict_proba"):
                raw   = fitness_clf.predict_proba(X)[0]
                probs = {i: round(float(v), 3) for i, v in enumerate(raw)}
            else:
                probs = {0:0.0, 1:0.0, 2:0.0}; probs[lvl] = 1.0
        except Exception:
            X5 = X[["age","weight_kg","height_cm","activity_level","gender"]]
            lvl = int(fitness_clf.predict(X5)[0])
            probs = {0:0.0, 1:0.0, 2:0.0}; probs[lvl] = 1.0

    return {
        "level_id":   lvl,
        "level_name": label_map.get(lvl, "Intermediate"),
        "probabilities": {label_map.get(k, str(k)): v for k, v in probs.items()},
    }


def get_recommendations(fitness_level_id: int, top_k: int = 5) -> list:
    if recommender is None:
        defaults = {
            0: ["Push-ups","Squats","Plank","Lunges","Burpees"],
            1: ["Bench Press","Deadlifts","Pull-ups","Row","OHP"],
            2: ["Deadlifts","OHP","Pull-ups","Bench Press","Squats"],
        }
        return [{"exercise": e, "muscle_group": "—", "difficulty": fitness_level_id+1}
                for e in defaults.get(fitness_level_id, defaults[1])[:top_k]]
    recs = recommender.get(fitness_level_id, recommender.get(1, []))
    return recs[:top_k]
