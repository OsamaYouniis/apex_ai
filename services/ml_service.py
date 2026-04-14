"""
services/ml_service.py
───────────────────────
APEX AI — ML Prediction Service (Production v5)

Loads trained models from /models (new) with fallback to /ai_models (legacy).
Provides prediction functions for:
  - Calorie burn prediction (XGBoost ensemble)
  - Fitness classification (4-class body performance)
  - Workout recommendations (content-based TF-IDF)
  - Weight change prediction
  - Comprehensive user analytics

All functions are pure (stateless) — safe for concurrent requests.
"""

import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("apex_ai.ml_service")

ROOT       = Path(__file__).parent.parent
MODEL_DIR  = ROOT / "models"
LEGACY_DIR = ROOT / "ai_models" / "ml_models"

try:
    import joblib
    _JOBLIB = True
except ImportError:
    _JOBLIB = False
    logger.error("joblib not installed — models cannot load")


# ─── MODEL LOADER ─────────────────────────────────────────────────────────────

def _load(filename: str, dirs: list[Path] = None):
    """Try loading a model from multiple dirs, first match wins."""
    if not _JOBLIB:
        return None
    search = dirs or [MODEL_DIR, LEGACY_DIR]
    for d in search:
        p = d / filename
        if p.exists():
            try:
                model = joblib.load(p)
                logger.info(f"✅ Loaded {filename} from {d.name}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")
    logger.warning(f"⚠️  {filename} not found in any search path")
    return None


def _load_json(filename: str, dirs: list[Path] = None) -> Optional[dict]:
    search = dirs or [MODEL_DIR, LEGACY_DIR]
    for d in search:
        p = d / filename
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


# ─── LOAD MODELS ONCE ─────────────────────────────────────────────────────────

calorie_model    = _load("calorie_model.pkl") or _load("calorie_regression.pkl")
fitness_model    = _load("fitness_model.pkl") or _load("fitness_classifier.pkl")
recommender      = _load("recommender.pkl")
weight_model     = _load("weight_regression.pkl")

calorie_features = _load_json("calorie_feature_list.json") or [
    "age", "height", "weight", "duration", "heart_rate", "body_temp",
    "gender_num", "bmi", "bmr", "hr_intensity", "weight_x_duration", "age_squared"
]
fitness_features = _load_json("fitness_feature_list.json") or [
    "age", "gender_num", "height_cm", "weight_kg", "bmi", "bmr", "age_squared",
    "gripforce", "sit-ups_counts", "broad_jump_cm", "strength_index"
]
fitness_labels   = _load_json("fitness_label_map.json") or {
    "3": "Advanced", "2": "Intermediate", "1": "Beginner", "0": "Untrained"
}

logger.info(f"Models loaded — calorie:{calorie_model is not None} | "
            f"fitness:{fitness_model is not None} | recommender:{recommender is not None}")


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def _compute_bmr(weight_kg: float, height_cm: float, age: int, gender: int) -> float:
    """Mifflin-St Jeor equation."""
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if gender == 1 else -161)


def _build_calorie_features(
    age: int, weight_kg: float, height_cm: float,
    activity_level: int, gender: int,
    duration: int = 45, heart_rate: int = 130, body_temp: float = 40.0,
) -> pd.DataFrame:
    h_m = height_cm / 100
    bmi = weight_kg / (h_m ** 2)
    bmr = _compute_bmr(weight_kg, height_cm, age, gender)
    feat = {
        "age":              age,
        "height":           height_cm,
        "weight":           weight_kg,
        "duration":         duration,
        "heart_rate":       heart_rate,
        "body_temp":        body_temp,
        "gender_num":       gender,
        "bmi":              round(bmi, 2),
        "bmr":              round(bmr, 1),
        "hr_intensity":     round(heart_rate / max(1, 220 - age), 4),
        "weight_x_duration": weight_kg * duration,
        "age_squared":      age ** 2,
    }
    # Only keep features the model knows about
    avail = {k: v for k, v in feat.items() if k in calorie_features}
    return pd.DataFrame([avail])


def _build_fitness_features(
    age: int, weight_kg: float, height_cm: float,
    activity_level: int, gender: int,
) -> pd.DataFrame:
    h_m = height_cm / 100
    bmi = weight_kg / (h_m ** 2)
    bmr = _compute_bmr(weight_kg, height_cm, age, gender)
    # Estimate performance metrics from activity level
    act_mult  = {1: 0.5, 2: 0.7, 3: 0.85, 4: 1.0, 5: 1.15}
    m         = act_mult.get(activity_level, 0.85)
    grip_est  = (35 + gender * 10) * m
    jump_est  = (160 + gender * 30) * m
    situps_est= (35 * m - age * 0.1)

    feat = {
        "age":                     age,
        "gender_num":              gender,
        "height_cm":               height_cm,
        "weight_kg":               weight_kg,
        "bmi":                     round(bmi, 2),
        "bmr":                     round(bmr, 1),
        "age_squared":             age ** 2,
        "gripforce":               round(grip_est, 1),
        "sit-ups_counts":          round(max(0, situps_est)),
        "broad_jump_cm":           round(jump_est),
        "sit_and_bend_forward_cm": 10.0,
        "strength_index":          round(grip_est + jump_est / 3, 1),
        "flexibility_idx":         10.0,
    }
    avail = {k: v for k, v in feat.items() if k in fitness_features}
    return pd.DataFrame([avail])


# ─── PREDICTION FUNCTIONS ─────────────────────────────────────────────────────

def predict_calories_burned(
    age: int, weight_kg: float, height_cm: float,
    activity_level: int, gender: int,
    duration_min: int = 45,
) -> float:
    """
    Predict calories burned during a workout session.
    Uses trained XGBoost ensemble if available, else MET formula.
    """
    if calorie_model is not None:
        X = _build_calorie_features(age, weight_kg, height_cm, activity_level, gender,
                                    duration=duration_min)
        try:
            return float(round(calorie_model.predict(X)[0]))
        except Exception as e:
            logger.warning(f"Calorie model predict error: {e}")

    # Fallback: MET-based formula
    met_map = {1: 3.5, 2: 5.0, 3: 6.5, 4: 8.0, 5: 10.0}
    met     = met_map.get(activity_level, 6.5)
    return round(met * weight_kg * duration_min / 60)


def predict_tdee(
    age: int, weight_kg: float, height_cm: float,
    activity_level: int, gender: int,
) -> float:
    """Predict Total Daily Energy Expenditure."""
    bmr  = _compute_bmr(weight_kg, height_cm, age, gender)
    mult = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
    return round(bmr * mult.get(activity_level, 1.375))


def classify_fitness(
    age: int, weight_kg: float, height_cm: float,
    activity_level: int, gender: int,
) -> dict:
    """
    Classify fitness level from 0 (Untrained) to 3 (Advanced).
    Returns level_id, level_name, probabilities.
    """
    if fitness_model is not None:
        X = _build_fitness_features(age, weight_kg, height_cm, activity_level, gender)
        try:
            # Try with new features first; fall back to legacy 5-feature set
            try:
                pred = int(fitness_model.predict(X)[0])
            except Exception:
                X_legacy = pd.DataFrame([{
                    "age": age, "weight_kg": weight_kg, "height_cm": height_cm,
                    "activity_level": activity_level, "gender": gender,
                }])
                pred = int(fitness_model.predict(X_legacy)[0])
            # Map 4-class to 3-class for legacy API
            legacy_map = {3: 2, 2: 1, 1: 1, 0: 0}
            legacy_id  = legacy_map.get(pred, 1)

            if hasattr(fitness_model[-1], "predict_proba"):
                raw   = fitness_model.predict_proba(X)[0]
                probs = {str(i): round(float(v), 3) for i, v in enumerate(raw)}
            else:
                probs = {str(i): 0.0 for i in range(4)}
                probs[str(pred)] = 1.0

            label = fitness_labels.get(str(pred), "Intermediate")
            return {
                "level_id":      legacy_id,
                "level_name":    label,
                "level_id_4":    pred,
                "probabilities": {
                    fitness_labels.get(str(k), str(k)): v
                    for k, v in {int(k): v for k, v in probs.items()}.items()
                },
            }
        except Exception as e:
            logger.warning(f"Fitness model predict error: {e}")

    # Fallback heuristic
    score = activity_level * 15 + max(0, 10 - abs(age - 28))
    if score < 30:   lvl = 0
    elif score < 50: lvl = 1
    elif score < 70: lvl = 2
    else:            lvl = 3
    names = {0: "Untrained", 1: "Beginner", 2: "Intermediate", 3: "Advanced"}
    return {
        "level_id":   min(lvl, 2),
        "level_name": names[lvl],
        "level_id_4": lvl,
        "probabilities": {names[i]: (1.0 if i == lvl else 0.0) for i in range(4)},
    }


def get_recommendations(
    fitness_level: int,
    goal: str = "maintain",
    age: int = 30,
    top_k: int = 6,
) -> list[dict]:
    """
    Get personalized workout recommendations.
    Uses content-based recommender if available.
    """
    if recommender is not None:
        try:
            # New recommender class
            if hasattr(recommender, "recommend"):
                return recommender.recommend(fitness_level, goal, age, top_k)
            # Legacy dict format
            elif isinstance(recommender, dict):
                recs = recommender.get(fitness_level, recommender.get(1, []))
                return recs[:top_k]
        except Exception as e:
            logger.warning(f"Recommender error: {e}")

    # Hardcoded fallback
    defaults = {
        0: ["Walking", "Bodyweight Squats", "Push-ups", "Plank", "Lunges"],
        1: ["Dumbbell Rows", "Goblet Squat", "Pull-ups", "Jump Rope", "Kettlebell Swings"],
        2: ["Barbell Squat", "Deadlift", "Bench Press", "OHP", "HIIT Sprints"],
    }
    exs = defaults.get(min(fitness_level, 2), defaults[1])
    return [{"exercise": e, "muscle_group": "—", "difficulty": fitness_level + 1,
             "equipment": "varies", "type": "strength", "calories_ph": 400}
            for e in exs[:top_k]]


def predict_weight_change(
    age: int, weight_kg: float, height_cm: float,
    activity_level: int, gender: int,
    tdee: Optional[float] = None,
    daily_intake: Optional[float] = None,
    days: int = 30,
) -> dict:
    """
    Predict weight change over N days.
    Returns kg_change, direction, target_date.
    """
    if tdee is None:
        tdee = predict_tdee(age, weight_kg, height_cm, activity_level, gender)
    if daily_intake is None:
        daily_intake = tdee  # maintenance

    deficit_surplus = daily_intake - tdee
    # 7700 kcal ≈ 1 kg body fat
    kg_change = round((deficit_surplus * days) / 7700, 2)

    if weight_model is not None:
        # Override with model prediction if available
        try:
            feat = {
                "age": age, "weight_kg": weight_kg, "height_cm": height_cm,
                "activity_level": activity_level, "gender": gender,
                "bmi": weight_kg / (height_cm / 100) ** 2,
                "bmr": _compute_bmr(weight_kg, height_cm, age, gender),
                "age_squared": age ** 2,
                "weight_height": weight_kg / height_cm,
                "calories_tdee": tdee,
            }
            X = pd.DataFrame([feat])
            kg_change = float(round(weight_model.predict(X)[0], 2))
        except Exception:
            pass

    return {
        "kg_change_30d":  kg_change,
        "direction":      "gain" if kg_change > 0.1 else ("loss" if kg_change < -0.1 else "maintain"),
        "new_weight_est": round(weight_kg + kg_change, 1),
        "tdee":           tdee,
        "deficit_daily":  round(deficit_surplus),
    }


def full_prediction(
    age: int, weight_kg: float, height_cm: float,
    activity_level: int, gender: int,
    goal: str = "maintain",
) -> dict:
    """
    Complete prediction pipeline — all metrics in one call.
    Used by the /predict API endpoint.
    """
    h_m        = height_cm / 100
    bmi        = round(weight_kg / (h_m ** 2), 2)
    bmr        = round(_compute_bmr(weight_kg, height_cm, age, gender))
    tdee       = predict_tdee(age, weight_kg, height_cm, activity_level, gender)
    fitness    = classify_fitness(age, weight_kg, height_cm, activity_level, gender)
    weight_chg = predict_weight_change(age, weight_kg, height_cm, activity_level, gender, tdee)
    recs       = get_recommendations(fitness["level_id"], goal, age)

    bmi_cat = ("Underweight" if bmi < 18.5 else
               "Normal weight" if bmi < 25 else
               "Overweight" if bmi < 30 else "Obese")

    protein_g   = round(weight_kg * 2.0)
    water_l     = round(weight_kg * 0.033, 1)
    cut_cals    = round(tdee - 500)
    bulk_cals   = round(tdee + 300)
    body_fat_est = max(5, min(45, 495 / bmi - 450 + (10 if gender == 0 else 0)))

    return {
        "calories_tdee":     tdee,
        "bmr":               bmr,
        "bmi":               bmi,
        "bmi_category":      bmi_cat,
        "body_fat_est":      round(body_fat_est, 1),
        "fitness_level":     fitness["level_name"],
        "fitness_level_id":  fitness["level_id"],
        "probabilities":     fitness["probabilities"],
        "weight_change_30d": weight_chg["kg_change_30d"],
        "new_weight_est":    weight_chg["new_weight_est"],
        "recommendations":   recs,
        "cut_calories":      cut_cals,
        "bulk_calories":     bulk_cals,
        "protein_g":         protein_g,
        "water_l":           water_l,
        "macros": {
            "protein_g":  protein_g,
            "carbs_g":    round((tdee * 0.45) / 4),
            "fat_g":      round((tdee * 0.25) / 9),
        },
    }
