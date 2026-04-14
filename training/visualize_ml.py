"""
training/visualize_ml.py
─────────────────────────
Generates human-readable reports and visualizations of all trained ML models.
Saves plots to ai_models/ml_models/reports/

Run:
    python training/visualize_ml.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "datasets"
MODEL_DIR= ROOT / "ai_models" / "ml_models"
OUT_DIR  = MODEL_DIR / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


def load_models():
    models = {}
    for name in ["calorie_regression","weight_regression","fitness_classifier","recommender"]:
        p = MODEL_DIR / f"{name}.pkl"
        if p.exists():
            models[name] = joblib.load(p)
            print(f"  ✅ Loaded {name}.pkl")
        else:
            print(f"  ⚠️  Missing {name}.pkl — run training/train_ml.py first")
    return models


def report_calorie_model(model, df):
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split

    FEATURES = ["age","weight_kg","height_cm","activity_level","gender"]
    X = df[FEATURES]; y = df["calories_tdee"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = model.predict(X_te)

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║  MODEL 1: Calorie Prediction (Linear Regression)            ║
╚══════════════════════════════════════════════════════════════╝

ALGORITHM : Linear Regression (with StandardScaler preprocessing)
INPUT     : Age, Weight (kg), Height (cm), Activity Level (1-5), Gender
OUTPUT    : Daily TDEE Calories (kcal)

PERFORMANCE METRICS (Test Set — 400 samples):
  Mean Absolute Error  (MAE) : {mean_absolute_error(y_te, preds):.2f} kcal
  Root Mean Sq. Error  (RMSE): {np.sqrt(mean_squared_error(y_te, preds)):.2f} kcal
  R² Score                   : {r2_score(y_te, preds):.4f}  (1.0 = perfect)
  Explained Variance         : {r2_score(y_te, preds)*100:.1f}%

INTERPRETATION:
  → The model explains {r2_score(y_te, preds)*100:.1f}% of variance in daily calorie needs
  → Average prediction error is only {mean_absolute_error(y_te, preds):.0f} kcal
  → Compare: manual Mifflin-St Jeor formula has ~100-150 kcal MAE

FEATURE WEIGHTS (coefficients):
"""
    if hasattr(model.named_steps.get("lr", model[-1]), "coef_"):
        coefs = model.named_steps.get("lr", model[-1]).coef_
        for feat, coef in zip(FEATURES, coefs):
            report += f"  {feat:<20} : {coef:+.4f}\n"
    
    report += f"""
SAMPLE PREDICTIONS vs ACTUAL:
  {'Age':>4} {'Weight':>7} {'Height':>7} {'Act':>4} {'Gen':>4} | {'Actual':>8} | {'Predicted':>10} | {'Error':>7}
  {'-'*65}
"""
    sample = X_te.iloc[:10].copy()
    actual_sample = y_te.iloc[:10]
    pred_sample   = model.predict(sample)
    for i in range(10):
        row = sample.iloc[i]
        report += f"  {int(row.age):>4} {row.weight_kg:>7.1f} {row.height_cm:>7.1f} {int(row.activity_level):>4} {int(row.gender):>4} | {actual_sample.iloc[i]:>8.0f} | {pred_sample[i]:>10.0f} | {abs(actual_sample.iloc[i]-pred_sample[i]):>7.0f}\n"

    path = OUT_DIR / "01_calorie_model_report.txt"
    path.write_text(report)
    print(f"\n📄 Calorie model report → {path}")
    print(report)
    return report


def report_weight_model(model, df):
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    FEATURES = ["age","weight_kg","height_cm","activity_level","gender","calories_tdee"]
    X = df[FEATURES]; y = df["weight_change_30d"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = model.predict(X_te)

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║  MODEL 2: Weight Change Prediction (Random Forest)          ║
╚══════════════════════════════════════════════════════════════╝

ALGORITHM : Random Forest Regressor (150 trees, max_depth=8)
INPUT     : Age, Weight, Height, Activity Level, Gender, TDEE Calories
OUTPUT    : Predicted weight change over 30 days (kg)

PERFORMANCE METRICS (Test Set — 400 samples):
  Mean Absolute Error  (MAE) : {mean_absolute_error(y_te, preds):.4f} kg
  R² Score                   : {r2_score(y_te, preds):.4f}

RANDOM FOREST DETAILS:
  Number of trees    : 150
  Max tree depth     : 8
  Min samples split  : 2
  Features per split : sqrt(n_features)

FEATURE IMPORTANCES (how much each variable matters):
"""
    rf = model.named_steps.get("rf", model[-1])
    if hasattr(rf, "feature_importances_"):
        importances = rf.feature_importances_
        for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
            bar = "█" * int(imp * 50)
            report += f"  {feat:<25}: {imp:.4f}  {bar}\n"

    report += f"""
INTERPRETATION:
  → Model can predict monthly weight change to within {mean_absolute_error(y_te, preds):.2f} kg
  → Useful for setting realistic 30-day weight loss/gain expectations
  → Based on calorie deficit/surplus: 7,700 kcal deficit ≈ 1 kg fat loss
"""
    path = OUT_DIR / "02_weight_model_report.txt"
    path.write_text(report)
    print(f"\n📄 Weight model report → {path}")
    print(report)


def report_fitness_classifier(model, df):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import train_test_split

    FEATURES = ["age","weight_kg","height_cm","activity_level","gender"]
    LABELS   = ["Beginner","Intermediate","Advanced"]
    X = df[FEATURES]; y = df["fitness_level"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preds = model.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    cm    = confusion_matrix(y_te, preds)

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║  MODEL 3: Fitness Level Classifier (Random Forest)          ║
╚══════════════════════════════════════════════════════════════╝

ALGORITHM : Random Forest Classifier (150 trees)
INPUT     : Age, Weight, Height, Activity Level, Gender
OUTPUT    : Fitness Level — Beginner (0) / Intermediate (1) / Advanced (2)

OVERALL ACCURACY: {acc:.4f} ({acc*100:.1f}%)

DETAILED CLASSIFICATION REPORT:
{classification_report(y_te, preds, target_names=LABELS, zero_division=0)}

CONFUSION MATRIX:
  (Rows = Actual, Columns = Predicted)
  
         {'':>4}  {'Beginner':>12}  {'Intermediate':>14}  {'Advanced':>10}
"""
    for i, label in enumerate(LABELS):
        report += f"  {label:<14}  "
        for j in range(3):
            val = cm[i][j]
            marker = " ✅" if i==j else " ❌"
            report += f"{val:>12}{marker if i==j else '   '}"
        report += "\n"

    report += f"""
WHAT THE CLASSES MEAN:
  Beginner     (0): Low activity, higher weight-to-height ratio, age 40+
  Intermediate (1): Moderate activity, average body composition
  Advanced     (2): High activity (4-5), good body composition, regular training

HOW IT'S USED IN APEX AI:
  The classifier automatically assigns your fitness tier when you enter your
  profile, then the recommendation engine uses that tier to suggest exercises
  matched to your actual ability level.
"""
    path = OUT_DIR / "03_fitness_classifier_report.txt"
    path.write_text(report)
    print(f"\n📄 Fitness classifier report → {path}")
    print(report)


def report_recommender(rec_dict):
    LABELS = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}
    report = """
╔══════════════════════════════════════════════════════════════╗
║  MODEL 4: Workout Recommendation Engine                     ║
╚══════════════════════════════════════════════════════════════╝

ALGORITHM : Content-Based Filtering (collaborative rating aggregation)
INPUT     : User fitness level (Beginner / Intermediate / Advanced)
OUTPUT    : Ranked list of exercises by average user rating

HOW IT WORKS:
  1. 200 synthetic users logged their workouts with ratings (1-5 stars)
  2. For each fitness level, we compute: avg_rating per exercise
  3. At inference, we return the top-K exercises for the user's level
  4. This ensures Beginners get accessible exercises, Advanced get challenging ones

RECOMMENDATIONS BY FITNESS LEVEL:

"""
    for lvl_id, lvl_name in LABELS.items():
        recs = rec_dict.get(lvl_id, [])
        report += f"  {'─'*50}\n"
        report += f"  {lvl_name.upper()} USERS (Level {lvl_id})\n"
        report += f"  {'─'*50}\n"
        report += f"  {'Rank':<5} {'Exercise':<20} {'Muscle Group':<15} {'Difficulty':<12} {'Avg Rating'}\n"
        for i, rec in enumerate(recs[:8], 1):
            stars = "★" * round(rec.get("avg_rating", 0)) + "☆" * (5 - round(rec.get("avg_rating", 0)))
            report += (f"  {i:<5} {rec.get('exercise',''):<20} "
                      f"{rec.get('muscle_group',''):<15} "
                      f"{rec.get('difficulty',''):<12} "
                      f"{rec.get('avg_rating',0):.2f}/5  {stars}\n")
        report += "\n"

    path = OUT_DIR / "04_recommender_report.txt"
    path.write_text(report)
    print(f"\n📄 Recommender report → {path}")
    print(report)


def generate_summary():
    summary = """
╔══════════════════════════════════════════════════════════════╗
║            APEX AI — ML MODELS SUMMARY REPORT              ║
╚══════════════════════════════════════════════════════════════╝

This file explains ALL machine learning models in plain English.
The models are saved as .pkl files (binary format) because that is
the industry standard for scikit-learn models. This report shows
you exactly what is INSIDE each .pkl file.

MODELS OVERVIEW:
─────────────────────────────────────────────────────────────────
│ File                        │ Algorithm           │ Purpose   │
─────────────────────────────────────────────────────────────────
│ calorie_regression.pkl      │ Linear Regression   │ TDEE calc │
│ weight_regression.pkl       │ Random Forest       │ Weight Δ  │
│ fitness_classifier.pkl      │ Random Forest       │ Level cls │
│ fitness_classifier_labels   │ Dict                │ Label map │
│ recommender.pkl             │ Content-based       │ Exercises │
─────────────────────────────────────────────────────────────────

WHY .pkl FILES?
  .pkl (pickle) is Python's native serialization format.
  It saves the entire trained model — including all learned 
  weights, trees, and preprocessors — into a single file.
  When the API loads it, it's instantly ready to make predictions
  without retraining. Think of it like saving a game.

PIPELINE FLOW:
  User enters profile → API calls ml_service.py → 
  Models loaded from .pkl → Predictions returned → 
  Frontend displays results

See individual report files in this folder for detailed metrics.
"""
    path = OUT_DIR / "00_SUMMARY.txt"
    path.write_text(summary)
    print(summary)


if __name__ == "__main__":
    print("=" * 65)
    print("  APEX AI — ML Model Visualization & Reports")
    print("=" * 65)

    df = pd.read_csv(DATA_DIR / "fitness_profiles.csv")
    models = load_models()

    generate_summary()

    if "calorie_regression" in models:
        report_calorie_model(models["calorie_regression"], df)
    if "weight_regression" in models:
        report_weight_model(models["weight_regression"], df)
    if "fitness_classifier" in models:
        report_fitness_classifier(models["fitness_classifier"], df)
    if "recommender" in models:
        report_recommender(models["recommender"])

    print(f"\n✅ All reports saved to: {OUT_DIR}")
    print("   Open the .txt files to read full model details!\n")
