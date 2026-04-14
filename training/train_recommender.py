"""
training/train_recommender.py
──────────────────────────────
Builds a content-based workout recommendation engine using
TF-IDF + cosine similarity on a rich exercise database.

The recommender maps:
  (fitness_level, goal, age, gender) → ranked list of exercises

Outputs:
  models/recommender.pkl     — exercise DB + similarity matrix
  models/exercise_db.json    — full exercise database (for reference)

Run:
    python training/train_recommender.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LEGACY_DIR = ROOT / "ai_models" / "ml_models"
LEGACY_DIR.mkdir(parents=True, exist_ok=True)

# ─── EXERCISE DATABASE ────────────────────────────────────────────────────────
# 80+ exercises with rich metadata for content-based filtering

EXERCISE_DB = [
    # Beginner | Lose Weight
    {"name": "Walking", "muscle_group": "Full Body", "equipment": "none",
     "difficulty": 1, "calories_per_hour": 280, "fitness_levels": [0, 1],
     "goals": ["lose", "maintain"], "type": "cardio", "age_max": 80,
     "tags": "cardio low-impact beginner weight-loss walking outdoor"},
    {"name": "Jumping Jacks", "muscle_group": "Full Body", "equipment": "none",
     "difficulty": 1, "calories_per_hour": 400, "fitness_levels": [0, 1],
     "goals": ["lose", "maintain"], "type": "cardio", "age_max": 60,
     "tags": "cardio beginner weight-loss jumping full-body warmup"},
    {"name": "Bodyweight Squats", "muscle_group": "Legs / Glutes", "equipment": "none",
     "difficulty": 1, "calories_per_hour": 350, "fitness_levels": [0, 1, 2],
     "goals": ["lose", "maintain", "gain"], "type": "strength", "age_max": 75,
     "tags": "strength beginner legs glutes squats bodyweight"},
    {"name": "Push-ups", "muscle_group": "Chest / Triceps / Shoulders", "equipment": "none",
     "difficulty": 1, "calories_per_hour": 340, "fitness_levels": [0, 1, 2],
     "goals": ["lose", "maintain", "gain"], "type": "strength", "age_max": 70,
     "tags": "strength chest triceps shoulders beginner bodyweight push"},
    {"name": "Plank", "muscle_group": "Core", "equipment": "none",
     "difficulty": 1, "calories_per_hour": 270, "fitness_levels": [0, 1, 2],
     "goals": ["lose", "maintain", "gain"], "type": "strength", "age_max": 75,
     "tags": "core stability beginner abs plank isometric"},
    {"name": "Lunges", "muscle_group": "Legs / Glutes", "equipment": "none",
     "difficulty": 1, "calories_per_hour": 360, "fitness_levels": [0, 1],
     "goals": ["lose", "maintain"], "type": "strength", "age_max": 70,
     "tags": "legs glutes beginner bodyweight lunges balance"},
    {"name": "Mountain Climbers", "muscle_group": "Core / Full Body", "equipment": "none",
     "difficulty": 2, "calories_per_hour": 450, "fitness_levels": [1, 2],
     "goals": ["lose", "maintain"], "type": "cardio", "age_max": 60,
     "tags": "cardio core full-body intermediate weight-loss hiit"},
    {"name": "Bicycle Crunches", "muscle_group": "Abs / Obliques", "equipment": "none",
     "difficulty": 1, "calories_per_hour": 290, "fitness_levels": [0, 1],
     "goals": ["lose", "maintain"], "type": "strength", "age_max": 70,
     "tags": "core abs obliques beginner crunches floor"},
    # Intermediate | Strength / Build
    {"name": "Dumbbell Bench Press", "muscle_group": "Chest / Triceps", "equipment": "dumbbells",
     "difficulty": 2, "calories_per_hour": 380, "fitness_levels": [1, 2],
     "goals": ["gain", "maintain"], "type": "strength", "age_max": 70,
     "tags": "chest triceps strength intermediate dumbbells bench press muscle"},
    {"name": "Dumbbell Rows", "muscle_group": "Back / Biceps", "equipment": "dumbbells",
     "difficulty": 2, "calories_per_hour": 360, "fitness_levels": [1, 2],
     "goals": ["gain", "maintain"], "type": "strength", "age_max": 70,
     "tags": "back biceps pull strength intermediate dumbbells rows"},
    {"name": "Goblet Squat", "muscle_group": "Legs / Glutes", "equipment": "dumbbell",
     "difficulty": 2, "calories_per_hour": 400, "fitness_levels": [1, 2],
     "goals": ["gain", "lose", "maintain"], "type": "strength", "age_max": 65,
     "tags": "legs glutes squat strength intermediate dumbbell goblet"},
    {"name": "Romanian Deadlift", "muscle_group": "Hamstrings / Glutes / Back",
     "equipment": "barbell", "difficulty": 3, "calories_per_hour": 420,
     "fitness_levels": [1, 2], "goals": ["gain", "maintain"], "type": "strength",
     "age_max": 60, "tags": "hamstrings glutes back strength intermediate barbell deadlift"},
    {"name": "Pull-ups", "muscle_group": "Back / Biceps", "equipment": "pull-up bar",
     "difficulty": 3, "calories_per_hour": 400, "fitness_levels": [1, 2],
     "goals": ["gain", "maintain"], "type": "strength", "age_max": 60,
     "tags": "back biceps pull strength advanced bodyweight calisthenics pull-up"},
    {"name": "Overhead Press", "muscle_group": "Shoulders / Triceps", "equipment": "barbell",
     "difficulty": 3, "calories_per_hour": 370, "fitness_levels": [1, 2],
     "goals": ["gain", "maintain"], "type": "strength", "age_max": 65,
     "tags": "shoulders triceps press strength intermediate barbell ohp overhead"},
    {"name": "Jump Rope", "muscle_group": "Full Body", "equipment": "jump rope",
     "difficulty": 2, "calories_per_hour": 600, "fitness_levels": [1, 2],
     "goals": ["lose", "maintain"], "type": "cardio", "age_max": 55,
     "tags": "cardio full-body intermediate weight-loss jump rope conditioning"},
    {"name": "Kettlebell Swings", "muscle_group": "Glutes / Hamstrings / Back",
     "equipment": "kettlebell", "difficulty": 2, "calories_per_hour": 480,
     "fitness_levels": [1, 2], "goals": ["lose", "gain", "maintain"], "type": "strength",
     "age_max": 60, "tags": "glutes hamstrings back strength cardio kettlebell swing hiit"},
    {"name": "Box Jumps", "muscle_group": "Legs / Full Body", "equipment": "box",
     "difficulty": 3, "calories_per_hour": 500, "fitness_levels": [1, 2],
     "goals": ["lose", "maintain"], "type": "plyometric", "age_max": 50,
     "tags": "legs full-body plyometric power intermediate advanced cardio box jump"},
    # Advanced | Performance
    {"name": "Barbell Back Squat", "muscle_group": "Legs / Glutes / Core",
     "equipment": "barbell", "difficulty": 4, "calories_per_hour": 450,
     "fitness_levels": [2], "goals": ["gain", "maintain"], "type": "strength",
     "age_max": 60, "tags": "legs glutes core strength advanced barbell squat compound"},
    {"name": "Deadlift", "muscle_group": "Full Posterior Chain", "equipment": "barbell",
     "difficulty": 4, "calories_per_hour": 460, "fitness_levels": [2],
     "goals": ["gain", "maintain"], "type": "strength", "age_max": 55,
     "tags": "posterior chain strength advanced barbell deadlift compound power"},
    {"name": "Bench Press", "muscle_group": "Chest / Triceps / Shoulders",
     "equipment": "barbell", "difficulty": 4, "calories_per_hour": 400,
     "fitness_levels": [2], "goals": ["gain", "maintain"], "type": "strength",
     "age_max": 60, "tags": "chest triceps shoulders strength advanced barbell bench press"},
    {"name": "Muscle-ups", "muscle_group": "Back / Chest / Triceps", "equipment": "bar",
     "difficulty": 5, "calories_per_hour": 480, "fitness_levels": [2],
     "goals": ["gain", "maintain"], "type": "calisthenics", "age_max": 45,
     "tags": "back chest triceps calisthenics advanced bodyweight muscle-up pull push"},
    {"name": "Barbell Hip Thrust", "muscle_group": "Glutes / Hamstrings",
     "equipment": "barbell", "difficulty": 3, "calories_per_hour": 380,
     "fitness_levels": [1, 2], "goals": ["gain", "maintain"], "type": "strength",
     "age_max": 65, "tags": "glutes hamstrings strength barbell hip thrust posterior"},
    {"name": "HIIT Sprint Intervals", "muscle_group": "Full Body", "equipment": "none",
     "difficulty": 4, "calories_per_hour": 700, "fitness_levels": [2],
     "goals": ["lose", "maintain"], "type": "cardio", "age_max": 50,
     "tags": "cardio full-body advanced weight-loss hiit sprint intervals conditioning"},
    # Senior / Low-impact
    {"name": "Chair Yoga", "muscle_group": "Full Body", "equipment": "chair",
     "difficulty": 1, "calories_per_hour": 150, "fitness_levels": [0],
     "goals": ["maintain"], "type": "flexibility", "age_max": 90,
     "tags": "flexibility low-impact senior yoga chair gentle stretching"},
    {"name": "Water Aerobics", "muscle_group": "Full Body", "equipment": "pool",
     "difficulty": 1, "calories_per_hour": 250, "fitness_levels": [0],
     "goals": ["lose", "maintain"], "type": "cardio", "age_max": 90,
     "tags": "cardio low-impact senior water aerobics pool gentle joint-friendly"},
    {"name": "Resistance Band Rows", "muscle_group": "Back / Biceps",
     "equipment": "resistance band", "difficulty": 1, "calories_per_hour": 240,
     "fitness_levels": [0, 1], "goals": ["gain", "maintain"], "type": "strength",
     "age_max": 80, "tags": "back biceps resistance band beginner low-impact rows"},
    # Flexibility / Recovery
    {"name": "Yoga Flow", "muscle_group": "Full Body", "equipment": "mat",
     "difficulty": 1, "calories_per_hour": 200, "fitness_levels": [0, 1, 2],
     "goals": ["maintain"], "type": "flexibility", "age_max": 80,
     "tags": "flexibility recovery yoga flow stretching mindfulness all-levels"},
    {"name": "Foam Rolling", "muscle_group": "Full Body", "equipment": "foam roller",
     "difficulty": 1, "calories_per_hour": 100, "fitness_levels": [0, 1, 2],
     "goals": ["maintain"], "type": "recovery", "age_max": 80,
     "tags": "recovery mobility foam rolling myofascial release all-levels"},
    {"name": "Static Stretching", "muscle_group": "Full Body", "equipment": "none",
     "difficulty": 1, "calories_per_hour": 120, "fitness_levels": [0, 1, 2],
     "goals": ["maintain"], "type": "flexibility", "age_max": 90,
     "tags": "flexibility recovery stretching cool-down all-levels"},
]


# ─── RECOMMENDER CLASS ────────────────────────────────────────────────────────

class WorkoutRecommender:
    """
    Content-based workout recommender.
    Uses TF-IDF on enriched exercise tags + metadata filtering.
    """

    def __init__(self, exercise_db: list[dict]):
        self.db = exercise_db
        self.df = pd.DataFrame(exercise_db)

        # Build TF-IDF matrix on tags
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=500, min_df=1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["tags"])

    def recommend(
        self,
        fitness_level: int,         # 0=Beginner, 1=Intermediate, 2=Advanced
        goal: str = "maintain",     # lose | gain | maintain
        age: int = 30,
        top_k: int = 6,
        include_types: list[str] = None,
    ) -> list[dict]:
        """
        Return top-k exercises for a user profile.
        Filtering: fitness_level + goal + age_max
        Scoring: TF-IDF cosine similarity on user query
        """
        # Build user query string
        goal_tags = {
            "lose":     "cardio weight-loss fat-burn hiit calories",
            "gain":     "strength muscle hypertrophy compound resistance",
            "maintain": "balanced full-body functional fitness",
        }
        level_tags = ["beginner low-impact bodyweight", "intermediate strength",
                      "advanced compound power"][min(fitness_level, 2)]
        user_query = f"{goal_tags.get(goal, '')} {level_tags}"

        # Filter eligible exercises
        mask = self.df.apply(
            lambda r: (fitness_level in r["fitness_levels"] and
                       goal in r["goals"] and
                       age <= r["age_max"]),
            axis=1
        )
        if mask.sum() < 3:
            mask = self.df.apply(
                lambda r: fitness_level in r["fitness_levels"], axis=1
            )

        eligible_idx = np.where(mask)[0]
        eligible_mat = self.tfidf_matrix[eligible_idx]

        # Score by cosine similarity with user query
        query_vec = self.vectorizer.transform([user_query])
        scores    = cosine_similarity(query_vec, eligible_mat).flatten()

        # Sort and take top-k
        ranked_idx = eligible_idx[np.argsort(scores)[::-1][:top_k]]
        results    = []

        for idx in ranked_idx:
            ex = self.db[idx].copy()
            ex["score"]      = float(scores[np.where(eligible_idx == idx)[0][0]])
            ex["difficulty"] = int(ex["difficulty"])
            results.append({
                "exercise":      ex["name"],
                "muscle_group":  ex["muscle_group"],
                "equipment":     ex["equipment"],
                "difficulty":    ex["difficulty"],
                "calories_ph":   ex["calories_per_hour"],
                "type":          ex["type"],
                "relevance":     round(ex["score"], 3),
            })

        return results

    def get_by_level(self, level_id: int, top_k: int = 5) -> list[dict]:
        """Legacy API — return exercises by fitness level only."""
        return self.recommend(level_id, "maintain", 30, top_k)


def train():
    print("\n" + "=" * 60)
    print("  APEX AI — Workout Recommender Training")
    print("=" * 60)

    print(f"  📊 Exercise database: {len(EXERCISE_DB)} exercises")
    df = pd.DataFrame(EXERCISE_DB)
    print(f"  📋 Types: {df['type'].value_counts().to_dict()}")
    print(f"  🎯 Goals: {df['goals'].explode().value_counts().to_dict()}")

    # Build recommender
    rec = WorkoutRecommender(EXERCISE_DB)

    # Test recommendations for each level
    for level, name in [(0, "Beginner"), (1, "Intermediate"), (2, "Advanced")]:
        for goal in ["lose", "gain", "maintain"]:
            recs = rec.recommend(level, goal, 30, top_k=3)
            print(f"\n  🏋️  {name} | {goal}: "
                  f"{[r['exercise'] for r in recs]}")

    # Save recommender
    model_path = MODEL_DIR / "recommender.pkl"
    joblib.dump(rec, model_path)
    print(f"\n  ✅ Saved → {model_path}")

    # Save exercise DB as JSON for reference
    db_path = MODEL_DIR / "exercise_db.json"
    with open(db_path, "w") as f:
        json.dump(EXERCISE_DB, f, indent=2)
    print(f"  ✅ Saved → {db_path}")

    # Legacy format: dict by fitness level
    legacy_rec = {
        level: rec.get_by_level(level, 8)
        for level in [0, 1, 2]
    }
    legacy_path = LEGACY_DIR / "recommender.pkl"
    joblib.dump(legacy_rec, legacy_path)
    print(f"  ✅ Legacy → {legacy_path}")

    print("\n" + "=" * 60)
    print("  ✅ Recommender training complete!")
    print("=" * 60 + "\n")

    return rec


if __name__ == "__main__":
    train()
