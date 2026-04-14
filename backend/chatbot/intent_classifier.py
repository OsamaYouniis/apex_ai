"""
backend/chatbot/intent_classifier.py
──────────────────────────────────────
NLP Intent Classifier for APEX AI Chatbot (NEW — v4)

Architecture:
    TF-IDF vectorizer → Logistic Regression classifier
    Trained on fitness Q&A intents from chatbot_training.ipynb

Intents detected:
    greeting       → rule_based (instant)
    bmi_calc       → rule_based (formulaic)
    calorie_calc   → rule_based (TDEE formula)
    macro_calc     → rule_based (protein/carb/fat)
    workout_plan   → api (complex, personalized)
    nutrition_plan → api (complex, personalized)
    supplement     → api (nuanced)
    injury_advice  → api (medical — always use best engine)
    motivation     → local_dl (conversational)
    general_fitness → local_dl (open-ended)

The classifier adds a confidence score and intent label to every message
BEFORE the response_router decides which engine to use — giving the router
a richer signal than keyword matching alone.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("apex_ai.intent")

ROOT      = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT / "ai_models" / "ml_models"

# ── Built-in training data (fallback if no trained model exists) ──────────────
# Each entry: (example_text, intent_label)
BUILTIN_TRAINING_DATA = [
    # greeting
    ("hello", "greeting"), ("hi there", "greeting"), ("hey", "greeting"),
    ("good morning", "greeting"), ("salam", "greeting"), ("marhaba", "greeting"),
    ("what's up", "greeting"), ("howdy", "greeting"),

    # bmi_calc
    ("what is my bmi", "bmi_calc"), ("calculate bmi", "bmi_calc"),
    ("how do I calculate body mass index", "bmi_calc"),
    ("am I overweight", "bmi_calc"), ("what is a healthy bmi", "bmi_calc"),

    # calorie_calc
    ("how many calories should I eat", "calorie_calc"),
    ("what is my tdee", "calorie_calc"), ("daily calorie needs", "calorie_calc"),
    ("caloric deficit for weight loss", "calorie_calc"),
    ("how many calories to lose weight", "calorie_calc"),
    ("maintenance calories", "calorie_calc"),

    # macro_calc
    ("how much protein should I eat", "macro_calc"),
    ("what are my macros", "macro_calc"), ("protein carbs fat ratio", "macro_calc"),
    ("macro breakdown", "macro_calc"), ("how much protein per day", "macro_calc"),
    ("keto macros", "macro_calc"),

    # workout_plan
    ("give me a workout plan", "workout_plan"),
    ("what exercises should I do", "workout_plan"),
    ("build muscle workout", "workout_plan"),
    ("weekly training program", "workout_plan"),
    ("best exercises for beginners", "workout_plan"),
    ("chest workout routine", "workout_plan"),
    ("leg day exercises", "workout_plan"),
    ("full body workout", "workout_plan"),

    # nutrition_plan
    ("what should I eat to lose weight", "nutrition_plan"),
    ("meal plan for muscle gain", "nutrition_plan"),
    ("healthy diet plan", "nutrition_plan"),
    ("what foods to avoid", "nutrition_plan"),
    ("intermittent fasting schedule", "nutrition_plan"),
    ("best diet for fat loss", "nutrition_plan"),

    # supplement
    ("should I take creatine", "supplement"),
    ("best protein powder", "supplement"),
    ("pre workout supplement", "supplement"),
    ("is creatine safe", "supplement"),
    ("whey vs casein protein", "supplement"),
    ("omega 3 benefits", "supplement"),

    # injury_advice
    ("my knee hurts when squatting", "injury_advice"),
    ("lower back pain from deadlifts", "injury_advice"),
    ("shoulder injury from bench press", "injury_advice"),
    ("how to avoid injury", "injury_advice"),
    ("muscle strain recovery", "injury_advice"),
    ("rotator cuff pain", "injury_advice"),

    # motivation
    ("I don't feel like working out", "motivation"),
    ("how to stay motivated", "motivation"),
    ("I want to give up", "motivation"),
    ("not seeing results", "motivation"),
    ("tips to stay consistent", "motivation"),
    ("how to build gym habit", "motivation"),

    # general_fitness
    ("how do I lose belly fat", "general_fitness"),
    ("best cardio for fat loss", "general_fitness"),
    ("how long to see results", "general_fitness"),
    ("should I do cardio or weights", "general_fitness"),
    ("how to track progress", "general_fitness"),
    ("what is progressive overload", "general_fitness"),
    ("how much sleep do I need", "general_fitness"),
    ("recovery tips after workout", "general_fitness"),
]

# Map intent → preferred engine
INTENT_TO_ENGINE = {
    "greeting":       "rule_based",
    "bmi_calc":       "rule_based",
    "calorie_calc":   "rule_based",
    "macro_calc":     "rule_based",
    "workout_plan":   "api",
    "nutrition_plan": "api",
    "supplement":     "api",
    "injury_advice":  "api",
    "motivation":     "local_dl",
    "general_fitness": "local_dl",
}


class IntentClassifier:
    """
    TF-IDF + Logistic Regression intent classifier.

    Trains on first call if no saved model exists.
    Uses built-in training data + any saved model from chatbot_training.ipynb.
    """

    _instance = None       # module-level singleton
    _model    = None
    _vectorizer = None
    _trained  = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not IntentClassifier._trained:
            self._load_or_train()

    # ── Load saved model or train from built-in data ───────────────────────

    def _load_or_train(self):
        """Try loading saved model first; fall back to training on built-in data."""
        clf_path = MODEL_DIR / "intent_classifier.pkl"
        vec_path = MODEL_DIR / "intent_vectorizer.pkl"

        try:
            import joblib
            if clf_path.exists() and vec_path.exists():
                IntentClassifier._model      = joblib.load(clf_path)
                IntentClassifier._vectorizer = joblib.load(vec_path)
                IntentClassifier._trained    = True
                logger.info("Intent classifier loaded from disk")
                return
        except Exception as e:
            logger.warning(f"Could not load saved intent model: {e}")

        # Train on built-in data
        self._train_builtin()

    def _train_builtin(self):
        """Train TF-IDF + LogisticRegression on built-in fitness Q&A examples."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            import numpy as np

            texts  = [t for t, _ in BUILTIN_TRAINING_DATA]
            labels = [l for _, l in BUILTIN_TRAINING_DATA]

            vec = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            )
            X = vec.fit_transform(texts)

            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(X, labels)

            IntentClassifier._vectorizer = vec
            IntentClassifier._model      = clf
            IntentClassifier._trained    = True
            logger.info("Intent classifier trained on built-in data")

        except ImportError:
            logger.warning("scikit-learn not available — intent classifier disabled")
        except Exception as e:
            logger.warning(f"Intent classifier training failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def is_trained(self) -> bool:
        return IntentClassifier._trained

    def classify(self, message: str) -> dict:
        """
        Classify a message into an intent.

        Returns:
            {
                "intent":     str,    # detected intent label
                "confidence": float,  # 0.0–1.0
                "engine":     str,    # suggested engine ("rule_based" / "local_dl" / "api")
            }
        """
        if not IntentClassifier._trained or IntentClassifier._model is None:
            return {"intent": "general_fitness", "confidence": 0.5, "engine": "local_dl"}

        try:
            import numpy as np
            X = IntentClassifier._vectorizer.transform([message.lower()])
            proba = IntentClassifier._model.predict_proba(X)[0]
            classes = IntentClassifier._model.classes_
            idx = int(np.argmax(proba))
            intent = classes[idx]
            confidence = float(proba[idx])
            engine = INTENT_TO_ENGINE.get(intent, "local_dl")
            return {"intent": intent, "confidence": confidence, "engine": engine}
        except Exception as e:
            logger.warning(f"Intent classification error: {e}")
            return {"intent": "general_fitness", "confidence": 0.5, "engine": "local_dl"}

    def classify_batch(self, messages: list) -> list:
        """Classify multiple messages at once."""
        return [self.classify(m) for m in messages]

    def save(self, model_dir: Optional[Path] = None):
        """Save the trained classifier to disk."""
        if not IntentClassifier._trained:
            return
        try:
            import joblib
            d = model_dir or MODEL_DIR
            d.mkdir(parents=True, exist_ok=True)
            joblib.dump(IntentClassifier._model,      d / "intent_classifier.pkl")
            joblib.dump(IntentClassifier._vectorizer, d / "intent_vectorizer.pkl")
            logger.info(f"Intent classifier saved to {d}")
        except Exception as e:
            logger.error(f"Failed to save intent classifier: {e}")
