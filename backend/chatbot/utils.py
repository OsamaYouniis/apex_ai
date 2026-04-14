"""
backend/chatbot/utils.py
─────────────────────────
Shared utility functions used across the chatbot package.

Includes:
  - BMR / TDEE calculation
  - BMI calculation and categorisation
  - Macro calculation
  - Text cleaning / normalisation
  - Confidence scoring helpers
  - Response caching
"""

import re
import time
import hashlib
from functools import lru_cache
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# FITNESS CALCULATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def calc_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """
    Mifflin-St Jeor BMR formula.
    Most accurate formula for general population.
    Male  : (10 × weight) + (6.25 × height) − (5 × age) + 5
    Female: (10 × weight) + (6.25 × height) − (5 × age) − 161
    """
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + 5 if gender in ("m", "male", 1, "1") else base - 161


def calc_tdee(weight_kg: float, height_cm: float, age: int,
              gender: str, activity_level: int) -> int:
    """
    Total Daily Energy Expenditure = BMR × activity multiplier.
    Activity levels:
      1 = Sedentary       (desk job, no exercise)
      2 = Lightly active  (1–3 workouts/week)
      3 = Moderately      (3–5 workouts/week)
      4 = Very active     (6–7 workouts/week)
      5 = Athlete         (2x/day training)
    """
    MULTIPLIERS = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
    bmr = calc_bmr(weight_kg, height_cm, age, gender)
    mult = MULTIPLIERS.get(int(activity_level), 1.375)
    return round(bmr * mult)


def calc_bmi(weight_kg: float, height_cm: float) -> float:
    """BMI = weight (kg) / height (m)²"""
    h_m = height_cm / 100
    return round(weight_kg / (h_m ** 2), 1)


def bmi_category(bmi: float) -> str:
    """WHO BMI classification."""
    if bmi < 18.5: return "Underweight"
    if bmi < 25.0: return "Normal weight"
    if bmi < 30.0: return "Overweight"
    if bmi < 35.0: return "Obese (Class I)"
    return "Obese (Class II+)"


def calc_macros(calories: int, goal: str, weight_kg: float) -> dict:
    """
    Calculate macro targets based on calorie goal.
    Protein  : 2.0g/kg bodyweight for lose/build, 1.6g/kg for maintain
    Fats     : 25% of calories
    Carbs    : remainder
    """
    protein_g = round(weight_kg * (2.0 if goal in ("lose", "build") else 1.6))
    fat_g     = round((calories * 0.25) / 9)
    carb_g    = round((calories - protein_g * 4 - fat_g * 9) / 4)
    return {
        "protein_g": protein_g,
        "fat_g":     max(fat_g, 30),        # minimum 30g fat for health
        "carb_g":    max(carb_g, 50),        # minimum 50g carbs
        "calories":  calories,
    }


def water_target_liters(weight_kg: float) -> float:
    """Daily water target: 33ml per kg bodyweight."""
    return round(weight_kg * 0.033, 1)


def target_heart_rate_zone(age: int) -> dict:
    """
    Calculate heart rate zones based on age.
    Max HR = 220 − age (Haskell & Fox formula)
    """
    max_hr = 220 - age
    return {
        "max_hr":      max_hr,
        "fat_burn":    (round(max_hr * 0.6), round(max_hr * 0.7)),
        "cardio":      (round(max_hr * 0.7), round(max_hr * 0.8)),
        "peak":        (round(max_hr * 0.8), round(max_hr * 0.9)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEXT PROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise and clean input text for model processing."""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'[^\w\s\?\!\.,\-]', '', text)  # remove special chars
    return text


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from user message."""
    # Remove stopwords
    STOPWORDS = {
        "i","me","my","the","a","an","is","it","to","do","for","and",
        "or","but","in","on","at","with","can","you","what","how","why",
        "when","where","who","will","should","would","could","please","tell",
        "give","want","need","help","about","some","any","this","that","make"
    }
    words = clean_text(text).split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def detect_language_tone(text: str) -> str:
    """
    Detect the tone of the message.
    Returns: 'greeting' | 'question' | 'motivational' | 'statement'
    """
    t = text.lower().strip()
    GREETINGS = ["hello","hi","hey","good morning","good evening","sup","what's up","greetings","salam","marhaba"]
    if any(t.startswith(g) or t == g for g in GREETINGS):
        return "greeting"
    if t.endswith("?") or any(t.startswith(w) for w in ["how","what","why","when","where","which","can","should","is"]):
        return "question"
    if any(w in t for w in ["motivat","inspire","give up","tired","lazy","stuck","help me"]):
        return "motivational"
    return "statement"


def build_user_context_string(user_data: dict) -> str:
    """
    Convert user profile dict into a readable context string
    used in model prompts and rule engine.
    """
    if not user_data:
        return ""
    w   = float(user_data.get("weight_kg", 70))
    h   = float(user_data.get("height_cm", 175))
    age = int(user_data.get("age", 25))
    act = int(user_data.get("activity_level", 2))
    g   = user_data.get("gender", "m")
    goal= user_data.get("goal", "maintain")
    name= user_data.get("name", "User")

    tdee = calc_tdee(w, h, age, g, act)
    bmi  = calc_bmi(w, h)

    return (
        f"User: {name} | Age: {age} | Weight: {w}kg | Height: {h}cm | "
        f"Gender: {'Male' if g in ('m','male',1,'1') else 'Female'} | "
        f"Activity: {act}/5 | Goal: {goal} | "
        f"TDEE: {tdee} kcal | BMI: {bmi} ({bmi_category(bmi)}) | "
        f"Protein target: {round(w*2)}g/day"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE CACHE  (in-memory, avoids re-computing identical questions)
# ─────────────────────────────────────────────────────────────────────────────

_response_cache: dict[str, tuple[str, float, str]] = {}
# key → (reply, confidence, source)
CACHE_TTL_SECONDS = 300   # 5-minute cache


def _cache_key(message: str, user_weight: float, user_goal: str) -> str:
    """
    Build a cache key from message + key profile fields.
    We include weight and goal so different users get different cached answers.
    """
    raw = f"{clean_text(message)}|{user_weight}|{user_goal}"
    return hashlib.md5(raw.encode()).hexdigest()


def cache_get(message: str, user_data: dict) -> Optional[tuple[str, float, str]]:
    """Return cached (reply, confidence, source) or None if miss/expired."""
    key = _cache_key(
        message,
        user_data.get("weight_kg", 0),
        user_data.get("goal", "")
    )
    entry = _response_cache.get(key)
    if entry and (time.time() - entry[3]) < CACHE_TTL_SECONDS:
        return entry[0], entry[1], entry[2]
    return None


def cache_set(message: str, user_data: dict,
              reply: str, confidence: float, source: str):
    """Store a response in the cache with timestamp."""
    key = _cache_key(
        message,
        user_data.get("weight_kg", 0),
        user_data.get("goal", "")
    )
    _response_cache[key] = (reply, confidence, source, time.time())


def clear_cache():
    """Clear all cached responses."""
    _response_cache.clear()
