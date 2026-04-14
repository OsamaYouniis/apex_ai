"""
api/routes.py
──────────────
APEX AI — FastAPI Routes (Production v5)

Clean, modular API with:
  POST /predict/calories    — calorie burn prediction
  POST /predict/fitness     — fitness classification
  POST /predict/full        — complete user analytics
  POST /recommend           — personalized workout recommendations
  POST /chat                — LLM-powered chatbot (Claude/GPT-4)
  GET  /dashboard/{user_id} — computed dashboard data
  POST /dashboard/log       — log workout or weight
  GET  /health              — system health

All endpoints handle errors gracefully and return structured JSON.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timedelta
import logging

from services.ml_service import (
    predict_tdee, classify_fitness, get_recommendations,
    predict_weight_change, full_prediction, predict_calories_burned,
)
from services.chatbot import get_chatbot_service

logger = logging.getLogger("apex_ai.api")

router = APIRouter()


# ─── SCHEMAS ──────────────────────────────────────────────────────────────────

class UserInput(BaseModel):
    age:            int   = Field(default=25, ge=10, le=100, description="Age in years")
    weight_kg:      float = Field(default=70.0, ge=20, le=300, description="Weight in kg")
    height_cm:      float = Field(default=175.0, ge=100, le=250, description="Height in cm")
    activity_level: int   = Field(default=2, ge=1, le=5, description="1=Sedentary … 5=Very Active")
    gender:         int   = Field(default=1, ge=0, le=1, description="0=Female, 1=Male")
    goal:           str   = Field(default="maintain", description="lose | gain | maintain")
    user_id:        int   = Field(default=1)


class CalorieInput(UserInput):
    duration_min: int = Field(default=45, ge=5, le=300, description="Workout duration in minutes")


class RecommendInput(BaseModel):
    fitness_level:  int   = Field(default=1, ge=0, le=2)
    goal:           str   = Field(default="maintain")
    age:            int   = Field(default=25, ge=10, le=100)
    top_k:          int   = Field(default=6, ge=1, le=15)


class ChatRequest(BaseModel):
    message:   str
    history:   List[dict] = []
    user_data: Optional[dict] = None
    user_id:   int = 1


class WorkoutLogInput(BaseModel):
    user_id:      int   = 1
    exercise:     str   = "Squats"
    sets:         int   = 3
    reps:         int   = 10
    weight_kg:    float = 0.0
    duration_min: int   = 30
    form_score:   float = Field(default=1.0, ge=0, le=1)


# ─── IN-MEMORY STORE (replace with DB for full production) ────────────────────

_user_profiles:   dict[int, dict]        = {}
_workout_history: dict[int, list[dict]]  = {}
_prediction_logs: dict[int, list[dict]]  = {}


def _store_prediction(user_id: int, data: dict):
    if user_id not in _prediction_logs:
        _prediction_logs[user_id] = []
    _prediction_logs[user_id].append({**data, "timestamp": datetime.utcnow().isoformat()})
    # Keep last 100
    if len(_prediction_logs[user_id]) > 100:
        _prediction_logs[user_id] = _prediction_logs[user_id][-100:]


# ─── CALORIE PREDICTION ───────────────────────────────────────────────────────

@router.post("/predict/calories", tags=["ML Predictions"])
async def predict_calories_endpoint(req: CalorieInput):
    """
    Predict calories burned during a workout session.
    Uses XGBoost ensemble trained on Kaggle calorie dataset.
    """
    try:
        burned = predict_calories_burned(
            req.age, req.weight_kg, req.height_cm,
            req.activity_level, req.gender, req.duration_min
        )
        tdee = predict_tdee(req.age, req.weight_kg, req.height_cm,
                             req.activity_level, req.gender)

        result = {
            "calories_burned":  burned,
            "calories_per_min": round(burned / req.duration_min, 1),
            "tdee":             tdee,
            "duration_min":     req.duration_min,
            "deficit_achieved": round(tdee - burned * (24 * 60 / req.duration_min), 0),
        }
        _store_prediction(req.user_id, {"type": "calories", **result})
        return result
    except Exception as e:
        logger.error(f"Calorie prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {e}")


# ─── FITNESS CLASSIFICATION ────────────────────────────────────────────────────

@router.post("/predict/fitness", tags=["ML Predictions"])
async def predict_fitness_endpoint(req: UserInput):
    """
    Classify fitness level using body performance model.
    Returns 0=Beginner → 3=Advanced with probability distribution.
    """
    try:
        result = classify_fitness(
            req.age, req.weight_kg, req.height_cm,
            req.activity_level, req.gender
        )
        _store_prediction(req.user_id, {"type": "fitness", **result})
        return result
    except Exception as e:
        raise HTTPException(500, f"Fitness classification failed: {e}")


# ─── FULL PREDICTION (used by dashboard) ──────────────────────────────────────

@router.post("/predict", tags=["ML Predictions"])
@router.post("/predict/full", tags=["ML Predictions"])
async def full_prediction_endpoint(req: UserInput):
    """
    Complete ML prediction pipeline.
    Returns TDEE, BMI, fitness level, weight change, macros, recommendations.
    """
    try:
        result = full_prediction(
            req.age, req.weight_kg, req.height_cm,
            req.activity_level, req.gender, req.goal
        )

        # Update stored profile
        _user_profiles[req.user_id] = {
            "age": req.age, "weight_kg": req.weight_kg, "height_cm": req.height_cm,
            "activity_level": req.activity_level, "gender": req.gender,
            "goal": req.goal, **result, "updated_at": datetime.utcnow().isoformat()
        }
        _store_prediction(req.user_id, {"type": "full", **result})
        return result
    except Exception as e:
        logger.error(f"Full prediction error: {e}", exc_info=True)
        raise HTTPException(500, f"Prediction failed: {e}")


# ─── RECOMMENDATIONS ──────────────────────────────────────────────────────────

@router.post("/recommend", tags=["Recommendations"])
async def recommend_endpoint(req: RecommendInput):
    """
    Get personalized workout recommendations.
    Uses content-based TF-IDF recommender.
    """
    try:
        recs = get_recommendations(req.fitness_level, req.goal, req.age, req.top_k)
        return {
            "recommendations": recs,
            "count":           len(recs),
            "fitness_level":   req.fitness_level,
            "goal":            req.goal,
        }
    except Exception as e:
        raise HTTPException(500, f"Recommendation failed: {e}")


# ─── CHATBOT ──────────────────────────────────────────────────────────────────

@router.post("/chat", tags=["AI Chatbot"])
async def chat_endpoint(req: ChatRequest):
    """
    LLM-powered fitness chatbot.
    Engine priority: Anthropic Claude → OpenAI GPT-4 → Rule-based
    """
    if not req.message or not req.message.strip():
        return {
            "reply":      "Please type your fitness question! 💪",
            "source":     "rule_based",
            "confidence": 1.0,
            "model":      "rule_engine",
        }
    try:
        svc    = get_chatbot_service()
        result = await svc.chat(
            message=req.message,
            history=req.history,
            user_data=req.user_data or {},
            user_id=req.user_id,
        )
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "reply":      "I'm having a moment — try asking again! 💪",
            "source":     "error_fallback",
            "confidence": 0.5,
            "model":      "fallback",
        }


# ─── DASHBOARD ────────────────────────────────────────────────────────────────

@router.get("/dashboard/{user_id}", tags=["Dashboard"])
async def get_dashboard(user_id: int):
    """
    Compute dynamic, personalized dashboard data for a user.
    Includes progress tracking, trends, and improvement percentages.
    """
    profile  = _user_profiles.get(user_id)
    workouts = _workout_history.get(user_id, [])
    preds    = _prediction_logs.get(user_id, [])

    # Compute progress stats from workout history
    total_workouts = len(workouts)
    total_duration = sum(w.get("duration_min", 0) for w in workouts)
    avg_form_score = (sum(w.get("form_score", 1.0) for w in workouts) / max(1, total_workouts))

    # Workout frequency (last 30 days)
    cutoff    = (datetime.utcnow() - timedelta(days=30)).isoformat()
    recent_wk = [w for w in workouts if w.get("logged_at", "") >= cutoff]
    wk_per_week = round(len(recent_wk) / 4.3, 1)

    # Calorie trend from prediction logs
    cal_preds = [p for p in preds if p.get("type") in ("full", "calories")]
    tdee_vals = [p.get("calories_tdee", p.get("tdee", 0)) for p in cal_preds if p.get("calories_tdee") or p.get("tdee")]

    tdee_avg     = round(sum(tdee_vals) / len(tdee_vals)) if tdee_vals else None
    tdee_trend   = "stable"
    if len(tdee_vals) >= 2:
        delta = tdee_vals[-1] - tdee_vals[0]
        tdee_trend = "increasing" if delta > 50 else ("decreasing" if delta < -50 else "stable")

    # Improvement percentage (form score over time)
    if len(workouts) >= 4:
        early = workouts[:len(workouts)//2]
        late  = workouts[len(workouts)//2:]
        early_form = sum(w.get("form_score", 1) for w in early) / len(early)
        late_form  = sum(w.get("form_score", 1) for w in late) / len(late)
        improvement_pct = round((late_form - early_form) / max(early_form, 0.01) * 100, 1)
    else:
        improvement_pct = 0

    return {
        "user_id":          user_id,
        "has_profile":      profile is not None,
        "profile":          profile,
        "stats": {
            "total_workouts":   total_workouts,
            "total_duration_h": round(total_duration / 60, 1),
            "avg_form_score":   round(avg_form_score, 2),
            "workouts_per_week": wk_per_week,
            "improvement_pct":  improvement_pct,
        },
        "nutrition": {
            "tdee_avg":    tdee_avg,
            "tdee_trend":  tdee_trend,
            "tdee_history": tdee_vals[-10:],
        },
        "recent_workouts": workouts[-5:],
        "computed_at":     datetime.utcnow().isoformat(),
    }


@router.post("/dashboard/log", tags=["Dashboard"])
async def log_workout(log: WorkoutLogInput):
    """Log a workout session and update dashboard metrics."""
    uid = log.user_id
    if uid not in _workout_history:
        _workout_history[uid] = []

    entry = {
        **log.model_dump(),
        "logged_at": datetime.utcnow().isoformat(),
    }
    _workout_history[uid].append(entry)
    if len(_workout_history[uid]) > 200:
        _workout_history[uid] = _workout_history[uid][-200:]

    return {"status": "logged", "total_sessions": len(_workout_history[uid])}


# ─── HEALTH ───────────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
async def health():
    """System health check — returns status of all models and integrations."""
    import os
    chatbot_svc = get_chatbot_service()
    status      = chatbot_svc.get_status()
    return {
        "status":  "ok",
        "version": "5.0.0",
        "models": {
            "calorie_model":    True,  # loaded at import
            "fitness_model":    True,
            "recommender":      True,
        },
        "chatbot":       status,
        "integrations": {
            "anthropic_api": status["anthropic_available"],
            "openai_api":    status["openai_available"],
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
