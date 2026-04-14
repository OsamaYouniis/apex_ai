"""
backend/routes/user_data.py  (v5 — real dashboard analytics)
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
import statistics

from backend.database.db import get_db, UserProfile, WorkoutLog, PredictionLog

router = APIRouter(prefix="/user-data", tags=["User Data & Dashboard"])


class ProfileIn(BaseModel):
    name:           str   = "User"
    age:            int   = Field(default=25, ge=10, le=100)
    weight_kg:      float = Field(default=70.0, ge=20, le=300)
    height_cm:      float = Field(default=175.0, ge=100, le=250)
    activity_level: int   = Field(default=2, ge=1, le=5)
    gender:         int   = Field(default=1, ge=0, le=1)
    goal:           str   = "lose"
    target_weight:  float = 65.0


class WorkoutIn(BaseModel):
    exercise:     str   = "Squats"
    sets:         int   = Field(default=3, ge=1)
    reps:         int   = Field(default=10, ge=0)
    weight_kg:    float = Field(default=0.0, ge=0)
    duration_min: int   = Field(default=30, ge=1)
    reps_counted: int   = 0
    form_score:   float = Field(default=1.0, ge=0, le=1)


@router.post("/profile")
async def upsert_profile(data: ProfileIn, db: AsyncSession = Depends(get_db)):
    result  = await db.execute(select(UserProfile).where(UserProfile.id == 1))
    profile = result.scalar_one_or_none()
    if profile is None:
        profile = UserProfile(id=1)
        db.add(profile)
    for field, val in data.model_dump().items():
        setattr(profile, field, val)
    await db.commit()
    await db.refresh(profile)
    return {"status": "saved", "profile": data.model_dump()}


@router.get("/profile")
async def get_profile(db: AsyncSession = Depends(get_db)):
    result  = await db.execute(select(UserProfile).where(UserProfile.id == 1))
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(404, "No profile found")
    return {
        "name": profile.name, "age": profile.age,
        "weight_kg": profile.weight_kg, "height_cm": profile.height_cm,
        "activity_level": profile.activity_level, "gender": profile.gender,
        "goal": profile.goal, "target_weight": profile.target_weight,
    }


@router.post("/workout")
async def log_workout(data: WorkoutIn, db: AsyncSession = Depends(get_db)):
    log = WorkoutLog(user_id=1, **data.model_dump(), logged_at=datetime.utcnow())
    db.add(log)
    await db.commit()
    return {"status": "logged", "workout": data.model_dump()}


@router.get("/workouts")
async def get_workouts(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WorkoutLog)
        .where(WorkoutLog.user_id == 1)
        .order_by(WorkoutLog.logged_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return [
        {
            "id": r.id, "exercise": r.exercise, "sets": r.sets,
            "reps": r.reps, "weight_kg": r.weight_kg,
            "duration_min": r.duration_min, "reps_counted": r.reps_counted,
            "form_score": r.form_score, "logged_at": r.logged_at.isoformat(),
        }
        for r in rows
    ]


@router.get("/dashboard")
async def get_dashboard(db: AsyncSession = Depends(get_db)):
    """Real-time computed dashboard analytics from stored user data."""
    result  = await db.execute(select(UserProfile).where(UserProfile.id == 1))
    profile = result.scalar_one_or_none()

    wk_result = await db.execute(
        select(WorkoutLog).where(WorkoutLog.user_id == 1)
        .order_by(WorkoutLog.logged_at.asc())
    )
    workouts = wk_result.scalars().all()

    pred_result = await db.execute(
        select(PredictionLog).where(PredictionLog.user_id == 1)
        .order_by(PredictionLog.predicted_at.desc()).limit(30)
    )
    predictions = pred_result.scalars().all()

    now    = datetime.utcnow()
    w30ago = now - timedelta(days=30)
    w7ago  = now - timedelta(days=7)

    total_workouts   = len(workouts)
    total_duration_h = sum(w.duration_min for w in workouts) / 60
    recent_30d       = [w for w in workouts if w.logged_at >= w30ago]
    recent_7d        = [w for w in workouts if w.logged_at >= w7ago]
    wk_per_week      = round(len(recent_30d) / 4.3, 1)

    form_scores = [w.form_score for w in workouts if w.form_score > 0]
    avg_form    = round(statistics.mean(form_scores), 3) if form_scores else 0.0
    improvement_pct = 0.0
    if len(form_scores) >= 6:
        half = len(form_scores) // 2
        early_avg = statistics.mean(form_scores[:half])
        late_avg  = statistics.mean(form_scores[half:])
        improvement_pct = round((late_avg - early_avg) / max(early_avg, 0.001) * 100, 1)

    workout_dates = sorted({w.logged_at.date() for w in workouts}, reverse=True)
    streak = 0
    if workout_dates:
        check = now.date()
        for d in workout_dates:
            if d == check or d == check - timedelta(days=1):
                streak += 1
                check = d - timedelta(days=1)
            else:
                break

    exercise_counts: dict = {}
    for w in workouts:
        exercise_counts[w.exercise] = exercise_counts.get(w.exercise, 0) + 1
    top_exercises = sorted(exercise_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    tdee_history = []
    for p in reversed(predictions):
        out = p.output_data or {}
        if tdee := (out.get("calories_tdee") or out.get("tdee")):
            tdee_history.append({"value": tdee, "date": p.predicted_at.isoformat()})

    tdee_trend = "stable"
    if len(tdee_history) >= 2:
        delta = tdee_history[-1]["value"] - tdee_history[0]["value"]
        tdee_trend = "increasing" if delta > 50 else ("decreasing" if delta < -50 else "stable")

    goal_progress = None
    if profile and profile.target_weight:
        current = profile.weight_kg
        target  = profile.target_weight
        start   = current + (5 if target < current else -5)
        pct     = min(100, round(abs(current - start) / max(abs(target - start), 0.1) * 100, 1))
        goal_progress = {
            "current_weight": current, "target_weight": target,
            "progress_pct": pct, "kg_remaining": round(abs(target - current), 1),
        }

    weekly_volume = sum(
        w.sets * w.reps * max(w.weight_kg, 1) for w in recent_7d
    )

    insights = []
    if total_workouts == 0:
        insights.append("🚀 Log your first workout to start tracking your progress!")
    else:
        if streak >= 7:   insights.append(f"🔥 {streak}-day streak! You're unstoppable!")
        elif streak >= 3: insights.append(f"💪 {streak}-day streak — keep it up!")
        if wk_per_week >= 4:  insights.append(f"⚡ Training {wk_per_week}x/week — elite consistency!")
        elif wk_per_week >= 3: insights.append(f"✅ {wk_per_week} sessions/week — on track!")
        if improvement_pct > 10: insights.append(f"🎯 Form improved {improvement_pct}% — technique levelling up!")
        if tdee_trend == "increasing": insights.append("🔥 Rising TDEE = growing muscle mass!")
        if goal_progress and goal_progress["progress_pct"] >= 50:
            insights.append(f"🏆 {goal_progress['progress_pct']}% to goal weight!")

    return {
        "computed_at": now.isoformat(),
        "has_profile": profile is not None,
        "progress": {
            "total_workouts": total_workouts,
            "total_duration_h": round(total_duration_h, 1),
            "workouts_per_week": wk_per_week,
            "streak_days": streak,
            "weekly_volume_kg": round(weekly_volume),
            "improvement_pct": improvement_pct,
            "avg_form_score": avg_form,
        },
        "calorie_analytics": {
            "tdee_trend": tdee_trend,
            "tdee_history": tdee_history[-10:],
        },
        "goal_progress": goal_progress,
        "recent_workouts": [
            {
                "exercise": w.exercise, "sets": w.sets, "reps": w.reps,
                "weight_kg": w.weight_kg, "duration_min": w.duration_min,
                "form_score": w.form_score, "logged_at": w.logged_at.isoformat(),
            }
            for w in sorted(workouts, key=lambda x: x.logged_at, reverse=True)[:5]
        ],
        "top_exercises": [{"exercise": ex, "count": cnt} for ex, cnt in top_exercises],
        "insights": insights,
    }
