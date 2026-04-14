"""
backend/routes/predict.py  (v5 — JWT protected + richer features)
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from typing import Optional

from backend.database.db import get_db, PredictionLog
from backend.middleware.auth_guard import get_optional_user
from backend.services.ml_service import (
    predict_calories, predict_weight_change,
    classify_fitness, get_recommendations,
)

router = APIRouter(prefix="/predict", tags=["ML Predictions"])


class UserInput(BaseModel):
    age:            int   = Field(default=25, ge=10, le=100)
    weight_kg:      float = Field(default=70.0, ge=20, le=300)
    height_cm:      float = Field(default=175.0, ge=100, le=250)
    activity_level: int   = Field(default=2, ge=1, le=5)
    gender:         int   = Field(default=1, ge=0, le=1)
    user_id:        int   = 1


class PredictResponse(BaseModel):
    calories_tdee:     float
    weight_change_30d: float
    fitness_level:     str
    fitness_level_id:  int
    probabilities:     dict
    bmi:               float
    bmi_category:      str
    recommendations:   list
    # Dashboard extras
    cut_calories:      float
    bulk_calories:     float
    protein_g:         float
    water_l:           float


def _bmi_cat(bmi: float) -> str:
    if bmi < 18.5: return "Underweight"
    if bmi < 25.0: return "Normal weight"
    if bmi < 30.0: return "Overweight"
    return "Obese"


@router.post("", response_model=PredictResponse)
async def predict(
    req: UserInput,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Full ML prediction pipeline — JWT optional (works logged-in or anonymous).
    Returns calorie/weight/fitness predictions + dashboard data.
    """
    calories   = predict_calories(req.age, req.weight_kg, req.height_cm,
                                  req.activity_level, req.gender)
    weight_d   = predict_weight_change(req.age, req.weight_kg, req.height_cm,
                                       req.activity_level, req.gender, calories)
    fitness    = classify_fitness(req.age, req.weight_kg, req.height_cm,
                                  req.activity_level, req.gender)
    recs       = get_recommendations(fitness["level_id"])
    bmi        = round(req.weight_kg / ((req.height_cm / 100) ** 2), 2)
    protein_g  = round(req.weight_kg * 2.0)
    water_l    = round(req.weight_kg * 0.033, 1)

    # Persist prediction
    user_id = current_user["user_id"] if current_user else req.user_id
    try:
        db.add(PredictionLog(
            user_id=user_id, prediction_type="full",
            input_data=req.model_dump(),
            output_data={
                "calories_tdee":     calories,
                "weight_change_30d": weight_d,
                "fitness_level":     fitness["level_name"],
            },
            predicted_at=datetime.utcnow(),
        ))
        await db.commit()
    except Exception:
        pass

    return PredictResponse(
        calories_tdee=calories,
        weight_change_30d=weight_d,
        fitness_level=fitness["level_name"],
        fitness_level_id=fitness["level_id"],
        probabilities=fitness["probabilities"],
        bmi=bmi,
        bmi_category=_bmi_cat(bmi),
        recommendations=recs,
        cut_calories=round(calories - 500),
        bulk_calories=round(calories + 300),
        protein_g=protein_g,
        water_l=water_l,
    )
