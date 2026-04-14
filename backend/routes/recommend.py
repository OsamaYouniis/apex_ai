"""
backend/routes/recommend.py
"""

from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.ml_service import classify_fitness, get_recommendations

router = APIRouter(prefix="/recommend", tags=["Recommendations"])


class RecommendRequest(BaseModel):
    age:            int   = 25
    weight_kg:      float = 70.0
    height_cm:      float = 175.0
    activity_level: int   = 2
    gender:         int   = 1
    top_k:          int   = 5


@router.post("")
async def recommend(req: RecommendRequest):
    """
    Classify user fitness level then return personalised workout recommendations.
    """
    fitness = classify_fitness(
        req.age, req.weight_kg, req.height_cm,
        req.activity_level, req.gender
    )
    recs = get_recommendations(fitness["level_id"], req.top_k)

    return {
        "fitness_level": fitness["level_name"],
        "fitness_level_id": fitness["level_id"],
        "recommendations": recs,
    }
