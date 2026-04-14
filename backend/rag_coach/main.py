from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.rag_coach.routes import _graph, _memory, _rag, router as rag_router


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    goal: Optional[str] = None
    activity_level: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    calories: Optional[int] = None
    macros: Optional[Dict[str, int]] = None
    workout_plan: List[str] = Field(default_factory=list)
    meal_plan: List[str] = Field(default_factory=list)

app = FastAPI(
    title="APEX RAG Fitness Coach",
    version="1.0.0",
    description="Standalone backend for LangGraph + LangChain + FAISS fitness coach.",
)

app.include_router(rag_router)


@app.on_event("startup")
def bootstrap_index() -> None:
    _rag.build_or_load_index(force_rebuild=False)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    profile = _memory.upsert_profile(
        req.user_id,
        {
            "age": req.age,
            "weight": req.weight,
            "height": req.height,
            "gender": req.gender,
            "goal": req.goal,
            "activity_level": req.activity_level,
        },
    )

    payload = _graph.invoke(
        message=req.message,
        profile={
            "age": profile.age,
            "weight": profile.weight,
            "height": profile.height,
            "gender": profile.gender,
            "goal": profile.goal,
            "activity_level": profile.activity_level,
        },
    )

    return ChatResponse(
        response=payload.get("response", "I could not generate a response."),
        calories=payload.get("calories"),
        macros=payload.get("macros"),
        workout_plan=payload.get("workout_plan", []),
        meal_plan=payload.get("meal_plan", []),
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag_fitness_coach"}
