from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.rag_coach.graph import FitnessGraph
from backend.rag_coach.memory import SessionMemory
from backend.rag_coach.rag import RAGService


router = APIRouter(prefix="/rag", tags=["RAG Fitness Coach"])

_docs_dir = os.getenv("RAG_DOCS_DIR", "knowledge_base/raw")
_index_dir = os.getenv("RAG_INDEX_DIR", "knowledge_base/faiss_index")

_memory = SessionMemory()
_rag = RAGService(docs_dir=_docs_dir, index_dir=_index_dir)
_graph = FitnessGraph(_rag)


class RagChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    goal: Optional[str] = None
    activity_level: Optional[str] = None


class RagChatResponse(BaseModel):
    response: str
    calories: Optional[int] = None
    macros: Optional[Dict[str, int]] = None
    workout_plan: List[str] = Field(default_factory=list)
    meal_plan: List[str] = Field(default_factory=list)


@router.on_event("startup")
def _startup_index() -> None:
    _rag.build_or_load_index(force_rebuild=False)


@router.post("/chat", response_model=RagChatResponse)
def rag_chat(req: RagChatRequest):
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

    if not payload:
        raise HTTPException(status_code=500, detail="RAG graph returned empty payload")

    return RagChatResponse(
        response=payload.get("response", "I could not generate a response."),
        calories=payload.get("calories"),
        macros=payload.get("macros"),
        workout_plan=payload.get("workout_plan", []),
        meal_plan=payload.get("meal_plan", []),
    )


@router.post("/reindex")
def rag_reindex(admin_key: str):
    expected_key = os.getenv("RAG_ADMIN_KEY")
    if expected_key and admin_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    chunk_count = _rag.build_or_load_index(force_rebuild=True)
    return {
        "status": "ok",
        "chunks_indexed": chunk_count,
        "docs_dir": _docs_dir,
    }


@router.get("/sources")
def rag_sources():
    return {
        "docs_dir": _docs_dir,
        "files": _rag.list_source_files(),
    }
