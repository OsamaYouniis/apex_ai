"""
backend/routes/chat.py
───────────────────────
Chat API endpoint — routes to the production hybrid chatbot.

Returns:
  {
    "reply":      str,    # chatbot response text
    "source":     str,    # "local_dl" | "api" | "rule_based" | "*_cached"
    "confidence": float,  # 0.0 – 1.0 confidence score
    "model_used": str,    # human-readable model name
  }
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from backend.database.db import get_db, ChatHistory
from backend.chatbot.chatbot_service import generate_response
from backend.chatbot.memory_manager import memory_stats, clear_memory
from backend.chatbot import model_loader

router = APIRouter(prefix="/chat", tags=["AI Chatbot"])


class ChatRequest(BaseModel):
    message:   str
    history:   list[dict] = []
    user_data: Optional[dict] = None
    user_id:   int = 1


class ChatResponse(BaseModel):
    reply:      str
    source:     str
    confidence: float
    model_used: str


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Hybrid AI chatbot endpoint.

    Priority order:
      1. Local DialoGPT-medium (conversational / motivational queries)
      2. Anthropic Claude API  (complex / nuanced queries)
      3. Smart rule-based engine (factual fitness / greetings / fallback)

    All responses include source and confidence score.
    """
    # Validate input
    if not req.message or not req.message.strip():
        return ChatResponse(
            reply="Please type a message! I'm here to help 💪",
            source="rule_based",
            confidence=1.0,
            model_used="rule_based_engine",
        )

    # Generate response through the hybrid engine
    result = await generate_response(
        message=req.message,
        history=req.history,
        user_data=req.user_data,
        user_id=req.user_id,
    )

    # Persist to database for chat history
    try:
        db.add(ChatHistory(
            user_id=req.user_id, role="user",
            content=req.message, timestamp=datetime.utcnow()
        ))
        db.add(ChatHistory(
            user_id=req.user_id, role="assistant",
            content=result["reply"], timestamp=datetime.utcnow()
        ))
        await db.commit()
    except Exception:
        pass  # DB persistence failure should not break the response

    return ChatResponse(
        reply=result["reply"],
        source=result["source"],
        confidence=result["confidence"],
        model_used=model_loader.get_model_name(),
    )


@router.get("/history/{user_id}")
async def get_history(user_id: int, limit: int = 50,
                      db: AsyncSession = Depends(get_db)):
    """Retrieve stored chat history for a user from the database."""
    result = await db.execute(
        select(ChatHistory)
        .where(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.timestamp.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return [
        {
            "role":      r.role,
            "content":   r.content,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in reversed(rows)
    ]


@router.delete("/history/{user_id}")
async def clear_history(user_id: int, db: AsyncSession = Depends(get_db)):
    """Clear chat history and in-memory conversation context for a user."""
    clear_memory(user_id)
    # Optionally also clear DB history
    return {"status": "ok", "message": f"Memory cleared for user {user_id}"}


@router.get("/stats")
async def chat_stats():
    """Return chatbot memory and model statistics."""
    return {
        "model":   model_loader.get_model_name(),
        "ready":   model_loader.is_model_available(),
        "memory":  memory_stats(),
    }
