"""
backend/chatbot/chatbot_service.py  (v5 — LLM-first hybrid)
─────────────────────────────────────────────────────────────
ENGINE PRIORITY:
  1. Knowledge Base (cosine similarity, ~0ms)
  2. Anthropic Claude  (primary LLM)
  3. OpenAI GPT-4o     (secondary LLM)
  4. Rule Engine       (offline fallback)
"""

import os
import logging
from pathlib import Path
from typing import Optional

from backend.chatbot.utils import build_user_context_string, cache_get, cache_set
from backend.chatbot.response_router import classify_message
from backend.chatbot.memory_manager import get_memory, add_message
from backend.chatbot.rule_engine import get_response as rule_response
from backend.chatbot.intent_classifier import IntentClassifier
from backend.chatbot.knowledge_base import FitnessKnowledgeBase

logger = logging.getLogger("apex_ai.chatbot")
ROOT   = Path(__file__).parent.parent.parent

_COMPLEX_INTENTS = {
    "workout_plan", "nutrition_plan", "injury_advice",
    "supplement", "periodization", "fat_loss_protocol"
}


def _run_kb(message: str, user_data: dict) -> Optional[tuple]:
    try:
        kb     = FitnessKnowledgeBase()
        result = kb.search(message)
        if result and result["confidence"] >= 0.20:
            return kb.render(result["entry"], user_data), result["confidence"]
    except Exception as e:
        logger.debug(f"KB error: {e}")
    return None


async def _run_llm(message: str, history: list, user_data: dict) -> Optional[tuple]:
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from services.chatbot import get_chatbot_service
        svc    = get_chatbot_service()
        result = await svc.chat(message=message, history=history,
                                user_data=user_data, use_cache=False)
        if result["source"] in ("anthropic_claude", "openai_gpt4",
                                 "anthropic_claude_cached", "openai_gpt4_cached"):
            return result["reply"], result["confidence"]
    except Exception as e:
        logger.warning(f"LLM error: {e}")
    return None


def _run_rules(message: str, user_data: dict) -> tuple:
    return rule_response(message, user_data), 0.82


def _should_use_llm(message: str, intent: dict, route) -> bool:
    if intent.get("intent") in _COMPLEX_INTENTS:
        return True
    if len(message.split()) > 20:
        return True
    if route.primary == "api":
        return True
    triggers = ["plan", "routine", "programme", "design", "create",
                "best way", "how do i", "should i", "explain", "what is"]
    return any(t in message.lower() for t in triggers)


async def generate_response(message: str, history: list,
                             user_data: Optional[dict] = None,
                             user_id: int = 1) -> dict:
    if not message or not message.strip():
        return {"reply": "Please type a message! 💪",
                "source": "rule_based", "confidence": 1.0}

    message = message.strip()
    ud      = user_data or {}
    memory  = get_memory(user_id, ud)

    cached = cache_get(message, ud)
    if cached:
        reply, confidence, source = cached
        add_message(user_id, "user", message, ud)
        add_message(user_id, "assistant", reply, ud)
        return {"reply": reply, "source": f"{source}_cached", "confidence": confidence}

    try:
        ic     = IntentClassifier()
        intent = ic.classify(message) if ic.is_trained() else {
            "intent": "general_fitness", "confidence": 0.5}
    except Exception:
        intent = {"intent": "general_fitness", "confidence": 0.5}

    route = classify_message(message)
    logger.info(f"[uid={user_id}] '{message[:45]}' | intent={intent['intent']} | route={route.primary}")

    reply = confidence = source = None

    # Engine 1: KB
    kb_result = _run_kb(message, ud)
    if kb_result:
        reply, confidence = kb_result
        source = "knowledge_base"

    # Engine 2+3: LLM
    if reply is None or (confidence < 0.40 and _should_use_llm(message, intent, route)):
        llm_result = await _run_llm(message, memory.build_claude_messages(), ud)
        if llm_result:
            reply, confidence = llm_result
            source = "api"

    # Engine 4: Rules
    if reply is None:
        reply, confidence = _run_rules(message, ud)
        source = "rule_based"

    add_message(user_id, "user", message, ud)
    add_message(user_id, "assistant", reply, ud)
    if source in ("rule_based", "knowledge_base"):
        cache_set(message, ud, reply, confidence, source)

    return {"reply": reply, "source": source, "confidence": confidence}
