"""
services/chatbot.py
────────────────────
APEX AI — Advanced LLM Chatbot Service (Production v5)

Architecture:
  1. Context builder   — assembles user profile + history for LLM
  2. Anthropic Claude  — primary LLM engine (claude-sonnet-4)
  3. OpenAI GPT-4      — fallback LLM engine
  4. Rule engine       — offline fallback (zero API cost)

Features:
  - Rich fitness coach system prompt with persona
  - Full conversation memory (sliding window)
  - User-data personalization in every response
  - Streaming-ready (async generator)
  - Automatic fallback chain
  - Response caching for repeated questions

Usage:
    from services.chatbot import ChatbotService
    svc = ChatbotService()
    reply = await svc.chat("How many calories should I eat?", user_data={...}, history=[...])
"""

import os
import logging
import hashlib
import json
from typing import Optional, AsyncGenerator
from datetime import datetime

logger = logging.getLogger("apex_ai.chatbot_service")

# ─── SYSTEM PROMPT (Fitness Coach Persona) ────────────────────────────────────

SYSTEM_PROMPT = """You are **APEX Coach** — an elite AI fitness coach combining:
- 20+ years as a personal trainer & strength coach
- PhD-level sports nutrition expertise
- Sports psychology and behavioral coaching
- Deep knowledge of anatomy, kinesiology, and exercise science

## Your Coaching Philosophy
You deliver **specific, actionable, science-backed advice** — never vague platitudes.
You treat every user as an individual with unique goals, limitations, and physiology.

## How You Respond
- Lead with the most impactful insight or recommendation
- Use the user's **exact stats** (weight, age, TDEE, goals) — not generic ranges
- Explain the WHY behind every recommendation (builds trust and adherence)
- Use **bold** for key numbers, bullet points for lists, and short paragraphs
- End every response with one **specific, immediate action step**
- Use occasional emojis (💪🔥⚡) for energy — don't overdo it
- Keep responses under 450 words unless the question genuinely requires more

## Your Expertise Covers
Hypertrophy programs, fat loss protocols, TDEE/macro calculation, periodization,
RPE/RIR training, progressive overload, injury prevention & rehab, sleep & recovery,
intermittent fasting, protein timing, supplement evidence, hydration, hormones & cortisol,
mind-muscle connection, form coaching, mental performance.

## Non-negotiables
- Never give medical diagnoses — always recommend a professional for injuries/illness
- Always acknowledge individual variation
- Base recommendations on evidence — cite research when relevant
- Be encouraging but honest: no false promises about results
"""


def _build_user_context(user_data: dict) -> str:
    """Build a rich user-profile string to inject into the system prompt."""
    if not user_data:
        return ""

    age    = user_data.get("age", "N/A")
    weight = user_data.get("weight", user_data.get("weight_kg", "N/A"))
    height = user_data.get("height", user_data.get("height_cm", "N/A"))
    goal   = user_data.get("goal", "N/A")
    act    = user_data.get("activity", user_data.get("activity_level", "N/A"))
    name   = user_data.get("name", "")
    tdee   = user_data.get("tdee", user_data.get("calories_tdee", "N/A"))
    bmi    = user_data.get("bmi", "N/A")
    level  = user_data.get("fitness_level", "N/A")

    activity_names = {1: "Sedentary", 2: "Light", 3: "Moderate", 4: "Active", 5: "Very Active"}

    ctx = f"""
## Current User Profile
- **Name**: {name or 'Anonymous'}
- **Age**: {age} years
- **Weight**: {weight} kg | **Height**: {height} cm | **BMI**: {bmi}
- **Fitness Level**: {level}
- **Activity Level**: {activity_names.get(act, act)}
- **Primary Goal**: {goal}
- **Estimated TDEE**: {tdee} kcal/day
"""
    if target := user_data.get("target_weight"):
        ctx += f"- **Target Weight**: {target} kg\n"
    if diet := user_data.get("dietary_pref"):
        ctx += f"- **Dietary Preference**: {diet}\n"

    ctx += "\nAlways reference these exact numbers in your responses."
    return ctx


def _cache_key(message: str, user_data: dict) -> str:
    """Deterministic cache key for a message+user_data pair."""
    payload = f"{message.lower().strip()}:{user_data.get('goal','')}:{user_data.get('weight','')}:{user_data.get('age','')}"
    return hashlib.md5(payload.encode()).hexdigest()


# ─── RESPONSE CACHE ────────────────────────────────────────────────────────────

_cache: dict[str, dict] = {}
CACHE_MAX_SIZE = 200


def _get_cached(key: str) -> Optional[dict]:
    return _cache.get(key)


def _set_cached(key: str, response: dict):
    if len(_cache) >= CACHE_MAX_SIZE:
        oldest = next(iter(_cache))
        del _cache[oldest]
    _cache[key] = response


# ─── RULE-BASED FALLBACK ───────────────────────────────────────────────────────

_FALLBACK_RESPONSES = [
    ("calorie|tdee|maintenance", lambda ud: (
        f"Your estimated TDEE is **{ud.get('tdee', '~2000')} kcal/day**. "
        f"For fat loss, target {int(ud.get('tdee', 2000)) - 500} kcal; "
        f"for muscle gain, target {int(ud.get('tdee', 2000)) + 300} kcal. "
        f"**Start with a 500 kcal deficit and track for 2 weeks.** 📊"
    )),
    ("protein|macros|nutrition", lambda ud: (
        f"For your weight of **{ud.get('weight', 70)} kg**, aim for "
        f"**{int(float(ud.get('weight', 70)) * 2.0)}g protein/day** "
        f"(2.0g per kg). Split this across 4-5 meals for optimal muscle protein synthesis."
    )),
    ("workout|training|exercise|program", lambda ud: (
        f"Based on your **{ud.get('fitness_level', 'current')} fitness level** and "
        f"**{ud.get('goal', 'fitness')} goal**, I recommend 4 sessions/week: "
        f"2 strength sessions + 2 cardio sessions. Start with compound movements "
        f"(squats, deadlifts, bench press, rows) and progressive overload."
    )),
    ("sleep|recovery|rest", lambda ud: (
        "Sleep is when muscle is built and fat is mobilised. Aim for **7-9 hours**. "
        "Optimise your sleep: dark room (19°C), no screens 1hr before bed, "
        "consistent sleep/wake time. Poor sleep increases cortisol and sabotages results. 😴"
    )),
    ("motivation|stuck|plateau|help", lambda ud: (
        "Plateaus are **normal** — they mean your body has adapted. Solutions: "
        "1) Increase training volume by 10%, 2) Cycle calories (refeed day), "
        "3) Deload week to recover, 4) Change exercise selection. "
        "Track everything — what gets measured gets improved. 💪"
    )),
]


def _rule_response(message: str, user_data: dict) -> str:
    """Simple keyword-matching fallback when APIs are unavailable."""
    import re
    msg_lower = message.lower()
    for pattern, template in _FALLBACK_RESPONSES:
        if re.search(pattern, msg_lower):
            try:
                return template(user_data)
            except Exception:
                continue
    return (
        "I'm here to help with your fitness journey! 💪 Ask me about "
        "nutrition, workouts, calorie targets, macros, recovery, or "
        "motivation. What's your biggest challenge right now?"
    )


# ─── ANTHROPIC CLIENT ─────────────────────────────────────────────────────────

_anthropic_client = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
        return _anthropic_client
    except ImportError:
        logger.warning("anthropic package not installed")
        return None
    except Exception as e:
        logger.warning(f"Anthropic client init failed: {e}")
        return None


# ─── OPENAI CLIENT ────────────────────────────────────────────────────────────

_openai_client = None


def _get_openai():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
        _openai_client = openai.OpenAI(api_key=api_key)
        return _openai_client
    except ImportError:
        logger.warning("openai package not installed")
        return None
    except Exception as e:
        logger.warning(f"OpenAI client init failed: {e}")
        return None


# ─── LLM ENGINES ──────────────────────────────────────────────────────────────

def _call_anthropic(
    message: str,
    history: list[dict],
    system: str,
    max_tokens: int = 800,
) -> Optional[tuple[str, float]]:
    """Call Anthropic Claude. Returns (text, confidence) or None."""
    client = _get_anthropic()
    if not client:
        return None
    try:
        messages = [*history[-8:], {"role": "user", "content": message}]
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        text = resp.content[0].text
        return text, 0.97
    except Exception as e:
        logger.warning(f"Anthropic API error: {e}")
        return None


def _call_openai(
    message: str,
    history: list[dict],
    system: str,
    max_tokens: int = 800,
) -> Optional[tuple[str, float]]:
    """Call OpenAI GPT-4. Returns (text, confidence) or None."""
    client = _get_openai()
    if not client:
        return None
    try:
        oai_msgs = [{"role": "system", "content": system}]
        for h in history[-8:]:
            oai_msgs.append({"role": h["role"], "content": h["content"]})
        oai_msgs.append({"role": "user", "content": message})

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=oai_msgs,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        text = resp.choices[0].message.content
        return text, 0.95
    except Exception as e:
        logger.warning(f"OpenAI API error: {e}")
        return None


# ─── CHATBOT SERVICE ──────────────────────────────────────────────────────────

class ChatbotService:
    """
    Production LLM chatbot service.
    Fallback chain: Anthropic → OpenAI → Rule-based
    """

    def __init__(self):
        self._check_apis()

    def _check_apis(self):
        """Log which APIs are available at startup."""
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai    = bool(os.environ.get("OPENAI_API_KEY"))
        if has_anthropic:
            logger.info("✅ Anthropic API key found — primary LLM: Claude")
        if has_openai:
            logger.info("✅ OpenAI API key found — fallback LLM: GPT-4o")
        if not has_anthropic and not has_openai:
            logger.warning("⚠️  No LLM API keys — using rule-based fallback only")

    async def chat(
        self,
        message: str,
        history: list[dict] = None,
        user_data: dict = None,
        user_id: int = 1,
        use_cache: bool = True,
    ) -> dict:
        """
        Generate a chatbot response.

        Args:
            message:   User's message
            history:   Previous messages [{"role": "user/assistant", "content": "..."}]
            user_data: User profile dict (age, weight, goal, tdee, etc.)
            user_id:   User identifier for logging
            use_cache: Whether to use response cache

        Returns:
            {"reply": str, "source": str, "confidence": float, "model": str}
        """
        if not message or not message.strip():
            return {
                "reply": "What's your fitness question? I'm ready to help! 💪",
                "source": "rule_based",
                "confidence": 1.0,
                "model": "rule_engine",
            }

        message    = message.strip()
        history    = history or []
        user_data  = user_data or {}

        # Check cache
        if use_cache:
            key    = _cache_key(message, user_data)
            cached = _get_cached(key)
            if cached:
                return {**cached, "source": cached["source"] + "_cached"}

        # Build system prompt with user context
        user_ctx = _build_user_context(user_data)
        system   = SYSTEM_PROMPT + (f"\n\n{user_ctx}" if user_ctx else "")

        # Engine 1: Anthropic Claude
        result = _call_anthropic(message, history, system)
        if result:
            reply, conf = result
            response = {"reply": reply, "source": "anthropic_claude",
                        "confidence": conf, "model": "claude-sonnet-4"}
            if use_cache:
                _set_cached(key, response)
            return response

        # Engine 2: OpenAI GPT-4
        result = _call_openai(message, history, system)
        if result:
            reply, conf = result
            response = {"reply": reply, "source": "openai_gpt4",
                        "confidence": conf, "model": "gpt-4o"}
            if use_cache:
                _set_cached(key, response)
            return response

        # Engine 3: Rule-based fallback
        reply = _rule_response(message, user_data)
        response = {"reply": reply, "source": "rule_based",
                    "confidence": 0.80, "model": "rule_engine"}
        if use_cache:
            _set_cached(key, response)
        return response

    async def stream(
        self,
        message: str,
        history: list[dict] = None,
        user_data: dict = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming version — yields tokens as they arrive from Anthropic.
        Falls back to single-shot if streaming unavailable.
        """
        history   = history or []
        user_data = user_data or {}
        user_ctx  = _build_user_context(user_data)
        system    = SYSTEM_PROMPT + (f"\n\n{user_ctx}" if user_ctx else "")

        client = _get_anthropic()
        if client:
            try:
                messages = [*history[-8:], {"role": "user", "content": message}]
                with client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=800,
                    system=system,
                    messages=messages,
                ) as stream:
                    for text in stream.text_stream:
                        yield text
                return
            except Exception as e:
                logger.warning(f"Streaming error: {e}")

        # Fallback: return full response as single chunk
        result = await self.chat(message, history, user_data)
        yield result["reply"]

    def get_status(self) -> dict:
        """Return service status for health checks."""
        return {
            "anthropic_available": _get_anthropic() is not None,
            "openai_available":    _get_openai() is not None,
            "cache_size":          len(_cache),
            "system_prompt_chars": len(SYSTEM_PROMPT),
        }


# ─── SINGLETON ────────────────────────────────────────────────────────────────

_service_instance: Optional[ChatbotService] = None


def get_chatbot_service() -> ChatbotService:
    global _service_instance
    if _service_instance is None:
        _service_instance = ChatbotService()
    return _service_instance
