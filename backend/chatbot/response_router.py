"""
backend/chatbot/response_router.py
────────────────────────────────────
Smart Response Router — classifies every incoming message and decides
which engine should handle it:

  ┌─────────────────────────────────────────────────────────┐
  │  ROUTING DECISION TREE                                  │
  │                                                         │
  │  Input message                                          │
  │      │                                                  │
  │      ├── GREETING?          → rule_based (instant)      │
  │      ├── FACTUAL FITNESS?   → rule_based (precise)      │
  │      │     (BMI, TDEE, macros, named exercises)         │
  │      ├── CONVERSATIONAL?    → local_dl (natural)        │
  │      │     (chit-chat, motivation, open questions)      │
  │      └── COMPLEX / ADVICE?  → api (most intelligent)    │
  │            (multi-part, medical, nuanced)               │
  │                                                         │
  │  If chosen engine fails → fall down to next engine      │
  └─────────────────────────────────────────────────────────┘

Confidence scoring:
  - rule_based: 0.90–0.99  (deterministic, very reliable)
  - local_dl:   0.65–0.85  (probabilistic, good for open questions)
  - api:        0.95–0.99  (most capable, network-dependent)
"""

import re
from dataclasses import dataclass
from typing import Literal

EngineType = Literal["rule_based", "local_dl", "api"]


@dataclass
class RouteDecision:
    """Result of routing analysis."""
    primary:    EngineType    # first engine to try
    secondary:  EngineType    # fallback if primary fails
    tertiary:   EngineType    # final fallback
    confidence: float         # expected confidence for primary route
    reason:     str           # human-readable explanation (for logging)
    topic:      str           # detected topic (for analytics)


# ── Keyword topic maps ────────────────────────────────────────────────────────

GREETING_TRIGGERS = {
    "hello","hi","hey","good morning","good evening","good afternoon",
    "good night","sup","what's up","whats up","howdy","greetings",
    "salam","marhaba","salamu","assalam","morning","evening","yo ","hiya"
}

FACTUAL_TRIGGERS = {
    # Calculations
    "bmi","tdee","bmr","calori","macro","protein","carb","fat","kcal",
    "caloric","deficit","surplus","maintenance",
    # Specific exercises (named)
    "squat","deadlift","bench press","overhead press","pull.up","push.up",
    "plank","lunge","curl","row","dip","crunch","burpee",
    # Specific topics with known answers
    "creatine","supplement","vitamin","whey","bcaa","pre.workout",
    "intermittent fasting","16:8","18:6","keto","vegan","vegetarian",
    "water","hydration","sleep","recovery","rep","set","1rm",
    "heart rate","target heart rate","vo2","metabolism",
    # Progress tracking
    "weight loss","fat loss","muscle gain","bulk","cut",
    "how much protein","how many calories","how much water",
    "calculate my","what is my",
}

MOTIVATIONAL_TRIGGERS = {
    "motivat","inspire","give up","tired","lazy","stuck","plateau",
    "not working","discouraged","feel like","struggling","cant do",
    "too hard","no progress","why bother","is it worth","keep going",
    "want to quit","demotivat","depressed","frustrated","fail",
}

COMPLEX_TRIGGERS = {
    # Multi-part questions
    "compare","difference between","vs","versus","better for","which is",
    "pros and cons","advantages","disadvantages","explain","elaborate",
    # Medical / injury
    "injury","pain","hurt","recover from","rehab","doctor","medical",
    "muscle pull","strain","sprain","inflammation","tendon","ligament",
    # Program design
    "design a program","create a plan","build me a","personalise",
    "periodiz","progressive overload program","powerlifting","olymp",
    # Nuanced nutrition
    "cardio tips","cardio for fat loss","cardio advice",
    "recomp","body recomposition","reverse diet","diet break",
    "cheat meal","refeed","anabolic","catabolic",
}

CONVERSATIONAL_TRIGGERS = {
    "how are you","what do you think","do you","can you","tell me about",
    "what should i","advice","suggest","recommend","opinion","thought",
    "what would","feels like","i am","i feel","my goal","my plan",
}


def classify_message(text: str) -> RouteDecision:
    """
    Analyse a user message and decide which engine should handle it.

    Priority order:
      1. Greeting           → rule_based  (fast, personalised)
      2. Factual fitness    → rule_based  (precise, data-driven)
      3. Motivational       → local_dl    (empathetic tone)
      4. Complex / nuanced  → api         (Claude is best here)
      5. Conversational     → local_dl    (natural dialogue)
      6. Default            → local_dl    (let DL model try)
    """
    t = text.lower().strip()

    # ── 1. Greeting ──────────────────────────────────────────────────────────
    if any(t.startswith(g) or t == g for g in GREETING_TRIGGERS):
        return RouteDecision(
            primary="rule_based", secondary="local_dl", tertiary="api",
            confidence=0.98, reason="Greeting detected",
            topic="greeting"
        )

    # ── 2. Factual fitness (keyword match) ───────────────────────────────────
    factual_hits = sum(1 for kw in FACTUAL_TRIGGERS if re.search(r'\b' + kw, t))
    if factual_hits >= 1:
        return RouteDecision(
            primary="rule_based", secondary="api", tertiary="local_dl",
            confidence=0.93,
            reason=f"Factual fitness query ({factual_hits} keyword hits)",
            topic="factual_fitness"
        )

    # ── 3. Motivational ──────────────────────────────────────────────────────
    motiv_hits = sum(1 for kw in MOTIVATIONAL_TRIGGERS if kw in t)
    if motiv_hits >= 1:
        return RouteDecision(
            primary="rule_based", secondary="local_dl", tertiary="api",
            confidence=0.88,
            reason=f"Motivational/emotional query ({motiv_hits} hits)",
            topic="motivational"
        )

    # ── 4. Complex / nuanced ────────────────────────────────────────────────
    complex_hits = sum(1 for kw in COMPLEX_TRIGGERS if re.search(r'\b' + re.escape(kw), t))
    long_question = len(t.split()) > 20
    if complex_hits >= 2 or (complex_hits >= 1 and long_question):
        return RouteDecision(
            primary="api", secondary="rule_based", tertiary="local_dl",
            confidence=0.95,
            reason=f"Complex query detected ({complex_hits} complex keywords, words={len(t.split())})",
            topic="complex"
        )

    # ── 5. Conversational ────────────────────────────────────────────────────
    conv_hits = sum(1 for kw in CONVERSATIONAL_TRIGGERS if kw in t)
    if conv_hits >= 1:
        return RouteDecision(
            primary="local_dl", secondary="rule_based", tertiary="api",
            confidence=0.72,
            reason=f"Conversational query ({conv_hits} hits)",
            topic="conversational"
        )

    # ── 6. Default → local DL ────────────────────────────────────────────────
    return RouteDecision(
        primary="local_dl", secondary="rule_based", tertiary="api",
        confidence=0.68,
        reason="Default routing — unclassified query sent to DL model",
        topic="general"
    )


def confidence_for_source(source: EngineType, route: RouteDecision) -> float:
    """
    Return appropriate confidence score based on which engine
    actually produced the answer vs which was originally recommended.
    """
    if source == route.primary:
        return round(route.confidence, 2)
    if source == route.secondary:
        return round(route.confidence * 0.85, 2)   # secondary is slightly less confident
    return round(route.confidence * 0.70, 2)        # tertiary fallback
