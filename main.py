"""
main.py
────────
APEX AI — Production Entry Point (v5)

Integrates both the new /api routes and the existing /backend routes.
Serves the unmodified frontend from /frontend.

Run:
    uvicorn main:app --reload --port 8000
    # OR the existing backend directly:
    uvicorn backend.main:app --reload --port 8000

Both work — this file wraps both for the unified graduation project.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from dotenv import load_dotenv

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("apex_ai")


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 65)
    print("  🚀 APEX AI — Production Backend v5  (Graduation Project)")
    print("=" * 65)

    # 1. Database
    try:
        from backend.database.db import init_db
        await init_db()
        print("  ✅ SQLite database initialised")
    except Exception as e:
        print(f"  ⚠️  Database: {e}")

    # 2. ML models (new /models path)
    try:
        from services import ml_service
        c = ml_service.calorie_model is not None
        f = ml_service.fitness_model is not None
        r = ml_service.recommender is not None
        print(f"  ✅ ML Models — calorie:{c} | fitness:{f} | recommender:{r}")
    except Exception as e:
        print(f"  ⚠️  ML service: {e}")

    # 3. Legacy ML models (backend/services)
    try:
        from backend.services import ml_service as legacy_ml
        print(f"  ✅ Legacy ML models loaded")
    except Exception as e:
        print(f"  ⚠️  Legacy ML: {e}")

    # 4. LLM chatbot
    try:
        from services.chatbot import get_chatbot_service
        svc    = get_chatbot_service()
        status = svc.get_status()
        llm    = ("Claude" if status["anthropic_available"] else
                  "GPT-4o" if status["openai_available"] else "Rule-based only")
        print(f"  ✅ LLM Chatbot — {llm}")
    except Exception as e:
        print(f"  ⚠️  Chatbot: {e}")

    # 5. PyTorch CV model
    try:
        from backend.services.cv_pytorch import load_pytorch_model
        load_pytorch_model()
        print("  ✅ PyTorch CV model loaded")
    except Exception as e:
        print(f"  ⚠️  PyTorch CV: {e}")

    # 6. NLP intent classifier
    try:
        from backend.chatbot.intent_classifier import IntentClassifier
        ic = IntentClassifier()
        print(f"  ✅ NLP Intent Classifier — trained:{ic.is_trained()}")
    except Exception as e:
        print(f"  ⚠️  Intent classifier: {e}")

    print(f"\n  📖 Swagger docs : http://localhost:8000/docs")
    print(f"  🌐 Frontend app : http://localhost:8000/app")
    print("=" * 65 + "\n")

    yield

    print("\n👋 APEX AI shutting down …")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="APEX AI — Fitness Intelligence Platform",
    description=(
        "Production-grade AI fitness system with real ML models, "
        "LLM chatbot, computer vision, and personalized analytics.\n\n"
        "**Models:** XGBoost calorie predictor, Ensemble fitness classifier, "
        "Content-based recommender, PyTorch exercise classifier\n\n"
        "**LLM:** Anthropic Claude (primary) → OpenAI GPT-4o (fallback) → Rule-based"
    ),
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routes ──────────────────────────────────────────────────────────────

# New production API (v5)
from api.routes import router as api_router
app.include_router(api_router)

# Existing backend routes (backward compatibility — v4)
try:
    from backend.routes import auth, chat, predict, vision, recommend, user_data
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(predict.router)
    app.include_router(vision.router)
    app.include_router(recommend.router)
    app.include_router(user_data.router)
    logger.info("Legacy backend routes mounted")
except Exception as e:
    logger.warning(f"Legacy routes partially unavailable: {e}")

try:
    from backend.rag_coach.routes import router as rag_router
    app.include_router(rag_router)
    print("✅ RAG router registered")
except Exception as e:
    print(f"⚠️ RAG router: {e}")

# ── Serve Frontend (DO NOT MODIFY) ────────────────────────────────────────────
frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/app", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(str(frontend_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
