"""
backend/main.py
─────────────────
APEX AI — FastAPI Backend Entry Point (v4 — Production Upgrade)

Run:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    POST /auth/signup         → Register new user
    POST /auth/signin         → Login + get JWT token
    POST /chat                → AI chatbot (DialoGPT / Claude API / Rule-based)
    POST /predict             → ML predictions (calories, weight, fitness level)
    POST /vision/pose         → TF MoveNet pose detection + rep counting
    POST /vision/predict      → PyTorch CNN exercise classification (NEW)
    POST /recommend           → Workout recommendations
    GET  /health              → Health check
    GET  /docs                → Swagger UI (auto-generated)
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.database.db import init_db
from backend.routes import chat, predict, vision, recommend, user_data, auth
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent


# ─── LIFESPAN (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("  🚀 APEX AI Backend Starting …  (v4 Production)")
    print("=" * 60)

    # 1. Database
    await init_db()
    print("  ✅ Database initialised")

    # 2. ML models
    try:
        from backend.services import ml_service  # noqa: F401
        print("  ✅ ML models loaded (scikit-learn)")
    except Exception as e:
        print(f"  ⚠️  ML models: {e}")

    # 3. DL chatbot model (DialoGPT)
    try:
        from backend.chatbot import model_loader
        model_loader.preload()
        print(f"  ✅ DL Chatbot: {model_loader.get_model_name()}")
    except Exception as e:
        print(f"  ⚠️  DL Chatbot: {e}")

    # 4. NLP intent classifier (NEW)
    try:
        from backend.chatbot.intent_classifier import IntentClassifier
        ic = IntentClassifier()
        status = "ready" if ic.is_trained() else "not trained (run chatbot_training.ipynb)"
        print(f"  ✅ NLP intent classifier: {status}")
    except Exception as e:
        print(f"  ⚠️  Intent classifier: {e}")

    # 5. PyTorch CV model (NEW)
    try:
        from backend.services.cv_pytorch import load_pytorch_model
        load_pytorch_model()
        print("  ✅ PyTorch CV model loaded")
    except Exception as e:
        print(f"  ⚠️  PyTorch CV: {e}")

    print("  ✅ Server ready at http://localhost:8000")
    print("  📖 Docs at        http://localhost:8000/docs")
    print("=" * 60 + "\n")

    yield

    print("\n👋 APEX AI Backend shutting down …")


# ─── APP ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="APEX AI",
    description="AI-powered fitness platform backend (v4 Production)",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ROUTES ────────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(predict.router)
app.include_router(vision.router)
app.include_router(recommend.router)
app.include_router(user_data.router)

# ── SERVE FRONTEND ─────────────────────────────────────────────────────────────
frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/app", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))


# ─── HEALTH CHECK ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Health check — returns status of all loaded models."""
    from backend.services.ml_service import calorie_model, fitness_clf, recommender
    try:
        from backend.services.cv_pytorch import get_model_status
        cv_ok = get_model_status()
    except Exception:
        cv_ok = False
    try:
        from backend.chatbot.intent_classifier import IntentClassifier
        intent_ok = IntentClassifier().is_trained()
    except Exception:
        intent_ok = False

    return {
        "status": "ok",
        "version": "4.0.0",
        "models": {
            "calorie_regression":    calorie_model is not None,
            "fitness_classifier":    fitness_clf is not None,
            "recommender":           recommender is not None,
            "pytorch_cv":            cv_ok,
            "nlp_intent_classifier": intent_ok,
        },
        "integrations": {
            "anthropic_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "jwt_secret_set":    bool(os.environ.get("JWT_SECRET")),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
