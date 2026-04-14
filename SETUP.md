# APEX AI — Setup & Run Guide (v4)

## Prerequisites
- Python 3.10+
- pip or conda
- (Optional) CUDA-capable GPU for faster model training

---

## Step 1 — Clone / Extract the Project

```bash
cd apex-ai-project-upgraded
```

---

## Step 2 — Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch installs the CPU version by default.
> For CUDA GPU support, visit https://pytorch.org/get-started/locally/ and install the matching version first.

---

## Step 4 — Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set:
- `ANTHROPIC_API_KEY` — your key from https://console.anthropic.com
- `JWT_SECRET` — a long random string (run: `python -c "import secrets; print(secrets.token_hex(32))"`)

---

## Step 5 — Train the AI Models (Notebooks)

Open Jupyter and run each notebook in order:

```bash
jupyter notebook notebooks/
```

### Notebook order:

| # | Notebook | What it does | Output files |
|---|----------|-------------|--------------|
| 1 | `ml_models.ipynb` | Trains 4 scikit-learn models | `calorie_regression.pkl`, `weight_regression.pkl`, `fitness_classifier.pkl`, `recommender.pkl` |
| 2 | `dl_models.ipynb` | Trains PyTorch FitnessNet MLP | `fitness_net.pth`, `fitness_net_scaler.pkl` |
| 3 | `chatbot_training.ipynb` | Trains TF-IDF intent classifier | `intent_classifier.pkl`, `intent_vectorizer.pkl` |
| 4 | `cv_model.ipynb` | Trains PyTorch ExerciseNet CNN | `exercise_classifier.pth`, `cv_keypoint_scaler.pkl` |

> **Skip to Step 6 if you want to run with the pre-trained `.pkl` models already in `ai_models/ml_models/`.**
> The backend has fallbacks for every model — it won't crash if a model is missing.

---

## Step 6 — Start the Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Expected startup output:
```
============================================================
  🚀 APEX AI Backend Starting …  (v4 Production)
============================================================
  ✅ Database initialised
  ✅ ML models loaded (scikit-learn)
  ✅ DL Chatbot: DialoGPT-medium
  ✅ NLP intent classifier: ready
  ✅ PyTorch CV model loaded
  ✅ Server ready at http://localhost:8000
  📖 Docs at        http://localhost:8000/docs
============================================================
```

---

## Step 7 — Open the Frontend

Visit: **http://localhost:8000/app**

Or open `frontend/index.html` directly in a browser (API calls go to `http://localhost:8000`).

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/signup` | Register new user → returns JWT |
| POST | `/auth/signin` | Login → returns JWT |
| GET  | `/auth/me` | Validate token |
| POST | `/predict` | Calorie + weight + fitness prediction |
| POST | `/chat` | AI chatbot (hybrid engine) |
| POST | `/vision/pose` | MoveNet pose detection + rep counting |
| POST | `/vision/predict` | PyTorch exercise classification (NEW) |
| POST | `/recommend` | Workout recommendations |
| GET  | `/health` | System health + model status |
| GET  | `/docs` | Interactive Swagger UI |

---

## Architecture Overview

```
frontend/index.html
        │
        │  fetch() API calls with JWT token
        ▼
backend/main.py  (FastAPI, port 8000)
        │
        ├── /auth          → auth.py + JWT (auth_guard.py)
        ├── /predict       → ml_service.py  (scikit-learn .pkl)
        ├── /chat          → chatbot_service.py
        │       ├─ Intent  → intent_classifier.py (TF-IDF)
        │       ├─ Engine1 → DialoGPT (local DL)
        │       ├─ Engine2 → Anthropic Claude API
        │       └─ Engine3 → rule_engine.py (fallback)
        ├── /vision/pose   → cv_service.py   (MoveNet TF)
        ├── /vision/predict → cv_pytorch.py  (ResNet18 PyTorch)
        └── /recommend     → ml_service.py
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch` | Run `pip install torch torchvision` |
| `ModuleNotFoundError: jose` | Run `pip install python-jose[cryptography]` |
| `Model not found` warning | Run the corresponding notebook to train it |
| JWT errors | Check `JWT_SECRET` is set in `.env` |
| Chatbot gives rule-based only | Set `ANTHROPIC_API_KEY` in `.env` |
| `/vision/pose` fails | Uncomment TensorFlow in `requirements.txt` and reinstall |
| Port 8000 in use | Use `--port 8001` and update frontend API_BASE |

---

## File Structure

```
apex-ai-project/
├── backend/
│   ├── main.py                      ← FastAPI entry point (FIXED)
│   ├── middleware/
│   │   └── auth_guard.py            ← JWT create/verify (NEW)
│   ├── routes/
│   │   ├── auth.py                  ← Signup/signin + JWT (UPGRADED)
│   │   ├── predict.py               ← ML predictions
│   │   ├── vision.py                ← Pose + /vision/predict (UPGRADED)
│   │   ├── chat.py                  ← Chatbot endpoint
│   │   └── recommend.py             ← Workout recommendations
│   ├── chatbot/
│   │   ├── intent_classifier.py     ← TF-IDF intent NLP (NEW)
│   │   ├── chatbot_service.py       ← Hybrid AI engine
│   │   ├── response_router.py       ← Engine routing
│   │   ├── rule_engine.py           ← Rule-based fallback
│   │   └── memory_manager.py        ← Conversation memory
│   ├── services/
│   │   ├── ml_service.py            ← scikit-learn model loading
│   │   └── cv_pytorch.py            ← PyTorch ResNet18 CV (NEW)
│   └── database/
│       └── db.py                    ← SQLite + SQLAlchemy models
├── notebooks/
│   ├── ml_models.ipynb              ← ML training (NEW)
│   ├── dl_models.ipynb              ← DL training — FitnessNet (NEW)
│   ├── chatbot_training.ipynb       ← NLP intent classifier (NEW)
│   └── cv_model.ipynb               ← PyTorch ExerciseNet CNN (NEW)
├── ai_models/
│   ├── ml_models/                   ← .pkl files (scikit-learn)
│   └── dl_models/                   ← .pth files (PyTorch)
├── datasets/
│   ├── fitness_profiles.csv
│   ├── workout_history.csv
│   └── pose_keypoints.csv
├── frontend/
│   └── index.html                   ← Existing UI (unchanged)
├── requirements.txt                 ← UPGRADED (added JWT, PyTorch, Jupyter)
├── .env.example                     ← Environment variable template
└── SETUP.md                         ← This file
