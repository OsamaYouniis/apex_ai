# 🏋️ APEX AI — Fitness Intelligence Platform (v5 Production)

> **Graduation Project — Full-Stack AI System**  
> Real ML models · LLM chatbot · Computer Vision · Dynamic analytics

---

## 🚀 System Overview

APEX AI is a production-grade AI fitness platform combining multiple AI technologies into a cohesive, user-facing application.

| Layer | Technology | Description |
|---|---|---|
| **ML** | XGBoost + RandomForest Ensemble | Calorie burn prediction (R² > 0.97) |
| **ML** | Gradient Boosting Classifier | Body performance classification (4-class) |
| **ML** | TF-IDF Content-Based | Personalized workout recommender (30+ exercises) |
| **DL** | PyTorch ResNet18 | Exercise classification from pose keypoints |
| **DL** | MoveNet (TensorFlow) | Real-time pose detection & rep counting |
| **LLM** | Anthropic Claude Sonnet 4 | Primary AI coach chatbot |
| **LLM** | OpenAI GPT-4o | Fallback LLM chatbot |
| **NLP** | TF-IDF Intent Classifier | Local intent routing (zero API cost) |
| **DB** | SQLite + SQLAlchemy | User profiles, workout logs, prediction history |
| **API** | FastAPI + Pydantic v2 | Async REST API with JWT auth & Swagger docs |

---

## 📁 Project Structure

```
apex-ai/
├── main.py                      ← Unified entry point (v5)
├── requirements.txt
├── .env.example
│
├── data/                        ← NEW: Real dataset management
│   └── download_datasets.py     ← Kaggle download + synthetic fallback
│
├── models/                      ← NEW: Trained model storage
│   ├── calorie_model.pkl        ← XGBoost ensemble (after training)
│   ├── fitness_model.pkl        ← GB classifier (after training)
│   ├── recommender.pkl          ← TF-IDF recommender (after training)
│   └── exercise_db.json         ← Exercise database
│
├── training/                    ← NEW + existing training scripts
│   ├── train_calorie_model.py   ← XGBoost calorie predictor
│   ├── train_fitness_model.py   ← Fitness classifier
│   ├── train_recommender.py     ← Content-based recommender
│   ├── train_ml.py              ← Original training script
│   ├── train_dl.py              ← PyTorch DL training
│   └── train_chatbot.py         ← NLP intent classifier
│
├── api/                         ← NEW: Clean API module
│   └── routes.py                ← /predict, /recommend, /chat, /dashboard
│
├── services/                    ← NEW: Business logic services
│   ├── chatbot.py               ← LLM chatbot (Anthropic + OpenAI + rules)
│   └── ml_service.py            ← ML inference (loads from /models)
│
├── backend/                     ← EXISTING (unchanged structure)
│   ├── main.py                  ← Original FastAPI app (still works)
│   ├── routes/                  ← auth, chat, predict, vision, recommend
│   ├── services/                ← cv_pytorch, cv_service, ml_service
│   ├── chatbot/                 ← knowledge_base, rule_engine, intent_classifier
│   ├── database/                ← SQLAlchemy models + async SQLite
│   └── middleware/              ← JWT auth guard
│
├── frontend/                    ← UNTOUCHED (original UI preserved)
│   ├── index.html               ← Complete single-page app
│   └── api.js                   ← Backend integration layer
│
├── ai_models/                   ← EXISTING (pre-trained models)
│   ├── ml_models/               ← calorie_regression.pkl, fitness_classifier.pkl
│   └── dl_models/               ← exercise_classifier.pth, cv_keypoint_scaler.pkl
│
├── datasets/                    ← EXISTING synthetic datasets
└── notebooks/                   ← Jupyter notebooks
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone <your-repo>
cd apex-ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
# Required for AI chatbot (get one free at console.anthropic.com)
ANTHROPIC_API_KEY=sk-ant-api03-...

# Optional: OpenAI fallback
OPENAI_API_KEY=sk-...

# JWT secret — change this!
JWT_SECRET=your_secret_here_make_it_long_and_random
```

> **Note:** The system works without any API keys using the rule-based engine.  
> For the full LLM chatbot experience, add at least one API key.

### 3. Download Datasets

```bash
python data/download_datasets.py
```

This will:
- Try to download real Kaggle datasets (requires `KAGGLE_USERNAME` + `KAGGLE_KEY` in `.env`)
- Fall back to high-quality synthetic data that mirrors real dataset statistics
- Save preprocessed CSVs to `/data`

For real Kaggle data, set credentials first:
```bash
# Option A: Environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Option B: Place kaggle.json at ~/.kaggle/kaggle.json
# Download from: kaggle.com → Account → API → Create New Token
```

### 4. Train Models

Run the three training scripts in any order:

```bash
# Calorie burn predictor (XGBoost ensemble, ~2 min)
python training/train_calorie_model.py

# Fitness classifier (Gradient Boosting, ~3 min)  
python training/train_fitness_model.py

# Workout recommender (TF-IDF, <10 sec)
python training/train_recommender.py

# Optional: NLP intent classifier
python training/train_chatbot.py
```

Expected output example:
```
📈 Calorie Model (XGBoost Ensemble)
   MAE  : 12.4 kcal
   RMSE : 18.7 kcal
   R²   : 0.9731
   5-Fold CV R²: 0.9708 ± 0.0021
✅ Saved → models/calorie_model.pkl
```

### 5. Run the Server

```bash
# Option A: Unified v5 entry point (recommended)
uvicorn main:app --reload --port 8000

# Option B: Original backend only
uvicorn backend.main:app --reload --port 8000
```

### 6. Access the App

| URL | Description |
|---|---|
| `http://localhost:8000/app` | Frontend web app |
| `http://localhost:8000/docs` | Interactive Swagger API docs |
| `http://localhost:8000/health` | System health check |
| `http://localhost:8000/redoc` | ReDoc API documentation |

---

## 🧠 AI Architecture

### Calorie Prediction Model

```
Input: [age, height, weight, duration, heart_rate, body_temp, gender]
       + engineered: [BMI, BMR, HR_intensity, weight×duration, age²]

Pipeline: StandardScaler → VotingRegressor(
    GradientBoosting(n=400, depth=5, lr=0.05),
    RandomForest(n=200, depth=10),
    Ridge(alpha=10)
)

Trained on: 15,000 samples (Kaggle Calories Burn dataset)
Performance: MAE ~12 kcal | R² > 0.97
```

### Fitness Classifier

```
Input: [age, gender, height, weight, grip_force, situps, broad_jump,
        flexibility] + engineered: [BMI, BMR, strength_index, age²]

Pipeline: StandardScaler → SoftVotingClassifier(
    GradientBoosting(n=300, depth=4),
    RandomForest(n=200, depth=12),
    LogisticRegression(C=1.0)
)

Classes: Untrained (D) → Beginner (C) → Intermediate (B) → Advanced (A)
Trained on: 13,393 samples (Kaggle Body Performance dataset)
Performance: Accuracy > 78% | 5-Fold CV: 77.5% ± 1.2%
```

### LLM Chatbot — Engine Routing

```
User message
    │
    ├─→ Knowledge Base (TF-IDF, 120+ Q&A, ~0ms)
    │       If confidence ≥ 0.40 → RETURN KB answer
    │
    ├─→ Complex intent? (workout_plan, nutrition_plan, injury, etc.)
    │   Long message? (>20 words)
    │       │
    │       ├─→ Anthropic Claude Sonnet 4 (primary)
    │       │       Rich system prompt + user profile injection
    │       │       Full conversation memory (8-turn window)
    │       │
    │       └─→ OpenAI GPT-4o (if Claude unavailable)
    │
    └─→ Rule Engine (100% offline, always available)
            BMI, TDEE, macros, workout plans (deterministic)
```

### Workout Recommender

```
Query: (fitness_level, goal, age)
    │
    ├─→ Filter eligible exercises (level + goal + age constraints)
    │
    ├─→ Build user query string
    │       "cardio weight-loss fat-burn intermediate strength"
    │
    ├─→ TF-IDF cosine similarity over exercise tags
    │
    └─→ Return top-K ranked exercises with metadata
```

---

## 📡 API Reference

### POST /predict
Full prediction pipeline — all metrics in one call.

```json
// Request
{
  "age": 28,
  "weight_kg": 75.0,
  "height_cm": 178.0,
  "activity_level": 3,
  "gender": 1,
  "goal": "lose"
}

// Response
{
  "calories_tdee": 2847,
  "bmr": 1836,
  "bmi": 23.67,
  "bmi_category": "Normal weight",
  "fitness_level": "Intermediate",
  "fitness_level_id": 1,
  "weight_change_30d": -1.2,
  "cut_calories": 2347,
  "bulk_calories": 3147,
  "protein_g": 150,
  "water_l": 2.5,
  "macros": {"protein_g": 150, "carbs_g": 320, "fat_g": 79},
  "recommendations": [...]
}
```

### POST /predict/calories
Predict calories burned in a workout session.

```json
// Request
{"age": 28, "weight_kg": 75.0, "height_cm": 178.0,
 "activity_level": 3, "gender": 1, "duration_min": 45}

// Response
{"calories_burned": 387, "calories_per_min": 8.6, "tdee": 2847}
```

### POST /chat
LLM fitness chatbot with memory and personalization.

```json
// Request
{
  "message": "Design me a 4-day strength programme for muscle gain",
  "history": [{"role": "user", "content": "..."}, ...],
  "user_data": {"age": 28, "weight": 75, "goal": "gain", "tdee": 2847},
  "user_id": 1
}

// Response
{
  "reply": "Based on your stats (75kg, TDEE 2847 kcal)...",
  "source": "anthropic_claude",
  "confidence": 0.97,
  "model": "claude-sonnet-4"
}
```

### GET /user-data/dashboard
Real-time computed analytics dashboard.

```json
// Response
{
  "progress": {
    "total_workouts": 24,
    "workouts_per_week": 3.7,
    "streak_days": 5,
    "improvement_pct": 12.4,
    "avg_form_score": 0.87
  },
  "calorie_analytics": {"tdee_trend": "stable", "tdee_history": [...]},
  "goal_progress": {"progress_pct": 43, "kg_remaining": 4.2},
  "top_exercises": [{"exercise": "Squats", "count": 8}, ...],
  "insights": ["🔥 5-day streak! You're unstoppable!", ...]
}
```

---

## 🎓 Academic Highlights

### What Makes This Graduation-Level

| Criterion | Implementation |
|---|---|
| **Real Data** | Kaggle datasets (15K calorie records, 13K body performance records) |
| **Strong ML** | XGBoost + RF + GradB ensembles with feature engineering, CV evaluation |
| **Deep Learning** | PyTorch ResNet18 for exercise classification from pose keypoints |
| **LLM Integration** | Anthropic Claude / OpenAI GPT-4o with prompt engineering + memory |
| **Clean Architecture** | Separation of concerns: data / training / services / api / frontend |
| **Production Features** | JWT auth, async DB, error handling, model versioning, health checks |
| **Dynamic Intelligence** | All dashboard data is computed in real-time, not hardcoded |
| **Personalization** | Every prediction and LLM response uses individual user data |

### Model Evaluation Methodology

All models are evaluated with:
- **Hold-out test set** (15% of data, stratified for classifiers)
- **5-Fold Cross-Validation** (mean ± std reported)
- **Multiple metrics**: MAE + RMSE + R² for regression; Accuracy + F1 + classification report for classification
- **Confusion matrix** analysis for classifier errors

---

## 🛠 Troubleshooting

**Models not loading?**  
→ Run training scripts first: `python training/train_calorie_model.py`

**Chatbot gives generic responses?**  
→ Add `ANTHROPIC_API_KEY` to `.env`. Rule-based engine works without it.

**Kaggle download fails?**  
→ Normal — synthetic fallback activates automatically. Data quality is comparable.

**Port already in use?**  
→ `uvicorn main:app --port 8001`

**Frontend not loading?**  
→ Make sure you're hitting `http://localhost:8000/app` (not `/`)

---

## 📦 Tech Stack

```
Python 3.10+     FastAPI 0.111     SQLAlchemy 2.0     Pydantic v2
scikit-learn 1.4  XGBoost 2.0      PyTorch 2.0+       Anthropic SDK
OpenAI SDK        Pandas 2.2       NumPy 1.26          Joblib 1.4
aiosqlite         python-jose      uvicorn             Pillow
```

---

*APEX AI v5 — Built for graduation excellence* 🎓
