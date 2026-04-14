"""
backend/database/db.py  — SQLAlchemy models + async SQLite
Includes User auth table for signup/signin memory.
"""

from sqlalchemy import Column, Integer, Float, String, Text, DateTime, JSON, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite+aiosqlite:///./apex_ai.db"
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    full_name       = Column(String, default="User")
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    gender          = Column(String, default="m")
    created_at      = Column(DateTime, default=datetime.utcnow)
    is_active       = Column(Boolean, default=True)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, default=1, index=True)
    name           = Column(String, default="User")
    age            = Column(Integer, default=25)
    weight_kg      = Column(Float,   default=70.0)
    height_cm      = Column(Float,   default=175.0)
    activity_level = Column(Integer, default=2)
    gender         = Column(Integer, default=1)
    goal           = Column(String,  default="lose")
    target_weight  = Column(Float,   default=65.0)
    dietary_pref   = Column(String,  default="No Restrictions")
    timeframe      = Column(String,  default="3-6 months")
    created_at     = Column(DateTime, default=datetime.utcnow)

class WorkoutLog(Base):
    __tablename__ = "workout_logs"
    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, default=1, index=True)
    exercise     = Column(String)
    sets         = Column(Integer, default=3)
    reps         = Column(Integer, default=10)
    weight_kg    = Column(Float,  default=0.0)
    duration_min = Column(Integer, default=30)
    reps_counted = Column(Integer, default=0)
    form_score   = Column(Float,  default=1.0)
    logged_at    = Column(DateTime, default=datetime.utcnow)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, default=1)
    prediction_type = Column(String)
    input_data      = Column(JSON)
    output_data     = Column(JSON)
    predicted_at    = Column(DateTime, default=datetime.utcnow)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id        = Column(Integer, primary_key=True, index=True)
    user_id   = Column(Integer, default=1, index=True)
    role      = Column(String)
    content   = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
