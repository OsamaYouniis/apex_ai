"""
backend/routes/auth.py  (v4 — JWT upgrade)
────────────────────────────────────────────
Signup + Signin endpoints.
- Passwords hashed with sha256 + salt (no extra libs) or bcrypt if passlib installed
- Returns JWT token on success (via auth_guard.create_token)
- /auth/me endpoint validates a token and returns user info
"""

import hashlib, secrets
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from backend.database.db import get_db, User, UserProfile
from backend.middleware.auth_guard import create_token, get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Password hashing ──────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str = None) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, hashed = stored.split(":")
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False


# ─── SCHEMAS ──────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    full_name:      str
    email:          str
    password:       str
    gender:         str   = "m"
    age:            int   = 25
    weight_kg:      float = 70.0
    height_cm:      float = 175.0
    activity_level: int   = 2
    goal:           str   = "lose"
    target_weight:  float = 65.0
    dietary_pref:   str   = "No Restrictions"
    timeframe:      str   = "3-6 months"


class SigninRequest(BaseModel):
    email:    str
    password: str


class AuthResponse(BaseModel):
    success:  bool
    message:  str
    token:    str  = ""   # JWT token — NEW
    user_id:  int  = 0
    name:     str  = ""
    email:    str  = ""
    gender:   str  = "m"
    profile:  dict = {}


# ─── SIGN UP ──────────────────────────────────────────────────────────────────

@router.post("/signup", response_model=AuthResponse)
async def signup(req: SignupRequest, db: AsyncSession = Depends(get_db)):
    """
    Register a new user.
    - Validates email uniqueness and password length
    - Hashes password with salt
    - Creates User + UserProfile rows
    - Returns JWT token immediately (user is logged in on signup)
    """
    # Check duplicate
    result = await db.execute(select(User).where(User.email == req.email.lower().strip()))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be ≥ 6 characters")

    # Create user
    user = User(
        full_name=req.full_name.strip(),
        email=req.email.lower().strip(),
        hashed_password=_hash_password(req.password),
        gender=req.gender,
        created_at=datetime.utcnow(),
    )
    db.add(user)
    await db.flush()

    # Create profile
    profile = UserProfile(
        user_id=user.id,
        name=req.full_name.strip(),
        age=req.age,
        weight_kg=req.weight_kg,
        height_cm=req.height_cm,
        activity_level=req.activity_level,
        gender=0 if req.gender == "f" else 1,
        goal=req.goal,
        target_weight=req.target_weight,
        dietary_pref=req.dietary_pref,
        timeframe=req.timeframe,
    )
    db.add(profile)
    await db.commit()

    # Generate JWT
    token = create_token(user.id, user.email, user.full_name)

    return AuthResponse(
        success=True,
        message=f"Welcome to APEX AI, {req.full_name.split()[0]}! 🎉",
        token=token,
        user_id=user.id,
        name=user.full_name,
        email=user.email,
        gender=user.gender,
        profile={
            "name": req.full_name, "age": req.age,
            "weight_kg": req.weight_kg, "height_cm": req.height_cm,
            "activity_level": req.activity_level, "goal": req.goal,
            "target_weight": req.target_weight, "gender": req.gender,
            "dietary_pref": req.dietary_pref,
        }
    )


# ─── SIGN IN ──────────────────────────────────────────────────────────────────

@router.post("/signin", response_model=AuthResponse)
async def signin(req: SigninRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate existing user.
    Returns JWT token + full profile for frontend to restore state.
    """
    result = await db.execute(select(User).where(User.email == req.email.lower().strip()))
    user = result.scalar_one_or_none()

    if not user or not _verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    # Load profile
    prof_result = await db.execute(
        select(UserProfile).where(UserProfile.user_id == user.id)
    )
    profile = prof_result.scalar_one_or_none()
    profile_dict = {}
    if profile:
        profile_dict = {
            "name": profile.name, "age": profile.age,
            "weight_kg": profile.weight_kg, "height_cm": profile.height_cm,
            "activity_level": profile.activity_level, "goal": profile.goal,
            "target_weight": profile.target_weight, "gender": user.gender,
            "dietary_pref": profile.dietary_pref,
        }

    # Generate JWT
    token = create_token(user.id, user.email, user.full_name)

    return AuthResponse(
        success=True,
        message=f"Welcome back, {user.full_name.split()[0]}! 💪",
        token=token,
        user_id=user.id,
        name=user.full_name,
        email=user.email,
        gender=user.gender,
        profile=profile_dict,
    )


# ─── VERIFY TOKEN ─────────────────────────────────────────────────────────────

@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    """
    Validate token and return current user info.
    Frontend can call this on page load to check if session is still valid.
    """
    return {"authenticated": True, **current_user}


# ─── CHECK EMAIL ──────────────────────────────────────────────────────────────

@router.get("/check-email")
async def check_email(email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == email.lower().strip()))
    exists = result.scalar_one_or_none() is not None
    return {"available": not exists}
