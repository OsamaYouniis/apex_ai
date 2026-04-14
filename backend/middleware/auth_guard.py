"""
backend/middleware/auth_guard.py
─────────────────────────────────
JWT authentication dependency for FastAPI routes.

Usage in any route:
    from backend.middleware.auth_guard import get_current_user

    @router.get("/protected")
    async def protected(user=Depends(get_current_user)):
        return {"user_id": user["user_id"]}
"""

import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

_bearer = HTTPBearer(auto_error=False)


def _get_secret() -> str:
    return os.environ.get("JWT_SECRET", "apex-ai-dev-secret-change-in-production")


def create_token(user_id: int, email: str, name: str) -> str:
    """Create a signed JWT. Falls back to base64 if python-jose not installed."""
    try:
        from jose import jwt as jose_jwt
        import time
        payload = {
            "sub": str(user_id),
            "email": email,
            "name": name,
            "iat": int(time.time()),
            "exp": int(time.time()) + 60 * 60 * 24 * 7,  # 7-day expiry
        }
        return jose_jwt.encode(payload, _get_secret(), algorithm="HS256")
    except ImportError:
        import base64, json, time
        payload = {"sub": str(user_id), "email": email, "name": name,
                   "exp": int(time.time()) + 60 * 60 * 24 * 7}
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def decode_token(token: str) -> Optional[dict]:
    """Decode + verify a JWT. Returns payload dict or None if invalid/expired."""
    try:
        from jose import jwt as jose_jwt, JWTError
        return jose_jwt.decode(token, _get_secret(), algorithms=["HS256"])
    except ImportError:
        try:
            import base64, json, time
            data = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
            if data.get("exp", 0) < int(time.time()):
                return None
            return data
        except Exception:
            return None
    except Exception:
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """FastAPI dependency — validates JWT and returns user info dict."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — include Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalid or expired — please sign in again",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "user_id": int(payload.get("sub", 0)),
        "email":   payload.get("email", ""),
        "name":    payload.get("name", ""),
    }


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[dict]:
    """Like get_current_user but returns None instead of raising — for optional auth."""
    if not credentials:
        return None
    payload = decode_token(credentials.credentials)
    if not payload:
        return None
    return {
        "user_id": int(payload.get("sub", 0)),
        "email":   payload.get("email", ""),
        "name":    payload.get("name", ""),
    }
