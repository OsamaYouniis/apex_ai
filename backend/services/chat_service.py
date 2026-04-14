"""
backend/services/chat_service.py
──────────────────────────────────
Compatibility shim — redirects to the new production chatbot package.
The actual implementation lives in backend/chatbot/chatbot_service.py
"""
from backend.chatbot.chatbot_service import generate_response

__all__ = ["generate_response"]
