# backend/chatbot/__init__.py
# APEX AI — Production Chatbot Package
# 
# Architecture:
#   1. LOCAL Deep Learning model  (DialoGPT-medium, fine-tuned on fitness Q&A)
#   2. Anthropic Claude API       (secondary — richer complex answers)
#   3. Smart Rule-Based Engine    (fallback — always works, fully personalised)
#
# Entry point: chatbot_service.py → generate_response()
