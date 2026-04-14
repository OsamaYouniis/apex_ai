"""
backend/chatbot/model_loader.py
─────────────────────────────────
Singleton model loader for the local Deep Learning chatbot.

Model: microsoft/DialoGPT-medium
  - A GPT-2 based conversational model fine-tuned on 147M Reddit dialogues
  - Supports multi-turn conversation via attention on past token ids
  - Small enough to run on CPU (≈1.5GB RAM), fast enough for real-time use

This module loads the model ONCE when first imported and keeps it in memory.
All subsequent calls to get_model() return the same instance — zero reload cost.

Fine-tuned version (from training/train_chatbot.py):
  If ai_models/dl_models/chatbot_model/ exists → load fine-tuned model
  Else                                         → load base DialoGPT-medium
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("apex_ai.model_loader")
logging.basicConfig(level=logging.INFO)

ROOT       = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT / "ai_models" / "dl_models" / "chatbot_model"
BASE_MODEL = "microsoft/DialoGPT-medium"

# ── Singleton state ───────────────────────────────────────────────────────────
_tokenizer  = None
_model      = None
_model_name = None
_load_error = None
_loaded     = False   # True once loading has been attempted


def _try_load() -> bool:
    """
    Attempt to load the tokenizer + model.
    Sets module-level singletons.
    Returns True on success, False on failure.
    """
    global _tokenizer, _model, _model_name, _load_error, _loaded
    _loaded = True

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        # Decide which model to load
        if MODEL_PATH.exists() and any(MODEL_PATH.iterdir()):
            source = str(MODEL_PATH)
            label  = "fine-tuned DialoGPT (local)"
        else:
            source = BASE_MODEL
            label  = "DialoGPT-medium (base)"

        logger.info(f"Loading {label} from '{source}' …")

        tok = AutoTokenizer.from_pretrained(
            source,
            padding_side="left",
        )
        # DialoGPT uses eos_token as pad_token
        tok.pad_token = tok.eos_token

        mdl = AutoModelForCausalLM.from_pretrained(source)
        mdl.eval()   # inference mode — disables dropout

        _tokenizer  = tok
        _model      = mdl
        _model_name = label
        logger.info(f"✅ {label} loaded successfully "
                    f"({sum(p.numel() for p in mdl.parameters())/1e6:.1f}M parameters)")
        return True

    except Exception as e:
        _load_error = str(e)
        logger.warning(f"⚠️  Could not load DL model: {e}")
        logger.warning("    Chatbot will fall back to API / rule-based engine.")
        return False


def get_model() -> Tuple[Optional[object], Optional[object]]:
    """
    Returns (tokenizer, model).
    Loads on first call; subsequent calls return cached singletons instantly.
    Returns (None, None) if loading failed.
    """
    global _loaded
    if not _loaded:
        _try_load()
    return _tokenizer, _model


def get_model_name() -> str:
    """Human-readable name of the currently loaded model."""
    if _model_name:
        return _model_name
    if _load_error:
        return f"unavailable ({_load_error[:60]})"
    return "not loaded yet"


def is_model_available() -> bool:
    """Returns True if the DL model loaded successfully."""
    if not _loaded:
        _try_load()
    return _model is not None


def preload():
    """
    Call this at server startup to load the model eagerly
    (so the first user request is not slow).
    """
    if not _loaded:
        _try_load()
