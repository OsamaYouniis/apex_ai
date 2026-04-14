"""
backend/chatbot/memory_manager.py
───────────────────────────────────
Manages per-user conversation context memory.

Two types of memory:
  1. Short-term  — the last N message turns (sliding window)
  2. Profile     — user fitness data (weight, goal, TDEE, etc.)

The memory is stored in-process (dict keyed by user_id).
For a production system with multiple servers you would store this in Redis,
but for a single-server graduation project this is perfectly fine.

Key design decisions:
  - MAX_TURNS = 10 → keeps context window manageable for the DL model
  - Profile is stored separately so it persists across cleared conversations
  - build_prompt() formats memory into the exact string the DL model expects
"""

from collections import deque
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


MAX_TURNS = 10     # maximum conversation turns stored per user
MAX_CHARS = 1500   # maximum total characters in context sent to DL model


@dataclass
class ConversationMemory:
    """
    Holds conversation state for one user.

    Attributes:
        user_id    : unique identifier
        turns      : deque of (role, content) tuples — newest at the right
        profile    : user fitness profile dict
        created_at : when this memory was first created
        updated_at : last time a message was added
    """
    user_id:    int
    turns:      deque = field(default_factory=lambda: deque(maxlen=MAX_TURNS))
    profile:    dict  = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_turn(self, role: str, content: str):
        """Append a message turn. role = 'user' | 'assistant'"""
        self.turns.append((role, content.strip()))
        self.updated_at = datetime.utcnow()

    def get_history(self) -> list[dict]:
        """Return turns as list of {role, content} dicts (newest last)."""
        return [{"role": r, "content": c} for r, c in self.turns]

    def build_dialogpt_prompt(self) -> str:
        """
        Build the token sequence DialoGPT expects:
          USER_MSG <|endoftext|> BOT_REPLY <|endoftext|> ... USER_MSG <|endoftext|>

        We truncate from the LEFT if the context is too long (keep newest turns).
        """
        EOS = "<|endoftext|>"
        parts = []
        for role, content in self.turns:
            parts.append(content)
            parts.append(EOS)
        prompt = " ".join(parts)

        # Truncate to MAX_CHARS keeping the END of the string (newest context)
        if len(prompt) > MAX_CHARS:
            prompt = prompt[-MAX_CHARS:]
        return prompt

    def build_claude_messages(self) -> list[dict]:
        """Format turns for Anthropic Claude API messages format."""
        return self.get_history()

    def clear(self):
        """Clear conversation turns (keep profile)."""
        self.turns.clear()

    def update_profile(self, profile: dict):
        """Update stored user profile data."""
        self.profile.update(profile)
        self.updated_at = datetime.utcnow()

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def last_user_message(self) -> Optional[str]:
        for role, content in reversed(self.turns):
            if role == "user":
                return content
        return None


# ── In-memory store: user_id → ConversationMemory ────────────────────────────
_memory_store: dict[int, ConversationMemory] = {}


def get_memory(user_id: int, profile: Optional[dict] = None) -> ConversationMemory:
    """
    Get or create a ConversationMemory for the given user_id.
    Optionally updates the stored profile with fresh data.
    """
    if user_id not in _memory_store:
        _memory_store[user_id] = ConversationMemory(user_id=user_id)

    mem = _memory_store[user_id]

    # Always sync the latest profile (user may have updated their stats)
    if profile:
        mem.update_profile(profile)

    return mem


def add_message(user_id: int, role: str, content: str,
                profile: Optional[dict] = None):
    """
    Add a message to a user's conversation memory.
    Also syncs profile if provided.
    """
    mem = get_memory(user_id, profile)
    mem.add_turn(role, content)


def clear_memory(user_id: int):
    """Clear conversation history for a user (keeps profile)."""
    if user_id in _memory_store:
        _memory_store[user_id].clear()


def get_all_user_ids() -> list[int]:
    """Return list of all users with active memory."""
    return list(_memory_store.keys())


def memory_stats() -> dict:
    """Return stats about current memory usage (for /health endpoint)."""
    return {
        "active_sessions": len(_memory_store),
        "total_turns":     sum(m.turn_count for m in _memory_store.values()),
    }
