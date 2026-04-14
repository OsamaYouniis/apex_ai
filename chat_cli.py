from __future__ import annotations

from backend.rag_coach.graph import FitnessGraph
from backend.rag_coach.rag import RAGService


def _get_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return int(raw) if raw else default


def _get_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    return float(raw) if raw else default


def _get_str(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def main() -> None:
    print("=== APEX RAG Chat (Terminal) ===")
    print("Type 'quit' to exit.\n")

    profile = {
        "age": _get_int("Age", 24),
        "weight": _get_float("Weight (kg)", 80.0),
        "height": _get_float("Height (cm)", 180.0),
        "gender": _get_str("Gender (male/female)", "male"),
        "goal": _get_str("Goal (cut/bulk/maintain)", "cut"),
        "activity_level": _get_str(
            "Activity (sedentary/light/moderate/active/very_active)",
            "moderate",
        ),
    }

    rag = RAGService(docs_dir="knowledge_base/raw", index_dir="knowledge_base/faiss_index")
    rag.build_or_load_index(force_rebuild=False)
    bot = FitnessGraph(rag)

    while True:
        msg = input("\nYou: ").strip()
        if msg.lower() in {"quit", "exit"}:
            print("Session ended.")
            break
        if not msg:
            continue

        resp = bot.invoke(message=msg, profile=profile)
        print("\nCoach:")
        print(resp.get("response", "No response"))
        if resp.get("calories") is not None:
            print(f"\nCalories target: {resp['calories']}")
        if resp.get("macros"):
            print(f"Macros: {resp['macros']}")


if __name__ == "__main__":
    main()
