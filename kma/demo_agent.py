"""Agentic-memory loop demo: write-after, retrieve-before.

Simulates an agent talking to a user across turns. Each turn is auto-placed into
the memory hierarchy; later, a query assembles a prompt-ready context block with
provenance -- showing recall the agent would inject before answering.

    python -m kma.demo_agent
"""

from __future__ import annotations

from kma.memory import AgenticMemory


def main() -> None:
    mem = AgenticMemory()
    scope = {"user_id": "alice", "session_id": "s1"}

    # --- the agent observes turns and writes memories (auto-placed) ----------
    turns = [
        ("the user is building a travel app for hikers", 0.9),
        ("they prefer python and fastapi for the backend", 0.8),
        ("they want offline maps so hikers can navigate without signal", 0.7),
        ("the user also mentioned they live in seattle", 0.5),
        ("they like rainy weather and strong coffee", 0.3),
    ]
    print("=== writing memories (auto-placed) ===")
    for text, importance in turns:
        r = mem.add(text, scope=scope, importance=importance)
        parent = r.path[-1][:34] if r.path else "(new root)"
        print(f"  d{r.node.depth}  {text[:42]:42}  -> under: {parent}")

    # --- before answering a new question, retrieve context -------------------
    query = "what backend stack should I scaffold for them?"
    print(f'\n=== retrieve-before-generate ===\nquery: "{query}"\n')
    print(mem.get_context(query, scope=scope, k=3))

    print("\n=== search with provenance ===")
    for r in mem.search("where does the user live", scope=scope, k=2):
        print(f"  {r.score:.2f}  {r.explain()}")


if __name__ == "__main__":
    main()
