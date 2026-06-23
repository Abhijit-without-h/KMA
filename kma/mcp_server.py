from __future__ import annotations

import os

from kma.memory import AgenticMemory

STORE_PATH = os.environ.get("KMA_STORE", "kma_memory.json")

# Singleton memory, persisted to JSON so it survives across tool calls / restarts.
MEM = AgenticMemory()
MEM.load(STORE_PATH)


def _scope(user_id: str | None, session_id: str | None) -> dict:
    scope = {}
    if user_id:
        scope["user_id"] = user_id
    if session_id:
        scope["session_id"] = session_id
    return scope


# --- tool implementations (pure, testable) ---------------------------------
def tool_add(text: str, user_id: str | None = None, session_id: str | None = None,
             importance: float = 0.5) -> dict:
    """Store a new memory; it is auto-placed under its nearest existing concept."""
    r = MEM.add(text, scope=_scope(user_id, session_id), importance=importance)
    MEM.save(STORE_PATH)
    return {"id": r.node.id, "text": r.node.text, "depth": r.node.depth,
            "parent": r.path[-1] if r.path else None}


def tool_search(query: str, user_id: str | None = None,
                session_id: str | None = None, k: int = 5) -> list[dict]:
    """Retrieve the most relevant memories, each with its provenance path."""
    rs = MEM.search(query, scope=_scope(user_id, session_id), k=k)
    return [{"id": r.node.id, "text": r.node.text, "score": round(r.score, 3),
             "via": r.path} for r in rs]


def tool_get_context(query: str, user_id: str | None = None,
                     session_id: str | None = None, k: int = 5) -> str:
    """Return a compact, prompt-ready memory block for the query."""
    return MEM.get_context(query, scope=_scope(user_id, session_id), k=k)


def tool_forget(node_id: str) -> dict:
    """Delete a memory; its children are reparented to its parent."""
    ok = MEM.forget(node_id)
    if ok:
        MEM.save(STORE_PATH)
    return {"forgotten": ok}


def tool_explain_address(node_id: str) -> dict:
    """Explain a memory's address: its root->node concept path and depth."""
    node = MEM.engine.store.get(node_id)
    if node is None:
        return {"error": "not found"}
    path = MEM._path(node)
    return {"id": node_id, "depth": node.depth, "address_path": [*path, node.text]}


def build_server():  # noqa: ANN201
    """Lazily build the FastMCP server with the tools registered."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("kma-memory")
    server.tool(name="memory.add")(tool_add)
    server.tool(name="memory.search")(tool_search)
    server.tool(name="memory.get_context")(tool_get_context)
    server.tool(name="memory.forget")(tool_forget)
    server.tool(name="memory.explain_address")(tool_explain_address)
    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
