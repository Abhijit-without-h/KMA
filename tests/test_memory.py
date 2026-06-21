"""AgenticMemory: auto-placement, lifecycle, scope, and the MCP tool wrappers."""

from kma.memory import AgenticMemory


def test_auto_placement_attaches_related_under_a_parent():
    mem = AgenticMemory()
    root = mem.add("the user is building a hiking and outdoor travel app")
    child = mem.add("the app needs offline trail maps for hikers without signal")
    # the closely-related second memory should attach under the first, not float free
    assert child.node.parent_id == root.node.id
    assert child.node.depth == 1
    assert root.node.text in child.path


def test_unrelated_memory_starts_new_root():
    mem = AgenticMemory()
    mem.add("a recipe for sourdough bread with rye flour")
    other = mem.add("quarterly tax filing deadlines for small businesses")
    assert other.node.parent_id is None
    assert other.node.depth == 0


def test_search_returns_provenance_and_finds_memory():
    mem = AgenticMemory()
    mem.add("the user prefers python and fastapi for backends")
    hits = mem.search("which web framework do they like", k=1)
    assert hits
    assert "fastapi" in hits[0].node.text
    assert isinstance(hits[0].path, list)


def test_lifecycle_forget_reparents_children():
    mem = AgenticMemory()
    a = mem.add("space exploration and planetary science")
    b = mem.add("missions that explore the planet mars in detail")
    c = mem.add("rovers drive across the martian surface collecting rocks")
    assert b.node.parent_id == a.node.id
    # forget the middle node -> its child should reparent upward, not vanish
    assert mem.forget(b.node.id)
    assert mem.engine.store.get(b.node.id) is None
    reparented = mem.engine.store.get(c.node.id)
    assert reparented is not None
    assert reparented.parent_id in (a.node.id, None)


def test_scope_isolation():
    mem = AgenticMemory()
    mem.add("alice loves mountain trails", scope={"user_id": "alice"})
    mem.add("bob loves city cafes", scope={"user_id": "bob"})
    hits = mem.search("what do they love", scope={"user_id": "alice"}, k=5)
    assert hits
    assert all(h.node.metadata["scope"].get("user_id") == "alice" for h in hits)


def test_get_context_respects_budget():
    mem = AgenticMemory()
    for t in ["a", "b longer memory text here", "c another memory line"]:
        mem.add(t)
    ctx = mem.get_context("memory", k=3, budget_chars=20)
    assert len(ctx) <= 40  # a couple of short lines at most


def test_mcp_tool_wrappers(tmp_path, monkeypatch):
    import kma.mcp_server as srv

    monkeypatch.setattr(srv, "MEM", AgenticMemory())
    monkeypatch.setattr(srv, "STORE_PATH", str(tmp_path / "mem.json"))

    added = srv.tool_add("the user enjoys hiking in the cascades", user_id="alice")
    assert "id" in added
    results = srv.tool_search("outdoor hobbies", user_id="alice", k=3)
    assert results and "hiking" in results[0]["text"]
    assert srv.tool_forget(added["id"])["forgotten"] is True
