"""LLM fact extraction: parsing robustness, ingest, and agent integration."""

from kma.connectors import OpenRouterAgent
from kma.extract import LLMFactExtractor, parse_facts
from kma.memory import AgenticMemory


def test_parse_facts_handles_clean_and_messy_output():
    assert parse_facts('["a", "b"]') == ["a", "b"]
    # tolerate code fences / prose around the array
    assert parse_facts('```json\n["x"]\n```') == ["x"]
    assert parse_facts("Sure! Here you go: [\"y\", \"z\"]") == ["y", "z"]
    # junk / non-list / empty -> no facts, no crash
    assert parse_facts("I could not find anything.") == []
    assert parse_facts('{"not": "a list"}') == []
    assert parse_facts('["keep", "", 5, "this"]') == ["keep", "this"]


def test_ingest_stores_extracted_facts_not_raw_turn():
    mem = AgenticMemory()

    def fake_llm(messages):
        return '["The user prefers FastAPI", "The user lives in Seattle"]'

    extractor = LLMFactExtractor(fake_llm)
    recalls = mem.ingest(
        "so anyway I usually reach for fastapi, and btw I'm up in seattle",
        extractor=extractor,
    )
    texts = [n.text for n in mem.engine.store.all()]
    assert len(recalls) == 2
    assert "The user prefers FastAPI" in texts
    assert "The user lives in Seattle" in texts
    # the raw, messy turn itself was NOT stored
    assert not any("so anyway" in t for t in texts)


def test_ingest_stores_nothing_when_no_facts():
    mem = AgenticMemory()
    extractor = LLMFactExtractor(lambda m: "[]")
    assert mem.ingest("what's the weather like?", extractor=extractor) == []
    assert len(mem.engine.store) == 0


def test_agent_extract_facts_flag():
    mem = AgenticMemory()

    def fake_llm(messages):
        # extraction calls have the extractor system prompt; chat calls don't
        if messages[0]["content"].startswith("You extract"):
            return '["The user is building a hiking app"]'
        return "Got it!"

    agent = OpenRouterAgent(mem, llm=fake_llm, extract_facts=True)
    agent.chat("yo so I'm hacking on this hiking app thing")
    texts = [n.text for n in mem.engine.store.all()]
    assert texts == ["The user is building a hiking app"]
