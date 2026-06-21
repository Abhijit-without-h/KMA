<p align="center">
  <img src="assets/logo.png" width="300" alt="KMA — Kleinian Memory Architecture" />
</p>

<h1 align="center">KMA · Kleinian Memory Architecture</h1>

<p align="center">
  <b>Memory that has a shape.</b><br/>
  Hyperbolic, hierarchy-aware <i>agentic memory</i> for LLMs — structured, navigable, and explainable.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="license" />
  <img src="https://img.shields.io/badge/python-3.12%2B-3776AB.svg" alt="python" />
  <img src="https://img.shields.io/badge/tests-36%20passing-2ea44f.svg" alt="tests" />
  <img src="https://img.shields.io/badge/status-research%20preview-orange.svg" alt="status" />
</p>

---

## The one-line pitch

> Most agent memory is a **flat bag of vectors** retrieved by cosine similarity. KMA gives your agent a **memory with structure** — a hyperbolic concept tree it can drill down, roll up, and *explain* — and it provably beats cosine on the one thing flat memory can't do: **hierarchy**.

It plugs **into** what you already use — any embedding model (OpenAI, Gemma, MiniLM), any vector DB (Pinecone, Chroma), any LLM (via MCP or OpenRouter) — and adds the structural layer on top at near-zero compute.

---

## Why this is different

Flat vector memory answers exactly one question: *"what is most similar?"* That's necessary but not sufficient for an agent that reasons over a growing, branching history.

| | Flat vector memory | **KMA** |
|---|---|---|
| Similarity recall | ✅ | ✅ (keeps cosine where it wins) |
| **Hierarchy** (parent / child / subtree) | ❌ structurally impossible | ✅ native, in hyperbolic space |
| **"More general than"** (asymmetric) | ❌ cosine is symmetric | ✅ generality = distance from center |
| **Explainability** ("why recalled?") | ❌ opaque nearest-neighbor | ✅ every hit carries its concept path |
| Auto-organization | ❌ undifferentiated pile | ✅ new memories self-place into a tree |

The key idea, stated plainly: hyperbolic space embeds **trees with almost no distortion**, where flat Euclidean space cannot. So instead of crushing meaning into a point, KMA places each memory by **direction** (what it's about) *and* **radius** (how general it is) — roots near the center, details toward the rim.

---

## The breakthrough, measured honestly

We trained a small projection (`φ`) that maps any embedding into a hyperbolic ball, supervised by the memory tree itself. On **ancestor retrieval** — reconstructing is-a / "what's the parent concept" relationships — it isn't a marginal gain:

```
Ancestor retrieval (MAP, higher is better)
  cosine (flat embeddings)      0.356
  KMA, untrained projection     0.187      ← geometry alone is not magic
  KMA, trained projection       0.90–1.00  ← +0.5–0.6 over cosine

Flat similarity (sibling AUC, the guardrail)
  cosine                        0.960
  KMA, trained projection       1.000      ← parity: we didn't break similarity
```

**Read this honestly.** We do **not** beat OpenAI/Gemma + cosine at flat semantic similarity — that's their home turf, and we aim for *parity* there. The win is on **hierarchical / asymmetric retrieval**, which flat search cannot represent at all. The untrained-projection row is in the table on purpose: it proves the gain comes from *learning*, not from hand-waving about geometry. (Current numbers are on a controlled taxonomy; larger real-world hierarchies are on the roadmap.)

---

## Quickstart

```bash
pip install kleinian-memory          # import as `kma`
# optional extras:
#   [embed] real sentence embeddings · [train] torch chart · [mcp] connector
```

**As an agent's memory** — memories auto-place into a hierarchy, recall comes with provenance:

```python
from kma.memory import AgenticMemory

mem = AgenticMemory()
mem.add("the user is building a travel app for hikers", scope={"user_id": "alice"})
mem.add("they prefer python and fastapi", scope={"user_id": "alice"})  # auto-placed

print(mem.get_context("what backend should I scaffold?", scope={"user_id": "alice"}))
# -> a compact, prompt-ready memory block, rolled up with its ancestors
```

**Wired to a live LLM** (OpenRouter — any model) with retrieve-before / write-after:

```python
from kma.connectors import OpenRouterAgent
from kma.memory import AgenticMemory

agent = OpenRouterAgent(AgenticMemory(), scope={"user_id": "alice"}, extract_facts=True)
reply, injected_memory = agent.chat("remind me what stack I'm using")
```

**As an MCP server** for Claude Desktop / Claude Code — drop into `.mcp.json`:

```json
{ "mcpServers": { "kma-memory": { "command": "kma-mcp" } } }
```

Exposes `memory.add · search · get_context · forget · explain_address`.

---

## How it works

```
text ─▶ embedding (any model) ─▶ φ chart ─▶ point in a Poincaré ball ─▶ memory tree ─▶ store (any DB)
                                    │                    │
                          direction = meaning    radius = generality
```

1. **Embed** with any model (a deterministic hashing fallback keeps it dependency-free).
2. **Project** into an *n*-D hyperbolic ball — semantics set the direction, hierarchy sets the radius.
3. **Place** the memory under its nearest concept automatically (no manual parenting).
4. **Retrieve** with a hybrid of cosine (similarity) + hyperbolic structure (branch / generality) + recency / importance — and return the **why** alongside the **what**.

---

## What's inside

| Module | Role |
|---|---|
| `kma/memory.py` | `AgenticMemory` — add / search / get_context / update / forget + auto-placement |
| `kma/engine.py` | embed → place → hybrid retrieval |
| `kma/geometry.py` | the math heart: Poincaré-ball ops with learnable curvature |
| `kma/chart.py` · `train.py` | the trainable projection `φ` and its objective |
| `kma/extract.py` | LLM fact extraction (distill turns into atomic memories) |
| `kma/connectors/` | `OpenRouterAgent` agent loop |
| `kma/mcp_server.py` | MCP connector for LLM hosts |

---

## Honest scope

- **Use KMA when structure matters:** agent memory, taxonomies, is-a retrieval, drill-down / roll-up, explainable recall.
- **Don't bother for flat RAG / dedup / FAQ** — there, an embedding + cosine + a vector DB is the right tool, and KMA adds nothing.
- KMA is a **layer**, not a replacement: it sits between your embedder and your store. It does not claim to be a faster embedding or a Pinecone competitor.

---

## Roadmap

- Train on large real hierarchies (WordNet, arXiv) with entailment-cone loss for unseen-concept generalization.
- LLM-driven branch consolidation (summarize dense subtrees) and decay-based forgetting.
- SQLite / pgvector / Pinecone storage backends behind the current interface.
- REST + LangChain / LlamaIndex adapters alongside MCP.

---

## License

**KMA — Copyright © 2026 Abhijit S R** ([@abhijit-without-h](https://github.com/Abhijit-without-h), git `now-im-inevitable`). Released under **GNU AGPL-3.0-or-later** — see [`LICENSE`](LICENSE).

Strong copyleft, chosen so credit is preserved and improvements stay open. Note **AGPL §13**: if you run a modified version as a network service (MCP server / hosted API), you must offer your users its complete source under the same license. For commercial or closed use, contact the author for a separate license.
