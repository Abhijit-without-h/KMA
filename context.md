# Kleinian / Hyperbolic Memory Research Brief

## Working Title

**Kleinian Memory Architecture (KMA)**

A mathematically structured memory system for LLMs where semantic items are first embedded in a vector space, projected into a hyperbolic chart, encoded as complex coordinates, and then organized by a discrete Möbius transformation group so that memory addresses become group words rather than flat IDs.

---

## 1. Problem Statement

Current AI memory systems mostly treat memory as flat vectors, chunks, or graph nodes with conventional nearest-neighbor retrieval. That works, but it does not naturally encode the recursive, branching, hierarchical nature of real conversation and reasoning.

This project explores a different question:

> Can we represent AI memory using a mathematically grounded hyperbolic + complex-analytic + group-theoretic address space, so that memory items are organized by recursive transformations instead of only by Euclidean distance?

The goal is not to beat vector databases immediately. The goal is to build a mathematically coherent memory representation that can later be evaluated for retrieval, compression, organization, and interpretability.

---

## 2. Core Research Hypothesis

Semantic items are not best represented as isolated points in flat Euclidean space. Instead, they can be represented as nodes in a hierarchical latent structure where:

1. a standard embedding model captures semantics,
2. a learned projection maps embeddings into a hyperbolic chart,
3. a complex coordinate inside a unit disk acts as the local geometric point,
4. a discrete subgroup of Möbius transformations acts on that disk,
5. a memory address is a word in the group rather than a UUID.

This makes memory recursive, structured, and mathematically explicit.

---

## 3. Main Idea in One Diagram

```text
Text / Conversation / Document
        ↓
Embedding model
        ↓
Latent semantic vector e ∈ R^d
        ↓
Chart map φ(e) → z ∈ D ⊂ C
        ↓
Hyperbolic normalization / projection
        ↓
Group generator assignment g₁, g₂, ...
        ↓
Memory address = word in a Kleinian-style group
        ↓
Stored memory node with parent, children, timestamp, weights
```

---

## 4. What We Are Actually Building

We are building a **memory engine**, not a new LLM.

The system has these parts:

### 4.1 Ingestion layer

Accepts text, metadata, timestamps, users, sessions, conversation turns, summaries, and derived sub-memories.

### 4.2 Embedding layer

Turns text into semantic vectors using a modern embedding model.

### 4.3 Geometry layer

Projects embeddings into a hyperbolic chart and then into a complex coordinate inside the unit disk.

### 4.4 Group-action layer

Defines a small set of Möbius generators and uses them to represent transitions, nesting, and recursive refinement.

### 4.5 Memory graph / group store

Stores nodes, generator words, parent-child relations, and metadata.

### 4.6 Query layer

Maps a query into the same geometric space, then navigates or expands candidate memory branches.

### 4.7 MCP / API layer

Exposes the memory engine as an external tool server to an LLM host.

### 4.8 Visualization layer

Shows disk layout, generator paths, branching structure, and memory depth.

---

## 5. Mathematical Foundation

## 5.1 Semantic embedding

Let a text item be represented as a high-dimensional vector:

[
e \in \mathbb{R}^d
]

This can come from a sentence embedding model.

The embedding should capture semantic similarity, topic, style, and context.

---

## 5.2 Latent chart map to the complex plane

We need a learned map from the embedding space into a 2D complex chart.

A practical version is:

[
u = W e + b \in \mathbb{R}^2
]

[
z_0 = u_1 + i u_2
]

Then normalize into the unit disk:

[
z = \frac{\tanh(|u|)}{|u| + \epsilon} (u_1 + i u_2)
]

So the final coordinate satisfies:

[
|z| < 1
]

This is the local geometric point.

---

## 5.3 Hyperbolic interpretation

The unit disk is used as a hyperbolic chart. Points near the boundary represent increasingly fine-grained structure. This is useful for recursive branching because hyperbolic space naturally expands capacity toward the boundary.

This gives a geometric mechanism for hierarchy:

* root ideas stay near the center,
* child concepts move outward,
* deeper refinements occupy nested regions.

---

## 5.4 Möbius transformations

A disk-preserving Möbius transformation has the form:

[
g(z) = e^{i\theta} \frac{z - \alpha}{1 - \overline{\alpha} z}, \quad |\alpha| < 1
]

This transformation preserves the unit disk.

In the project, each generator is a structured semantic move such as:

* topic shift,
* detail refinement,
* correction,
* temporal update,
* parent-to-child expansion,
* sibling transition.

---

## 5.5 Group words as memory addresses

Instead of storing a memory as only a point (z), store it as:

[
M = (z, w)
]

where (w) is a word in the generator set:

[
w = g_{i_n} \circ g_{i_{n-1}} \circ \cdots \circ g_{i_1}
]

This means the memory can be addressed by its transformation history.

Example:

```text
g₁ → g₂ → g₄ → g₂ → g₁
```

is not just a path. It is the address.

---

## 5.6 Reflection interpretation

If a generator is chosen to represent inversion or reflection across a circle boundary, then hierarchical recursion becomes a sequence of reflected regions.

That gives a fractal / Kleinian-style interpretation:

* the parent region contains a concept,
* reflection creates a child region,
* recursive reflection creates deeper branches,
* the memory map becomes self-similar.

---

## 5.7 Stability and canonicalization

A useful research problem is to define:

* whether two different group words can map to nearly the same memory,
* how to canonicalize equivalent or near-equivalent addresses,
* how to measure drift when the embedding of a memory changes.

These are not optional. They are part of the mathematical definition of the system.

---

## 6. Proposed Data Model

Every memory node should store:

```json
{
  "id": "stable internal identifier",
  "text": "raw text or summary",
  "embedding": "float vector",
  "complex_coordinate": "complex point inside unit disk",
  "generator_word": ["g1", "g2", "g4"],
  "parent_id": "optional",
  "children_ids": ["..."],
  "topic_label": "optional",
  "timestamp": "ISO-8601",
  "source": "chat / document / tool / agent",
  "confidence": 0.0,
  "importance": 0.0,
  "version": 1,
  "metadata": {}
}
```

Suggested storage split:

* relational store for canonical metadata,
* document store for content,
* graph store for parent-child and generator relations,
* optional vector index for fallback search.

---

## 7. What Exactly Must Be Implemented

## 7.1 Embedding pipeline

Implement a reusable embedding service that can accept:

* raw text,
* chunked text,
* summaries,
* dialog turns,
* tool outputs.

The service should output:

* a dense vector,
* token counts,
* a stable content hash,
* optional semantic tags.

---

## 7.2 Projection network

Implement a learned projection module:

[
\phi : \mathbb{R}^d \rightarrow \mathbb{D}
]

Practical implementation:

* a small MLP,
* or a linear head + normalization,
* with a disk projection step.

The projection must be differentiable if training is planned.

---

## 7.3 Möbius generator library

Implement a library of generator objects with:

* parameters (\alpha, \theta),
* forward map,
* inverse map,
* composition,
* numerical projection back into the disk,
* stability checks.

Each generator should represent an interpretable semantic action.

---

## 7.4 Word-to-address encoding

Implement:

* assignment of a memory item to a path of generators,
* encoding of the path as a canonical word,
* optional compression of repeated subpaths,
* decoding back to the node.

---

## 7.5 Memory insertion

When a new memory arrives:

1. embed it,
2. project it to the disk,
3. compare with nearby nodes or parent concepts,
4. assign a generator path,
5. store node + metadata,
6. update parent-child relations,
7. optionally create a summary node.

---

## 7.6 Memory query

When a query arrives:

1. embed the query,
2. project it to the disk,
3. find a coarse candidate region,
4. expand through generator neighbors,
5. return top candidate nodes,
6. attach paths and explanation traces.

At the start, retrieval efficiency is not the main goal. Correct mathematical mapping and structural consistency are.

---

## 7.7 Rebalancing / reorganization

Implement a reorganization routine that can:

* merge nearly identical branches,
* split overloaded branches,
* reassign nodes if a branch becomes semantically dense,
* preserve old addresses through redirects or aliases.

---

## 7.8 Visualization

Implement at least one visual debugger:

* disk plot,
* branch tree view,
* generator path explorer,
* node density heatmap,
* reflection depth view.

Visualization is essential because this project is geometric.

---

## 7.9 MCP server layer

Expose the memory engine as tools via MCP so an LLM can use it as an external memory system.

Suggested tools:

* `memory.insert`
* `memory.query`
* `memory.expand_branch`
* `memory.summarize_branch`
* `memory.link_nodes`
* `memory.visualize`
* `memory.reindex`
* `memory.explain_address`

---

## 8. Recommended Modern Stack

This is a practical, current stack for building the project.

### Core language/runtime

* **Python 3.12+**
* **uv** for dependency and environment management
* **ruff** for linting
* **pytest** for tests
* **mypy** or **pyright** for static typing

### API and schemas

* **FastAPI** for HTTP APIs
* **Pydantic v2** for typed data models and validation
* **MCP** for exposing memory tools to LLM hosts

### Embeddings and ML

* **SentenceTransformers** for embedding and reranking
* **PyTorch** for trainable components
* **Geoopt** or **Geomstats** for manifold-aware math and hyperbolic optimization

### Data and storage

* **PostgreSQL** for canonical metadata
* **Redis** for caching and short-lived working state
* **Qdrant / pgvector / another vector backend** as a fallback semantic index if needed
* **NetworkX** for prototype graph operations

### Visualization and UI

* **Plotly** or **Matplotlib** for geometric plots
* **Streamlit** or **Next.js** for a research dashboard
* **React + TypeScript** if a richer UI is needed

### Optional orchestration

* **Celery / Dramatiq / Arq** for background jobs
* **Docker** for reproducible deployment
* **OpenTelemetry** for tracing

---

## 9. Implementation Strategy

## Phase A — Mathematical prototype

Build a minimal Python prototype with:

* embedding input,
* 2D chart map,
* disk projection,
* Möbius generator definitions,
* storage of nodes with paths.

No database complexity yet. Just prove the math pipeline works.

## Phase B — Memory engine

Add:

* insertion,
* query,
* parent-child linking,
* path canonicalization,
* node versioning,
* branch summaries.

## Phase C — MCP server

Wrap the engine as an MCP server so external LLM clients can call it.

## Phase D — Persistence

Move from in-memory storage to Postgres / graph store / vector fallback.

## Phase E — Visualization and debugging

Build the visual tools needed to inspect branches, words, and reflections.

## Phase F — Research evaluation

Measure:

* reconstruction consistency,
* address stability,
* branching compactness,
* semantic locality,
* drift under updates,
* interpretability of group words.

---

## 10. Training / Learning Plan

If the system is learned rather than hand-designed, train these components:

### 10.1 Chart map training

Train the projection (\phi) so semantically similar items land near each other in the disk.

### 10.2 Generator learning

Learn generator parameters so that each generator corresponds to a useful semantic transition.

### 10.3 Regularizers

Use regularization to enforce:

* disk boundary constraints,
* smoothness,
* group consistency,
* path sparsity,
* stability over time.

### 10.4 Contrastive objectives

Use pairs and triplets:

* same topic vs different topic,
* parent vs child,
* answer vs correction,
* original memory vs updated memory.

---

## 11. Important Research Risks

1. **Information loss in 2D charting**

   * A 768D embedding compressed into a single complex coordinate will lose information.
   * This is acceptable only if the system is designed as a chart, not as the full semantic container.

2. **Unstable generator assignments**

   * Small semantic changes might cause large address changes.

3. **Collision / aliasing**

   * Different memories may map to similar coordinates or group words.

4. **Boundary numerical issues**

   * Möbius maps can be unstable near the unit-circle boundary.

5. **Path explosion**

   * Deep recursion may create long generator words that need canonical compression.

6. **Overfitting to geometry**

   * The system may look elegant but not preserve useful semantics.

7. **Complexity creep**

   * The first prototype must stay small enough to test.

---

## 12. What Success Looks Like

The project is successful if it can demonstrate that:

* semantic items can be mapped consistently into a hyperbolic-complex chart,
* memory items can be represented by generator words,
* recursive structure can be navigated algebraically,
* the address scheme is interpretable,
* the system remains stable under updates,
* the representation is useful as a research artifact even before retrieval optimizations.

---

## 13. Agent Prompt to Think and Implement

Paste the following into your build agent.

```text
You are a research-grade implementation agent working on a project called Kleinian Memory Architecture (KMA).

Your task is to design and implement a mathematically grounded memory engine for LLMs where semantic items are embedded in Euclidean space, projected into a hyperbolic chart, represented as complex coordinates inside the unit disk, and then organized by a discrete set of Möbius transformations. A memory address is not a flat ID; it is a canonical word in a generator set.

Primary objective:
Build a clean, modular, well-typed, testable prototype that proves the mathematical pipeline and supports insertion, querying, linking, visualization, and MCP exposure.

You must think in layers:
1. semantic embedding,
2. chart projection into the complex unit disk,
3. Möbius generator definitions,
4. generator-word address encoding,
5. memory storage and versioning,
6. query/navigation,
7. visualization,
8. MCP tool exposure.

Do not jump directly to an optimized production system. First prove the math and architecture with a minimal, deterministic prototype.

Requirements:
- Use Python 3.12+.
- Use uv for environment/dependency management.
- Use FastAPI for the API layer.
- Use Pydantic v2 for schemas and validation.
- Use SentenceTransformers for embeddings.
- Use PyTorch for trainable components.
- Use Geoopt or Geomstats for hyperbolic/manifold-aware operations.
- Use MCP to expose tools to an external LLM host.
- Use PostgreSQL or SQLite for persistent metadata during the prototype, with room to migrate to a full production store.
- Use Redis only if caching is truly needed.
- Use NetworkX for initial graph/prototype operations.
- Use Plotly or Matplotlib for visualization.
- Use pytest, ruff, and type checking.

Mathematical requirements:
- Define an embedding vector e in R^d.
- Define a learned chart map phi: R^d -> D where D is the unit disk in the complex plane.
- Ensure the chart output z satisfies |z| < 1.
- Define disk-preserving Möbius transformations g(z)=exp(i theta) * (z-alpha)/(1-conj(alpha) z) with |alpha|<1.
- Represent memory addresses as generator words, e.g. g2 o g1 o g4.
- Store both the coordinate and the word.
- Provide inverse and composition operations.
- Provide canonicalization logic for repeated or equivalent paths.
- Keep numerical stability near the disk boundary.

Implementation tasks:
1. Create a modular repository structure with separate packages for embeddings, geometry, generators, storage, API, MCP tools, and visualization.
2. Build a data model for memory nodes that stores raw text, embeddings, complex coordinates, generator words, parent-child links, metadata, timestamps, confidence, importance, and version.
3. Implement insertion: embed text, project to disk, choose a generator path, store node, link to parents, and emit metadata.
4. Implement query: embed query, project it, retrieve candidate branches, expand through generator neighbors, and return ranked nodes with explanation paths.
5. Implement a minimal mathematical visualization of the disk, generator paths, and recursion depth.
6. Implement MCP tools: memory.insert, memory.query, memory.expand_branch, memory.explain_address, memory.reindex, memory.visualize.
7. Add tests for projection validity, Möbius invariants, serialization, insertion/query round trips, and stability.
8. Add documentation explaining the math and the system architecture.

Agent behavior rules:
- Prefer simple, readable code over clever code.
- Keep all math explicit and typed.
- Explain design choices in code comments and docs.
- When a design choice is ambiguous, choose the simplest version that preserves the research hypothesis.
- Preserve the distinction between semantic meaning and geometric address.
- Do not claim the system is already superior to vector databases. Treat performance as a later evaluation question.
- If a piece of math is only approximate, say so clearly.

Deliverables:
- A working prototype.
- A clear README.
- A research note describing the math.
- A small visual demo.
- A minimal MCP server.
- Tests.
- A list of open research questions.

Definition of done:
The prototype is done when a user can insert text, see its embedding, see its complex coordinate, see its generator word, query nearby memory, and inspect the resulting branch structure through a visualization or MCP tool.
```

---

## 14. Suggested Research Questions to Keep Open

* What is the best chart map from high-dimensional semantics to the complex disk?
* Should generator selection be learned or rule-based?
* What is the canonical form of a group word?
* How should temporal updates alter generator words?
* When do two addresses count as the same semantic memory?
* Can the representation survive continual learning?
* Can the model reconstruct hierarchy from group words alone?

---

## 15. Final Positioning Statement

This project is best framed as:

> A new algebraic-hyperbolic memory representation for LLMs, where semantic items are mapped to complex coordinates and organized via Möbius-group actions, enabling recursive and interpretable memory addressing.

That phrasing keeps the project mathematically serious, research-focused, and implementation-ready.

