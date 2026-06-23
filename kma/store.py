from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kma.models import MemoryNode


class MemoryStore:
    def __init__(self) -> None:
        self._nodes: dict[str, MemoryNode] = {}
        # Monotonic counter bumped on every mutation. Caches (e.g. the engine's
        # retriever snapshot) compare against it to know when to rebuild, so
        # read-heavy query traffic never rebuilds an unchanged index.
        self._version = 0

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def version(self) -> int:
        return self._version

    def add(self, node: MemoryNode) -> None:
        self._nodes[node.id] = node
        if node.parent_id and node.parent_id in self._nodes:
            parent = self._nodes[node.parent_id]
            if node.id not in parent.children_ids:
                parent.children_ids.append(node.id)
        self._version += 1

    def get(self, node_id: str) -> MemoryNode | None:
        return self._nodes.get(node_id)

    def remove(self, node_id: str) -> None:
        """Delete a node; reparent its children to its parent (forget op)."""
        node = self._nodes.pop(node_id, None)
        if node is None:
            return
        self._version += 1
        new_parent = node.parent_id
        if new_parent and new_parent in self._nodes:
            p = self._nodes[new_parent]
            if node_id in p.children_ids:
                p.children_ids.remove(node_id)
        for cid in node.children_ids:
            child = self._nodes.get(cid)
            if not child:
                continue
            child.parent_id = new_parent
            if new_parent in self._nodes:
                parent = self._nodes[new_parent]
                child.depth = parent.depth + 1
                if cid not in parent.children_ids:
                    parent.children_ids.append(cid)
            else:
                child.depth = 0

    def all(self) -> list[MemoryNode]:
        return list(self._nodes.values())

    # --- hierarchy traversal -------------------------------------------------
    def ancestors(self, node_id: str) -> list[MemoryNode]:
        out: list[MemoryNode] = []
        node = self._nodes.get(node_id)
        while node and node.parent_id:
            node = self._nodes.get(node.parent_id)
            if node:
                out.append(node)
        return out

    def descendants(self, node_id: str) -> list[MemoryNode]:
        out: list[MemoryNode] = []
        stack = list(self._nodes[node_id].children_ids) if node_id in self._nodes else []
        while stack:
            cid = stack.pop()
            child = self._nodes.get(cid)
            if child:
                out.append(child)
                stack.extend(child.children_ids)
        return out

    def branch(self, node_id: str) -> list[MemoryNode]:
        """The node plus its ancestors and descendants (its whole branch)."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [node, *self.ancestors(node_id), *self.descendants(node_id)]

    # --- vectorized views ----------------------------------------------------
    def matrices(self) -> tuple[list[str], NDArray, NDArray]:
        """(ids, embeddings[N,d], coords[N,n]) in a stable order."""
        nodes = self.all()
        ids = [n.id for n in nodes]
        embs = np.array([n.embedding for n in nodes], dtype=np.float64)
        coords = np.array([n.ball_coord for n in nodes], dtype=np.float64)
        return ids, embs, coords

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        data = [n.model_dump() for n in self._nodes.values()]
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self._nodes = {d["id"]: MemoryNode(**d) for d in data}
        self._version += 1
