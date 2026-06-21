# KMA (Kleinian Memory Architecture)
# Copyright (C) 2026 Abhijit S R (@abhijit-without-h, git: now-im-inevitable)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version. It is distributed WITHOUT ANY WARRANTY; see the GNU AGPL for
# details: <https://www.gnu.org/licenses/>.
"""Kleinian Memory Architecture (KMA) — hyperbolic memory engine for LLMs.

v1 scope (no MCP, no DB, no training):
  * embed text with a sentence model (hashing fallback if unavailable),
  * place each node in an n-D Poincare ball using its semantic direction
    and its *given* hierarchy depth (root near center, detail near boundary),
  * retrieve with a HYBRID strategy: flat cosine for recall, hyperbolic
    geometry for re-ranking + branch expansion.

The retrieval win over flat embeddings comes from hierarchy awareness, not
from low dimension. The Mobius/Kleinian addressing layer (mobius.py) is an
optional, separate bookkeeping layer and is NOT what makes retrieval better.
"""

from kma.models import MemoryNode
from kma.store import MemoryStore
from kma.engine import KMAEngine

__all__ = ["MemoryNode", "MemoryStore", "KMAEngine"]
