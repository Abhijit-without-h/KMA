from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryNode(BaseModel):
    id: str
    text: str
    content_hash: str
    embedding: list[float]                       # L2-normalized semantic vector
    ball_coord: list[float]                       # point in the Poincare ball
    depth: int = 0                                # hierarchy depth (0 = root)
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    generator_word: list[str] = Field(default_factory=list)  # optional address
    topic_label: str | None = None
    timestamp: str = Field(default_factory=_now)
    source: str = "chat"
    confidence: float = 1.0
    importance: float = 0.5
    version: int = 1
    metadata: dict = Field(default_factory=dict)

    @property
    def emb(self) -> NDArray[np.float64]:
        return np.asarray(self.embedding, dtype=np.float64)

    @property
    def coord(self) -> NDArray[np.float64]:
        return np.asarray(self.ball_coord, dtype=np.float64)
