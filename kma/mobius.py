"""OPTIONAL addressing layer — Mobius generators & group-word addresses.

This is the "Kleinian" flavor from the brief: each parent->child edge is
labeled with a generator, so a node's address is the canonical word of
generators from the root. It is interpretable bookkeeping; it does NOT drive
retrieval (engine.py does). Kept separate on purpose so the retrieval win
never depends on this riskier piece.

A disk-preserving Mobius map (2D / complex form, for the visual chart):
    g(z) = e^{i.theta} (z - alpha) / (1 - conj(alpha) z),   |alpha| < 1
"""

from __future__ import annotations

from dataclasses import dataclass

# Interpretable semantic moves, one generator symbol each.
GENERATORS = {
    "g1": "child / refinement",
    "g2": "sibling / topic shift",
    "g3": "correction",
    "g4": "temporal update",
    "g5": "summary / parent",
}


@dataclass(frozen=True)
class Mobius:
    """2D disk-preserving Mobius transform on a complex coordinate z."""

    alpha: complex   # |alpha| < 1
    theta: float = 0.0

    def __post_init__(self) -> None:
        if abs(self.alpha) >= 1.0:
            raise ValueError("|alpha| must be < 1 to preserve the disk")

    def __call__(self, z: complex) -> complex:
        import cmath

        return cmath.exp(1j * self.theta) * (z - self.alpha) / (1 - self.alpha.conjugate() * z)

    def inverse(self) -> "Mobius":
        import cmath

        a = self.alpha * cmath.exp(1j * self.theta)
        return Mobius(alpha=-a, theta=-self.theta)


def address_word(child_index_path: list[int]) -> list[str]:
    """Turn a path of child indices into a canonical generator word.

    Convention: descending to the i-th child emits g1 (refine) then i copies
    of g2 (sibling shift). Repeated runs could be compressed later
    (brief 7.4 path compression) — left explicit for clarity in v1.
    """
    word: list[str] = []
    for idx in child_index_path:
        word.append("g1")
        word.extend(["g2"] * idx)
    return word
