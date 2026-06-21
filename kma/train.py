"""Train the HyperbolicChart on the taxonomy's is-a tree.

Objective (the heart of beating cosine):
  L_hier  parent<->child close + child farther from origin than parent, and
          closer than a cross-branch negative  -> encodes asymmetric is-a.
  L_sem   siblings close, cross-branch far                 -> preserves semantics.
  L_depth radius grows with tree depth                     -> generality ordering.

Supervision is the parent/child structure we already store -- free labels. Base
embeddings are frozen; only phi (and curvature c) train. Plain Adam on the
tangent-space parameters (no geoopt needed).

Run:  python -m kma.train
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from kma.chart import HyperbolicChart, t_dist_c
from kma.data import build_store

CHECKPOINT = "kma_chart.pt"
SEED = 0


@dataclass
class TrainData:
    embs: torch.Tensor          # [N, d] frozen embeddings
    depth: torch.Tensor         # [N]
    root: np.ndarray            # [N] top-ancestor index (branch id)
    edges: torch.Tensor         # [E, 2] (parent_idx, child_idx)
    sibs: torch.Tensor          # [S, 2] same-parent pairs


def build_training_data(store) -> TrainData:
    nodes = store.all()
    idx = {n.id: i for i, n in enumerate(nodes)}
    embs = np.array([n.embedding for n in nodes], dtype=np.float32)
    depth = np.array([n.depth for n in nodes], dtype=np.float32)

    root = np.zeros(len(nodes), dtype=np.int64)
    for n in nodes:
        anc = store.ancestors(n.id)
        root[idx[n.id]] = idx[anc[-1].id] if anc else idx[n.id]

    edges, sibs = [], []
    for n in nodes:
        if n.parent_id in idx:
            edges.append((idx[n.parent_id], idx[n.id]))
        kids = [idx[c] for c in n.children_ids if c in idx]
        for a in range(len(kids)):
            for b in range(a + 1, len(kids)):
                sibs.append((kids[a], kids[b]))

    return TrainData(
        embs=torch.tensor(embs),
        depth=torch.tensor(depth),
        root=root,
        edges=torch.tensor(edges, dtype=torch.long),
        sibs=torch.tensor(sibs, dtype=torch.long),
    )


def _cross_branch_negatives(anchor_idx: torch.Tensor, root: np.ndarray,
                            n_neg: int, rng: np.random.Generator) -> torch.Tensor:
    """For each anchor, sample n_neg node indices from a DIFFERENT branch."""
    n = len(root)
    out = np.empty((len(anchor_idx), n_neg), dtype=np.int64)
    for i, a in enumerate(anchor_idx.tolist()):
        pool = np.where(root != root[a])[0]
        out[i] = rng.choice(pool, size=n_neg, replace=len(pool) < n_neg)
    return torch.tensor(out)


def train(epochs: int = 400, lr: float = 5e-3, n_neg: int = 5,
          hier_margin: float = 1.0, sem_margin: float = 1.0,
          radius_margin: float = 0.12, verbose: bool = True) -> HyperbolicChart:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    store = build_store().store
    data = build_training_data(store)
    chart = HyperbolicChart(in_dim=data.embs.shape[1], dim=64)
    opt = torch.optim.Adam(chart.parameters(), lr=lr)

    pu, cv = data.edges[:, 0], data.edges[:, 1]
    sa, sb = data.sibs[:, 0], data.sibs[:, 1]
    max_depth = float(data.depth.max())

    for ep in range(epochs):
        chart.train()
        coords = chart(data.embs)
        c = chart.c
        sc = torch.sqrt(c)
        norms = coords.norm(dim=-1)

        # --- L_hier: edge attract + radius order + cross-branch ranking ------
        d_edge = t_dist_c(coords[pu], coords[cv], c)
        neg_h = _cross_branch_negatives(pu, data.root, n_neg, rng)
        d_neg_h = t_dist_c(coords[pu].repeat_interleave(n_neg, 0),
                           coords[neg_h.reshape(-1)], c).reshape(-1, n_neg)
        l_attract = d_edge.mean()
        l_rank_h = torch.relu(d_edge[:, None] - d_neg_h + hier_margin).mean()
        l_radius = torch.relu(norms[pu] - norms[cv] + radius_margin).mean()

        # --- L_sem: sibling attract vs cross-branch ranking ------------------
        d_sib = t_dist_c(coords[sa], coords[sb], c)
        neg_s = _cross_branch_negatives(sa, data.root, n_neg, rng)
        d_neg_s = t_dist_c(coords[sa].repeat_interleave(n_neg, 0),
                           coords[neg_s.reshape(-1)], c).reshape(-1, n_neg)
        l_sem = torch.relu(d_sib[:, None] - d_neg_s + sem_margin).mean()

        # --- L_depth: radius grows with depth --------------------------------
        target = data.depth / (max_depth + 1.0)
        l_depth = ((sc * norms - target) ** 2).mean()

        # Heavier radius/depth weight so radius cleanly stratifies by tree depth;
        # this is what makes the generality signal reliable at retrieval time.
        loss = l_attract + l_rank_h + 3.0 * l_radius + l_sem + 3.0 * l_depth
        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"ep {ep:4d}  loss {loss.item():.3f}  "
                  f"attract {l_attract.item():.2f}  rankH {l_rank_h.item():.2f}  "
                  f"sem {l_sem.item():.2f}  depth {l_depth.item():.3f}  "
                  f"c {chart.curvature():.3f}")

    chart.save(CHECKPOINT)
    if verbose:
        print(f"saved chart -> {Path(CHECKPOINT).resolve()}")
    return chart


if __name__ == "__main__":
    train()
