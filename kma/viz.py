from __future__ import annotations

import numpy as np

from kma.store import MemoryStore


def plot_disk(store: MemoryStore, path: str = "kma_disk.png") -> str:
    import matplotlib.pyplot as plt  # lazy import

    nodes = store.all()
    if not nodes:
        raise ValueError("nothing to plot")

    coords = np.array([n.ball_coord for n in nodes])
    # PCA to 2D for display, then rescale into the unit disk.
    coords = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords, full_matrices=False)
    pts = coords @ vt[:2].T
    span = np.abs(pts).max() + 1e-9
    pts = 0.95 * pts / span
    pos = {n.id: pts[i] for i, n in enumerate(nodes)}

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#bbb"))
    for n in nodes:
        if n.parent_id in pos:
            x0, y0 = pos[n.parent_id]
            x1, y1 = pos[n.id]
            ax.plot([x0, x1], [y0, y1], color="#cccccc", lw=0.8, zorder=1)
    depths = np.array([n.depth for n in nodes])
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=depths, cmap="viridis", s=60, zorder=2)
    for n, p in zip(nodes, pts):
        label = (n.topic_label or n.text)[:18]
        ax.annotate(label, p, fontsize=7, alpha=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title("KMA Poincare disk (PCA projection; color = depth)")
    fig.colorbar(sc, ax=ax, label="depth", shrink=0.7)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
