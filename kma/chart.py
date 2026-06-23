from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

EPS = 1e-9
MAX_TANH_ARG = 1.0 - 1e-5


# --- torch curvature-c geometry (differentiable) ---------------------------
def t_expmap0_c(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """exp_0^c(v) = tanh(sqrt(c)|v|/2)/sqrt(c) * v/|v|, then project into ball."""
    sc = torch.sqrt(c)
    norm = v.norm(dim=-1, keepdim=True).clamp_min(EPS)
    x = torch.tanh(sc * norm / 2.0) / sc * v / norm
    return project_c(x, c)


def project_c(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    sc = torch.sqrt(c)
    max_norm = (1.0 - 1e-5) / sc
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    factor = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * factor


def t_mobius_add_c(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    xy = (x * y).sum(dim=-1, keepdim=True)
    xx = (x * x).sum(dim=-1, keepdim=True)
    yy = (y * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * yy) * x + (1.0 - c * xx) * y
    den = 1.0 + 2.0 * c * xy + c * c * xx * yy
    return num / (den + EPS)


def t_dist_c(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Pairwise (row-aligned) geodesic distance in the curvature-c ball."""
    sc = torch.sqrt(c)
    diff = t_mobius_add_c(-x, y, c)
    dn = diff.norm(dim=-1).clamp(0.0, MAX_TANH_ARG / sc.item())
    return (2.0 / sc) * torch.atanh((sc * dn).clamp(max=MAX_TANH_ARG))


class HyperbolicChart(nn.Module):
    def __init__(self, in_dim: int, dim: int = 64, hidden: int = 256,
                 init_c: float = 1.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # c = softplus(raw_c); invert softplus to start near init_c.
        raw = float(np.log(np.expm1(init_c)))
        self.raw_c = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    @property
    def c(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_c) + 1e-4

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return t_expmap0_c(self.net(emb), self.c)

    @torch.no_grad()
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        self.eval()
        emb = torch.as_tensor(np.asarray(embeddings, dtype=np.float32))
        if emb.ndim == 1:
            emb = emb[None, :]
        return self.forward(emb).cpu().numpy().astype(np.float64)

    def curvature(self) -> float:
        return float(self.c.detach())

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        torch.save({"state": self.state_dict(), "in_dim": self.in_dim,
                    "dim": self.dim}, path)

    @classmethod
    def load(cls, path: str | Path) -> "HyperbolicChart":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(in_dim=ckpt["in_dim"], dim=ckpt["dim"])
        model.load_state_dict(ckpt["state"])
        model.eval()
        return model
