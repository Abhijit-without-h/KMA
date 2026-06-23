from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from kma.retrievers import Candidate

RRF_K = 60                  # standard RRF constant
DEFAULT_TAU_EVIDENCE = 0.30
MIN_CORROBORATION = 2       # need >= this many retrievers to be "confident"


@dataclass
class FusedHit:
    node_id: str
    fused: float                       # raw RRF score
    evidence: float                    # confidence in [0, 1]
    agreement: int                     # # of distinct retrievers that returned it
    voters: tuple[str, ...]
    ranks: dict[str, int] = field(default_factory=dict)   # retriever -> rank


@dataclass
class FusionResult:
    hits: list[FusedHit]
    n_retrievers: int
    abstain: bool                      # True => no corroborated, confident answer
    reason: str = ""

    @property
    def top(self) -> FusedHit | None:
        return self.hits[0] if self.hits else None


def reciprocal_rank_fusion(lists: list[list[Candidate]], k: int = RRF_K
                           ) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for cands in lists:
        for cand in cands:
            scores[cand.node_id] += 1.0 / (k + cand.rank + 1)
    return dict(scores)


def fuse(lists: list[list[Candidate]], *, n_retrievers: int | None = None,
         tau_evidence: float = DEFAULT_TAU_EVIDENCE,
         min_corroboration: int = MIN_CORROBORATION) -> FusionResult:
    """Fuse ranked candidate lists into evidence-scored hits + an abstain flag.

    evidence = norm_rrf * (0.5 + 0.5 * agreement / n_retrievers)
      - norm_rrf  : RRF score scaled to [0, 1] across this candidate set
      - agreement : how many distinct retrievers returned the candidate
    so a candidate that several methods rank highly scores near 1.0, while a
    lone-retriever candidate is capped well below it.
    """
    lists = [lst for lst in lists if lst]
    n = n_retrievers if n_retrievers is not None else len(lists)
    if not lists or n == 0:
        return FusionResult(hits=[], n_retrievers=n, abstain=True,
                            reason="no retrievers returned candidates")

    rrf = reciprocal_rank_fusion(lists)
    voters: dict[str, list[str]] = defaultdict(list)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for cands in lists:
        for cand in cands:
            voters[cand.node_id].append(cand.retriever)
            ranks[cand.node_id][cand.retriever] = cand.rank

    hi = max(rrf.values())
    lo = min(rrf.values())

    hits: list[FusedHit] = []
    for nid, score in rrf.items():
        distinct = sorted(set(voters[nid]))
        agreement = len(distinct)
        # When every candidate ties (e.g. a single hit), each is "the best" -> 1.0.
        norm = 1.0 if hi == lo else (score - lo) / (hi - lo)
        evidence = norm * (0.5 + 0.5 * min(agreement, n) / n)
        hits.append(FusedHit(node_id=nid, fused=score, evidence=round(evidence, 4),
                             agreement=agreement, voters=tuple(distinct),
                             ranks=dict(ranks[nid])))
    # sort by corroboration first, then evidence -- corroborated answers lead.
    hits.sort(key=lambda h: (h.agreement, h.evidence, h.fused), reverse=True)

    top = hits[0]
    can_corroborate = n >= min_corroboration
    abstain = can_corroborate and (
        top.agreement < min_corroboration or top.evidence < tau_evidence
    )
    reason = ""
    if abstain:
        reason = (f"top hit corroborated by {top.agreement}/{n} retrievers "
                  f"(evidence {top.evidence:.2f} < {tau_evidence:.2f})")
    return FusionResult(hits=hits, n_retrievers=n, abstain=abstain, reason=reason)


class CrossEncoderReranker:
    """Optional precision booster: a cross-encoder re-scores the fused shortlist.

    Lazy/optional -- needs `sentence-transformers`. The evidence gate from fuse()
    still governs whether the result is presented as confident.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # lazy import

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[FusedHit], texts: dict[str, str],
               top_n: int = 20) -> list[FusedHit]:
        import numpy as np

        shortlist = hits[:top_n]
        if not shortlist:
            return hits
        pairs = [(query, texts.get(h.node_id, "")) for h in shortlist]
        raw = np.asarray(self.model.predict(pairs), dtype=np.float64)
        ce = 1.0 / (1.0 + np.exp(-raw))                  # sigmoid -> [0, 1]
        for h, s in zip(shortlist, ce):
            # blend model precision with corroboration so a confident cross-encoder
            # score still can't fully override "only one retriever found this".
            h.evidence = round(float(0.6 * s + 0.4 * h.evidence), 4)
        shortlist.sort(key=lambda h: (h.agreement, h.evidence), reverse=True)
        return shortlist + hits[top_n:]
