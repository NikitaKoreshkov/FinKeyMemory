# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""RRF fusion and MMR diversification for multi-vector retrieval."""

from __future__ import annotations

from typing import Sequence

_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """
    ``ranked_lists`` — several lists of (doc_key, raw_score) ordered best-first.
    Returns merged list of (doc_key, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (key, _raw) in enumerate(lst):
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def cosine_sim_dense(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na ** 0.5 * (nb ** 0.5))


def mmr_select(
    query_vec: Sequence[float],
    candidates: list[tuple[str, str, Sequence[float]]],
    *,
    k: int,
    lambda_mult: float = 0.5,
) -> list[tuple[str, str]]:
    """
    Maximal Marginal Relevance.

    ``candidates``: list of (doc_key, text, embedding_vector).
    Returns up to ``k`` (doc_key, text) picks.
    """
    if k <= 0 or not candidates:
        return []
    chosen: list[tuple[str, str, Sequence[float]]] = []
    remaining = list(candidates)

    def relevance(vec: Sequence[float]) -> float:
        return cosine_sim_dense(query_vec, vec)

    while remaining and len(chosen) < k:
        best_i = -1
        best_score = -1.0
        for i, (_key, _text, vec) in enumerate(remaining):
            rel = relevance(vec)
            div = 0.0
            for _ck, _ct, v2 in chosen:
                div = max(div, cosine_sim_dense(vec, v2))
            score = lambda_mult * rel - (1.0 - lambda_mult) * div
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            break
        chosen.append(remaining.pop(best_i))
    return [(c[0], c[1]) for c in chosen]
