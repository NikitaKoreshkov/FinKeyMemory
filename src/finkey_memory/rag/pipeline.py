# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Company RAG retrieval pipeline (dense bge-m3 via ``embed_fn``).

Stages (all optional via env, default **on** where cheap):

1. **Multi-query** — a few deterministic paraphrases / substrings of the user query
   to reduce vocabulary mismatch (no keyword lists in any natural language).
2. **Per-query dense search** in ``finkey_rag`` with slightly relaxed ``score_min``.
3. **RRF fusion** across ranked lists (``finkey_memory.rag.fusion``).
4. **Dedup** by normalized text prefix.
5. **Redis cache** (short TTL) when a Redis client is available on ``MemoryManager``.

This does **not** require re-creating Qdrant collections: it only uses existing dense vectors
(``bge-m3`` recommended via ``build_embedder_from_env``).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Optional

from finkey_memory.rag.cache import cache_get, cache_set
from finkey_memory.rag.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], list[float]]


def _env_bool(name: str, default: str = "1") -> bool:
    v = (os.getenv(name, default) or default).strip().lower()
    return v not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _query_variants(query: str) -> list[str]:
    """Language-agnostic query expansion (structure only, no word lists)."""
    q = (query or "").strip()
    if not q:
        return []
    out: list[str] = [q]
    if _env_bool("FINKEY_RAG_MULTI_QUERY", "1"):
        low = q.lower()
        if low != q and len(low) > 2:
            out.append(low)
        words = re.split(r"\s+", q)
        if len(words) > 6:
            out.append(" ".join(words[:6]))
            out.append(" ".join(words[-8:]))
        if len(q) > 24:
            out.append(q[: max(32, len(q) // 2)])
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        x = x.strip()
        if len(x) < 2 or x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq[: _env_int("FINKEY_RAG_MULTI_QUERY_MAX", 5)]


def _norm_key(text: str) -> str:
    t = (text or "").strip().lower()
    return t[:480]


def retrieve_company_rag(
    *,
    semantic: Any,
    company_id: str,
    query: str,
    redis_client: Any = None,
) -> list[str]:
    """
    High-recall RAG over ``SemanticMemory.search_rag`` / ``search_rag_scored``.

    Falls back to plain ``search_rag`` if ``search_rag_scored`` is missing.
    """
    if semantic is None or not (query or "").strip():
        return []

    mode = (os.getenv("FINKEY_RAG_PIPELINE", "advanced") or "advanced").strip().lower()
    if mode in ("simple", "legacy", "off", "0"):
        try:
            return semantic.search_rag(
                company_id,
                query.strip(),
                top_k=_env_int("FINKEY_RAG_SIMPLE_TOP_K", 5),
            )
        except Exception as exc:
            logger.warning("search_rag (simple) failed: %s", exc)
            return []

    fp = "rrf_v1"
    if redis_client is not None:
        cached = cache_get(redis_client, company_id, fp, query.strip())
        if cached is not None:
            return cached

    score_floor = float(os.getenv("FINKEY_RAG_DENSE_SCORE_MIN", "0.42") or "0.42")
    per_q_k = _env_int("FINKEY_RAG_PER_QUERY_TOP_K", 14)
    final_cap = _env_int("FINKEY_RAG_FINAL_CHUNKS", 10)

    ranked_lists: list[list[tuple[str, float]]] = []
    variants = _query_variants(query)

    for v in variants:
        try:
            if hasattr(semantic, "search_rag_scored"):
                rows = semantic.search_rag_scored(
                    company_id,
                    v,
                    top_k=per_q_k,
                    score_min=score_floor,
                )
            else:
                texts = semantic.search_rag(company_id, v, top_k=per_q_k, score_min=score_floor)
                rows = [
                    {"id": f"t{i}", "score": 1.0, "content": t}
                    for i, t in enumerate(texts)
                    if t
                ]
        except Exception as exc:
            logger.debug("RAG variant search failed (%r): %s", v[:48], exc)
            continue
        lst: list[tuple[str, float]] = []
        for row in rows:
            content = (row.get("content") or "").strip()
            if not content:
                continue
            did = row.get("doc_id")
            ch = row.get("chunk_index")
            if did is not None and ch is not None:
                key = f"doc:{did}:chunk:{ch}"
            else:
                key = f"id:{row.get('id', _norm_key(content))}"
            lst.append((key, float(row.get("score") or 0.0)))
        if lst:
            ranked_lists.append(lst)

    if not ranked_lists:
        try:
            out = semantic.search_rag(
                company_id,
                query.strip(),
                top_k=final_cap,
                score_min=0.55,
            )
        except Exception as exc:
            logger.warning("RAG fallback search_rag failed: %s", exc)
            return []
        cache_set(redis_client, company_id, fp, query.strip(), out)
        return out

    fused = reciprocal_rank_fusion(ranked_lists)
    key_order = [k for k, _s in fused[: final_cap * 3]]

    text_by_key: dict[str, str] = {}
    for v in variants:
        try:
            if hasattr(semantic, "search_rag_scored"):
                rows = semantic.search_rag_scored(
                    company_id,
                    v,
                    top_k=per_q_k,
                    score_min=score_floor,
                )
            else:
                continue
        except Exception:
            continue
        for row in rows:
            content = (row.get("content") or "").strip()
            if not content:
                continue
            did = row.get("doc_id")
            ch = row.get("chunk_index")
            if did is not None and ch is not None:
                key = f"doc:{did}:chunk:{ch}"
            else:
                key = f"id:{row.get('id', _norm_key(content))}"
            if key not in text_by_key:
                text_by_key[key] = content

    picked: list[str] = []
    seen_norm: set[str] = set()
    for key in key_order:
        txt = text_by_key.get(key)
        if not txt:
            continue
        nk = _norm_key(txt)
        if nk in seen_norm:
            continue
        seen_norm.add(nk)
        picked.append(txt)
        if len(picked) >= final_cap:
            break

    if not picked:
        try:
            picked = semantic.search_rag(company_id, query.strip(), top_k=final_cap)
        except Exception:
            picked = []

    cache_set(redis_client, company_id, fp, query.strip(), picked)
    return picked


class MemoryRAGPipeline:
    """Thin wrapper for symmetry / future HyDE-rerank hooks."""

    def __init__(self, *, semantic: Any, redis_client: Any = None) -> None:
        self._semantic = semantic
        self._redis = redis_client

    def retrieve_company_kb(self, company_id: str, query: str) -> list[str]:
        return retrieve_company_rag(
            semantic=self._semantic,
            company_id=company_id,
            query=query,
            redis_client=self._redis,
        )
