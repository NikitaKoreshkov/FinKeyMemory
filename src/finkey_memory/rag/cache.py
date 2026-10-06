# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Optional Redis cache for RAG query results (short TTL)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _ttl_sec() -> int:
    try:
        return max(0, int(os.getenv("FINKEY_RAG_CACHE_TTL_SEC", "120")))
    except ValueError:
        return 120


def cache_key(company_id: str, pipeline_fingerprint: str, query: str) -> str:
    h = hashlib.sha256(
        f"{company_id}|{pipeline_fingerprint}|{query}".encode("utf-8"),
    ).hexdigest()[:32]
    return f"finkey:rag:{company_id}:{h}"


def cache_get(redis_client: Any, company_id: str, fp: str, query: str) -> Optional[list[str]]:
    ttl = _ttl_sec()
    if ttl <= 0 or redis_client is None:
        return None
    key = cache_key(company_id, fp, query)
    try:
        raw = redis_client.get(key)
    except Exception as exc:
        logger.debug("RAG cache get failed: %s", exc)
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    return None


def cache_set(redis_client: Any, company_id: str, fp: str, query: str, chunks: list[str]) -> None:
    ttl = _ttl_sec()
    if ttl <= 0 or redis_client is None:
        return
    key = cache_key(company_id, fp, query)
    try:
        redis_client.setex(key, ttl, json.dumps(chunks, ensure_ascii=False))
    except Exception as exc:
        logger.debug("RAG cache set failed: %s", exc)
