# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Composition root for FinKeyMemory: build a fully-wired ``MemoryManager``
from connection URLs (or env), with graceful degradation per substrate.

    from finkey_memory.factory import build_memory_manager
    mm = build_memory_manager(
        database_url="postgresql://user:pass@host/db",
        redis_url="redis://host:6379",
        qdrant_url="http://host:6333",
    )

Any substrate left unset (and absent from env) simply stays disabled:
  * no redis  → sessions live in-process
  * no PG     → facts land in the volatile store only
  * no Qdrant → semantic recall off, keyword/as-of still work

The extraction runner is deliberately NOT wired here: it needs
application-level providers (recent-turns, result sinks). Build
``AsyncExtractorRunner`` yourself and attach it via
``mm.set_extractor_runner(...)``.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from .embedder import build_embedder_from_env
from .manager import MemoryManager

logger = logging.getLogger(__name__)


def build_memory_manager(
    *,
    database_url: Optional[str] = None,
    redis_url:    Optional[str] = None,
    qdrant_url:   Optional[str] = None,
    embed_fn:     Optional[Callable[[str], list[float]]] = None,
) -> MemoryManager:
    """
    Wire session/long-term/semantic layers from explicit URLs or env
    (``DATABASE_URL`` / ``REDIS_URL`` / ``QDRANT_URL``).

    Never raises for a missing substrate — the manager degrades to what
    exists. Infrastructure import errors (library not installed) and
    connection failures are logged and the layer is skipped.
    """
    from .long_term import LongTermMemory
    from .semantic import SemanticMemory
    from .session import SessionMemory

    session = None
    rurl = (redis_url or os.getenv("REDIS_URL") or "").strip()
    if rurl:
        try:
            import redis as _redis

            client = _redis.Redis.from_url(rurl, decode_responses=True)
            client.ping()
            session = SessionMemory(client)
        except Exception as exc:  # noqa: BLE001 - degrade by design
            logger.warning("SessionMemory disabled (%s)", exc)

    long_term = None
    dsn = (database_url or os.getenv("DATABASE_URL") or "").strip()
    if dsn:
        try:
            import psycopg2.pool

            pool = psycopg2.pool.ThreadedConnectionPool(1, 10, dsn=dsn)
            long_term = LongTermMemory(conn_factory=pool.getconn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LongTermMemory disabled (%s)", exc)

    semantic = None
    qurl = (qdrant_url or os.getenv("QDRANT_URL") or "").strip()
    if qurl:
        try:
            from qdrant_client import QdrantClient

            emb = embed_fn or build_embedder_from_env()
            semantic = SemanticMemory(QdrantClient(url=qurl), emb)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SemanticMemory disabled (%s)", exc)

    return MemoryManager(
        session=session,
        long_term=long_term,
        semantic=semantic,
        extractor_runner=None,
    )
