# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
IdentityPinner — компактный «пин» идентичности пользователя в Redis.

Зачем:
  * ``load_context`` / MindEngine не должны каждый раз бить Postgres ради 5–10
    ключевых фактов (имя, компания, роль, текущий проект).
  * После upsert фактов (MemoryExtractor или ручная правка) мы **пересобираем**
    один текстовый блок и кладём в Redis под ``RedisKeys.identity_pin``.
  * TTL — несколько часов; любой новый факт триггерит refresh.

Это **не** замена PG — только горячий кэш для промпта. Источник правды остаётся
в ``user_facts``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from finkey_memory.schema import RedisKeys, TTL_IDENTITY_PIN

logger = logging.getLogger("finkey.memory.identity_pin")

_PIN_CATEGORIES = (
    "identity",
    "affiliation",
    "context",
    "preference",
    "goal",
)


def _ttl_sec() -> int:
    try:
        return max(60, int(os.getenv("FINKEY_IDENTITY_PIN_TTL_SEC", str(TTL_IDENTITY_PIN))))
    except ValueError:
        return TTL_IDENTITY_PIN


def refresh_identity_pin(
    redis_client: Any,
    *,
    fact_store: Any,
    company_id: str,
    user_id:    str,
) -> None:
    """Перечитать топ-факты из PG и обновить Redis-ключ (или удалить, если пусто)."""
    if redis_client is None:
        return
    key = RedisKeys.identity_pin(company_id, user_id)
    try:
        facts = fact_store.list_user_facts(
            company_id=company_id,
            user_id=user_id,
            categories=list(_PIN_CATEGORIES),
            limit=32,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("identity_pin refresh: list_user_facts failed: %s", exc)
        return

    if not facts:
        try:
            redis_client.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.debug("identity_pin delete empty failed: %s", exc)
        return

    lines: list[str] = []
    for f in facts:
        cat = f.get("category") or "other"
        kn  = f.get("key_normalized") or ""
        val = (f.get("value") or "").strip()
        if not val:
            continue
        lines.append(f"- [{cat}] {kn}: {val}")
    if not lines:
        try:
            redis_client.delete(key)
        except Exception:
            pass
        return

    body = (
        "Identity and durable user context (cached; refreshed automatically):\n"
        + "\n".join(lines[:24])
    )
    try:
        redis_client.set(key, body, ex=_ttl_sec())
    except Exception as exc:  # noqa: BLE001
        logger.warning("identity_pin SET failed: %s", exc)


def clear_identity_pin(redis_client: Optional[Any], *, company_id: str, user_id: str) -> None:
    """Сбросить пин (например, после ``memory.purge_user``)."""
    if redis_client is None:
        return
    try:
        redis_client.delete(RedisKeys.identity_pin(company_id, user_id))
    except Exception as exc:  # noqa: BLE001
        logger.debug("identity_pin clear failed: %s", exc)
