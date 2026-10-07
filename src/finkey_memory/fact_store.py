# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey FactStore — единая точка записи durable-фактов о пользователе.

Зачем не писать напрямую через ``LongTermMemory``:
  * Источников несколько (MemoryExtractor, ручные правки пользователя из приложения, заливка
    «факт-импорт» из CRM), и им нужен один контракт.
  * Параллельные слои: PG (авторитетный), Qdrant (семантический recall),
    VolatileKnowledgeStore (горячий кэш для prompts/load_context), audit_log
    (трэк изменений) — должны обновляться согласованно.
  * Soft-delete и privacy-операции (право быть забытым) централизуем здесь.

Контракт:
  * ``upsert_extracted_facts(...)`` — batch-запись `ExtractedFact[]` из экстрактора;
    идемпотентна, не падает, если PG/Qdrant временно недоступны.
  * ``list_user_facts(...)`` — чтение с фильтрами для UI/prompt-блока.
  * ``soft_delete_fact(...)`` / ``soft_delete_all(...)`` — privacy.

PostgreSQL обязателен для ``FactStore`` (конструктор бросает ``ValueError``, если
``long_term is None``). Qdrant, volatile и Redis — опциональны (graceful degrade).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from finkey_memory.extractor import ExtractedFact, FactCategory
from finkey_memory.identity_pinner import clear_identity_pin, refresh_identity_pin
from finkey_memory.long_term import LongTermMemory
from finkey_memory.metrics import (
    FACT_DELETES,
    FACT_PURGES,
    FACT_UPSERT_PG_FAIL,
    FACT_UPSERT_PG_OK,
    FACT_UPSERT_QDRANT_FAIL,
    FACT_UPSERT_QDRANT_OK,
    FACT_UPSERT_TOTAL,
    log_event,
)
from finkey_memory.semantic import SemanticMemory
from finkey_memory.temporal import (
    TemporalStamp,
    archive_key,
    as_datetime,
    stamp_evidence,
    values_conflict,
    _now as _temporal_now,
)
from finkey_memory.volatile_store import VolatileKnowledgeStore
from finkey_memory.pii import mask_sensitive

logger = logging.getLogger("finkey.memory.fact_store")

_MAX_AUDIT_FACTS = 20

#: Потолок ключей в одном batched UPDATE — держим запрос коротким.
_MAX_TOUCH_KEYS = 64

_touch_unavailable_warned = False


def _warn_touch_unavailable_once(reason: str) -> None:
    """Шум на каждый ход не нужен: предупреждаем ровно один раз на процесс."""
    global _touch_unavailable_warned
    if _touch_unavailable_warned:
        return
    _touch_unavailable_warned = True
    logger.warning(
        "FactStore.touch_facts degraded to no-op (%s) — user_facts.access_count "
        "is not in the schema; frequency revival stays off until the column ships.",
        reason,
    )


def _pii_mask_enabled() -> bool:
    import os
    raw = (os.getenv("FINKEY_MEMORY_PII_MASK") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on", "")


def _float_of(value: Any, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass
    try:
        return float(default)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _epoch_of(value: Any) -> Optional[float]:
    """PG/Redis timestamp → unix float; None, если значение совсем отсутствует."""
    if value is None:
        return None
    try:
        return as_datetime(value).timestamp()
    except Exception:  # noqa: BLE001
        return None


def _temporal_enabled() -> bool:
    import os
    raw = (os.getenv("FINKEY_MEMORY_TEMPORAL") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on", "")


@dataclass
class UpsertReport:
    """Итог одного вызова ``upsert_extracted_facts`` — удобно для метрик и логов."""
    accepted: int = 0
    pg_ok:    int = 0
    pg_fail:  int = 0
    qdrant_ok:   int = 0
    qdrant_fail: int = 0
    volatile_ok: int = 0
    audit_ok:    bool = False
    written_keys: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class DecayReport:
    """Итог одного прогона ``run_decay`` — для DecayScheduler.stats и метрик."""
    pg_decayed:    int = 0
    qdrant_ok:     int = 0
    qdrant_fail:   int = 0
    volatile_ok:   int = 0
    audit_records: int = 0
    errors:        list[str] = field(default_factory=list)


class FactStore:
    """
    Унифицированный writer/reader для ``user_facts``.

    ``volatile`` обычно берётся из ``MemoryManager._volatile`` (через ``MemoryManager``
    или явный передачей). ``semantic`` опционален.
    """

    def __init__(
        self,
        *,
        long_term:     LongTermMemory,
        volatile:      Optional[VolatileKnowledgeStore] = None,
        semantic:      Optional[SemanticMemory]        = None,
        redis_client:  Any                             = None,
    ) -> None:
        if long_term is None:
            raise ValueError("FactStore requires LongTermMemory (PostgreSQL).")
        self._lt   = long_term
        self._vol  = volatile
        self._sem  = semantic
        self._redis = redis_client

    def _refresh_entity_index(self, company_id: str, user_id: str) -> None:
        from finkey_memory.entity_memory import (
            persist_entity_index_redis,
            rebuild_entity_index,
        )

        rows = self.list_user_facts(company_id=company_id, user_id=user_id, limit=200)
        pairs = [
            (str(r.get("key_normalized") or ""), str(r.get("value") or ""))
            for r in rows
            if r.get("key_normalized") and "__asof__" not in str(r.get("key_normalized") or "")
        ]
        idx = rebuild_entity_index(company_id, user_id, pairs)
        persist_entity_index_redis(self._redis, company_id, user_id, idx)


    def upsert_extracted_facts(
        self,
        *,
        company_id:      str,
        user_id:         str,
        conversation_id: Optional[str],
        facts:           list[ExtractedFact],
    ) -> UpsertReport:
        """
        Batch upsert. PG — основной writer; Qdrant и volatile — best-effort.

        В audit_log пишется ОДНА запись на весь batch с краткой выжимкой по фактам,
        чтобы не флудить лог при больших экстракциях.
        """
        report = UpsertReport()
        if not facts:
            return report

        # PII mask + temporal supersession prep
        prepared: list[ExtractedFact] = []
        for fact in facts:
            value = fact.value
            evidence = fact.evidence_snippet or ""
            if _pii_mask_enabled():
                value, _st = mask_sensitive(value, aggressive=True)
                if evidence:
                    evidence, _ = mask_sensitive(evidence, aggressive=True)
            if _temporal_enabled():
                try:
                    existing = self._lt.get_user_fact(company_id, user_id, fact.key_normalized)
                except Exception:
                    existing = None
                if existing and values_conflict(str(existing.get("value") or ""), value):
                    # Archive previous value under temporal key, close validity window
                    try:
                        now = _temporal_now()
                        old_ev = stamp_evidence(
                            str(existing.get("evidence_snippet") or ""),
                            TemporalStamp(
                                valid_from=as_datetime(existing.get("created_at"), fallback=now),
                                valid_to=now,
                                supersedes=None,
                            ),
                        )

                        arch = ExtractedFact(
                            key_normalized=archive_key(fact.key_normalized, now),
                            value=str(existing.get("value") or ""),
                            category=FactCategory.parse(str(existing.get("category") or "other")),
                            confidence=float(existing.get("confidence") or 0.5),
                            evidence_snippet=old_ev,
                            priority=float(existing.get("priority") or 0.4),
                            source=str(existing.get("source") or "inferred"),
                            source_conv_id=existing.get("source_conv_id"),
                        )
                        prepared.append(arch)
                        evidence = stamp_evidence(
                            evidence,
                            TemporalStamp(valid_from=now, valid_to=None, supersedes=arch.key_normalized),
                        )
                    except Exception as exc:
                        logger.debug("temporal archive skipped: %s", exc)
            if value != fact.value or evidence != (fact.evidence_snippet or ""):
                from dataclasses import replace as _replace

                prepared.append(
                    _replace(fact, value=value, evidence_snippet=evidence)
                )
            else:
                prepared.append(fact)

        facts = prepared
        report.accepted = len(facts)

        upserted_rows: list[dict] = []
        fact_by_key = {f.key_normalized: f for f in facts}
        for fact in facts:
            try:
                row = self._lt.upsert_user_fact(
                    company_id           = company_id,
                    user_id_or_external  = user_id,
                    key_normalized       = fact.key_normalized,
                    value                = fact.value,
                    category             = (fact.category.value
                                            if isinstance(fact.category, FactCategory)
                                            else str(fact.category)),
                    confidence           = float(fact.confidence),
                    source               = fact.source or "inferred",
                    priority             = float(fact.priority),
                    evidence_snippet     = fact.evidence_snippet,
                    source_conv_id       = fact.source_conv_id or conversation_id,
                    source_message_id    = fact.source_message_id,
                    expires_at           = fact.expires_at,
                )
                if row:
                    report.pg_ok += 1
                    report.written_keys.append(row.get("key_normalized") or fact.key_normalized)
                    upserted_rows.append(row)
                else:
                    report.pg_fail += 1
                    report.errors.append(f"pg_upsert_returned_none:{fact.key_normalized}")
            except Exception as exc:  # noqa: BLE001
                report.pg_fail += 1
                report.errors.append(f"pg_upsert_exc:{fact.key_normalized}:{type(exc).__name__}")
                logger.warning(
                    "FactStore PG upsert failed company=%s user=%s key=%s: %s",
                    company_id, user_id, fact.key_normalized, exc,
                )

        if self._vol is not None and upserted_rows:
            for row in upserted_rows:
                fact = fact_by_key.get(row["key_normalized"])
                if not fact:
                    continue
                try:
                    self._vol.upsert_fact(
                        company_id=company_id,
                        user_id=user_id,
                        raw_key=row["key_normalized"],
                        value=row["value"],
                        confidence=float(row.get("confidence") or fact.confidence),
                        source=row.get("source") or fact.source or "inferred",
                        priority=float(row.get("priority") or fact.priority),
                        expires_at=row.get("expires_at") or fact.expires_at,
                        confirmed_at=row.get("confirmed_at"),
                    )
                    report.volatile_ok += 1
                except Exception as exc:  # noqa: BLE001
                    report.errors.append(f"volatile_exc:{fact.key_normalized}:{type(exc).__name__}")
                    logger.debug("FactStore volatile upsert failed: %s", exc)

        if self._sem is not None and upserted_rows:
            for row in upserted_rows:
                fact = fact_by_key.get(row["key_normalized"])
                if not fact:
                    continue
                try:
                    self._sem.upsert_user_fact(
                        company_id     = company_id,
                        user_id        = self._resolve_uuid_for_payload(company_id, user_id),
                        point_id       = row["id"],
                        key_normalized = row["key_normalized"],
                        value          = row["value"],
                        category       = row.get("category")
                                         or (fact.category.value
                                             if isinstance(fact.category, FactCategory)
                                             else str(fact.category)),
                        confidence     = float(row.get("confidence") or fact.confidence),
                        source         = row.get("source") or fact.source,
                        source_conv_id = row.get("source_conv_id") or conversation_id,
                        evidence       = row.get("evidence_snippet") or fact.evidence_snippet,
                        # Read-time rerank живёт в payload — иначе recall'у неоткуда
                        # брать свежесть, кроме как лишним round-trip в PG.
                        updated_at     = _epoch_of(row.get("updated_at") or row.get("touched_at")),
                        priority       = _float_of(row.get("priority"), fact.priority),
                        access_count   = _float_of(row.get("access_count"), 0.0),
                    )
                    report.qdrant_ok += 1
                except Exception as exc:  # noqa: BLE001
                    report.qdrant_fail += 1
                    report.errors.append(f"qdrant_exc:{fact.key_normalized}:{type(exc).__name__}")
                    logger.debug("FactStore Qdrant upsert failed: %s", exc)

        try:
            self._lt.audit(
                company_id  = company_id,
                user_id     = user_id,
                action      = "memory.upsert",
                entity_type = "user_fact",
                entity_id   = conversation_id,
                metadata    = {
                    "facts": [
                        {
                            "key":        f.key_normalized,
                            "value":      f.value[:240],
                            "category":   (f.category.value
                                            if isinstance(f.category, FactCategory)
                                            else str(f.category)),
                            "confidence": round(float(f.confidence), 3),
                            "source":     f.source,
                        }
                        for f in facts[:_MAX_AUDIT_FACTS]
                    ],
                    "pg_ok":      report.pg_ok,
                    "pg_fail":    report.pg_fail,
                    "qdrant_ok":  report.qdrant_ok,
                    "qdrant_fail": report.qdrant_fail,
                },
            )
            report.audit_ok = True
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"audit_exc:{type(exc).__name__}")
            logger.warning("FactStore audit write failed: %s", exc)

        if self._redis is not None and report.pg_ok > 0:
            try:
                refresh_identity_pin(
                    self._redis,
                    fact_store=self,
                    company_id=company_id,
                    user_id=user_id,
                )
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"identity_pin_exc:{type(exc).__name__}")
                logger.debug("identity_pin refresh failed: %s", exc)

        if report.pg_ok > 0:
            try:
                self._refresh_entity_index(company_id, user_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("entity index refresh skipped: %s", exc)

        FACT_UPSERT_TOTAL.inc(report.accepted, company_id=company_id)
        if report.pg_ok:
            FACT_UPSERT_PG_OK.inc(report.pg_ok, company_id=company_id)
        if report.pg_fail:
            FACT_UPSERT_PG_FAIL.inc(report.pg_fail, company_id=company_id)
        if report.qdrant_ok:
            FACT_UPSERT_QDRANT_OK.inc(report.qdrant_ok, company_id=company_id)
        if report.qdrant_fail:
            FACT_UPSERT_QDRANT_FAIL.inc(report.qdrant_fail, company_id=company_id)

        log_event(
            logger, logging.INFO, "memory.fact_store.upsert",
            company_id=company_id, user_id=user_id, conv_id=conversation_id,
            accepted=report.accepted, pg_ok=report.pg_ok, pg_fail=report.pg_fail,
            qdrant_ok=report.qdrant_ok, qdrant_fail=report.qdrant_fail,
            volatile_ok=report.volatile_ok,
        )
        return report


    def list_user_facts(
        self,
        *,
        company_id: str,
        user_id:    str,
        categories: Optional[list[str]] = None,
        min_confidence: Optional[float] = None,
        limit:      int = 100,
    ) -> list[dict]:
        return self._lt.list_user_facts(
            company_id, user_id,
            categories=categories,
            min_confidence=min_confidence,
            limit=limit,
        )

    def list_user_fact_keys(self, company_id: str, user_id: str) -> set[str]:
        return self._lt.list_user_fact_keys(company_id, user_id)

    def get_user_fact(self, company_id: str, user_id: str, key_normalized: str) -> Optional[dict]:
        return self._lt.get_user_fact(company_id, user_id, key_normalized)


    def touch_facts(self, company_id: str, user_id: str, keys: list[str]) -> int:
        """
        Один batched инкремент ``access_count`` за ход (частотное воскрешение).

        Вызывается ПОСЛЕ сборки контекста, вне пула параллельной загрузки: это
        единственный write в hot-path, и он должен быть одним SQL-запросом.

        Канонический PG (миграция 003) колонку ``access_count`` на ``user_facts``
        не имеет, а миграции тут запрещены — поэтому при отсутствии метода/колонки
        это no-op с ОДНИМ warning на процесс, а не исключение на каждый ход.
        """
        clean: list[str] = []
        seen: set[str] = set()
        for raw in keys or []:
            key = str(raw or "").strip()
            if not key or key in seen or "__asof__" in key:
                continue
            seen.add(key)
            clean.append(key)
            if len(clean) >= _MAX_TOUCH_KEYS:
                break
        if not clean or self._lt is None:
            return 0

        bump = getattr(self._lt, "touch_user_facts", None)
        if not callable(bump):
            _warn_touch_unavailable_once("LongTermMemory has no touch_user_facts()")
            return 0
        try:
            touched = bump(
                company_id       = company_id,
                user_id_or_external = user_id,
                keys_normalized  = clean,
            )
        except Exception as exc:  # noqa: BLE001 — доступность памяти важнее счётчика
            _warn_touch_unavailable_once(f"{type(exc).__name__}: {exc}")
            return 0
        try:
            return int(touched) if touched is not None else len(clean)
        except (TypeError, ValueError):
            return len(clean)


    def soft_delete_fact(
        self,
        *,
        company_id: str,
        user_id: str,
        key_normalized: str,
        reason: Optional[str] = None,
        by_user: bool = False,
    ) -> bool:
        """Soft-delete одного факта + audit."""
        existing = self._lt.get_user_fact(company_id, user_id, key_normalized)
        ok = self._lt.soft_delete_user_fact(company_id, user_id, key_normalized)
        try:
            self._lt.audit(
                company_id  = company_id,
                user_id     = user_id,
                action      = "memory.delete",
                entity_type = "user_fact",
                entity_id   = (existing or {}).get("id"),
                metadata    = {
                    "key":     key_normalized,
                    "by_user": bool(by_user),
                    "reason":  reason,
                    "ok":      bool(ok),
                },
            )
        except Exception as exc:
            logger.warning("FactStore audit (delete) failed: %s", exc)
        if ok and existing and self._sem is not None:
            try:
                self._sem.delete_user_fact(point_id=existing["id"])
            except Exception as exc:
                logger.debug("Qdrant delete_user_fact failed: %s", exc)
        if ok and self._vol is not None:
            try:
                self._vol.delete_fact(company_id, user_id, key_normalized)
            except Exception as exc:
                logger.debug("volatile delete_fact failed: %s", exc)
        if ok and self._redis is not None:
            try:
                refresh_identity_pin(
                    self._redis, fact_store=self, company_id=company_id, user_id=user_id,
                )
            except Exception as exc:
                logger.debug("identity_pin refresh after delete failed: %s", exc)
        if ok:
            FACT_DELETES.inc(company_id=company_id)
            log_event(
                logger, logging.INFO, "memory.fact_store.soft_delete",
                company_id=company_id, user_id=user_id, key=key_normalized,
                by_user=by_user, reason=reason,
            )
        return ok

    def run_decay(
        self,
        *,
        company_id: Optional[str] = None,
        max_rows:   int           = 500,
        grace_sec:  int           = 0,
    ) -> "DecayReport":
        """
        Soft-delete всех фактов с истёкшим ``expires_at``, синхронизируя
        Qdrant / volatile / identity_pin / audit_log.

        Логика
        ------
        1. PostgreSQL: ``LongTermMemory.decay_expired_user_facts`` — атомарно
           помечает строки и возвращает список ``{id, company_id, user_id,
           key_normalized, category, expires_at}``.
        2. Qdrant: удаляем point по ``id`` (point_id у нас == PG-id).
        3. Volatile: вычищаем тот же ключ из in-process кеша.
        4. Redis identity_pin: если в decay попал identity-факт пользователя —
           ререфрешим pin (или чистим, если фактов не осталось).
        5. audit_log: одна запись «memory.decay» на тенант с агрегированной
           статистикой (чтобы не плодить тысячи строк).

        Возвращает ``DecayReport`` со счётчиками и затронутыми пользователями.
        """
        from collections import defaultdict

        report = DecayReport()
        try:
            rows = self._lt.decay_expired_user_facts(
                company_id = company_id,
                max_rows   = max_rows,
                grace_sec  = grace_sec,
            )
        except Exception as exc:
            logger.warning("run_decay: PG step failed: %s", exc)
            report.errors.append(f"pg:{type(exc).__name__}")
            return report

        report.pg_decayed = len(rows)
        if not rows:
            return report

        identity_categories = {FactCategory.IDENTITY.value, FactCategory.AFFILIATION.value}

        by_tenant: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            by_tenant[(r["company_id"], r["user_id"])].append(r)

        for (cid, uid), batch in by_tenant.items():
            for r in batch:
                if self._sem is not None:
                    try:
                        self._sem.delete_user_fact(point_id=r["id"])
                        report.qdrant_ok += 1
                    except Exception as exc:
                        logger.debug("Qdrant delete (decay) failed: %s", exc)
                        report.qdrant_fail += 1
                if self._vol is not None:
                    try:
                        self._vol.delete_fact(cid, uid, r["key_normalized"])
                        report.volatile_ok += 1
                    except Exception as exc:
                        logger.debug("volatile delete (decay) failed: %s", exc)

            if self._redis is not None and any((b.get("category") in identity_categories) for b in batch):
                try:
                    refresh_identity_pin(
                        self._redis, fact_store=self, company_id=cid, user_id=uid,
                    )
                except Exception as exc:
                    logger.debug("identity_pin refresh after decay failed: %s", exc)

            try:
                self._lt.audit(
                    company_id  = cid,
                    user_id     = uid,
                    action      = "memory.decay",
                    entity_type = "user_fact",
                    metadata    = {
                        "count": len(batch),
                        "keys":  [b["key_normalized"] for b in batch[:_MAX_AUDIT_FACTS]],
                    },
                )
                report.audit_records += 1
            except Exception as exc:
                logger.debug("audit (decay) failed: %s", exc)

        FACT_DELETES.inc(report.pg_decayed, company_id=(company_id or ""))
        log_event(
            logger, logging.INFO, "memory.fact_store.decay",
            company_id=company_id or "",
            pg_decayed=report.pg_decayed,
            qdrant_ok=report.qdrant_ok, qdrant_fail=report.qdrant_fail,
            volatile_ok=report.volatile_ok,
            audit_records=report.audit_records,
        )
        return report

    def soft_delete_all(
        self,
        *,
        company_id: str,
        user_id: str,
        reason: Optional[str] = None,
    ) -> int:
        """«Забудь меня»: soft-delete всех активных фактов пользователя."""
        n = self._lt.soft_delete_all_user_facts(company_id, user_id)
        if self._vol is not None and n:
            try:
                self._vol.clear_facts(company_id, user_id)
            except Exception as exc:
                logger.debug("volatile clear_facts failed: %s", exc)
        if self._sem is not None and n:
            try:
                uid = self._resolve_uuid_for_payload(company_id, user_id)
                self._sem.delete_all_user_facts_vectors(company_id, uid)
            except Exception as exc:
                logger.debug("Qdrant purge user facts failed: %s", exc)
        if self._redis is not None:
            clear_identity_pin(self._redis, company_id=company_id, user_id=user_id)
        try:
            self._lt.audit(
                company_id  = company_id,
                user_id     = user_id,
                action      = "memory.purge_user",
                entity_type = "user_fact",
                metadata    = {"count": int(n), "reason": reason},
            )
        except Exception as exc:
            logger.warning("FactStore audit (purge) failed: %s", exc)
        FACT_PURGES.inc(company_id=company_id)
        log_event(
            logger, logging.INFO, "memory.fact_store.purge_user",
            company_id=company_id, user_id=user_id, count=int(n), reason=reason,
        )
        return n


    def _resolve_uuid_for_payload(self, company_id: str, user_id_or_external: str) -> str:
        """Возвращает UUID-строку пользователя; для Qdrant payload — стабильный ID."""
        uid = self._lt._resolve_user_id(company_id, user_id_or_external)  # noqa: SLF001
        return uid or user_id_or_external
