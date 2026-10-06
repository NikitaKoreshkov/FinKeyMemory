# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Dream worker — background memory garden.

Periodically:
  1. Load active user facts from PG (queued tenants or recent writers)
  2. Score importance (Ebbinghaus + source + observed access frequency)
  3. Semantic-dedupe near-duplicates
  4. Soft-delete pruned / merged keys via FactStore
  5. Mine recurring patterns and persist each as a durable ``pattern:<hash>`` fact
  6. L2: place still-unassigned facts into scene blocks, then L3: check the
     persona trigger ladder and regenerate the portrait when it fires
  7. Optional: refresh identity pin + volatile

Env:
  FINKEY_MEMORY_DREAM_ENABLED       default 1
  FINKEY_MEMORY_DREAM_INTERVAL_SEC  default 900
  FINKEY_MEMORY_DREAM_PRUNE         default 0.15
  FINKEY_MEMORY_DREAM_DEDUP         default 0.88

L2/L3 идут именно здесь, а не в ``AsyncExtractorRunner``: тот runner умеет ровно
одну задачу (экстракция) и не расширяется новыми job kind'ами, а LLM-вызовы
консолидации дороже секунды — в hot-path они стоили бы TTFT.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Deque, Optional, Tuple

from finkey_memory.smart_consolidation import (
    SemanticDeduplicator,
    mine_patterns_heuristic,
    run_memory_garden,
    score_importance,
)

logger = logging.getLogger("finkey.memory.dream")

if TYPE_CHECKING:
    from finkey_memory.fact_store import FactStore
    from finkey_memory.semantic import SemanticMemory

#: Минный паттерн живёт отдельным durable-фактом под своим ключом; он же —
#: маркер «это моё отражение», которое нельзя снова подкармливать в майнер.
_PATTERN_KEY_PREFIX = "pattern:"
_PATTERN_PRIORITY   = 0.45
_PATTERN_MAX_VALUE  = 480


def _access_of(row: dict) -> int:
    """Наблюдённая частота; в каноническом PG колонки ``access_count`` нет → 0."""
    try:
        return max(0, int(float(row.get("access_count") or 0)))
    except (TypeError, ValueError):
        return 0


def _pattern_key(pattern: str) -> str:
    digest = hashlib.sha1((pattern or "").strip().lower().encode("utf-8")).hexdigest()
    return f"{_PATTERN_KEY_PREFIX}{digest[:12]}"


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


@dataclass
class DreamStats:
    ticks: int = 0
    tenants: int = 0
    pruned: int = 0
    merged: int = 0
    insights: int = 0
    last_run_ts: float = 0.0
    scenes_consolidated: int = 0
    personas_generated: int = 0


@dataclass
class DreamWorker:
    fact_store: "FactStore"
    embed_fn: Optional[Callable[[str], list[float]]] = None
    semantic: Optional["SemanticMemory"] = None
    merge_fn: Optional[Callable] = None
    #: L2/L3-сервис (:class:`finkey_memory.scene_persona.ScenePersonaService`).
    #: None = слои выключены целиком, dream работает по-старому.
    scene_persona: Optional[Any] = None
    interval_sec: int = field(default_factory=lambda: _env_int("FINKEY_MEMORY_DREAM_INTERVAL_SEC", 900))
    prune_threshold: float = field(default_factory=lambda: _env_float("FINKEY_MEMORY_DREAM_PRUNE", 0.15))
    dedup_threshold: float = field(default_factory=lambda: _env_float("FINKEY_MEMORY_DREAM_DEDUP", 0.88))
    enabled: bool = field(default_factory=lambda: _env_bool("FINKEY_MEMORY_DREAM_ENABLED", True))
    initial_delay: int = field(default_factory=lambda: _env_int("FINKEY_MEMORY_DREAM_INITIAL_SEC", 90))
    stats: DreamStats = field(default_factory=DreamStats)

    _queue: Deque[Tuple[str, str]] = field(default_factory=deque)
    _seen: set[Tuple[str, str]] = field(default_factory=set)
    _thread: Optional[threading.Thread] = None
    _stop_evt: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def enqueue(self, company_id: str, user_id: str) -> None:
        key = (company_id, user_id)
        with self._lock:
            if key in self._seen:
                return
            self._seen.add(key)
            self._queue.append(key)
            if len(self._queue) > 500:
                old = self._queue.popleft()
                self._seen.discard(old)

    def start(self) -> bool:
        with self._lock:
            if not self.enabled:
                logger.info("DreamWorker disabled by env")
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._loop, name="finkey-dream-worker", daemon=True)
            self._thread.start()
            logger.info("DreamWorker started interval=%ss", self.interval_sec)
            return True

    def stop(self, *, wait: bool = False, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        t = self._thread
        if wait and t is not None:
            t.join(timeout=timeout)

    def run_once(self, *, max_tenants: int = 4) -> int:
        processed = 0
        for _ in range(max(1, max_tenants)):
            with self._lock:
                if not self._queue:
                    break
                company_id, user_id = self._queue.popleft()
                self._seen.discard((company_id, user_id))
            try:
                self.dream_user(company_id, user_id)
                processed += 1
            except Exception as exc:
                logger.warning("dream_user failed %s/%s: %s", company_id, user_id, exc)
            # Отдельный try: падение сада не должно прятать L2/L3, и наоборот.
            try:
                self.run_scene_persona_pass(company_id, user_id)
            except Exception as exc:
                logger.warning("scene/persona pass failed %s/%s: %s", company_id, user_id, exc)
        self.stats.ticks += 1
        self.stats.last_run_ts = time.time()
        return processed

    def run_scene_persona_pass(self, company_id: str, user_id: str) -> dict:
        """
        L2 → L3 в том же тике: сначала консолидация сцен, потом проверка
        триггеров портрета. Порядок важен — портрет должен видеть свежие сцены,
        иначе P2/P3 опаздывают ровно на один прогон.

        Без сервиса / без PG сервис сам деградирует ('disabled_or_no_pg'), и ни
        один LLM-вызов не уходит наружу из этого метода.
        """
        svc = self.scene_persona
        if svc is None or not getattr(svc, "usable", False):
            return {}
        out: dict[str, Any] = {}
        try:
            scenes = svc.consolidate_scenes(company_id, user_id)
            out["scenes"] = scenes.to_dict()
            if scenes.changed():
                self.stats.scenes_consolidated += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("consolidate_scenes failed %s/%s: %s", company_id, user_id, exc)
            out["scenes"] = {"skipped_reason": f"{type(exc).__name__}"}
        try:
            persona = svc.maybe_regenerate_persona(company_id, user_id)
            out["persona"] = persona.to_dict()
            if persona.generated:
                self.stats.personas_generated += 1
                logger.info(
                    "persona regenerated %s/%s v%d (%d chars, reason=%s)",
                    company_id, user_id, persona.version, persona.chars, persona.reason,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("persona trigger failed %s/%s: %s", company_id, user_id, exc)
            out["persona"] = {"skipped_reason": f"{type(exc).__name__}"}
        return out

    def _loop(self) -> None:
        if self.initial_delay > 0:
            self._stop_evt.wait(self.initial_delay)
        while not self._stop_evt.is_set():
            try:
                self.run_once()
            except Exception as exc:
                logger.warning("DreamWorker tick failed: %s", exc)
            self._stop_evt.wait(max(30, self.interval_sec))

    def dream_user(self, company_id: str, user_id: str) -> dict:
        rows = self.fact_store.list_user_facts(company_id=company_id, user_id=user_id, limit=200)
        facts = [(str(r.get("key_normalized") or ""), str(r.get("value") or "")) for r in rows]
        # Отражения (наши же pattern:*) не должны попадать в сырьё следующего
        # прогона — иначе «Recurring topic» размножает сам себя.
        facts = [(k, v) for k, v in facts if k and v and not k.startswith(_PATTERN_KEY_PREFIX)]
        if len(facts) < 2:
            return {"kept": len(facts), "pruned": 0, "merged": 0}

        now = datetime.now(timezone.utc)
        # Частота — фактические наблюдения, нормированные по максимуму внутри
        # тенанта, а не заглушка max_access=1. Без колонки access_count все нули и
        # max=1 → поведение совпадает со старым (frequency = 0).
        access_by_key: dict[str, int] = {}
        for r in rows:
            k = str(r.get("key_normalized") or "")
            if k:
                access_by_key[k] = _access_of(r)
        max_access = max([1] + list(access_by_key.values()))
        scores = {}
        for r in rows:
            k = str(r.get("key_normalized") or "")
            if not k:
                continue
            touched = r.get("touched_at") or r.get("updated_at") or now
            if isinstance(touched, str):
                try:
                    touched = datetime.fromisoformat(touched.replace("Z", "+00:00"))
                except Exception:
                    touched = now
            scores[k] = score_importance(
                value=str(r.get("value") or ""),
                source=r.get("source"),
                touched_at=touched if isinstance(touched, datetime) else now,
                access_count=access_by_key.get(k, 0),
                max_access_count=max_access,
                sibling_count=0,
                now=now,
            )

        groups = []
        if self.embed_fn is not None and len(facts) >= 2:
            deduper = SemanticDeduplicator(
                self.embed_fn,
                merge_fn=self.merge_fn,
                threshold=self.dedup_threshold,
            )
            groups = deduper.find_duplicates(facts)
            # Apply merges: keep canonical, soft-delete others after writing merged value
            fact_map = {k: v for k, v in facts}
            for g in groups:
                merged_val = deduper.merge_group(g, fact_map)
                if merged_val and g.canonical_key in fact_map:
                    try:
                        from finkey_memory.extractor import ExtractedFact, FactCategory

                        cat_raw = "preference"
                        for r in rows:
                            if r.get("key_normalized") == g.canonical_key:
                                cat_raw = str(r.get("category") or "preference")
                                break
                        self.fact_store.upsert_extracted_facts(
                            company_id=company_id,
                            user_id=user_id,
                            conversation_id=None,
                            facts=[
                                ExtractedFact(
                                    key_normalized=g.canonical_key,
                                    value=merged_val,
                                    category=FactCategory.parse(cat_raw),
                                    confidence=0.85,
                                    source="verified",
                                    priority=0.7,
                                )
                            ],
                        )
                    except Exception as exc:
                        logger.debug("merge upsert skipped: %s", exc)
                    for m in g.members[1:]:
                        try:
                            self.fact_store.soft_delete_fact(
                                company_id=company_id, user_id=user_id, key_normalized=m
                            )
                        except Exception:
                            pass
                self.stats.merged += 1

        cleaned, garden = run_memory_garden(
            facts,
            importance_scores=scores,
            duplicate_groups=groups,
            prune_threshold=self.prune_threshold,
        )
        # Soft-delete pruned keys that weren't already merged away
        merged_away = set()
        for g in groups:
            merged_away.update(g.members[1:])
        for k in garden.pruned_keys:
            if k in merged_away:
                continue
            # Only prune truly low-importance, not just merged
            if k in scores and scores[k].total < self.prune_threshold:
                try:
                    self.fact_store.soft_delete_fact(
                        company_id=company_id, user_id=user_id, key_normalized=k
                    )
                except Exception:
                    pass

        insights = mine_patterns_heuristic(cleaned, min_occurrence=3)
        self.stats.tenants += 1
        self.stats.pruned += garden.pruned
        self.stats.insights += len(insights)
        persisted = self._persist_pattern_insights(company_id, user_id, insights)
        return {
            "kept": garden.kept,
            "pruned": garden.pruned,
            "merged": garden.merged_groups,
            "insights": [i.to_dict() for i in insights[:5]],
            "insights_persisted": persisted,
        }

    def _persist_pattern_insights(self, company_id: str, user_id: str, insights: list) -> int:
        """
        Минные паттерны раньше только считались в ``stats.insights`` и пропадали
        вместе с прогоном. Каждый становится durable-фактом ``pattern:<sha1>``:
        ключ уникален на паттерн, поэтому повторный прогон обновляет строку на
        месте, а не плодит дубли.

        Value держим стабильным (только текст паттерна): счётчик-подпорка живёт в
        ``evidence_snippet``/``confidence``. Иначе меняющееся число каждый раз
        выглядело бы конфликтом и пловило ``__asof__``-архивы.
        """
        if not insights:
            return 0
        try:
            from finkey_memory.extractor import (
                ExtractedFact,
                FactCategory,
                _is_transient_channel_glitch,  # noqa: SLF001
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("pattern persist import failed: %s", exc)
            return 0

        # Таксономия = зеркало CHECK на user_facts.category: parse() сам решит,
        # есть ли уже 'reflection', и иначе вернёт безопасный 'other'.
        category = FactCategory.parse("reflection")

        payload: list = []
        seen: set[str] = set()
        for ins in insights:
            pattern = str(getattr(ins, "pattern", "") or "").strip()
            if not pattern:
                continue
            key = _pattern_key(pattern)
            if key in seen:
                continue
            evidence_keys = [str(k) for k in (getattr(ins, "evidence_keys", None) or []) if k]
            if _is_transient_channel_glitch(key, pattern):
                continue
            seen.add(key)
            try:
                support = float(getattr(ins, "confidence", 0.5) or 0.5)
            except (TypeError, ValueError):
                support = 0.5
            payload.append(ExtractedFact(
                key_normalized   = key,
                value            = pattern[:_PATTERN_MAX_VALUE],
                category         = category,
                # Confidence — из поддержки (сколько фактов её дали), с потолком:
                # эвристика не должна перекрикивать слова самого человека.
                confidence       = round(min(0.85, max(0.4, support)), 3),
                evidence_snippet = (
                    f"reflection support={len(evidence_keys) or 1}: "
                    + "; ".join(evidence_keys[:5])
                )[:300],
                priority         = _PATTERN_PRIORITY,
                source           = "inferred",
            ))
        if not payload:
            return 0
        try:
            report = self.fact_store.upsert_extracted_facts(
                company_id      = company_id,
                user_id         = user_id,
                conversation_id = None,
                facts           = payload,
            )
            return int(getattr(report, "pg_ok", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("pattern insights upsert skipped: %s", exc)
            return 0
