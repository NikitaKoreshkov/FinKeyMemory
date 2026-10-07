# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey memory decay — фоновое истечение durable-фактов с TTL.

Бизнес-смысл:
  * Категории ``state`` (7д), ``commitment`` (14д), ``goal`` (180д) при создании
    получают ``expires_at``. Без активной чистки они продолжали бы висеть
    «истёкшими, но не удалёнными» — забивая prompt-блок и индексы.
  * ``DecayScheduler`` — однопоточный фон-тик: раз в N секунд зовёт
    ``FactStore.run_decay`` пачкой до ``MAX_ROWS``. Под капотом — PG
    ``UPDATE ... WHERE expires_at < NOW() FOR UPDATE SKIP LOCKED``, поэтому
    несколько процессов могут крутить scheduler одновременно без
    дубликатов.
  * Сценарий аварии: если PG / Qdrant временно недоступны — тик ловит
    исключение, логирует и идёт спать дальше; следующий тик попробует снова.

Env (Production-defaults — включено):

    FINKEY_MEMORY_DECAY_ENABLED       1 | 0    (default 1)
    FINKEY_MEMORY_DECAY_INTERVAL_SEC  int      (default 3600 — раз в час)
    FINKEY_MEMORY_DECAY_BATCH         int      (default 500 строк за тик)
    FINKEY_MEMORY_DECAY_GRACE_SEC     int      (default 0 — без льготы)
    FINKEY_MEMORY_DECAY_INITIAL_SEC   int      (default 60 — задержка первого тика)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from finkey_memory.metrics import log_event

if TYPE_CHECKING:
    from finkey_memory.fact_store import DecayReport, FactStore

logger = logging.getLogger("finkey.memory.decay")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class DecaySchedulerStats:
    ticks:        int = 0
    decayed_total: int = 0
    qdrant_ok:    int = 0
    qdrant_fail:  int = 0
    errors:       int = 0
    last_run_ts:  float = 0.0
    last_report:  Optional["DecayReport"] = None


@dataclass
class DecayScheduler:
    """
    Фоновый поток, периодически зовущий ``FactStore.run_decay()``.

    Пример::

        sch = DecayScheduler(fact_store=fs)
        sch.start()
        sch.stop(wait=True)
    """
    fact_store:    "FactStore"
    interval_sec:  int  = field(default_factory=lambda: _env_int("FINKEY_MEMORY_DECAY_INTERVAL_SEC", 3600))
    batch_size:    int  = field(default_factory=lambda: _env_int("FINKEY_MEMORY_DECAY_BATCH", 500))
    grace_sec:     int  = field(default_factory=lambda: _env_int("FINKEY_MEMORY_DECAY_GRACE_SEC", 0))
    initial_delay: int  = field(default_factory=lambda: _env_int("FINKEY_MEMORY_DECAY_INITIAL_SEC", 60))
    enabled:       bool = field(default_factory=lambda: _env_bool("FINKEY_MEMORY_DECAY_ENABLED", True))
    stats:         DecaySchedulerStats = field(default_factory=DecaySchedulerStats)

    _thread:    Optional[threading.Thread] = None
    _stop_evt:  threading.Event = field(default_factory=threading.Event)
    _wake_evt:  threading.Event = field(default_factory=threading.Event)
    _lock:      threading.Lock  = field(default_factory=threading.Lock)


    def start(self) -> bool:
        """Запустить фон-поток (no-op если не enabled или уже запущен)."""
        with self._lock:
            if not self.enabled:
                logger.info("DecayScheduler disabled by env — not starting")
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_evt.clear()
            self._wake_evt.clear()
            self._thread = threading.Thread(
                target      = self._run_loop,
                name        = "finkey-decay-scheduler",
                daemon      = True,
            )
            self._thread.start()
            logger.info(
                "DecayScheduler started: interval=%ds batch=%d grace=%ds initial=%ds",
                self.interval_sec, self.batch_size, self.grace_sec, self.initial_delay,
            )
            return True

    def stop(self, *, wait: bool = False, timeout: float = 5.0) -> None:
        """Остановить фон-поток. ``wait=True`` — дождаться завершения текущего тика."""
        with self._lock:
            self._stop_evt.set()
            self._wake_evt.set()
            t = self._thread
        if wait and t is not None and t.is_alive():
            t.join(timeout=timeout)

    def kick(self) -> None:
        """Разбудить scheduler досрочно (полезно из тестов / админ-ручек)."""
        self._wake_evt.set()


    def tick_once(self) -> "DecayReport":
        """Один тик decay без фон-потока. Используется в тестах и админ-эндпоинтах."""
        return self._run_one_tick()


    def _run_loop(self) -> None:
        if self.initial_delay > 0:
            self._wait(self.initial_delay)
        while not self._stop_evt.is_set():
            try:
                self._run_one_tick()
            except Exception as exc:
                self.stats.errors += 1
                logger.exception("DecayScheduler tick crashed: %s", exc)
            if self._stop_evt.is_set():
                break
            self._wait(self.interval_sec)

    def _wait(self, seconds: int) -> None:
        """Прерываемый sleep: wake_evt или stop_evt разбудит раньше."""
        deadline = time.time() + max(0, seconds)
        while not self._stop_evt.is_set():
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            if self._wake_evt.wait(timeout=min(remaining, 1.0)):
                self._wake_evt.clear()
                return

    def _run_one_tick(self) -> "DecayReport":
        t0 = time.time()
        try:
            report = self.fact_store.run_decay(
                company_id = None,
                max_rows   = self.batch_size,
                grace_sec  = self.grace_sec,
            )
        except Exception as exc:
            self.stats.errors += 1
            log_event(
                logger, logging.WARNING, "memory.decay.tick_failed",
                error_type=type(exc).__name__,
            )
            raise

        self.stats.ticks         += 1
        self.stats.decayed_total += report.pg_decayed
        self.stats.qdrant_ok     += report.qdrant_ok
        self.stats.qdrant_fail   += report.qdrant_fail
        self.stats.last_run_ts    = time.time()
        self.stats.last_report    = report

        if report.pg_decayed > 0 or self.stats.ticks <= 1:
            log_event(
                logger, logging.INFO, "memory.decay.tick_done",
                tick=self.stats.ticks,
                pg_decayed=report.pg_decayed,
                qdrant_ok=report.qdrant_ok,
                qdrant_fail=report.qdrant_fail,
                elapsed_ms=int((time.time() - t0) * 1000),
            )
        return report
