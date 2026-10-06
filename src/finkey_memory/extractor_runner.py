# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
AsyncExtractorRunner — неблокирующий запускальщик ``MemoryExtractor``-а.

Зачем отдельный объект:
  1. ``MemoryExtractor.extract()`` делает LLM-call (сотни мс — секунды). Запускать
     это синхронно в горячем пути save_turn нельзя — задержим ответ юзеру.
  2. Нужен **debounce**: запускаем не на каждом ходу, а раз в ``every_n_turns``
     (по умолчанию 3). Логика уже в ``MemoryExtractor.should_run_this_turn``.
  3. Нужен **lock**: иначе на параллельных тиках save_turn мы получим два экстракта
     для одного и того же conv → лишние LLM-калла и потенциальный конфликт мёрджа.
     Используем Redis ``SET NX EX`` (atomic). Если Redis недоступен — fallback на
     in-process per-(company,user,conv) lock.
  4. Нужен **timeout**: одну экстракцию режем по таймауту, ниже которого ставится
     ``ExtractionResult(parse_ok=False, skipped_reason="timeout")``.

Сам Runner не пишет результаты никуда — он только зовёт ``on_result`` коллбек.
Куда писать факты — решает потребитель (в Фазе 4 это будет ``FactStore.upsert_extracted_facts``).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from finkey_memory.extractor import (
    ExtractionResult,
    ExtractorConfig,
    MemoryExtractor,
    parse_extractor_response,
)
from finkey_memory.metrics import (
    RUNNER_COMPLETED,
    RUNNER_ERRORS,
    RUNNER_INFLIGHT,
    RUNNER_SCHEDULED,
    RUNNER_SKIPPED_DEBOUNCE,
    RUNNER_SKIPPED_LOCKED,
    RUNNER_TIMEOUTS,
    log_event,
)

logger = logging.getLogger("finkey.memory.extractor.runner")


RecentTurnsProvider = Callable[[str, str, str], list[dict]]

KnownKeysProvider = Callable[[str, str], set[str]]

OnResultCallback = Callable[[str, str, str, ExtractionResult], None]


def _noop_known_keys(_c: str, _u: str) -> set[str]:
    return set()


def _noop_on_result(*_args, **_kwargs) -> None:
    return




@dataclass
class _AcquiredLock:
    key:     str
    token:   str
    backend: str
    released: bool = False


class _InProcLocks:
    """Простой in-process per-key lock, fallback при отсутствии Redis."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: dict[str, str] = {}

    def acquire(self, key: str, token: str) -> bool:
        with self._lock:
            if key in self._keys:
                return False
            self._keys[key] = token
            return True

    def release(self, key: str, token: str) -> None:
        with self._lock:
            if self._keys.get(key) == token:
                self._keys.pop(key, None)


class ExtractionLockManager:
    """
    Объединяет Redis-lock и in-proc lock. Redis имеет приоритет, если клиент работоспособен.
    """

    def __init__(self, redis_client: Any = None, *, ttl_sec: int = 30):
        self._redis = redis_client
        self._ttl   = max(5, ttl_sec)
        self._mem   = _InProcLocks()

    def try_acquire(self, key: str) -> Optional[_AcquiredLock]:
        token = f"{int(time.time() * 1000)}:{threading.get_ident()}"
        if self._redis is not None:
            try:
                ok = self._redis.set(key, token, nx=True, ex=self._ttl)
                if ok:
                    return _AcquiredLock(key=key, token=token, backend="redis")
                return None
            except Exception as exc:
                logger.warning("Redis lock SET failed (%s); falling back to in-proc lock", exc)
        return _AcquiredLock(key=key, token=token, backend="memory") \
            if self._mem.acquire(key, token) else None

    def release(self, lock: _AcquiredLock) -> None:
        if lock.released:
            return
        lock.released = True
        if lock.backend == "redis" and self._redis is not None:
            try:
                v = None
                try:
                    v = self._redis.getdel(lock.key)
                except Exception:
                    self._redis.delete(lock.key)
                    return
                if v is not None and v != lock.token:
                    logger.debug("Redis lock %s changed owner before release", lock.key)
            except Exception as exc:
                logger.warning("Redis lock release failed: %s", exc)
        else:
            self._mem.release(lock.key, lock.token)




@dataclass
class RunnerStats:
    """Лёгкая телеметрия для метрик Фазы 7."""
    scheduled:        int = 0
    started:          int = 0
    completed:        int = 0
    skipped_locked:   int = 0
    skipped_disabled: int = 0
    timeouts:         int = 0
    errors:           int = 0
    facts_total:      int = 0


class AsyncExtractorRunner:
    """
    Неблокирующий запуск экстрактора.

    Использование::

        runner = AsyncExtractorRunner(
            extractor=memory_extractor,
            recent_turns_provider=lambda c,u,v: redis.get_recent_messages(c,v),
            known_keys_provider=lambda c,u: fact_store.list_user_fact_keys(c,u),
            on_result=lambda c,u,v,res: fact_store.upsert_extracted_facts(c,u,v,res.facts),
            redis_client=redis_client,
        )
        runner.schedule(company_id=..., user_id=..., conversation_id=..., turn_index=session.turn_number)
    """

    def __init__(
        self,
        *,
        extractor:             MemoryExtractor,
        recent_turns_provider: RecentTurnsProvider,
        known_keys_provider:   KnownKeysProvider     = _noop_known_keys,
        on_result:             OnResultCallback     = _noop_on_result,
        redis_client:          Any                  = None,
        lock_ttl_sec:          int                  = 30,
        timeout_sec:           Optional[float]      = None,
        max_workers:           int                  = 2,
    ) -> None:
        self._extractor   = extractor
        self._turns_prov  = recent_turns_provider
        self._keys_prov   = known_keys_provider
        self._on_result   = on_result
        self._lock_mgr    = ExtractionLockManager(redis_client=redis_client, ttl_sec=lock_ttl_sec)
        self._timeout     = timeout_sec if timeout_sec is not None else _default_timeout()
        self._executor    = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="finkey-extract")
        self.stats        = RunnerStats()
        self._stats_lock  = threading.Lock()


    @property
    def enabled(self) -> bool:
        return self._extractor.config.enabled

    def schedule(
        self,
        *,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
        turn_index:      int,
        user_locale:     Optional[str] = None,
        fallback_recent_turns: Optional[list[dict]] = None,
    ) -> bool:
        """
        Запланировать прогон экстрактора. Возвращает True, если задача поставлена в очередь
        (False — отключено / не пора / lock не взят / pool закрыт).
        """
        if not self._extractor.config.enabled:
            with self._stats_lock:
                self.stats.skipped_disabled += 1
            return False
        if not self._extractor.should_run_this_turn(turn_index):
            RUNNER_SKIPPED_DEBOUNCE.inc(company_id=company_id)
            return False

        lock_key = f"finkey:extract:{company_id}:{user_id}:{conversation_id}"
        lock = self._lock_mgr.try_acquire(lock_key)
        if lock is None:
            with self._stats_lock:
                self.stats.skipped_locked += 1
            RUNNER_SKIPPED_LOCKED.inc(company_id=company_id)
            logger.debug("extractor lock busy for %s", lock_key)
            return False

        with self._stats_lock:
            self.stats.scheduled += 1
        RUNNER_SCHEDULED.inc(company_id=company_id)

        try:
            self._executor.submit(
                self._run_one,
                lock, company_id, user_id, conversation_id, turn_index,
                user_locale, fallback_recent_turns or [],
            )
        except RuntimeError:
            self._lock_mgr.release(lock)
            return False
        return True

    def shutdown(self, *, wait: bool = False) -> None:
        """Останавливаем пул. ``wait=True`` дожидается всех текущих задач."""
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def wait_idle(self, timeout_sec: float = 5.0) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self._stats_lock:
                pending = self.stats.scheduled - (self.stats.completed + self.stats.errors + self.stats.timeouts)
            if pending <= 0:
                return
            time.sleep(0.01)


    def _run_one(
        self,
        lock:           _AcquiredLock,
        company_id:     str,
        user_id:        str,
        conversation_id: str,
        turn_index:     int,                                     # noqa: ARG002 — нужно для дебага, оставляем
        user_locale:    Optional[str],
        fallback_recent_turns: list[dict],
    ) -> None:
        t0 = time.time()
        with self._stats_lock:
            self.stats.started += 1
        RUNNER_INFLIGHT.inc(company_id=company_id)

        try:
            try:
                recent = self._turns_prov(company_id, user_id, conversation_id) or []
            except Exception as exc:  # noqa: BLE001
                log_event(
                    logger, logging.WARNING, "memory.runner.recent_turns_failed",
                    company_id=company_id, user_id=user_id, conv_id=conversation_id,
                    error_type=type(exc).__name__,
                )
                recent = []
            if not recent and fallback_recent_turns:
                recent = fallback_recent_turns

            try:
                known_keys = self._keys_prov(company_id, user_id) or set()
            except Exception as exc:  # noqa: BLE001
                log_event(
                    logger, logging.WARNING, "memory.runner.known_keys_failed",
                    company_id=company_id, user_id=user_id, conv_id=conversation_id,
                    error_type=type(exc).__name__,
                )
                known_keys = set()

            timer_fired = threading.Event()
            timer = threading.Timer(self._timeout, timer_fired.set) if self._timeout else None
            if timer:
                timer.daemon = True
                timer.start()
            try:
                result = self._extractor.extract(
                    company_id=company_id, user_id=user_id, conversation_id=conversation_id,
                    recent_turns=recent, known_keys=known_keys,
                    user_locale=user_locale,
                )
            finally:
                if timer:
                    timer.cancel()

            if timer_fired.is_set() and not result.facts:
                with self._stats_lock:
                    self.stats.timeouts += 1
                RUNNER_TIMEOUTS.inc(company_id=company_id)
                log_event(
                    logger, logging.WARNING, "memory.runner.timeout",
                    company_id=company_id, user_id=user_id, conv_id=conversation_id,
                )
                return

            with self._stats_lock:
                self.stats.completed += 1
                self.stats.facts_total += len(result.facts)
            RUNNER_COMPLETED.inc(company_id=company_id)

            try:
                self._on_result(company_id, user_id, conversation_id, result)
            except Exception as exc:  # noqa: BLE001
                log_event(
                    logger, logging.WARNING, "memory.runner.callback_failed",
                    company_id=company_id, user_id=user_id, conv_id=conversation_id,
                    error_type=type(exc).__name__,
                )
        except Exception as exc:  # noqa: BLE001
            with self._stats_lock:
                self.stats.errors += 1
            RUNNER_ERRORS.inc(company_id=company_id)
            logger.exception("extractor worker crashed: %s", exc)
        finally:
            RUNNER_INFLIGHT.dec(company_id=company_id)
            self._lock_mgr.release(lock)
            log_event(
                logger, logging.DEBUG, "memory.runner.worker_done",
                company_id=company_id, user_id=user_id, conv_id=conversation_id,
                elapsed_ms=int((time.time() - t0) * 1000),
            )




def _default_timeout() -> float:
    try:
        v = float(os.getenv("FINKEY_MEMORY_EXTRACTOR_TIMEOUT_SEC", "30"))
    except ValueError:
        v = 30.0
    return max(2.0, v)


def build_default_extractor(completer) -> MemoryExtractor:
    """
    Конструирует ``MemoryExtractor`` со стандартной конфигурацией из env.
    Тонкое место для wiring в composition root приложения.
    """
    return MemoryExtractor(completer=completer, config=ExtractorConfig())
