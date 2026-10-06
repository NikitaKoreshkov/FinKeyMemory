# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Unit-тесты на AsyncExtractorRunner.

Не используем ни LLM, ни сеть. Реализуем in-process fake-completer и fake-Redis,
проверяем lock/debounce/timeout/colback-pipeline.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from finkey_memory.extractor import (
    ExtractorConfig,
    MemoryExtractor,
)
from finkey_memory.extractor_runner import (
    AsyncExtractorRunner,
    ExtractionLockManager,
)




class FakeCompleter:
    """Возвращает заранее заданный JSON; считает количество вызовов."""

    def __init__(self, payload: dict, *, delay_sec: float = 0.0):
        self.payload   = payload
        self.delay_sec = delay_sec
        self.calls     = 0
        self.lock      = threading.Lock()

    def __call__(self, messages, *, max_tokens, temperature):
        with self.lock:
            self.calls += 1
        if self.delay_sec:
            time.sleep(self.delay_sec)
        return json.dumps(self.payload, ensure_ascii=False)


class FakeRedis:
    """Минимальная Redis-like: set(nx, ex), getdel, delete."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        now = time.time()
        expired = [k for k, (_v, exp) in self._store.items() if exp <= now]
        for k in expired:
            del self._store[k]

    def set(self, key, value, *, nx=False, ex=None):
        with self._lock:
            self._purge()
            if nx and key in self._store:
                return False
            exp = time.time() + (ex if ex else 3600)
            self._store[key] = (value, exp)
            return True

    def getdel(self, key):
        with self._lock:
            self._purge()
            v = self._store.pop(key, None)
            return v[0] if v else None

    def delete(self, key):
        with self._lock:
            self._store.pop(key, None)



def test_lock_redis_acquires_and_releases():
    r = FakeRedis()
    mgr = ExtractionLockManager(redis_client=r)
    a = mgr.try_acquire("k1")
    assert a is not None and a.backend == "redis"
    b = mgr.try_acquire("k1")
    assert b is None
    mgr.release(a)
    c = mgr.try_acquire("k1")
    assert c is not None


def test_lock_inproc_fallback_when_no_redis():
    mgr = ExtractionLockManager(redis_client=None)
    a = mgr.try_acquire("k")
    assert a is not None and a.backend == "memory"
    b = mgr.try_acquire("k")
    assert b is None
    mgr.release(a)
    assert mgr.try_acquire("k") is not None


def test_lock_falls_back_to_memory_when_redis_throws():
    class _Broken:
        def set(self, *a, **kw): raise RuntimeError("redis down")
    mgr = ExtractionLockManager(redis_client=_Broken())
    a = mgr.try_acquire("k")
    assert a is not None and a.backend == "memory"



def _make_runner(
    *,
    payload: dict | None  = None,
    delay_sec: float       = 0.0,
    enabled: bool          = True,
    every_n_turns: int     = 3,
    timeout_sec: float     = 5.0,
    redis_client=None,
    on_result=None,
    recent_turns_provider=None,
    known_keys_provider=None,
):
    payload = payload or {"facts": [
        {"key": "identity.name", "value": "Никита", "category": "identity",
         "confidence": 0.97, "evidence": "Я Никита"},
    ]}
    completer = FakeCompleter(payload, delay_sec=delay_sec)
    ext = MemoryExtractor(
        completer=completer,
        config=ExtractorConfig(enabled=enabled, every_n_turns=every_n_turns),
    )
    runner = AsyncExtractorRunner(
        extractor=ext,
        recent_turns_provider=recent_turns_provider
            or (lambda c, u, v: [{"role": "user", "content": "Я Никита"}]),
        known_keys_provider=known_keys_provider or (lambda c, u: set()),
        on_result=on_result or (lambda *a, **kw: None),
        redis_client=redis_client,
        timeout_sec=timeout_sec,
        max_workers=2,
    )
    return runner, completer


def test_schedule_triggers_extraction_and_callback():
    seen: list[tuple[str, str, str, int]] = []

    def on_res(c, u, v, res):
        seen.append((c, u, v, len(res.facts)))

    runner, _ = _make_runner(on_result=on_res)
    try:
        assert runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=1) is True
        runner.wait_idle(timeout_sec=3.0)
        assert seen == [("a", "u", "c", 1)]
        assert runner.stats.completed == 1
        assert runner.stats.facts_total == 1
    finally:
        runner.shutdown(wait=True)


def test_debounce_skips_off_turns():
    runner, completer = _make_runner(every_n_turns=3)
    try:
        assert runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=2) is False
        assert runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=4) is False
        assert runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=5) is False
        assert runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=1) is True
        runner.wait_idle()
        assert runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=6) is True
        runner.wait_idle()
        assert completer.calls == 2
    finally:
        runner.shutdown(wait=True)


def test_disabled_extractor_does_nothing():
    runner, completer = _make_runner(enabled=False)
    try:
        assert runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=1) is False
        runner.wait_idle()
        assert completer.calls == 0
        assert runner.stats.skipped_disabled >= 1
    finally:
        runner.shutdown(wait=True)


def test_concurrent_same_conv_only_one_runs(monkeypatch):
    """Параллельные schedule() на одном (user, conv) — второй должен быть skipped_locked."""
    redis = FakeRedis()
    runner, completer = _make_runner(delay_sec=0.15, redis_client=redis)
    try:
        ok1 = runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=1)
        time.sleep(0.02)
        ok2 = runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=1)
        runner.wait_idle()
        assert ok1 is True
        assert ok2 is False
        assert completer.calls == 1
        assert runner.stats.skipped_locked >= 1
    finally:
        runner.shutdown(wait=True)


def test_recent_turns_provider_used_for_extraction():
    seen = []

    def recents(c, u, v):
        return [
            {"role": "user", "content": "Меня зовут Алиса"},
            {"role": "assistant", "content": "Привет, Алиса!"},
            {"role": "user", "content": "Я PM в Stripe"},
        ]

    runner, completer = _make_runner(recent_turns_provider=recents,
                                     on_result=lambda *a, **k: seen.append(a))
    try:
        runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=1)
        runner.wait_idle()
        last_msgs = completer
        out = runner._extractor.extract(
            company_id="a", user_id="u", conversation_id="c",
            recent_turns=recents("a", "u", "c"),
        )
        assert out.parse_ok is True
        assert seen and seen[0][:3] == ("a", "u", "c")
    finally:
        runner.shutdown(wait=True)


def test_callback_exception_does_not_break_runner():
    def bad_on_result(*a, **kw):
        raise RuntimeError("boom")
    runner, completer = _make_runner(on_result=bad_on_result)
    try:
        runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=1)
        runner.wait_idle()
        assert runner.stats.completed == 1
        assert runner.stats.errors == 0
    finally:
        runner.shutdown(wait=True)


def test_provider_exception_uses_fallback_recent_turns():
    seen = []

    def bad_provider(*_):
        raise RuntimeError("redis down")

    runner, completer = _make_runner(
        recent_turns_provider=bad_provider,
        on_result=lambda *a, **k: seen.append(a),
    )
    try:
        runner.schedule(
            company_id="a", user_id="u", conversation_id="c", turn_index=1,
            fallback_recent_turns=[{"role": "user", "content": "Я Боб"}],
        )
        runner.wait_idle()
        assert len(seen) == 1
        assert runner.stats.completed == 1
    finally:
        runner.shutdown(wait=True)


def test_shutdown_then_schedule_returns_false():
    runner, _ = _make_runner()
    runner.shutdown(wait=True)
    assert runner.schedule(company_id="a", user_id="u", conversation_id="c", turn_index=1) is False
