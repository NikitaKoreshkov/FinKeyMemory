# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Интеграция: ``MemoryManager.save_turn`` корректно дёргает ``AsyncExtractorRunner``
с правильным turn_index и не блокирует основной поток.
"""

from __future__ import annotations

import json
import threading
import time

from finkey_memory.extractor import ExtractorConfig, MemoryExtractor
from finkey_memory.extractor_runner import AsyncExtractorRunner
from finkey_memory.manager import MemoryManager
from finkey_memory.schema import ConversationTurn


class _FakeCompleter:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, messages, *, max_tokens, temperature):
        with self.lock:
            self.calls.append((messages, max_tokens, temperature))
        return json.dumps({"facts": [
            {"key": "identity.name", "value": "Никита",
             "category": "identity", "confidence": 0.95,
             "evidence": "Меня зовут Никита"},
        ]}, ensure_ascii=False)


def _make_runner_and_results():
    completer = _FakeCompleter()
    extractor = MemoryExtractor(
        completer=completer,
        config=ExtractorConfig(enabled=True, every_n_turns=2),
    )
    results: list = []
    runner = AsyncExtractorRunner(
        extractor=extractor,
        recent_turns_provider=lambda *_: [],
        known_keys_provider=lambda *_: set(),
        on_result=lambda c, u, v, r: results.append((c, u, v, r)),
        max_workers=2,
    )
    return runner, completer, results


def test_save_turn_schedules_on_first_turn():
    runner, completer, results = _make_runner_and_results()
    try:
        mm = MemoryManager(extractor_runner=runner)
        mm.save_turn(
            company_id="acme", user_id="u-1", conversation_id="c-1",
            user_turn      = ConversationTurn(role="user", content="Привет, я Никита"),
            assistant_turn = ConversationTurn(role="assistant", content="Здравствуй!"),
            updated_arc    = [],
        )
        runner.wait_idle(timeout_sec=3.0)
        assert len(completer.calls) == 1
        assert len(results) == 1
        c, u, v, res = results[0]
        assert (c, u, v) == ("acme", "u-1", "c-1")
        assert res.facts and res.facts[0].value == "Никита"
    finally:
        mm.shutdown(wait=True)


def test_save_turn_debounces_on_off_turns():
    runner, completer, results = _make_runner_and_results()
    try:
        mm = MemoryManager(extractor_runner=runner)
        for _ in range(4):
            mm.save_turn(
                company_id="acme", user_id="u-1", conversation_id="c-2",
                user_turn=ConversationTurn(role="user", content="X"),
                assistant_turn=ConversationTurn(role="assistant", content="Y"),
                updated_arc=[],
            )
            runner.wait_idle(timeout_sec=2.0)
        assert len(completer.calls) == 3
    finally:
        mm.shutdown(wait=True)


def test_save_turn_does_not_block_main_thread(monkeypatch):
    completer = _FakeCompleter()
    real_call = completer.__call__

    def slow(*a, **kw):
        time.sleep(0.5)
        return real_call(*a, **kw)

    completer.__call__ = slow  # type: ignore[assignment]
    extractor = MemoryExtractor(
        completer=completer,
        config=ExtractorConfig(enabled=True, every_n_turns=1),
    )
    runner = AsyncExtractorRunner(
        extractor=extractor,
        recent_turns_provider=lambda *_: [{"role": "user", "content": "slow"}],
        known_keys_provider=lambda *_: set(),
        on_result=lambda *a, **k: None,
        max_workers=2,
    )
    try:
        mm = MemoryManager(extractor_runner=runner)
        t0 = time.time()
        mm.save_turn(
            company_id="acme", user_id="u-1", conversation_id="c-3",
            user_turn=ConversationTurn(role="user", content="hi"),
            assistant_turn=ConversationTurn(role="assistant", content="hello"),
            updated_arc=[],
        )
        elapsed = time.time() - t0
        assert elapsed < 0.2, f"save_turn blocked for {elapsed:.3f}s — must be non-blocking"
        runner.wait_idle(timeout_sec=3.0)
    finally:
        mm.shutdown(wait=True)


def test_extractor_disabled_skips_silently():
    completer = _FakeCompleter()
    extractor = MemoryExtractor(
        completer=completer,
        config=ExtractorConfig(enabled=False),
    )
    runner = AsyncExtractorRunner(
        extractor=extractor,
        recent_turns_provider=lambda *_: [],
        known_keys_provider=lambda *_: set(),
        on_result=lambda *a, **k: None,
        max_workers=1,
    )
    try:
        mm = MemoryManager(extractor_runner=runner)
        for _ in range(5):
            mm.save_turn(
                company_id="acme", user_id="u-1", conversation_id="c",
                user_turn=ConversationTurn(role="user", content="x"),
                assistant_turn=ConversationTurn(role="assistant", content="y"),
                updated_arc=[],
            )
        runner.wait_idle(timeout_sec=2.0)
        assert completer.calls == []
    finally:
        mm.shutdown(wait=True)
