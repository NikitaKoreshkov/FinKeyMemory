# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Тесты wiring'a ``MemoryManager.load_context`` под Фазу 5:

  * identity_pin читается из Redis-клиента (через ``SessionMemory.client``);
  * durable_facts берутся из ``FactStore.list_user_facts``;
  * cognition substrate'ы не пишутся, если флаг выключен;
  * substrate'ы пишутся, если флаг включён.

Никакого Postgres — используем «фейковые» адаптеры с нужным интерфейсом.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from unittest.mock import patch

from finkey_memory.manager import MemoryManager
from finkey_memory.schema import ConversationTurn, RedisKeys




class _FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value


class _FakeSession:
    def __init__(self, redis: _FakeRedis) -> None:
        self._r = redis

    @property
    def client(self):
        return self._r

    def load_session(self, *a, **kw):     return None
    def load_arc(self, *a, **kw):         return []
    def load_dims(self, *a, **kw):        return None
    def load_recent_messages(self, *a, **kw): return []
    def push_message(self, *a, **kw):     pass
    def save_arc(self, *a, **kw):         pass
    def touch_session(self, *a, **kw):    pass


class _FakeLongTerm:
    """Эмулирует только то, что нужно load_context-у."""
    def __init__(self, facts: Optional[list[dict]] = None) -> None:
        self._facts = list(facts or [])

    def get_user_profile(self, company_id, user_id):
        return None

    def get_user_impressions(self, company_id, user_id):
        return []

    def list_user_facts(self, company_id, user_id, **kw):
        return list(self._facts)

    def list_user_fact_keys(self, company_id, user_id):
        return {f["key_normalized"] for f in self._facts}

    def upsert_user_fact(self, *a, **kw):
        return None

    def get_user_fact(self, *a, **kw):
        return None

    def soft_delete_user_fact(self, *a, **kw):
        return False

    def soft_delete_all_user_facts(self, *a, **kw):
        return 0

    def audit(self, *a, **kw):
        pass

    def _resolve_user_id(self, company_id, user_id):
        return user_id




def test_identity_pin_read_from_redis():
    redis = _FakeRedis()
    redis.data[RedisKeys.identity_pin("acme", "u-1")] = "PIN_PAYLOAD"
    mm = MemoryManager(session=_FakeSession(redis), long_term=_FakeLongTerm())
    ctx = mm.load_context("acme", "u-1", "c-1", user_message="hi")
    assert ctx.identity_pin == "PIN_PAYLOAD"


def test_durable_facts_loaded_from_fact_store():
    lt = _FakeLongTerm(facts=[
        {"key_normalized": "identity.name", "value": "Никита", "category": "identity"},
        {"key_normalized": "affiliation.company", "value": "FinKey", "category": "affiliation"},
    ])
    mm = MemoryManager(long_term=lt)
    ctx = mm.load_context("acme", "u-1", "c-1", user_message="hi")
    assert len(ctx.durable_facts) == 2
    keys = {f["key_normalized"] for f in ctx.durable_facts}
    assert keys == {"identity.name", "affiliation.company"}


def test_cognition_substrates_flag_default_on():
    """Production-default: substrate'ы включены, если env-переменная не задана."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FINKEY_COGNITION_SUBSTRATES_ENABLED", None)
        mm = MemoryManager(long_term=_FakeLongTerm())
        ctx = mm.load_context("acme", "u-1", "c-1", user_message="hi")
        assert ctx.cognition_substrates_enabled is True


def test_cognition_substrates_flag_explicitly_off():
    """Сайт может явно выключить substrate'ы (A/B-тесты)."""
    with patch.dict(os.environ, {"FINKEY_COGNITION_SUBSTRATES_ENABLED": "0"}, clear=False):
        mm = MemoryManager(long_term=_FakeLongTerm())
        ctx = mm.load_context("acme", "u-1", "c-1", user_message="hi")
        assert ctx.cognition_substrates_enabled is False


def test_cognition_substrates_flag_explicitly_on():
    with patch.dict(os.environ, {"FINKEY_COGNITION_SUBSTRATES_ENABLED": "1"}, clear=False):
        mm = MemoryManager(long_term=_FakeLongTerm())
        ctx = mm.load_context("acme", "u-1", "c-1", user_message="hi")
        assert ctx.cognition_substrates_enabled is True




class _CountingVolatile:
    """Подменяем _volatile, чтобы засечь, писались ли substrate-данные."""

    def __init__(self):
        self.episode_writes = 0
        self.emotional_trace_writes = 0
        self.fact_writes = 0
        self.procedural_writes = 0
        self.impressions: list = []
        self._counts = 0

    def bump_interaction(self, *a, **kw):
        self._counts += 1
        return self._counts

    def save_memory_episode_snapshot(self, *a, **kw):
        self.episode_writes += 1

    def append_emotional_trace(self, *a, **kw):
        self.emotional_trace_writes += 1

    def upsert_fact(self, *a, **kw):
        self.fact_writes += 1

    def upsert_procedural_memory(self, *a, **kw):
        self.procedural_writes += 1

    def tenant_keys(self):
        return iter([])


def _mk_cognition():
    from finkey_memory.cognition_snapshot import CognitionTurnSnapshot
    return CognitionTurnSnapshot(
        perception_literal="спрашивает про память",
        primary_emotion="curious",
        emotion_intensity=0.6,
        topic_type="memory",
        urgency=0.2,
        emotional_read="смесь любопытства и собранности",
        subtext_insight="хочет system design ясности",
        core_need="понять как устроена память",
        inner_feeling="вижу любопытство",
        tone_name="empathic",
        approach="ask + validate",
        validate_first=True,
        tenderness=0.5,
        concern=0.3,
        temperature=0.7,
    )


def test_save_turn_skips_substrates_when_flag_explicitly_off():
    """Когда сайт явно выключает substrate'ы — save_turn не пишет episodic/semantic/..."""
    with patch.dict(os.environ, {"FINKEY_COGNITION_SUBSTRATES_ENABLED": "0"}, clear=False):
        mm = MemoryManager()
        counting = _CountingVolatile()
        mm._volatile = counting
        mm.save_turn(
            company_id="acme", user_id="u-1", conversation_id="c-1",
            user_turn=ConversationTurn(role="user", content="как у тебя устроена память?"),
            assistant_turn=ConversationTurn(role="assistant", content="вот так"),
            updated_arc=[],
            cognition=_mk_cognition(),
        )
        assert counting.episode_writes == 0
        assert counting.emotional_trace_writes == 0
        assert counting.fact_writes == 0
        assert counting.procedural_writes == 0


def test_save_turn_persists_substrates_by_default():
    """Production-default: substrate'ы пишутся, env-переменная не задана."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FINKEY_COGNITION_SUBSTRATES_ENABLED", None)
        mm = MemoryManager()
        counting = _CountingVolatile()
        mm._volatile = counting
        mm.save_turn(
            company_id="acme", user_id="u-1", conversation_id="c-1",
            user_turn=ConversationTurn(role="user", content="как у тебя устроена память?"),
            assistant_turn=ConversationTurn(role="assistant", content="вот так"),
            updated_arc=[],
            cognition=_mk_cognition(),
        )
        assert counting.episode_writes == 1
        assert counting.emotional_trace_writes == 1
        assert counting.fact_writes >= 1
        assert counting.procedural_writes == 1


def test_save_turn_persists_substrates_when_flag_explicitly_on():
    with patch.dict(os.environ, {"FINKEY_COGNITION_SUBSTRATES_ENABLED": "1"}, clear=False):
        mm = MemoryManager()
        counting = _CountingVolatile()
        mm._volatile = counting
        mm.save_turn(
            company_id="acme", user_id="u-1", conversation_id="c-1",
            user_turn=ConversationTurn(role="user", content="как у тебя устроена память?"),
            assistant_turn=ConversationTurn(role="assistant", content="вот так"),
            updated_arc=[],
            cognition=_mk_cognition(),
        )
        assert counting.episode_writes == 1
        assert counting.emotional_trace_writes == 1
        assert counting.fact_writes >= 1
        assert counting.procedural_writes == 1
