# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey session memory — Redis layer.

Stores everything about an active conversation that needs to survive
between HTTP requests but doesn't need to live forever.

Isolation: every key is prefixed with company_id — Redis is shared,
but a company can only read/write its own namespace.

What lives here:
  • Emotional arc         — list of detected emotions in this session
  • EmotionalDimensions   — FinKey's 4D inner state for this conversation
  • Session metadata      — turn count, last topic, relationship tone
  • Recent messages       — last N raw messages (for context window)

All keys expire automatically — no cleanup needed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Optional

from finkey_memory.consciousness_state import EmotionalDimensions, EmotionType
from finkey_memory.schema import (
    RedisKeys,
    SessionState,
    ConversationTurn,
    TTL_ARC,
    TTL_DIMS,
    TTL_MESSAGES,
    TTL_SESSION,
)

logger = logging.getLogger(__name__)

MAX_REDIS_MESSAGES = 20


class SessionMemory:
    """
    Redis-backed session memory.

    Accepts a redis client on init — supports both redis-py (sync)
    and any compatible interface. Pass ``redis.Redis(...)`` or
    ``aioredis.from_url(...)`` depending on your application stack.
    """

    def __init__(self, redis_client) -> None:
        self._r = redis_client

    @property
    def client(self):
        """Низкоуровневый redis-клиент (для locks / счётчиков из других модулей)."""
        return self._r


    def load_session(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
    ) -> Optional[SessionState]:
        key  = RedisKeys.session(company_id, user_id, conversation_id)
        raw  = self._r.get(key)
        if not raw:
            return None
        try:
            d = json.loads(raw)
            return SessionState(
                company_id        = d["company_id"],
                user_id           = d["user_id"],
                conversation_id   = d["conversation_id"],
                turn_number       = d.get("turn_number", 0),
                emotional_arc     = d.get("emotional_arc", []),
                last_topic        = d.get("last_topic"),
                relationship_tone = d.get("relationship_tone", "neutral"),
            )
        except Exception as exc:
            logger.warning("Failed to deserialize session: %s", exc)
            return None

    def save_session(self, state: SessionState) -> None:
        key  = RedisKeys.session(state.company_id, state.user_id, state.conversation_id)
        data = {
            "company_id":        state.company_id,
            "user_id":           state.user_id,
            "conversation_id":   state.conversation_id,
            "turn_number":       state.turn_number,
            "emotional_arc":     state.emotional_arc,
            "last_topic":        state.last_topic,
            "relationship_tone": state.relationship_tone,
        }
        self._r.setex(key, TTL_SESSION, json.dumps(data, ensure_ascii=False))

    def delete_session(self, company_id: str, user_id: str, conversation_id: str) -> None:
        key = RedisKeys.session(company_id, user_id, conversation_id)
        self._r.delete(key)


    def load_arc(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
    ) -> list[EmotionType]:
        key = RedisKeys.emotional_arc(company_id, user_id, conversation_id)
        raw = self._r.get(key)
        if not raw:
            return []
        try:
            values = json.loads(raw)
            return [EmotionType(v) for v in values if v in EmotionType._value2member_map_]  # type: ignore[attr-defined]
        except Exception:
            return []

    def save_arc(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
        arc:             list[EmotionType],
    ) -> None:
        key = RedisKeys.emotional_arc(company_id, user_id, conversation_id)
        self._r.setex(key, TTL_ARC, json.dumps([e.value for e in arc]))


    def load_dims(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
    ) -> Optional[EmotionalDimensions]:
        key = RedisKeys.emotional_dims(company_id, user_id, conversation_id)
        raw = self._r.get(key)
        if not raw:
            return None
        try:
            d = json.loads(raw)
            return EmotionalDimensions(
                valence    = float(d["valence"]),
                arousal    = float(d["arousal"]),
                tenderness = float(d["tenderness"]),
                concern    = float(d["concern"]),
            )
        except Exception:
            return None

    def save_dims(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
        dims:            EmotionalDimensions,
    ) -> None:
        key  = RedisKeys.emotional_dims(company_id, user_id, conversation_id)
        data = {
            "valence":    dims.valence,
            "arousal":    dims.arousal,
            "tenderness": dims.tenderness,
            "concern":    dims.concern,
        }
        self._r.setex(key, TTL_DIMS, json.dumps(data))


    def push_message(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
        turn:            ConversationTurn,
    ) -> None:
        key = RedisKeys.conv_messages(company_id, user_id, conversation_id)
        self._r.rpush(key, json.dumps(asdict(turn), ensure_ascii=False))
        self._r.ltrim(key, -MAX_REDIS_MESSAGES, -1)
        self._r.expire(key, TTL_MESSAGES)

    def load_recent_messages(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
        limit:           int = MAX_REDIS_MESSAGES,
    ) -> list[ConversationTurn]:
        key  = RedisKeys.conv_messages(company_id, user_id, conversation_id)
        raws = self._r.lrange(key, -limit, -1)
        turns: list[ConversationTurn] = []
        for raw in raws:
            try:
                d = json.loads(raw)
                turns.append(ConversationTurn(**d))
            except Exception:
                continue
        return turns


    def touch_session(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
    ) -> None:
        """Reset TTL on all session keys (user is still active)."""
        for key in [
            RedisKeys.session(company_id, user_id, conversation_id),
            RedisKeys.emotional_arc(company_id, user_id, conversation_id),
            RedisKeys.emotional_dims(company_id, user_id, conversation_id),
            RedisKeys.conv_messages(company_id, user_id, conversation_id),
        ]:
            self._r.expire(key, TTL_SESSION)
