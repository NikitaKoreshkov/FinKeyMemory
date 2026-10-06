# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Episodic slice for persistence (JSONB snapshots + vector summaries).

Dataclasses for episodic snapshots persisted by ``episode_codec`` / ``MemoryManager``.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Episode:
    """A single meaningful exchange + FinKey's read of that moment."""

    id:              str            = field(default_factory=_new_id)
    company_id:      str            = ""
    user_id:         str            = ""
    conversation_id: str            = ""
    timestamp:       datetime       = field(default_factory=_now)

    user_message:    str            = ""
    finkey_response: str            = ""

    inner_thought:   str            = ""
    inner_feeling:   str            = ""
    emotional_read:  str            = ""
    subtext_insight: str            = ""
    core_need_read:  str            = ""

    topic:           str            = ""
    topic_type:      str            = "work_process"
    user_emotion:    str            = "neutral"
    emotion_intensity: float        = 0.5
    urgency:         float          = 0.5

    temperature_used: float         = 0.7
    tone_used:       str            = "warm_supportive"
    empathy_depth:   float          = 0.5
    validated_first: bool           = False
    response_structure: str         = "conversational"

    perceived_outcome: str          = "unknown"
    what_worked:     Optional[str]  = None
    what_missed:     Optional[str]  = None
    user_sentiment_shift: Optional[float] = None

    importance_score: float         = 0.5
    confidence:       float         = 1.0
    last_accessed:    Optional[datetime] = None
    access_count:     int           = 0
    decay_factor:     float         = 1.0

    tags:            list[str]      = field(default_factory=list)

    def touch(self) -> None:
        self.last_accessed = _now()
        self.access_count += 1
        self.decay_factor = min(1.0, self.decay_factor + 0.05)

    def effective_importance(self) -> float:
        return round(self.importance_score * self.decay_factor * self.confidence, 4)

    def age_days(self) -> float:
        return ((_now() - self.timestamp).total_seconds()) / 86400

    def is_significant(self) -> bool:
        return (
            self.effective_importance() >= 0.3
            or self.perceived_outcome in ("resolved", "helped")
            or bool(self.what_worked)
            or self.emotion_intensity >= 0.75
        )

    def to_text_summary(self) -> str:
        parts = [f"[{self.timestamp.strftime('%d.%m.%Y')}]"]
        if self.topic:
            parts.append(f"Тема: {self.topic}")
        parts.append(f"Эмоция: {self.user_emotion} (интенсивность {self.emotion_intensity:.1f})")
        if self.core_need_read:
            parts.append(f"Реальная потребность: {self.core_need_read}")
        if self.what_worked:
            parts.append(f"Что сработало: {self.what_worked}")
        if self.what_missed:
            parts.append(f"Что не попало: {self.what_missed}")
        if self.perceived_outcome != "unknown":
            parts.append(f"Итог: {self.perceived_outcome}")
        return " | ".join(parts)


@dataclass
class EpisodeBuilder:
    """Fluent builder for creating an Episode from a conversation turn."""

    _ep: Episode = field(default_factory=Episode)

    def for_user(self, company_id: str, user_id: str, conv_id: str) -> "EpisodeBuilder":
        self._ep.company_id      = company_id
        self._ep.user_id         = user_id
        self._ep.conversation_id = conv_id
        return self

    def with_exchange(self, user_msg: str, finkey_resp: str) -> "EpisodeBuilder":
        self._ep.user_message    = user_msg
        self._ep.finkey_response = finkey_resp
        return self

    def with_inner_state(
        self,
        thought: str,
        feeling: str,
        emotional_read: str,
        subtext: str = "",
        core_need: str = "",
    ) -> "EpisodeBuilder":
        self._ep.inner_thought   = thought
        self._ep.inner_feeling   = feeling
        self._ep.emotional_read  = emotional_read
        self._ep.subtext_insight = subtext
        self._ep.core_need_read  = core_need
        return self

    def with_context(
        self,
        topic:     str,
        emotion:   str,
        intensity: float,
        urgency:   float = 0.5,
        topic_type: str  = "work_process",
    ) -> "EpisodeBuilder":
        self._ep.topic             = topic
        self._ep.user_emotion      = emotion
        self._ep.emotion_intensity = intensity
        self._ep.urgency           = urgency
        self._ep.topic_type        = topic_type
        return self

    def with_quality(
        self,
        temperature: float,
        tone:        str,
        empathy:     float,
        validated:   bool = False,
        structure:   str  = "conversational",
    ) -> "EpisodeBuilder":
        self._ep.temperature_used   = temperature
        self._ep.tone_used          = tone
        self._ep.empathy_depth      = empathy
        self._ep.validated_first    = validated
        self._ep.response_structure = structure
        return self

    def with_importance(self, score: float) -> "EpisodeBuilder":
        self._ep.importance_score = max(0.0, min(1.0, score))
        return self

    def with_tags(self, *tags: str) -> "EpisodeBuilder":
        self._ep.tags.extend(tags)
        return self

    def build(self) -> Episode:
        return self._ep
