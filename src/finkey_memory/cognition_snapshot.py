# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Serializable slice of cognition for persisting structured memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from finkey_memory.consciousness_state import ConsciousnessContext


@dataclass(frozen=True)
class CognitionTurnSnapshot:
    """
    Flat snapshot built after MindEngine.think — safe to persist each turn.

    Does not retain the giant system_prompt or references to mutable graph state.
    """

    perception_literal: str
    primary_emotion: str
    emotion_intensity: float
    topic_type: str
    urgency: float

    emotional_read: str
    subtext_insight: str
    core_need: str
    inner_feeling: str

    tone_name: str
    approach: str
    validate_first: bool

    tenderness: float
    concern: float
    temperature: float


def cognition_from_context(ctx: "ConsciousnessContext") -> CognitionTurnSnapshot:
    """Extract a persistence-friendly snapshot."""
    perc = ctx.perception
    emo = ctx.emotional_state
    thought = ctx.inner_thought
    fk = getattr(ctx.finkey_emotions, "tenderness", 0.0) or 0.0
    fc = getattr(ctx.finkey_emotions, "concern", 0.45) if ctx.finkey_emotions else 0.45

    return CognitionTurnSnapshot(
        perception_literal   = (perc.literal or "").strip(),
        primary_emotion      = perc.primary_emotion.value if hasattr(perc.primary_emotion, "value") else str(perc.primary_emotion),
        emotion_intensity    = float(perc.emotion_intensity or 0.0),
        topic_type           = perc.topic_type.value if hasattr(perc.topic_type, "value") else str(perc.topic_type),
        urgency              = float(perc.urgency or 0.0),
        emotional_read       = (thought.emotional_read or "").strip(),
        subtext_insight      = (thought.subtext_insight or "").strip(),
        core_need            = (thought.core_need or "").strip(),
        inner_feeling        = (thought.inner_feeling or "").strip(),
        tone_name            = thought.tone.value if hasattr(thought.tone, "value") else str(thought.tone),
        approach             = (thought.approach or "").strip(),
        validate_first       = bool(thought.validate_first),
        tenderness           = float(fk),
        concern              = float(fc),
        temperature          = float(thought.temperature or 0.7),
    )
