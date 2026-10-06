# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Episode dataclass ↔ JSONB for PostgreSQL."""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from finkey_memory.cognition_snapshot import CognitionTurnSnapshot
from finkey_memory.episode import Episode, EpisodeBuilder


def sanitize_for_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    if is_dataclass(obj):
        return sanitize_for_json(asdict(obj))
    return str(obj)


def episode_to_plain_dict(ep: Episode) -> dict:
    """Plain JSON-serializable dict for snapshot column."""
    return sanitize_for_json(asdict(ep))  # type: ignore[arg-type]


def episode_from_snapshot(
    company_key: str,
    user_external: str,
    conv_external: str,
    user_message: str,
    assistant_reply: str,
    snap: CognitionTurnSnapshot,
) -> Episode:
    importance = min(
        1.0,
        0.32 + snap.emotion_intensity * 0.38 + (0.12 if snap.validate_first else 0.0),
    )

    thought_line = snap.approach[:500] if snap.approach else ""
    tt = snap.topic_type
    if "." in tt:
        tt = tt.split(".")[-1]
    tt_clean = "".join(c if c.isalnum() or c == "_" else "_" for c in tt.lower())[:80] or "work_process"

    return (
        EpisodeBuilder()
        .for_user(company_key, user_external, conv_external)
        .with_exchange(user_message, assistant_reply)
        .with_inner_state(
            thought          = thought_line,
            feeling          = snap.inner_feeling,
            emotional_read   = snap.emotional_read,
            subtext          = snap.subtext_insight,
            core_need        = snap.core_need,
        )
        .with_context(
            topic     = snap.perception_literal[:400] if snap.perception_literal else "",
            emotion   = snap.primary_emotion,
            intensity = snap.emotion_intensity,
            urgency   = snap.urgency,
            topic_type= tt_clean,
        )
        .with_quality(
            temperature = snap.temperature,
            tone        = snap.tone_name,
            empathy     = snap.tenderness,
            validated   = snap.validate_first,
        )
        .with_importance(importance)
        .with_tags("episodic", snap.topic_type)
        .build()
    )
