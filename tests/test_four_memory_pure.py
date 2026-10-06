# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Unit tests for consolidation + episode codecs (no DB)."""

from __future__ import annotations

from finkey_memory.cognition_snapshot import CognitionTurnSnapshot
from finkey_memory.consolidation import merge_fact_versions, normalize_fact_key
from finkey_memory.episode_codec import episode_from_snapshot, episode_to_plain_dict


def test_normalize_fact_key() -> None:
    k = normalize_fact_key("  Привет-Мир !!  ")
    assert "привет" in k and "мир" in k.replace(" ", "")


def test_merge_conflict_prefers_user() -> None:
    v, conf, src = merge_fact_versions(
        existing_value      = "A",
        existing_confidence = 0.8,
        existing_source     = "inferred",
        incoming_value      = "B",
        incoming_confidence = 0.6,
        incoming_source     = "user",
    )
    assert "B" in v or src == "user"
    assert conf >= 0.05


def test_episode_snapshot_roundtrip_shape() -> None:
    snap = CognitionTurnSnapshot(
        perception_literal   = "нужно сделать отчёт",
        primary_emotion      = "anxious",
        emotion_intensity    = 0.7,
        topic_type           = "work_process",
        urgency              = 0.5,
        emotional_read       = "давление по срокам",
        subtext_insight      = "боится ошибиться",
        core_need            = "ясный первый шаг",
        inner_feeling        = "хочется поддержать",
        tone_name            = "warm_supportive",
        approach             = "сначала валидация, потом маленький план из трёх шагов",
        validate_first       = True,
        tenderness           = 0.82,
        concern              = 0.41,
        temperature          = 0.72,
    )
    ep = episode_from_snapshot(
        "demo_co", "u1", "conv_x",
        "помогите с отчётом",
        "Давай уточним срок и разобьём на шаги…",
        snap,
    )
    d = episode_to_plain_dict(ep)
    assert d["conversation_id"] == "conv_x"
    assert "помогите" in (d["user_message"] or "")
    assert isinstance(d["timestamp"], str)
