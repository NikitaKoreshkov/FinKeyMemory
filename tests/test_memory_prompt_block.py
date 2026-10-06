# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Unit tests for ``MemoryContext.memory_prompt_block`` (phase 5).

No external dependencies — pure text assembly only.
"""

from __future__ import annotations

import datetime as dt

from finkey_memory.consciousness_state import UserProfile
from finkey_memory.schema import (
    ConversationTurn,
    MemoryContext,
    SessionState,
    UserImpression,
)


def _mk_ctx(**kwargs) -> MemoryContext:
    base = dict(
        company_id="acme",
        user_id="u-1",
        conversation_id="c-1",
        user_profile=UserProfile(user_id="u-1"),
        session=SessionState(company_id="acme", user_id="u-1", conversation_id="c-1"),
    )
    base.update(kwargs)
    return MemoryContext(**base)


def test_empty_context_yields_empty_block():
    assert _mk_ctx().memory_prompt_block == ""


def test_identity_pin_first_and_durable_facts_also_rendered():
    """Pin идёт первым, но durable-факты из PG не подавляются: pin в Redis
    может протухнуть (TTL 6ч), а PG — источник истины."""
    ctx = _mk_ctx(
        identity_pin=(
            "Identity and durable user context (cached; refreshed automatically):\n"
            "- [identity] identity.name: Никита"
        ),
        durable_facts=[
            {"key_normalized": "identity.name", "value": "Никита", "category": "identity"},
        ],
    )
    block = ctx.memory_prompt_block
    assert "Identity and durable user context" in block
    assert "Memory I keep and use" in block
    assert block.find("Identity and durable user context") < block.find(
        "Memory I keep and use"
    )


def test_durable_facts_used_when_no_pin_and_grouped_by_category():
    ctx = _mk_ctx(durable_facts=[
        {"key_normalized": "identity.name",      "value": "Никита",   "category": "identity"},
        {"key_normalized": "affiliation.company","value": "FinKey",   "category": "affiliation"},
        {"key_normalized": "context.project",    "value": "AI-память","category": "context"},
        {"key_normalized": "preference.tone",    "value": "коротко",  "category": "preference"},
        {"key_normalized": "other.x",            "value": "OTHER",    "category": "other"},
    ])
    block = ctx.memory_prompt_block
    assert "Memory I keep and use" in block
    pos_id   = block.find("Identity:")
    pos_aff  = block.find("Affiliation:")
    pos_ctx  = block.find("Context / projects:")
    assert 0 <= pos_id < pos_aff < pos_ctx
    assert "OTHER" not in block
    assert "DIRECTIVE (how you speak" in block
    assert "коротко" in block
    assert "Preferences:" not in block


def test_relevant_fact_hits_block_present_and_dedup():
    ctx = _mk_ctx(
        identity_pin="pin-here",
        relevant_fact_hits=[
            {"value": "Работаю в FinKey"},
            {"value": "Работаю в FinKey"},
            {"value": "Люблю короткие ответы"},
        ],
    )
    block = ctx.memory_prompt_block
    assert "Recalled for this message" in block
    assert block.count("Работаю в FinKey") == 1


def test_user_impressions_rendered_compactly():
    ctx = _mk_ctx(user_impressions=[
        UserImpression(user_id="u-1", company_id="acme", impression_type="communication_style",
                       content="прямой и сухой стиль", confidence=0.9, source="user", priority=0.7),
    ])
    block = ctx.memory_prompt_block
    assert "DIRECTIVE (how you speak" in block
    assert "прямой и сухой стиль" in block
    assert "Communication style and observations" not in block


def test_substrates_are_hidden_by_default():
    ctx = _mk_ctx(
        episodic_highlights=["Эпизод 1"],
        semantic_facts=["Семантика 1"],
        emotional_memory=["Эмо 1"],
        procedural_hints=["Процедура 1"],
        cognition_substrates_enabled=False,
    )
    block = ctx.memory_prompt_block
    assert "Episodic memory (meaningful beats)" not in block
    assert "Semantic facts about context"   not in block
    assert "Recent emotional trajectory" not in block
    assert "Procedural cues (how to steer the dialogue)" not in block


def test_substrates_appear_when_flag_on():
    ctx = _mk_ctx(
        episodic_highlights=["Эпизод 1"],
        semantic_facts=["Семантика 1"],
        emotional_memory=["Эмо 1"],
        procedural_hints=["Процедура 1"],
        cognition_substrates_enabled=True,
    )
    block = ctx.memory_prompt_block
    assert "Episodic memory (meaningful beats)" in block
    assert "Semantic facts about context"   in block
    assert "Recent emotional trajectory" in block
    assert "Procedural cues (how to steer the dialogue)" in block


def test_durable_facts_total_cap_respected():
    facts = [
        {"key_normalized": f"context.k{i}", "value": f"v{i}", "category": "context"}
        for i in range(30)
    ]
    ctx = _mk_ctx(durable_facts=facts, durable_facts_per_group=10, durable_facts_total_cap=6)
    block = ctx.memory_prompt_block
    fact_lines = [ln for ln in block.splitlines() if ln.strip().startswith("—")]
    assert len(fact_lines) <= 6


def test_prompt_order_identity_first_then_impressions_then_rag():
    ctx = _mk_ctx(
        identity_pin="IDENTITY_PIN",
        user_impressions=[UserImpression(
            user_id="u-1", company_id="acme", impression_type="communication_style",
            content="IMPR", confidence=0.8, source="user", priority=0.6,
        )],
        relevant_rag_docs=["RAG_DOC"],
        relevant_memories=["MEM"],
    )
    b = ctx.memory_prompt_block
    pos_voice = b.find("DIRECTIVE (how you speak")
    pos_pin  = b.find("IDENTITY_PIN")
    pos_impr = b.find("IMPR")
    pos_rag  = b.find("RAG_DOC")
    pos_mem  = b.find("MEM")
    assert 0 <= pos_voice < pos_impr < pos_pin < pos_rag < pos_mem


def test_memory_voice_directive_appended_when_any_memory_present():
    ctx = _mk_ctx(user_impressions=[UserImpression(
        user_id="u-1", company_id="acme", impression_type="communication_style",
        content="коротко", confidence=0.9, source="user", priority=0.7,
    )])
    block = ctx.memory_prompt_block
    assert "DIRECTIVE (memory voice)" in block
    assert "based on your profile" in block
    assert "sensitive life details" in block
    assert "How-you-speak" in block
    assert "default warm" in block
    assert "Working notes I stored" in block


def test_voice_contract_is_orders_not_trivia():
    ctx = _mk_ctx(durable_facts=[
        {
            "key_normalized": "preference.tone",
            "value": "строго в академическом и нейтральном стиле",
            "category": "preference",
            "label": "Tone",
        },
        {
            "key_normalized": "preference.response_language",
            "value": "на русском языке",
            "category": "preference",
            "label": "Response language",
        },
        {
            "key_normalized": "identity.city",
            "value": "Алматы",
            "category": "identity",
        },
    ])
    block = ctx.memory_prompt_block
    assert ctx.has_voice_contract is True
    voice_at = block.find("DIRECTIVE (how you speak")
    bio_at = block.find("Алматы")
    assert voice_at >= 0
    assert voice_at < bio_at
    assert "строго в академическом и нейтральном стиле" in block
    assert "default voice" in block
    assert "Алматы" in block
    # City stays biography; tone is not dumped as optional Preferences trivia.
    assert "Preferences:" not in block


def test_programming_language_is_not_a_voice_contract():
    ctx = _mk_ctx(durable_facts=[
        {
            "key_normalized": "context.programming_language",
            "value": "Python",
            "category": "context",
        },
    ])
    assert ctx.has_voice_contract is False
    assert "DIRECTIVE (how you speak" not in ctx.memory_prompt_block
    assert "Python" in ctx.memory_prompt_block
    assert "Memory I keep and use" in ctx.memory_prompt_block
    assert "act on them this turn" in ctx.memory_prompt_block


def test_current_query_is_working_memory_not_optional_biography():
    ctx = _mk_ctx(durable_facts=[
        {
            "key_normalized": "context.current_query",
            "value": "findMedianSortedArrays + asyncio bug",
            "category": "context",
            "label": "Current query",
        },
    ])
    block = ctx.memory_prompt_block
    assert ctx.has_voice_contract is False
    assert "findMedianSortedArrays" in block
    assert "apply when it helps this turn" not in block
    assert "Working notes I stored" in block


def test_memory_grounding_includes_selective_apply_and_forbidden_phrases():
    ctx = _mk_ctx(
        identity_pin=(
            "Identity and durable user context (cached; refreshed automatically):\n"
            "- [identity] identity.name: Никита"
        ),
        durable_facts=[
            {"key_normalized": "identity.name", "value": "Никита", "category": "identity"},
        ],
    )
    block = ctx.memory_prompt_block
    assert "DIRECTIVE (memory grounding)" in block
    assert "Do not dump personal biography" in block
    assert "Working memory I stored" in block
    assert "Forbidden phrases" in block
    assert "based on your profile/data/memories" in block
    assert "DIRECTIVE (memory voice)" in block
