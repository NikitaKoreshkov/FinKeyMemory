# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Unit-тесты на MemoryExtractor.

Тесты не зависят ни от Postgres, ни от LLM-провайдера — используем stub-completer.
"""

from __future__ import annotations

import json

import pytest

from finkey_memory.extractor import (
    ExtractedFact,
    ExtractionResult,
    ExtractorConfig,
    FactCategory,
    MemoryExtractor,
    normalize_key,
    parse_extractor_response,
)
from finkey_memory.extractor_prompts import build_extraction_messages


def test_parse_extractor_strips_leading_cot_before_json():
    raw = (
        "Thinking Process: 1. Analyze. 2. Output JSON.\n"
        '{"facts": [{"key": "identity.name", "value": "Никита", "category": "identity", '
        '"confidence": 0.98, "evidence": "меня зовут Никита"}]}'
    )
    facts, ok = parse_extractor_response(raw, min_confidence=0.3)
    assert ok
    assert len(facts) == 1
    assert facts[0].key_normalized == "identity.name"
    assert facts[0].value == "Никита"



@pytest.mark.parametrize("raw, expected", [
    ("identity.name",              "identity.name"),
    ("Identity.Name",              "identity.name"),
    ("Context.Project FinKey",     "context.project.finkey"),
    ("affiliation:company",        "affiliation.company"),
    ("  state.workload  ",         "state.workload"),
    ("user@key#1",                 "user.key.1"),
    ("",                           ""),
    ("...",                        ""),
    ("a" * 200,                    "a" * 80),
])
def test_normalize_key(raw, expected):
    assert normalize_key(raw) == expected



_GOOD_JSON = json.dumps({
    "facts": [
        {"key": "identity.name", "value": "Никита", "category": "identity",
         "confidence": 0.98, "evidence": "Меня зовут Никита"},
        {"key": "affiliation.company", "value": "FinKey", "category": "affiliation",
         "confidence": 0.95, "evidence": "Я фаундер FinKey"},
        {"key": "context.project", "value": "Эмпатичный AI", "category": "context",
         "confidence": 0.8, "evidence": "Мы строим эмпатичный AI"},
    ],
}, ensure_ascii=False)


def test_parse_happy_path():
    facts, ok = parse_extractor_response(_GOOD_JSON, source_conv_id="c-1")
    assert ok is True
    assert len(facts) == 3
    assert facts[0].key_normalized == "identity.name"
    assert facts[0].value == "Никита"
    assert facts[0].category == FactCategory.IDENTITY
    assert facts[0].source_conv_id == "c-1"
    assert facts[0].source == "inferred"
    assert facts[0].priority >= 0.9


def test_parse_code_fenced_json():
    fenced = f"Here you go:\n```json\n{_GOOD_JSON}\n```\nThat's it."
    facts, ok = parse_extractor_response(fenced)
    assert ok is True
    assert len(facts) == 3


def test_parse_garbage_prefix_and_suffix():
    noisy = "Окей, вот результат:\n" + _GOOD_JSON + "\n\nГотово."
    facts, ok = parse_extractor_response(noisy)
    assert ok is True
    assert len(facts) == 3


def test_parse_empty_facts_list():
    facts, ok = parse_extractor_response('{"facts": []}')
    assert ok is True
    assert facts == []


def test_parse_invalid_json_returns_parse_failed():
    facts, ok = parse_extractor_response("totally not json")
    assert ok is False
    assert facts == []



def test_parse_drops_facts_without_evidence_or_value():
    bad = json.dumps({"facts": [
        {"key": "identity.name", "value": "Алиса", "evidence": ""},
        {"key": "identity.name", "value": "", "evidence": "что-то"},
        {"key": "",              "value": "Боб", "evidence": "Я Боб"},
        {"key": "x.y",           "value": "Z",   "evidence": "Z",
         "confidence": 0.1},
    ]})
    facts, ok = parse_extractor_response(bad, min_confidence=0.35)
    assert ok is True
    assert len(facts) == 1
    assert facts[0].value == "Алиса"


def test_parse_caps_max_facts():
    many = {"facts": [
        {"key": f"context.k{i}", "value": f"v{i}", "evidence": f"e{i}",
         "category": "context", "confidence": 0.9}
        for i in range(50)
    ]}
    facts, ok = parse_extractor_response(json.dumps(many), max_facts=5)
    assert ok is True
    assert len(facts) == 5


def test_parse_unknown_category_falls_back_to_other():
    raw = json.dumps({"facts": [
        {"key": "weird.thing", "value": "foo", "evidence": "bar",
         "category": "made-up-cat", "confidence": 0.9},
    ]})
    facts, ok = parse_extractor_response(raw)
    assert ok is True
    assert facts[0].category == FactCategory.OTHER


def test_state_and_commitment_get_expiry():
    raw = json.dumps({"facts": [
        {"key": "state.workload", "value": "перегружен", "evidence": "перегружен",
         "category": "state", "confidence": 0.7},
        {"key": "commitment.send_report", "value": "отчёт к пятнице",
         "evidence": "отчёт к пятнице", "category": "commitment", "confidence": 0.8},
    ]})
    facts, _ = parse_extractor_response(raw)
    assert all(f.expires_at is not None for f in facts)
    assert facts[1].expires_at > facts[0].expires_at


def test_image_load_glitch_is_not_remembered_as_a_user_fact():
    raw = json.dumps({"facts": [
        {"key": "state.image_upload_issue",
         "value": "изображение не загрузилось или осталось пустым",
         "category": "state", "confidence": 0.95, "evidence": "агент сказал"},
        {"key": "identity.name", "value": "Никита", "category": "identity",
         "confidence": 0.9, "evidence": "я Никита"},
    ]})
    facts, ok = parse_extractor_response(raw)
    assert ok
    assert [f.key_normalized for f in facts] == ["identity.name"]


def test_identity_facts_have_no_expiry():
    raw = json.dumps({"facts": [
        {"key": "identity.name", "value": "Никита", "evidence": "Я Никита",
         "category": "identity", "confidence": 0.95},
    ]})
    facts, _ = parse_extractor_response(raw)
    assert facts[0].expires_at is None


class _Completer:
    """Stub LLM completer: можно задать ответ или исключение."""
    def __init__(self, response: str | Exception):
        self._response = response
        self.last_messages: list[dict] | None = None
        self.last_kwargs: dict | None = None

    def __call__(self, messages, *, max_tokens, temperature):
        self.last_messages = messages
        self.last_kwargs   = {"max_tokens": max_tokens, "temperature": temperature}
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _ext(completer):
    return MemoryExtractor(completer=completer, config=ExtractorConfig(enabled=True))


def test_extract_happy_path():
    completer = _Completer(_GOOD_JSON)
    ext = _ext(completer)
    result = ext.extract(
        company_id="acme", user_id="u-1", conversation_id="c-1",
        recent_turns=[
            {"role": "user", "content": "Привет, меня зовут Никита, я фаундер FinKey"},
            {"role": "assistant", "content": "Здравствуй!"},
            {"role": "user", "content": "Мы строим эмпатичный AI"},
        ],
        known_keys=set(),
    )
    assert result.parse_ok is True
    assert len(result.facts) == 3
    assert all(isinstance(f, ExtractedFact) for f in result.facts)
    assert result.facts[0].source_conv_id == "c-1"
    assert result.elapsed_ms >= 0
    msgs = completer.last_messages
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "user: Привет, меня зовут Никита" in msgs[1]["content"]


def test_extract_passes_known_keys_into_prompt():
    completer = _Completer('{"facts": []}')
    ext = _ext(completer)
    ext.extract(
        company_id="acme", user_id="u-1", conversation_id="c-1",
        recent_turns=[{"role": "user", "content": "Ок"}],
        known_keys={"identity.name", "affiliation.company"},
    )
    user_msg = completer.last_messages[1]["content"]
    assert "identity.name" in user_msg
    assert "affiliation.company" in user_msg


def test_extract_llm_failure_is_graceful():
    completer = _Completer(RuntimeError("provider 5xx"))
    ext = _ext(completer)
    result = ext.extract(
        company_id="acme", user_id="u-1", conversation_id="c-1",
        recent_turns=[{"role": "user", "content": "Я Никита"}],
    )
    assert result.facts == []
    assert result.parse_ok is False
    assert "llm_error" in (result.skipped_reason or "")


def test_extract_skips_when_no_turns():
    ext = _ext(_Completer('{"facts": []}'))
    result = ext.extract(
        company_id="acme", user_id="u-1", conversation_id="c-1",
        recent_turns=[],
    )
    assert result.skipped_reason == "no_turns"
    assert result.facts == []


def test_extract_clips_to_max_recent_turns():
    cfg = ExtractorConfig(enabled=True, max_recent_turns=3)
    completer = _Completer('{"facts": []}')
    ext = MemoryExtractor(completer=completer, config=cfg)
    turns = [{"role": "user", "content": f"msg-{i}"} for i in range(10)]
    ext.extract(company_id="a", user_id="u", conversation_id="c",
                recent_turns=turns)
    user_msg = completer.last_messages[1]["content"]
    assert "msg-9" in user_msg
    assert "msg-8" in user_msg
    assert "msg-7" in user_msg
    assert "msg-6" not in user_msg


def test_should_run_this_turn_debouncing():
    cfg = ExtractorConfig(enabled=True, every_n_turns=3)
    ext = MemoryExtractor(completer=_Completer('{"facts": []}'), config=cfg)
    assert ext.should_run_this_turn(1) is True
    assert ext.should_run_this_turn(2) is False
    assert ext.should_run_this_turn(3) is True
    assert ext.should_run_this_turn(4) is False
    assert ext.should_run_this_turn(5) is False
    assert ext.should_run_this_turn(6) is True


def test_should_run_returns_false_when_disabled():
    cfg = ExtractorConfig(enabled=False, every_n_turns=1)
    ext = MemoryExtractor(completer=_Completer('{"facts": []}'), config=cfg)
    assert ext.should_run_this_turn(1) is False
    assert ext.should_run_this_turn(99) is False



def test_build_extraction_messages_shape():
    msgs = build_extraction_messages(
        recent_turns=[
            {"role": "user", "content": "Я Никита"},
            {"role": "assistant", "content": "Привет"},
        ],
        known_keys=["identity.name"],
        user_locale="ru-RU",
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].lstrip().startswith("/no_think")
    assert "CATEGORY TAXONOMY" in msgs[0]["content"]
    assert "LANGUAGE POLICY" in msgs[0]["content"]
    assert "Никита" in msgs[0]["content"]
    assert "Alice"  in msgs[0]["content"]
    assert "李伟"    in msgs[0]["content"]
    assert "Diego"  in msgs[0]["content"]
    assert "ru-RU" in msgs[0]["content"]
    assert "known_keys" in msgs[1]["content"]
    assert "identity.name" in msgs[1]["content"]
    assert msgs[1]["content"].lstrip().startswith("/no_think")


def test_build_messages_filters_empty_turns():
    msgs = build_extraction_messages(
        recent_turns=[
            {"role": "user", "content": ""},
            {"role": "user", "content": "  "},
            {"role": "user", "content": "Привет"},
        ],
        known_keys=[],
    )
    assert "Привет" in msgs[1]["content"]
    assert "(empty)" not in msgs[1]["content"]


def test_build_messages_empty_marker_when_all_empty():
    msgs = build_extraction_messages(
        recent_turns=[{"role": "user", "content": ""}],
        known_keys=[],
    )
    assert "(empty)" in msgs[1]["content"]
