# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Тесты language-agnostic-промпта MemoryExtractor.

Мы не вызываем живую LLM (это интеграционный путь). Вместо этого мы:
  * проверяем, что system-промпт реально многоязычный по содержанию (есть
    все 4 few-shot — RU/EN/ZH/ES, есть инструкция ``LANGUAGE POLICY``);
  * через **fake-LLM** имитируем ответы модели на разных языках и убеждаемся,
    что parser сохраняет value/evidence в исходной кодировке (UTF-8) и
    нормализует ключи в latin/snake-with-dots независимо от языка;
  * проверяем, что ``user_locale`` — это soft hint, а не жёсткое требование.
"""

from __future__ import annotations

import json

from finkey_memory.extractor import (
    ExtractorConfig,
    MemoryExtractor,
    normalize_key,
    parse_extractor_response,
)
from finkey_memory.extractor_prompts import build_extraction_messages




def test_system_prompt_contains_language_policy_and_four_examples():
    """Промпт должен явно инструктировать модель работать с любым языком + иметь 4 few-shot языков."""
    msgs = build_extraction_messages(
        recent_turns=[{"role": "user", "content": "test"}],
        known_keys=[],
    )
    system = msgs[0]["content"]
    assert "LANGUAGE POLICY" in system
    assert "CATEGORY TAXONOMY" in system

    assert "Никита" in system
    assert "Alice"  in system
    assert "李伟"    in system
    assert "Diego"  in system
    assert '{"facts": []}' in system


def test_user_locale_is_soft_hint_not_overriding_dialogue():
    """user_locale попадает как HINT, не как hard-override."""
    msgs = build_extraction_messages(
        recent_turns=[{"role": "user", "content": "Hello, I'm Bob"}],
        known_keys=[],
        user_locale="ja-JP",
    )
    system = msgs[0]["content"]
    assert "ja-JP" in system
    assert "HINT" in system
    assert "trust the dialogue" in system




def test_normalize_key_keeps_ascii_dotted():
    assert normalize_key("identity.name") == "identity.name"
    assert normalize_key("Identity.Name") == "identity.name"


def test_normalize_key_strips_non_ascii_to_dots():
    """Даже если LLM прислала ключ на нелатинице — нормализатор должен это пережить."""
    result = normalize_key("идентичность.имя")
    assert result == "" or all(ord(c) < 128 for c in result), \
        "key MUST be ASCII regardless of input language"
    result_mixed = normalize_key("identity.имя.name")
    assert all(ord(c) < 128 for c in result_mixed), \
        "mixed-language key must be reduced to ASCII"
    assert "identity" in result_mixed and "name" in result_mixed




def _payload(facts: list[dict]) -> str:
    return json.dumps({"facts": facts}, ensure_ascii=False)


def test_parser_preserves_values_in_any_language():
    raw = _payload([
        {"key": "identity.name", "value": "Никита", "category": "identity",
         "confidence": 0.97, "evidence": "меня зовут Никита"},
        {"key": "identity.name", "value": "李伟", "category": "identity",
         "confidence": 0.96, "evidence": "我叫李伟"},
        {"key": "identity.name", "value": "حسن", "category": "identity",
         "confidence": 0.95, "evidence": "اسمي حسن"},
        {"key": "identity.location", "value": "Madrid", "category": "identity",
         "confidence": 0.95, "evidence": "vivo en Madrid"},
        {"key": "identity.name", "value": "佐藤健", "category": "identity",
         "confidence": 0.96, "evidence": "私は佐藤健です"},
    ])
    facts, ok = parse_extractor_response(raw)
    assert ok is True
    assert len(facts) == 5
    stored = {f.value for f in facts}
    assert {"Никита", "李伟", "حسن", "Madrid", "佐藤健"}.issubset(stored)


def test_parser_handles_mixed_language_dialogue_facts():
    """Реальный кейс: один диалог содержит факты в нескольких языках."""
    raw = _payload([
        {"key": "identity.name", "value": "Maria", "category": "identity",
         "confidence": 0.95, "evidence": "I'm Maria"},
        {"key": "context.project.name", "value": "Проект Аврора",
         "category": "context", "confidence": 0.88,
         "evidence": "наш проект Аврора"},
        {"key": "affiliation.company", "value": "字节跳动",
         "category": "affiliation", "confidence": 0.9,
         "evidence": "字节跳动的产品经理"},
    ])
    facts, ok = parse_extractor_response(raw)
    assert ok and len(facts) == 3
    by_key = {f.key_normalized: f for f in facts}
    assert by_key["identity.name"].value == "Maria"
    assert by_key["context.project.name"].value == "Проект Аврора"
    assert by_key["affiliation.company"].value == "字节跳动"




def _make_extractor(payload: str) -> MemoryExtractor:
    cfg = ExtractorConfig(
        enabled=True, model_label="fake-multilingual",
        max_tokens=400, temperature=0.0, min_confidence=0.0, max_facts=20,
    )
    return MemoryExtractor(completer=lambda *_a, **_kw: payload, config=cfg)


def test_extract_arabic_dialogue_returns_arabic_values():
    payload = _payload([
        {"key": "identity.name", "value": "حسن", "category": "identity",
         "confidence": 0.97, "evidence": "اسمي حسن"},
        {"key": "affiliation.company", "value": "أرامكو", "category": "affiliation",
         "confidence": 0.93, "evidence": "أعمل في أرامكو"},
    ])
    ext = _make_extractor(payload)
    res = ext.extract(
        company_id="acme", user_id="u-ar", conversation_id="c-1",
        recent_turns=[{"role": "user", "content": "مرحبا، اسمي حسن، أعمل في أرامكو."}],
        known_keys=set(),
        user_locale="ar-SA",
    )
    assert res.parse_ok and len(res.facts) == 2
    values = {f.value for f in res.facts}
    assert "حسن" in values
    assert "أرامكو" in values


def test_extract_hindi_dialogue_returns_devanagari_values():
    payload = _payload([
        {"key": "identity.name", "value": "अनिल", "category": "identity",
         "confidence": 0.97, "evidence": "मेरा नाम अनिल है"},
        {"key": "context.project.description", "value": "मोबाइल ऐप", "category": "context",
         "confidence": 0.85, "evidence": "हम एक मोबाइल ऐप बना रहे हैं"},
    ])
    ext = _make_extractor(payload)
    res = ext.extract(
        company_id="acme", user_id="u-hi", conversation_id="c-1",
        recent_turns=[{"role": "user", "content": "नमस्ते, मेरा नाम अनिल है। हम एक मोबाइल ऐप बना रहे हैं।"}],
        known_keys=set(),
    )
    assert res.parse_ok and len(res.facts) == 2
    assert res.facts[0].value == "अनिल"


def test_extract_kazakh_dialogue_keys_remain_ascii():
    """Даже если факт о казахоязычном юзере — key всё равно ASCII."""
    payload = _payload([
        {"key": "identity.name", "value": "Айнур", "category": "identity",
         "confidence": 0.96, "evidence": "менің атым Айнур"},
        {"key": "affiliation.company", "value": "Kaspi.kz", "category": "affiliation",
         "confidence": 0.95, "evidence": "Kaspi.kz-те жұмыс істеймін"},
    ])
    ext = _make_extractor(payload)
    res = ext.extract(
        company_id="acme", user_id="u-kk", conversation_id="c-1",
        recent_turns=[{"role": "user", "content": "Сәлем, менің атым Айнур, Kaspi.kz-те жұмыс істеймін."}],
        known_keys=set(),
    )
    assert res.parse_ok and len(res.facts) == 2
    for f in res.facts:
        assert all(ord(c) < 128 for c in f.key_normalized), \
            f"key must be pure ASCII regardless of user's language, got {f.key_normalized!r}"
