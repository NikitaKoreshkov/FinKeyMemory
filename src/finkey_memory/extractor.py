# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey MemoryExtractor — LLM-driven автономный выделитель фактов о пользователе.

Идея: после нескольких реплик чата мы прогоняем **отдельный, дешёвый** LLM-call с
жёстко структурированным промптом и просим извлечь только **durable**-факты о
человеке (имя, профессия, текущий проект, цели, постоянные предпочтения, ...).

Этот модуль **чистый** — он не лезет в БД, Redis или Qdrant. Возвращает список
``ExtractedFact``-ов; решение, как их мёрджить и куда писать, принимает ``FactStore``
из Фазы 4 (через ``MemoryManager.upsert_extracted_facts``).

Контракт LLM-клиента — обычная коллабельная функция:

    completer(messages: list[dict], *, max_tokens: int, temperature: float) -> str

Возвращает сырой текст ответа модели. Парсер сам вытащит JSON, даже если модель
завернула его в code-fence или дописала комментарий снаружи.

Все параметры можно крутить через env:

  * ``FINKEY_MEMORY_EXTRACT_ENABLED``       — feature flag, по умолчанию false.
  * ``FINKEY_MEMORY_EXTRACTOR_MODEL``       — slug модели (информативно, для логов).
  * ``FINKEY_MEMORY_EXTRACTOR_MAX_TOKENS``  — лимит output (default 1024).
  * ``FINKEY_MEMORY_EXTRACTOR_TEMPERATURE`` — температура (default 0.0 — строгая).
  * ``FINKEY_MEMORY_EXTRACT_MAX_RECENT``    — сколько реплик показывать (default 8).
  * ``FINKEY_MEMORY_EXTRACT_MIN_CONFIDENCE``— нижний порог отсечки (default 0.35).
  * ``FINKEY_MEMORY_EXTRACT_MAX_FACTS``     — потолок числа фактов за один прогон (default 12).
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from finkey_memory.extractor_prompts import build_extraction_messages
from finkey_memory.metrics import (
    EXTRACTOR_FACTS,
    EXTRACTOR_LATENCY,
    EXTRACTOR_LLM_ERRORS,
    EXTRACTOR_PARSE_FAIL,
    EXTRACTOR_RUNS,
    log_event,
)

logger = logging.getLogger("finkey.memory.extractor")



class FactCategory(str, Enum):
    """
    Таксономия фактов о пользователе.

    Совпадает с CHECK-constraint на ``user_facts.category`` в каноническом PG.
    """
    IDENTITY     = "identity"
    AFFILIATION  = "affiliation"
    CONTEXT      = "context"
    PREFERENCE   = "preference"
    GOAL         = "goal"
    CONSTRAINT   = "constraint"
    COMMITMENT   = "commitment"
    RELATIONSHIP = "relationship"
    STATE        = "state"
    OTHER        = "other"

    @classmethod
    def parse(cls, raw: str) -> "FactCategory":
        v = (raw or "").strip().lower()
        try:
            return cls(v)
        except ValueError:
            return cls.OTHER


_PRIORITY_BY_CATEGORY: dict[FactCategory, float] = {
    FactCategory.IDENTITY:     0.95,
    FactCategory.AFFILIATION:  0.85,
    FactCategory.CONTEXT:      0.80,
    FactCategory.GOAL:         0.75,
    FactCategory.COMMITMENT:   0.70,
    FactCategory.RELATIONSHIP: 0.65,
    FactCategory.PREFERENCE:   0.60,
    FactCategory.CONSTRAINT:   0.60,
    FactCategory.STATE:        0.35,
    FactCategory.OTHER:        0.30,
}


_DEFAULT_TTL_DAYS_BY_CATEGORY: dict[FactCategory, Optional[int]] = {
    FactCategory.STATE:      7,
    FactCategory.COMMITMENT: 14,
    FactCategory.GOAL:       180,
}


@dataclass
class ExtractedFact:
    """Один factоид, выделенный экстрактором."""
    key_normalized:    str
    value:             str
    category:          FactCategory       = FactCategory.OTHER
    confidence:        float              = 0.5
    evidence_snippet:  str                = ""
    priority:          float              = 0.5
    source:            str                = "inferred"
    source_conv_id:    Optional[str]      = None
    source_message_id: Optional[str]      = None
    expires_at:        Optional[datetime] = None
    created_at:        datetime           = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExtractionResult:
    """Результат одного прогона экстрактора."""
    facts:           list[ExtractedFact]
    raw_response:    str
    parse_ok:        bool
    elapsed_ms:      int
    model_label:     str
    skipped_reason:  Optional[str] = None
    fact_count_raw:  int = 0


LLMCompleter = Callable[..., str]



_KEY_ALLOWED_RE = re.compile(r"[^a-z0-9._\-]+")
_KEY_DOTS_RE    = re.compile(r"\.+")


def normalize_key(raw_key: str) -> str:
    """
    Приводим key к канонической форме:
      * lowercase
      * пробелы → `.`
      * любые недопустимые символы → `.`
      * многократные точки → одна
      * trim `.`, `_`, `-` по краям
      * максимум 80 символов

    Примеры::
        "Identity.Name"            → "identity.name"
        "context.project FinKey"   → "context.project.finkey"
        "пользователь:работа"       → "context.work"  (если LLM так и отдал)
    """
    if not raw_key:
        return ""
    k = raw_key.strip().lower()
    k = k.replace(" ", ".")
    k = _KEY_ALLOWED_RE.sub(".", k)
    k = _KEY_DOTS_RE.sub(".", k)
    k = k.strip("._-")
    return k[:80]



_CODE_FENCE_RE  = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_FIRST_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)
_FACTS_OBJ_RE   = re.compile(r"\{\s*\"facts\"\s*:")


def _slice_balanced_brace_object(s: str, start: int) -> Optional[str]:
    """
    Вырезать сбалансированный JSON-объект ``{...}`` с позиции ``start``,
    учитывая строки в двойных кавычках (экранирование ``\\`` / ``\"``).
    """
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _try_parse_json(text: str) -> Optional[dict]:
    import json
    if not text:
        return None
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = _CODE_FENCE_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    for m in _FACTS_OBJ_RE.finditer(s):
        blob = _slice_balanced_brace_object(s, m.start())
        if not blob:
            continue
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    m = _FIRST_BRACE_RE.search(s)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _coerce_float(value: Any, default: float = 0.5) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _clean_text(value: Any, max_len: int) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return s[:max_len]


_TRANSIENT_CHANNEL_NEEDLES = (
    "image_upload",
    "image_load",
    "media_unavail",
    "изображение не загрузил",
    "изображение пуст",
    "не удалось загрузить изображ",
    "image did not load",
    "image is empty",
    "failed to load the image",
)


def _is_transient_channel_glitch(key: str, value: str) -> bool:
    """Do not remember a one-off media fetch miss as a durable user fact.

    Otherwise the next iPhone photo is answered from memory: «фото не загрузилось».
    """
    blob = f"{key} {value}".lower()
    return any(needle in blob for needle in _TRANSIENT_CHANNEL_NEEDLES)


def parse_extractor_response(
    raw: str,
    *,
    min_confidence: float    = 0.35,
    max_facts:      int      = 12,
    source_conv_id: Optional[str] = None,
) -> tuple[list[ExtractedFact], bool]:
    """
    Разбирает ответ модели и возвращает ``(facts, parse_ok)``.

    ``parse_ok=False`` значит, что JSON-структура не распозналась вообще.
    Если структура есть, но отдельные факты невалидны — они тихо отбрасываются,
    а флаг остаётся True.
    """
    obj = _try_parse_json(raw)
    if not isinstance(obj, dict):
        return [], False

    raw_facts = obj.get("facts") or obj.get("items") or []
    if not isinstance(raw_facts, list):
        return [], True

    extracted: list[ExtractedFact] = []
    for rf in raw_facts:
        if not isinstance(rf, dict):
            continue
        key_raw   = rf.get("key") or rf.get("key_normalized") or ""
        value_raw = rf.get("value") or ""
        ev_raw    = rf.get("evidence") or rf.get("evidence_snippet") or ""
        if not key_raw or not value_raw:
            continue
        key   = normalize_key(str(key_raw))
        if not key:
            continue
        value = _clean_text(value_raw, 512)
        if not value:
            continue
        if _is_transient_channel_glitch(key, value):
            continue
        ev    = _clean_text(ev_raw, 512)
        cat   = FactCategory.parse(rf.get("category") or "")
        conf  = _coerce_float(rf.get("confidence"), default=0.5)
        if conf < min_confidence:
            continue
        prio  = _coerce_float(rf.get("priority"), default=_PRIORITY_BY_CATEGORY.get(cat, 0.5))

        ttl_days = _DEFAULT_TTL_DAYS_BY_CATEGORY.get(cat)
        expires_at: Optional[datetime] = None
        if ttl_days:
            expires_at = datetime.now(timezone.utc).replace(microsecond=0)
            from datetime import timedelta
            expires_at = expires_at + timedelta(days=ttl_days)

        extracted.append(ExtractedFact(
            key_normalized   = key,
            value            = value,
            category         = cat,
            confidence       = conf,
            evidence_snippet = ev,
            priority         = prio,
            source           = "inferred",
            source_conv_id   = source_conv_id,
            expires_at       = expires_at,
        ))
        if len(extracted) >= max_facts:
            break
    return extracted, True




def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


@dataclass
class ExtractorConfig:
    enabled:            bool   = field(default_factory=lambda: _env_bool("FINKEY_MEMORY_EXTRACT_ENABLED", True))
    model_label:        str    = field(default_factory=lambda: os.getenv("FINKEY_MEMORY_EXTRACTOR_MODEL", "qwen/qwen3.7-flash"))
    max_tokens:         int    = field(default_factory=lambda: _env_int("FINKEY_MEMORY_EXTRACTOR_MAX_TOKENS", 1024))
    temperature:        float  = field(default_factory=lambda: _env_float("FINKEY_MEMORY_EXTRACTOR_TEMPERATURE", 0.0))
    max_recent_turns:   int    = field(default_factory=lambda: _env_int("FINKEY_MEMORY_EXTRACT_MAX_RECENT", 8))
    min_confidence:     float  = field(default_factory=lambda: _env_float("FINKEY_MEMORY_EXTRACT_MIN_CONFIDENCE", 0.35))
    max_facts:          int    = field(default_factory=lambda: _env_int("FINKEY_MEMORY_EXTRACT_MAX_FACTS", 12))
    every_n_turns:      int    = field(default_factory=lambda: _env_int("FINKEY_MEMORY_EXTRACT_EVERY_N_TURNS", 3))




class MemoryExtractor:
    """
    Извлекает durable-факты о пользователе из последних реплик диалога.

    Пример::

        def my_completer(messages, *, max_tokens, temperature):
            return openrouter_chat(messages, max_tokens=max_tokens, temperature=temperature)

        ext = MemoryExtractor(completer=my_completer)
        result = ext.extract(
            company_id="acme", user_id="u-1", conversation_id="c-1",
            recent_turns=[
                {"role": "user", "content": "Привет, я Никита, фаундер FinKey"},
                {"role": "assistant", "content": "Здравствуй!"},
                {"role": "user", "content": "Мы строим AI с эмпатией"},
            ],
            known_keys=set(),
        )
        for f in result.facts:
            print(f.key_normalized, "=", f.value, f"(conf={f.confidence:.2f})")
    """

    def __init__(
        self,
        *,
        completer:   LLMCompleter,
        config:      Optional[ExtractorConfig] = None,
    ) -> None:
        self._completer = completer
        self._config    = config or ExtractorConfig()

    @property
    def config(self) -> ExtractorConfig:
        return self._config

    def extract(
        self,
        *,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
        recent_turns:    list[dict],
        known_keys:      Optional[set[str]] = None,
        user_locale:     Optional[str] = None,
    ) -> ExtractionResult:
        """
        Запускает LLM-call и возвращает структурированный результат.

        ``known_keys`` — нормализованные ключи фактов, которые уже хорошо известны;
        прокидываются в промпт, чтобы модель не плодила дубли (это soft-сигнал, не enforce).
        """
        t0 = time.time()
        cfg = self._config

        if not recent_turns:
            return ExtractionResult(
                facts=[], raw_response="", parse_ok=True,
                elapsed_ms=0, model_label=cfg.model_label,
                skipped_reason="no_turns", fact_count_raw=0,
            )

        clipped = [
            t for t in recent_turns
            if isinstance(t, dict) and (t.get("role") in {"user", "assistant"})
            and isinstance(t.get("content"), str) and t.get("content", "").strip()
        ][-cfg.max_recent_turns:]
        if not clipped:
            return ExtractionResult(
                facts=[], raw_response="", parse_ok=True,
                elapsed_ms=int((time.time() - t0) * 1000),
                model_label=cfg.model_label,
                skipped_reason="no_textual_turns", fact_count_raw=0,
            )

        messages = build_extraction_messages(
            recent_turns = clipped,
            known_keys   = sorted(known_keys or set()),
            user_locale  = user_locale,
        )

        try:
            raw = self._completer(
                messages,
                max_tokens  = cfg.max_tokens,
                temperature = cfg.temperature,
            )
        except Exception as exc:  # noqa: BLE001 — никогда не падаем наружу из extract()
            EXTRACTOR_LLM_ERRORS.inc(company_id=company_id)
            err_txt = str(exc)[:500]
            log_event(
                logger, logging.WARNING, "memory.extractor.llm_error",
                company_id=company_id, user_id=user_id, conv_id=conversation_id,
                error_type=type(exc).__name__, model=cfg.model_label,
                error_message=err_txt,
            )
            logger.warning(
                "memory.extractor.llm_error model=%s err=%s",
                cfg.model_label,
                err_txt,
            )
            return ExtractionResult(
                facts=[], raw_response="", parse_ok=False,
                elapsed_ms=int((time.time() - t0) * 1000),
                model_label=cfg.model_label,
                skipped_reason=f"llm_error:{type(exc).__name__}",
                fact_count_raw=0,
            )

        facts, parse_ok = parse_extractor_response(
            raw or "",
            min_confidence = cfg.min_confidence,
            max_facts      = cfg.max_facts,
            source_conv_id = conversation_id,
        )
        elapsed_sec = time.time() - t0
        elapsed_ms = int(elapsed_sec * 1000)

        EXTRACTOR_RUNS.inc(company_id=company_id)
        EXTRACTOR_LATENCY.observe(elapsed_sec, company_id=company_id)
        if not parse_ok:
            EXTRACTOR_PARSE_FAIL.inc(company_id=company_id)
        if facts:
            EXTRACTOR_FACTS.inc(len(facts), company_id=company_id)
        log_event(
            logger, logging.INFO, "memory.extractor.done",
            company_id=company_id, user_id=user_id, conv_id=conversation_id,
            model=cfg.model_label, parse_ok=parse_ok,
            facts=len(facts), elapsed_ms=elapsed_ms,
        )
        raw_preview = (raw or "").replace("\n", " ")[:220]
        logger.info(
            "memory.extractor.done model=%s parse_ok=%s facts=%d elapsed_ms=%d raw_preview=%s",
            cfg.model_label, parse_ok, len(facts), elapsed_ms, raw_preview,
        )
        return ExtractionResult(
            facts           = facts,
            raw_response    = raw or "",
            parse_ok        = parse_ok,
            elapsed_ms      = elapsed_ms,
            model_label     = cfg.model_label,
            fact_count_raw  = len(facts),
        )

    def should_run_this_turn(self, turn_index: int) -> bool:
        """
        Хелпер для debouncing: возвращает True, если на этом ходу пора запустить экстрактор.

        Логика: бежим **каждые N ходов** (default 3). На первом ходу — да, чтобы
        моментально подхватить имя/представление.
        """
        if not self._config.enabled:
            return False
        if turn_index <= 1:
            return True
        return (turn_index % max(1, self._config.every_n_turns)) == 0




def fact_id() -> str:
    """Удобный генератор UUID для внешних потребителей."""
    return str(uuid.uuid4())
