# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Минимальный PII-редактор до индексации (телефоны/email/номера карт — маскирование).
Расширяй правила без тяжёлых NLP-зависимостей.

``mask_sensitive`` — единственная точка входа для всех callers (safety rails,
rag pipeline, memory fact_store). При ``FINKEY_PII_BACKEND=presidio`` она
дополнительно прогоняет текст через локальный Presidio-бэкенд
(``finkey_memory.pii_presidio``); без флага или при любой ошибке Presidio
поведение остаётся строго regex-ным.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PIIMaskStats:
    phone_hits: int = 0
    email_hits: int = 0
    card_hits: int = 0
    inn_kz_hints: int = 0


_RE_PHONE = re.compile(r"(\+?7|8)?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b")
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_RE_CARD = re.compile(r"\b(?:\d[ -]*?){13,16}\d\b")
_RE_INN = re.compile(r"\b(?:ИНН|инн|TIN)[\s:]*(\d{10}|\d{12})\b", re.I)


def _mask_regex(text: str, *, aggressive: bool = True) -> tuple[str, PIIMaskStats]:
    """
    Возвращает очищенный текст и счётчики срабатываний.
    ``aggressive`` — маскировать подозрительные длинные цифровые блоки как карты (осторожно с артикулами).
    """
    s = text
    phone_hits = 0
    for m in list(_RE_PHONE.finditer(s)):
        phone_hits += 1
        s = s.replace(m.group(0), "[PHONE]")
    email_hits = 0

    def _em(sub: re.Match[str]) -> str:
        nonlocal email_hits
        email_hits += 1
        return "[EMAIL]"

    s = _RE_EMAIL.sub(_em, s)
    card_hits = 0
    if aggressive:
        for m in list(_RE_CARD.finditer(s)):
            raw = re.sub(r"\D", "", m.group(0))
            if len(raw) >= 13:
                card_hits += 1
                s = s.replace(m.group(0), "[CARD]")
    inn = 0
    for _ in _RE_INN.finditer(s):
        inn += 1
    stats = PIIMaskStats(
        phone_hits=phone_hits,
        email_hits=email_hits,
        card_hits=card_hits,
        inn_kz_hints=inn,
    )
    return s, stats


def mask_sensitive(text: str, *, aggressive: bool = True) -> tuple[str, PIIMaskStats]:
    """Единая точка входа. По умолчанию — regex; при ``FINKEY_PII_BACKEND=presidio``
    дополнительно подключается Presidio-бэкенд (aditive слой; при его недоступности
    результат = чистый regex, без исключений)."""
    if (os.getenv("FINKEY_PII_BACKEND") or "").strip().lower() == "presidio":
        try:
            from finkey_memory.rag import pii_presidio

            return pii_presidio.mask_sensitive(text, aggressive=aggressive)
        except Exception as exc:  # pragma: no cover — preserve regex behaviour
            logging.getLogger("finkey.rag.pii").warning(
                "presidio PII backend failed; regex masking used: %s", exc
            )
    return _mask_regex(text, aggressive=aggressive)


def chunk_text_maybe_mask(text: str, *, enabled: bool) -> tuple[str, PIIMaskStats | None]:
    if not enabled:
        return text, None
    return mask_sensitive(text)
