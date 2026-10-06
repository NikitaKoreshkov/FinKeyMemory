# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
LLM-completer для MemoryExtractor — провайдер-агностичный.

Поддерживаемые backend-ы (контракт один — ``callable(messages, max_tokens, temperature) -> str``):

  * ``OpenRouterCompleter`` — POST на ``openrouter.ai/api/v1/chat/completions``,
    использует ``OPENROUTER_API_KEY`` из env, JSON-mode (response_format).
  * ``OllamaCompleter``     — POST на локальный Ollama (``/api/chat``),
    ``format=json`` для строгого JSON.
  * ``OpenAICompleter``     — POST на ``api.openai.com/v1/chat/completions``,
    тот же JSON-mode, на случай если у пользователя нет OpenRouter, но есть OpenAI.

Унифицированное поведение:
  * Если ``FINKEY_MEMORY_EXTRACTOR_BACKEND`` не задан — auto-detect:
      OPENROUTER_API_KEY → openrouter
      OPENAI_API_KEY     → openai
      Ollama doable      → ollama   (по умолчанию для локалки)
  * Если ``FINKEY_MEMORY_EXTRACTOR_MODEL`` не задан — берём ту же модель, что и
    основной чат (``OPENROUTER_MODEL`` / ``OLLAMA_MODEL``), чтобы качество
    память == качество ответа.
  * Перевод slug-а между провайдерами: ``qwen/qwen3.7-flash`` ↔ ``qwen3.5:9b`` (Ollama — свой тег).
    Делает ``resolve_model_for_backend`` (см. ниже) — но если юзер явно задал
    модель под нужный backend, она не трогается.

Все клиенты блокирующие; использовать строго из background-потока
(см. ``AsyncExtractorRunner``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger("finkey.memory.completer")


def _default_ssl_context() -> ssl.SSLContext:
    """
    На свежих сборках macOS / Python 3.14 системные CA-roots для urllib часто
    отсутствуют — handshake к ``openrouter.ai`` падает с
    ``CERTIFICATE_VERIFY_FAILED``. Берём пакет ``certifi`` (всегда есть в venv
    gateway), если он импортируется; иначе — системный default.
    """
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL_CTX = _default_ssl_context()

# Keep in sync with llm-gateway openrouter_provider._QWEN3_HYBRID_THINK_PREFIXES.
# qwen/qwen3.7-flash is the VPS default: without effort=none it spends the whole
# max_tokens budget on hidden reasoning and leaves content empty.
_QWEN3_HYBRID_THINK_PREFIXES = (
    "qwen/qwen3.8",
    "qwen/qwen3.7",
    "qwen/qwen3.6",
    "qwen/qwen3.5",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-30b-a3b",
    "qwen/qwen3-14b",
    "qwen/qwen3-8b",
    "qwen/qwen3.5-9b",
    "qwen/qwen3-1.7b",
    "qwen/qwen3-0.6b",
)

# OpenRouter 400 if we send reasoning.effort=none for these (Alibaba Qwen3.8*).
_QWEN3_MANDATORY_REASONING_PREFIXES = (
    "qwen/qwen3.8",
)


def _is_qwen3_hybrid_openrouter_slug(model: str) -> bool:
    mb = (model or "").split(":")[0].strip().lower()
    return any(mb.startswith(p) for p in _QWEN3_HYBRID_THINK_PREFIXES)


def _qwen3_reasoning_is_mandatory(model: str) -> bool:
    mb = (model or "").split(":")[0].strip().lower()
    return any(mb.startswith(p) for p in _QWEN3_MANDATORY_REASONING_PREFIXES)


def openrouter_think_off_payload(model: str) -> dict:
    """Stop Qwen 3.7 from spending max_tokens on hidden reasoning.

    ``reasoning.exclude`` only hides the trace — content stays empty.
    ``effort: none`` is what the chat completer already uses on VPS.
    """
    if not _is_qwen3_hybrid_openrouter_slug(model):
        return {}
    if _qwen3_reasoning_is_mandatory(model):
        return {}
    raw_sup = (os.getenv("FINKEY_OPENROUTER_QWEN_SUPPRESS_REASONING", "1") or "1").strip().lower()
    if raw_sup in ("0", "false", "no", "off"):
        return {}
    return {"reasoning": {"effort": "none"}, "enable_thinking": False}


def openrouter_think_on_payload(model: str, *, max_tokens: int = 2048) -> dict:
    """Enable hybrid thinking with a hard cap so JSON/content still fits.

    OpenRouter counts reasoning toward ``max_tokens``. Uncapped Qwen 3.7
    can spend the whole budget on hidden thought and truncate the answer.
    """
    cap = max(128, int(max_tokens or 2048))
    if _qwen3_reasoning_is_mandatory(model):
        return {"reasoning": {"enabled": True, "max_tokens": cap}}
    if not _is_qwen3_hybrid_openrouter_slug(model):
        return {"reasoning": {"max_tokens": cap}}
    return {"reasoning": {"max_tokens": cap}, "enable_thinking": True}


def _inject_think_control(messages: list[dict], *, thinking: bool, model: str) -> list[dict]:
    """Prepend /think or /no_think on the last user turn for Qwen3 hybrid slugs."""
    if not _is_qwen3_hybrid_openrouter_slug(model):
        return messages
    if _qwen3_reasoning_is_mandatory(model):
        thinking = True
    token = "/think" if thinking else "/no_think"
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") != "user":
            continue
        content = out[i].get("content", "")
        if isinstance(content, str):
            stripped = content.lstrip()
            if stripped.startswith("/think") or stripped.startswith("/no_think"):
                return out
            out[i] = {**out[i], "content": token + "\n" + content}
        break
    return out


_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENAI_URL     = "https://api.openai.com/v1/chat/completions"


def _ollama_chat_url() -> str:
    return (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/") + "/api/chat"


def _ollama_base_url() -> str:
    return (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")


_RETRYABLE_CODES = (429, 500, 502, 503, 504)
_DEFAULT_RETRY_DELAY_SEC = 1.5
_MAX_RETRY_DELAY_SEC     = 6.0
_DEFAULT_RETRIES         = 3


def _build_openrouter_provider_block() -> Optional[dict]:
    """
    Тот же провайдер-роутинг для OpenRouter, что в основном чате (см.
    ``app/providers/openrouter_provider.py:_build_provider_routing_block``).
    Читает env: OPENROUTER_PROVIDER_ORDER / IGNORE / SORT / ALLOW_FALLBACKS.
    """
    def _csv(name: str) -> list[str]:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return []
        return [s.strip().lower() for s in raw.split(",") if s.strip()]

    order   = _csv("OPENROUTER_PROVIDER_ORDER")
    ignore  = _csv("OPENROUTER_PROVIDER_IGNORE")
    sort_r  = (os.getenv("OPENROUTER_PROVIDER_SORT") or "").strip().lower()
    allow_r = (os.getenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS") or "").strip().lower()

    if not order and not ignore and not sort_r and not allow_r:
        return None

    block: dict = {}
    if order:
        block["order"] = order
    if ignore:
        block["ignore"] = ignore
    if sort_r in ("price", "throughput", "latency"):
        block["sort"] = sort_r
    block["allow_fallbacks"] = False if allow_r in ("0", "false", "no", "off") else True
    return block


def _parse_retry_after_seconds(body: str) -> float:
    """Если OpenRouter сам подсказал retry_after_seconds — берём оттуда, иначе дефолт."""
    try:
        data = json.loads(body)
        meta = data.get("error", {}).get("metadata", {}) or {}
        for key in ("retry_after_seconds_raw", "retry_after_seconds"):
            raw = meta.get(key)
            if raw is not None:
                return max(_DEFAULT_RETRY_DELAY_SEC, min(float(raw) + 1.0, _MAX_RETRY_DELAY_SEC))
    except Exception:
        pass
    return _DEFAULT_RETRY_DELAY_SEC


def _normalize_openrouter_content(raw_content: object) -> str:
    """Нормализует ``message.content`` (строка или multimodal list) в plain text."""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, dict):
                text_part = item.get("text")
                if isinstance(text_part, str):
                    parts.append(text_part)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    if isinstance(raw_content, (int, float, bool)):
        return ""
    return str(raw_content or "")


_PLACEHOLDER_CONTENT = re.compile(
    r"^(?:null|none|undefined|\[\]|\{\}|[\(\[]\s*\d+\s*[\]\)]|\d{1,3})$",
    re.I,
)


def _placeholder_openrouter_content(text: str) -> bool:
    """Qwen sometimes leaves content as [0] / [] while JSON sits in reasoning."""
    return bool(_PLACEHOLDER_CONTENT.match((text or "").strip()))


def _reasoning_from_openrouter_message(message: dict) -> str:
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    details = message.get("reasoning_details")
    if isinstance(details, list):
        parts_rd: list[str] = []
        for item in details:
            if isinstance(item, dict) and item.get("type") == "reasoning.text":
                tx = item.get("text")
                if isinstance(tx, str) and tx:
                    parts_rd.append(tx)
        if parts_rd:
            return "".join(parts_rd).strip()
    for key in ("thinking", "thought"):
        t = message.get(key)
        if isinstance(t, str) and t.strip():
            return t.strip()
    return ""


def _json_parses(text: str) -> bool:
    blob = (text or "").strip()
    if not blob:
        return False
    try:
        json.loads(blob)
        return True
    except Exception:
        return False


def _first_parseable_json(*blobs: str) -> str:
    fallback = ""
    for blob in blobs:
        text = (blob or "").strip()
        if not text:
            continue
        if not fallback:
            fallback = text
        if _json_parses(text):
            return text
    return fallback


def _extract_openrouter_assistant_text(message: dict | None, *, json_mode: bool) -> str:
    """
    Достаёт финальный текст ответа ассистента из non-stream ``chat/completions``.

    Qwen 3.x на OpenRouter при ``response_format: json_object`` иногда кладёт
    валидный JSON **только** в ``reasoning`` / ``reasoning_details``, оставляя
    ``content`` пустым (см. ``OpenRouterProvider.generate`` — тот же кейс для
    классификатора). Для MemoryExtractor это критично: пустой ``content`` →
    ``parse_ok=False`` и ноль фактов в PG.

    If thinking truncates ``content`` mid-JSON, prefer a parseable blob from
    reasoning instead of the broken prefix.
    """
    if not isinstance(message, dict):
        return ""
    content = _normalize_openrouter_content(message.get("content")).strip()
    if _placeholder_openrouter_content(content):
        content = ""
    reasoning = _reasoning_from_openrouter_message(message)
    if json_mode:
        picked = _first_parseable_json(content, reasoning)
        if picked:
            return picked
        return content or reasoning
    if "{" in content and (_json_parses(content) or not _json_parses(reasoning)):
        return content
    if _json_parses(reasoning):
        return reasoning
    if "{" in content:
        return content
    if "{" in reasoning:
        return reasoning
    if content:
        return content
    return reasoning




class OpenRouterCompleter:
    """
    Простой блокирующий клиент к OpenRouter Chat Completions.

    Использовать строго из background-потока (см. ``AsyncExtractorRunner``).
    """

    def __init__(
        self,
        *,
        model:      str,
        api_key:    Optional[str] = None,
        timeout:    float        = 30.0,
        json_mode:  bool         = True,
        site_url:   Optional[str] = None,
        app_name:   str          = "FinKey",
    ) -> None:
        self._model    = model
        self._api_key  = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        self._timeout  = max(2.0, float(timeout))
        self._json_mode = bool(json_mode)
        self._site_url = (site_url or os.getenv("OPENROUTER_SITE_URL", "")).strip()
        self._app_name = app_name

    @property
    def model(self) -> str:
        return self._model

    def __call__(
        self,
        messages: list[dict],
        *,
        max_tokens:  int,
        temperature: float,
        thinking: bool | None = None,
        reasoning_max_tokens: int | None = None,
    ) -> str:
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set; cannot call OpenRouter completer")

        working = list(messages)
        payload: dict = {
            "model":       self._model,
            "max_tokens":  int(max_tokens),
            "temperature": float(temperature),
        }
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}

        if thinking is True:
            working = _inject_think_control(working, thinking=True, model=self._model)
            cap = int(reasoning_max_tokens or 2048)
            payload.update(openrouter_think_on_payload(self._model, max_tokens=cap))
        else:
            payload.update(openrouter_think_off_payload(self._model))
            if (os.getenv("FINKEY_MEMORY_EXTRACTOR_OPENROUTER_REASONING_EXCLUDE", "") or "").strip().lower() in (
                "1", "true", "yes", "on",
            ):
                payload["reasoning"] = {"exclude": True}

        payload["messages"] = working

        provider_block = _build_openrouter_provider_block()
        if provider_block:
            payload["provider"] = provider_block

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        def _build_request():
            return urllib.request.Request(
                _OPENROUTER_URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type":  "application/json",
                    "HTTP-Referer":  self._site_url,
                    "X-Title":       self._app_name,
                },
                method="POST",
            )

        last_exc: Optional[BaseException] = None
        last_detail = ""
        last_code = -1
        for attempt in range(_DEFAULT_RETRIES):
            try:
                with urllib.request.urlopen(_build_request(), timeout=self._timeout, context=_SSL_CTX) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                break
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                last_exc, last_detail, last_code = e, detail, int(e.code)
                if e.code in _RETRYABLE_CODES and attempt < _DEFAULT_RETRIES - 1:
                    delay = _parse_retry_after_seconds(detail)
                    logger.warning(
                        "MemoryExtractor OpenRouter HTTP %d (attempt %d/%d), retry in %.1fs",
                        e.code, attempt + 1, _DEFAULT_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"OpenRouter HTTP {e.code}: {detail}") from e
            except urllib.error.URLError as e:
                raise RuntimeError(f"OpenRouter URL error: {e.reason}") from e
        else:
            raise RuntimeError(f"OpenRouter HTTP {last_code}: {last_detail}") from last_exc

        try:
            data = json.loads(raw)
            message = (data.get("choices") or [{}])[0].get("message") or {}
            out = _extract_openrouter_assistant_text(message, json_mode=self._json_mode)
            if not out.strip():
                logger.warning(
                    "OpenRouterCompleter: empty assistant text (json_mode=%s model=%s msg_keys=%s)",
                    self._json_mode,
                    self._model,
                    list(message.keys())[:16],
                )
            return out
        except Exception as exc:
            raise RuntimeError(f"OpenRouter bad response: {exc}; raw={raw[:200]}") from exc




class OllamaCompleter:
    """
    Блокирующий клиент к локальному Ollama (POST /api/chat).
    Полезно для дев-режима, чтобы не жечь токены OpenRouter.
    """

    def __init__(
        self,
        *,
        model:   str,
        timeout: float = 60.0,
    ) -> None:
        self._model   = model
        self._timeout = max(2.0, float(timeout))

    @property
    def model(self) -> str:
        return self._model

    def __call__(
        self,
        messages: list[dict],
        *,
        max_tokens:  int,
        temperature: float,
    ) -> str:
        payload = {
            "model":    self._model,
            "messages": messages,
            "options":  {"temperature": float(temperature), "num_predict": int(max_tokens)},
            "format":   "json",
            "stream":   False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _ollama_chat_url(),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout, context=_SSL_CTX) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama unreachable: {e.reason}") from e
        try:
            data = json.loads(raw)
            return (data.get("message") or {}).get("content", "") or ""
        except Exception as exc:
            raise RuntimeError(f"Ollama bad response: {exc}; raw={raw[:200]}") from exc




class OpenAICompleter:
    """Тот же контракт, что OpenRouter — нужный для пользователей без OpenRouter."""

    def __init__(
        self,
        *,
        model:   str,
        api_key: Optional[str] = None,
        timeout: float         = 30.0,
        json_mode: bool        = True,
    ) -> None:
        self._model   = model
        self._api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self._timeout = max(2.0, float(timeout))
        self._json_mode = bool(json_mode)

    @property
    def model(self) -> str:
        return self._model

    def __call__(self, messages: list[dict], *, max_tokens: int, temperature: float) -> str:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot call OpenAI completer")

        payload: dict = {
            "model":       self._model,
            "messages":    messages,
            "max_tokens":  int(max_tokens),
            "temperature": float(temperature),
        }
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _OPENAI_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type":  "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout, context=_SSL_CTX) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise RuntimeError(f"OpenAI HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenAI URL error: {e.reason}") from e

        try:
            data = json.loads(raw)
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        except Exception as exc:
            raise RuntimeError(f"OpenAI bad response: {exc}; raw={raw[:200]}") from exc




_DEFAULT_MODELS = {
    "openrouter": "qwen/qwen3.7-flash",
    "openai":     "gpt-4o-mini",
    "ollama":     "qwen3.5:9b",
}


def detect_extractor_backend() -> str:
    """
    Авто-выбор backend'а для extractor'а.

    Приоритет (если ни одно явно не задано):
      1. ``FINKEY_MEMORY_EXTRACTOR_BACKEND`` (явный override).
      2. ``OPENROUTER_API_KEY`` → openrouter (cloud, тот же что основной чат).
      3. ``OPENAI_API_KEY``     → openai     (cloud fallback).
      4. otherwise              → ollama     (локальный fallback).
    """
    explicit = (os.getenv("FINKEY_MEMORY_EXTRACTOR_BACKEND") or "").strip().lower()
    if explicit in _DEFAULT_MODELS:
        return explicit
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return "openrouter"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    return "ollama"


def resolve_model_for_backend(backend: str) -> str:
    """
    Резолвим slug модели под выбранный backend.

    Если ``FINKEY_MEMORY_EXTRACTOR_MODEL`` задан — используем его как есть
    (пользователь знает, что делает).

    Если не задан — пытаемся подхватить slug **основной чат-модели**, чтобы
    extractor работал на том же мозге, что и юзер видит в ответе:
      * openrouter → ``OPENROUTER_MODEL`` (например ``qwen/qwen3.7-flash``)
      * ollama     → ``OLLAMA_MODEL``     (например ``qwen3.5:9b``)
      * openai     → ``OPENAI_MODEL`` (если задана) или ``gpt-4o-mini``

    Иначе — дефолт OpenRouter ``qwen/qwen3.7-flash`` / Ollama ``qwen3.5:9b``.
    """
    explicit = (os.getenv("FINKEY_MEMORY_EXTRACTOR_MODEL") or "").strip()
    if explicit:
        return explicit
    if backend == "openrouter":
        return (os.getenv("OPENROUTER_MODEL") or _DEFAULT_MODELS["openrouter"]).strip()
    if backend == "ollama":
        return (os.getenv("OLLAMA_MODEL") or _DEFAULT_MODELS["ollama"]).strip()
    if backend == "openai":
        return (os.getenv("OPENAI_MODEL") or _DEFAULT_MODELS["openai"]).strip()
    return _DEFAULT_MODELS.get(backend, _DEFAULT_MODELS["openrouter"])




def build_extractor_completer_from_env():
    """
    Унифицированная фабрика completer'а.

    Контракт:
      * Возвращает callable ``(messages, max_tokens, temperature) -> str``.
      * Никаких побочных эффектов на построение — конструкторы лениво читают env.
      * Любой backend (openrouter / openai / ollama) выдаёт JSON-ready ответ,
        который parser отдельно расхлёбывает в ``ExtractedFact``-ы.

    Поведение задаётся через env (см. .env.example):
      FINKEY_MEMORY_EXTRACTOR_BACKEND   openrouter | openai | ollama  (auto)
      FINKEY_MEMORY_EXTRACTOR_MODEL     slug под backend              (auto)
      FINKEY_MEMORY_EXTRACTOR_TIMEOUT_SEC                              30
    """
    backend = detect_extractor_backend()
    model   = resolve_model_for_backend(backend)
    try:
        timeout = float(os.getenv("FINKEY_MEMORY_EXTRACTOR_TIMEOUT_SEC", "30"))
    except ValueError:
        timeout = 30.0

    logger.info("MemoryExtractor backend=%s model=%s timeout=%.1fs", backend, model, timeout)

    if backend == "ollama":
        return OllamaCompleter(model=model, timeout=timeout)
    if backend == "openai":
        return OpenAICompleter(model=model, timeout=timeout)
    return OpenRouterCompleter(model=model, timeout=timeout)
