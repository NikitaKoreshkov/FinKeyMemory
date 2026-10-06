# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Тесты backend-агностичности памяти.

Память должна работать одинаково через любой LLM-провайдер (OpenRouter / OpenAI /
Ollama) и через любой embedder (Ollama bge-m3 / OpenAI / noop). Эти тесты
гарантируют:

  * ``detect_extractor_backend()`` корректно выбирает backend по env.
  * ``resolve_model_for_backend()`` выдаёт правильный slug под каждый backend
    (включая трансляцию типичных Qwen-слагов OpenRouter ↔ Ollama).
  * Каждый completer формирует **одинаковую** структуру запроса
    (system+user messages, JSON-mode, max_tokens, temperature) — независимо
    от провайдера. Это значит, что MemoryExtractor выдаёт идентичный JSON
    локально и в облаке.
  * ``detect_embedder_backend()`` корректно auto-detect: cloud / hybrid / local.
  * ``build_embedder_from_env()`` всегда возвращает callable с ``vector_size``,
    даже когда ничего недоступно (noop-fallback).
  * Реальные HTTP-вызовы не делаются — всё через monkeypatch ``urlopen``.
"""

from __future__ import annotations

import io
import json
import time
from contextlib import contextmanager
from typing import Any

import pytest

from finkey_memory import embedder as emb_mod
from finkey_memory import llm_completer as llm_mod




class _Resp:
    """Минимальный context-manager, подражающий ``urllib.urlopen``-response."""

    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status   = status

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _Capture:
    """Перехватывает все исходящие POST-ы из urllib.request.urlopen."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def make_fake_urlopen(self, response_payload: dict | None = None, status: int = 200):
        def _fake(req, timeout=None, context=None):
            url      = getattr(req, "full_url", None) or getattr(req, "_full_url", None) or str(req)
            body     = req.data.decode("utf-8") if getattr(req, "data", None) else ""
            try:
                parsed_body = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed_body = {}
            self.calls.append({
                "url":     url,
                "headers": dict(req.headers) if hasattr(req, "headers") else {},
                "body":    parsed_body,
                "timeout": timeout,
            })
            payload = response_payload if response_payload is not None else {"ok": True}
            return _Resp(payload, status=status)

        return _fake

    def make_fake_session(self, response_payload: dict | None = None, status: int = 200):
        """Фейковая ``requests.Session`` — embedder ходит через keep-alive пул."""
        capture = self

        class _FakeResp:
            def __init__(self, payload: dict, status_code: int) -> None:
                self._payload = payload
                self.status_code = status_code
                self.headers: dict[str, str] = {}
                self.content = json.dumps(payload).encode("utf-8")

            def json(self) -> dict:
                return self._payload

        class _FakeSession:
            def post(self, url, data=None, headers=None, timeout=None):
                try:
                    parsed_body = json.loads(data.decode("utf-8")) if data else {}
                except (json.JSONDecodeError, AttributeError):
                    parsed_body = {}
                capture.calls.append({
                    "url":     url,
                    "headers": dict(headers or {}),
                    "body":    parsed_body,
                    "timeout": timeout,
                })
                payload = response_payload if response_payload is not None else {"ok": True}
                return _FakeResp(payload, status)

        return _FakeSession()


@contextmanager
def _clean_env(monkeypatch: pytest.MonkeyPatch, **env: str | None):
    """
    Изолированно подменяет env-переменные. ``value=None`` означает «удалить».
    """
    known = [
        "FINKEY_MEMORY_EXTRACTOR_BACKEND",
        "FINKEY_MEMORY_EXTRACTOR_MODEL",
        "FINKEY_MEMORY_EXTRACTOR_TIMEOUT_SEC",
        "FINKEY_EMBEDDER_BACKEND",
        "FINKEY_EMBEDDER_MODEL",
        "FINKEY_EMBEDDER_VECTOR_SIZE",
        "FINKEY_EMBEDDER_TIMEOUT_SEC",
        "FINKEY_MEMORY_OPENAI_EMBED",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
    ]
    for k in known:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    yield




class TestExtractorBackendDetection:
    """``detect_extractor_backend()`` — приоритет: explicit → OpenRouter → OpenAI → Ollama."""

    def test_explicit_override_openrouter(self, monkeypatch):
        with _clean_env(monkeypatch, FINKEY_MEMORY_EXTRACTOR_BACKEND="openrouter"):
            assert llm_mod.detect_extractor_backend() == "openrouter"

    def test_explicit_override_ollama(self, monkeypatch):
        with _clean_env(monkeypatch,
                         FINKEY_MEMORY_EXTRACTOR_BACKEND="ollama",
                         OPENROUTER_API_KEY="sk-or-test"):
            assert llm_mod.detect_extractor_backend() == "ollama"

    def test_explicit_override_openai(self, monkeypatch):
        with _clean_env(monkeypatch, FINKEY_MEMORY_EXTRACTOR_BACKEND="openai"):
            assert llm_mod.detect_extractor_backend() == "openai"

    def test_unknown_explicit_falls_through_to_autodetect(self, monkeypatch):
        with _clean_env(monkeypatch,
                         FINKEY_MEMORY_EXTRACTOR_BACKEND="cohere",
                         OPENROUTER_API_KEY="sk-or-test"):
            assert llm_mod.detect_extractor_backend() == "openrouter"

    def test_auto_openrouter_when_only_or_key(self, monkeypatch):
        with _clean_env(monkeypatch, OPENROUTER_API_KEY="sk-or-test"):
            assert llm_mod.detect_extractor_backend() == "openrouter"

    def test_auto_openai_when_only_openai_key(self, monkeypatch):
        with _clean_env(monkeypatch, OPENAI_API_KEY="sk-openai"):
            assert llm_mod.detect_extractor_backend() == "openai"

    def test_auto_ollama_when_nothing_set(self, monkeypatch):
        with _clean_env(monkeypatch):
            assert llm_mod.detect_extractor_backend() == "ollama"

    def test_openrouter_wins_when_both_keys_present(self, monkeypatch):
        with _clean_env(monkeypatch,
                         OPENROUTER_API_KEY="sk-or-test",
                         OPENAI_API_KEY="sk-openai"):
            assert llm_mod.detect_extractor_backend() == "openrouter"




class TestModelResolution:
    """Slug-translation: то же семейство модели на любом backend-е."""

    def test_explicit_model_passes_through_for_openrouter(self, monkeypatch):
        with _clean_env(monkeypatch, FINKEY_MEMORY_EXTRACTOR_MODEL="my/custom-llm"):
            assert llm_mod.resolve_model_for_backend("openrouter") == "my/custom-llm"

    def test_explicit_model_passes_through_for_ollama(self, monkeypatch):
        with _clean_env(monkeypatch, FINKEY_MEMORY_EXTRACTOR_MODEL="custom:tag"):
            assert llm_mod.resolve_model_for_backend("ollama") == "custom:tag"

    def test_openrouter_picks_chat_model_when_extractor_unset(self, monkeypatch):
        with _clean_env(monkeypatch, OPENROUTER_MODEL="qwen/qwen3.7-flash"):
            assert llm_mod.resolve_model_for_backend("openrouter") == "qwen/qwen3.7-flash"

    def test_ollama_picks_chat_model_when_extractor_unset(self, monkeypatch):
        with _clean_env(monkeypatch, OLLAMA_MODEL="qwen3.5:9b"):
            assert llm_mod.resolve_model_for_backend("ollama") == "qwen3.5:9b"

    def test_openrouter_default_is_qwen3p5_9b_slug(self, monkeypatch):
        with _clean_env(monkeypatch):
            assert llm_mod.resolve_model_for_backend("openrouter") == "qwen/qwen3.7-flash"

    def test_ollama_default_is_qwen3p5_9b_tag(self, monkeypatch):
        with _clean_env(monkeypatch):
            assert llm_mod.resolve_model_for_backend("ollama") == "qwen3.5:9b"

    def test_openai_default_is_gpt4o_mini(self, monkeypatch):
        with _clean_env(monkeypatch):
            assert llm_mod.resolve_model_for_backend("openai") == "gpt-4o-mini"

    def test_openai_picks_custom_chat_model(self, monkeypatch):
        with _clean_env(monkeypatch, OPENAI_MODEL="gpt-4o"):
            assert llm_mod.resolve_model_for_backend("openai") == "gpt-4o"




class TestExtractorFactory:
    """Фабрика должна возвращать готовый callable с правильным backend-ом."""

    def test_factory_returns_openrouter_when_or_key_set(self, monkeypatch):
        with _clean_env(monkeypatch, OPENROUTER_API_KEY="sk-or-test"):
            c = llm_mod.build_extractor_completer_from_env()
            assert isinstance(c, llm_mod.OpenRouterCompleter)
            assert c.model == "qwen/qwen3.7-flash"

    def test_factory_returns_openai_when_only_openai_key(self, monkeypatch):
        with _clean_env(monkeypatch, OPENAI_API_KEY="sk-openai"):
            c = llm_mod.build_extractor_completer_from_env()
            assert isinstance(c, llm_mod.OpenAICompleter)
            assert c.model == "gpt-4o-mini"

    def test_factory_returns_ollama_when_no_keys(self, monkeypatch):
        with _clean_env(monkeypatch):
            c = llm_mod.build_extractor_completer_from_env()
            assert isinstance(c, llm_mod.OllamaCompleter)
            assert c.model == "qwen3.5:9b"

    def test_factory_honors_explicit_model_under_any_backend(self, monkeypatch):
        with _clean_env(monkeypatch,
                         FINKEY_MEMORY_EXTRACTOR_BACKEND="ollama",
                         FINKEY_MEMORY_EXTRACTOR_MODEL="llama3:8b"):
            c = llm_mod.build_extractor_completer_from_env()
            assert isinstance(c, llm_mod.OllamaCompleter)
            assert c.model == "llama3:8b"




_SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are FinKey memory extractor."},
    {"role": "user",   "content": "Меня зовут Никита."},
]


class TestProviderParity:
    """Все три completer-а должны слать одинаковую структуру сообщений и max_tokens/temperature."""

    def test_openrouter_payload_shape(self, monkeypatch):
        cap = _Capture()
        ok_resp = {"choices": [{"message": {"content": '{"facts":[]}'}}]}
        monkeypatch.setattr("urllib.request.urlopen", cap.make_fake_urlopen(ok_resp))

        c = llm_mod.OpenRouterCompleter(model="qwen/qwen3.5-9b", api_key="sk-or-test")
        out = c(_SAMPLE_MESSAGES, max_tokens=800, temperature=0.1)
        assert out == '{"facts":[]}'
        assert len(cap.calls) == 1
        call = cap.calls[0]
        assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
        assert call["body"]["model"]       == "qwen/qwen3.5-9b"
        assert call["body"]["messages"]    == _SAMPLE_MESSAGES
        assert call["body"]["max_tokens"]  == 800
        assert call["body"]["temperature"] == 0.1
        assert call["body"]["response_format"] == {"type": "json_object"}
        assert call["body"]["reasoning"] == {"effort": "none"}
        headers_lower = {k.lower(): v for k, v in call["headers"].items()}
        assert headers_lower.get("authorization") == "Bearer sk-or-test"

    def test_openrouter_thinking_on_caps_reasoning(self, monkeypatch):
        cap = _Capture()
        ok_resp = {"choices": [{"message": {"content": '{"phase":"apply"}'}}]}
        monkeypatch.setattr("urllib.request.urlopen", cap.make_fake_urlopen(ok_resp))
        c = llm_mod.OpenRouterCompleter(model="qwen/qwen3.7-flash", api_key="sk-or-test")
        out = c(_SAMPLE_MESSAGES, max_tokens=10240, temperature=0.2, thinking=True, reasoning_max_tokens=2048)
        assert out == '{"phase":"apply"}'
        body = cap.calls[0]["body"]
        assert body["reasoning"]["max_tokens"] == 2048
        assert body.get("enable_thinking") is True
        assert str(body["messages"][-1]["content"]).startswith("/think")
        assert body["max_tokens"] == 10240

    def test_openai_payload_shape(self, monkeypatch):
        cap = _Capture()
        ok_resp = {"choices": [{"message": {"content": '{"facts":[]}'}}]}
        monkeypatch.setattr("urllib.request.urlopen", cap.make_fake_urlopen(ok_resp))

        c = llm_mod.OpenAICompleter(model="gpt-4o-mini", api_key="sk-openai")
        out = c(_SAMPLE_MESSAGES, max_tokens=800, temperature=0.1)
        assert out == '{"facts":[]}'
        assert cap.calls[0]["url"] == "https://api.openai.com/v1/chat/completions"
        assert cap.calls[0]["body"]["messages"] == _SAMPLE_MESSAGES
        assert cap.calls[0]["body"]["response_format"] == {"type": "json_object"}

    def test_openrouter_qwen_reasoning_exclude_env_overrides_effort(self, monkeypatch):
        monkeypatch.setenv("FINKEY_MEMORY_EXTRACTOR_OPENROUTER_REASONING_EXCLUDE", "1")
        cap = _Capture()
        ok_resp = {"choices": [{"message": {"content": '{"facts":[]}'}}]}
        monkeypatch.setattr("urllib.request.urlopen", cap.make_fake_urlopen(ok_resp))
        c = llm_mod.OpenRouterCompleter(model="qwen/qwen3.5-9b", api_key="x")
        c(_SAMPLE_MESSAGES, max_tokens=100, temperature=0.0)
        assert cap.calls[0]["body"]["reasoning"] == {"exclude": True}

    def test_openrouter_non_qwen_slug_has_no_reasoning_block(self, monkeypatch):
        cap = _Capture()
        ok_resp = {"choices": [{"message": {"content": '{"facts":[]}'}}]}
        monkeypatch.setattr("urllib.request.urlopen", cap.make_fake_urlopen(ok_resp))
        c = llm_mod.OpenRouterCompleter(model="anthropic/claude-3.5-haiku", api_key="x")
        c(_SAMPLE_MESSAGES, max_tokens=50, temperature=0.0)
        assert "reasoning" not in cap.calls[0]["body"]

    def test_ollama_payload_shape(self, monkeypatch):
        cap = _Capture()
        ok_resp = {"message": {"content": '{"facts":[]}'}}
        monkeypatch.setattr("urllib.request.urlopen", cap.make_fake_urlopen(ok_resp))
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

        c = llm_mod.OllamaCompleter(model="qwen3.5:9b")
        out = c(_SAMPLE_MESSAGES, max_tokens=800, temperature=0.1)
        assert out == '{"facts":[]}'
        call = cap.calls[0]
        assert call["url"].endswith("/api/chat")
        assert call["body"]["model"]    == "qwen3.5:9b"
        assert call["body"]["messages"] == _SAMPLE_MESSAGES
        assert call["body"]["format"]   == "json"
        assert call["body"]["options"]["num_predict"]  == 800
        assert call["body"]["options"]["temperature"]  == 0.1

    def test_all_backends_produce_identical_extracted_json(self, monkeypatch):
        """Контракт MemoryExtractor.parse_extractor_response должен принимать любой ответ."""
        ok = {"facts": [{"key": "user.first_name", "value": "Никита",
                          "category": "identity", "confidence": 0.95}]}
        ok_text = json.dumps(ok, ensure_ascii=False)

        cap1 = _Capture()
        monkeypatch.setattr("urllib.request.urlopen",
                            cap1.make_fake_urlopen({"choices": [{"message": {"content": ok_text}}]}))
        r1 = llm_mod.OpenRouterCompleter(model="qwen/qwen3.5-9b", api_key="x")(
            _SAMPLE_MESSAGES, max_tokens=800, temperature=0.1)

        cap2 = _Capture()
        monkeypatch.setattr("urllib.request.urlopen",
                            cap2.make_fake_urlopen({"choices": [{"message": {"content": ok_text}}]}))
        r2 = llm_mod.OpenAICompleter(model="gpt-4o-mini", api_key="x")(
            _SAMPLE_MESSAGES, max_tokens=800, temperature=0.1)

        cap3 = _Capture()
        monkeypatch.setattr("urllib.request.urlopen",
                            cap3.make_fake_urlopen({"message": {"content": ok_text}}))
        r3 = llm_mod.OllamaCompleter(model="qwen3.5:9b")(
            _SAMPLE_MESSAGES, max_tokens=800, temperature=0.1)

        assert r1 == r2 == r3 == ok_text




class TestEmbedderBackendDetection:
    """``detect_embedder_backend()`` — приоритет: explicit → legacy → Ollama-probe → OpenAI → noop."""

    def test_explicit_ollama(self, monkeypatch):
        with _clean_env(monkeypatch, FINKEY_EMBEDDER_BACKEND="ollama"):
            assert emb_mod.detect_embedder_backend() == "ollama"

    def test_explicit_openai(self, monkeypatch):
        with _clean_env(monkeypatch, FINKEY_EMBEDDER_BACKEND="openai"):
            assert emb_mod.detect_embedder_backend() == "openai"

    def test_explicit_noop(self, monkeypatch):
        with _clean_env(monkeypatch, FINKEY_EMBEDDER_BACKEND="noop"):
            assert emb_mod.detect_embedder_backend() == "noop"

    def test_legacy_openai_env_flag(self, monkeypatch):
        with _clean_env(monkeypatch,
                         FINKEY_MEMORY_OPENAI_EMBED="1",
                         OPENAI_API_KEY="sk-openai"):
            assert emb_mod.detect_embedder_backend() == "openai"

    def test_auto_ollama_when_probe_succeeds(self, monkeypatch):
        with _clean_env(monkeypatch):
            monkeypatch.setattr(emb_mod, "_probe_ollama_alive", lambda timeout=1.5: True)
            assert emb_mod.detect_embedder_backend() == "ollama"

    def test_auto_openai_when_no_ollama_but_key(self, monkeypatch):
        with _clean_env(monkeypatch, OPENAI_API_KEY="sk-openai"):
            monkeypatch.setattr(emb_mod, "_probe_ollama_alive", lambda timeout=1.5: False)
            assert emb_mod.detect_embedder_backend() == "openai"

    def test_auto_noop_when_nothing_available(self, monkeypatch):
        with _clean_env(monkeypatch):
            monkeypatch.setattr(emb_mod, "_probe_ollama_alive", lambda timeout=1.5: False)
            assert emb_mod.detect_embedder_backend() == "noop"


class TestEmbedderFactories:
    """Каждый embedder возвращает callable c ``vector_size``/``backend``/``model``."""

    def test_ollama_embedder_returns_vector_of_declared_size(self, monkeypatch):
        cap = _Capture()
        monkeypatch.setattr(emb_mod, "_get_session",
                            lambda: cap.make_fake_session({"embeddings": [[0.1] * 1024]}))

        emb = emb_mod.make_ollama_embedder(base_url="http://x:11434", model="bge-m3", vector_size=1024)
        vec = emb("Hello")
        assert len(vec) == 1024
        assert emb.backend == "ollama"             # type: ignore[attr-defined]
        assert emb.vector_size == 1024              # type: ignore[attr-defined]
        assert cap.calls[0]["body"]["model"] == "bge-m3"
        assert cap.calls[0]["body"]["input"] == "Hello"

    def test_ollama_embedder_handles_legacy_embedding_field(self, monkeypatch):
        cap = _Capture()
        monkeypatch.setattr(emb_mod, "_get_session",
                            lambda: cap.make_fake_session({"embedding": [0.2] * 1024}))

        emb = emb_mod.make_ollama_embedder(base_url="http://x:11434", model="bge-m3", vector_size=1024)
        vec = emb("Hello")
        assert len(vec) == 1024
        assert all(abs(x - 0.2) < 1e-6 for x in vec)

    def test_ollama_embedder_falls_back_to_urllib_without_pool(self, monkeypatch):
        """Без requests (пул недоступен) embedder обязан работать через urllib."""
        cap = _Capture()
        monkeypatch.setattr(emb_mod, "_get_session", lambda: None)
        monkeypatch.setattr("urllib.request.urlopen",
                            cap.make_fake_urlopen({"embeddings": [[0.1] * 1024]}))

        emb = emb_mod.make_ollama_embedder(base_url="http://x:11434", model="bge-m3", vector_size=1024)
        vec = emb("Hello")
        assert len(vec) == 1024
        assert cap.calls[0]["body"]["input"] == "Hello"

    def test_ollama_embedder_returns_zero_vector_on_failure(self, monkeypatch):
        def _fail(*_args, **_kwargs):
            raise RuntimeError("network down")

        class _FailSession:
            post = _fail

        monkeypatch.setattr(emb_mod, "_get_session", lambda: _FailSession())
        monkeypatch.setattr("urllib.request.urlopen", _fail)

        emb = emb_mod.make_ollama_embedder(base_url="http://x:11434", model="bge-m3", vector_size=1024)
        vec = emb("Hello")
        assert vec == [0.0] * 1024

    def test_openai_embedder_returns_vector(self, monkeypatch):
        cap = _Capture()
        monkeypatch.setattr(emb_mod, "_get_session",
                            lambda: cap.make_fake_session({"data": [{"embedding": [0.3] * 1536}]}))

        emb = emb_mod.make_openai_embedder(api_key="sk", model="text-embedding-3-small", vector_size=1536)
        vec = emb("Hello")
        assert len(vec) == 1536
        assert emb.backend == "openai"          # type: ignore[attr-defined]
        assert cap.calls[0]["body"]["input"] == "Hello"
        assert cap.calls[0]["body"]["model"] == "text-embedding-3-small"

    def test_openai_embedder_returns_zeros_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        emb = emb_mod.make_openai_embedder(api_key="", model="text-embedding-3-small", vector_size=1536)
        assert emb("Hello") == [0.0] * 1536

    def test_noop_embedder_is_deterministic_and_no_nan(self):
        emb = emb_mod.make_noop_embedder(vector_size=1024)
        v1 = emb("Same input")
        v2 = emb("Same input")
        v3 = emb("Different input")
        assert v1 == v2
        assert v1 != v3
        assert len(v1) == 1024
        assert all(-1.0 <= x <= 1.0 for x in v1)
        assert all(x == x for x in v1)
        norm = sum(x * x for x in v1) ** 0.5
        assert abs(norm - 1.0) < 1e-3

    def test_embed_cache_collapses_parallel_identical_calls(self):
        """
        load_context эмбеддит один и тот же текст из ~12 параллельных воркеров.
        Без single-flight все они промахиваются мимо LRU и бьют в Ollama
        одновременно — N×0.4 с прямо в TTFT.
        """
        import threading

        calls: list[str] = []
        calls_lock = threading.Lock()
        started = threading.Event()

        def slow_embed(text: str) -> list[float]:
            with calls_lock:
                calls.append(text)
            started.set()
            time.sleep(0.05)
            return [0.5, 0.5]

        cached = emb_mod._with_embed_cache(slow_embed, 2)
        results: list[list[float]] = []
        results_lock = threading.Lock()

        def worker() -> None:
            vec = cached("одинаковый вопрос")
            with results_lock:
                results.append(vec)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(calls) == 1, f"ожидался 1 HTTP-вызов, было {len(calls)}"
        assert len(results) == 8
        assert all(v == [0.5, 0.5] for v in results)

    def test_embed_cache_does_not_cache_failed_vectors(self):
        """Нулевой вектор (embed упал) не должен залипать в кэше."""
        calls: list[str] = []

        def failing_embed(text: str) -> list[float]:
            calls.append(text)
            return [0.0, 0.0]

        cached = emb_mod._with_embed_cache(failing_embed, 2)
        cached("x")
        cached("x")
        assert len(calls) == 2

    def test_embed_cache_keeps_distinct_texts_independent(self):
        calls: list[str] = []

        def embed(text: str) -> list[float]:
            calls.append(text)
            return [float(len(text)), 0.0]

        cached = emb_mod._with_embed_cache(embed, 2)
        assert cached("aa") == [2.0, 0.0]
        assert cached("bbb") == [3.0, 0.0]
        assert cached("aa") == [2.0, 0.0]
        assert calls == ["aa", "bbb"]

    def test_shared_embedder_is_one_instance_per_process(self, monkeypatch):
        """
        SemanticCache, SemanticMemory и RagEngine должны делить один embedder:
        LRU и single-flight живут внутри экземпляра, и с отдельными копиями
        один и тот же текст эмбеддится ими параллельно (N запросов в Ollama).
        """
        monkeypatch.setattr(emb_mod, "_shared_embedder", None)
        with _clean_env(monkeypatch):
            monkeypatch.setattr(emb_mod, "_probe_ollama_alive", lambda timeout=1.5: False)
            first = emb_mod.get_shared_embedder()
            second = emb_mod.get_shared_embedder()
            assert first is second
            assert emb_mod.build_embedder_from_env() is not first

    def test_build_embedder_from_env_never_returns_none(self, monkeypatch):
        with _clean_env(monkeypatch):
            monkeypatch.setattr(emb_mod, "_probe_ollama_alive", lambda timeout=1.5: False)
            emb = emb_mod.build_embedder_from_env()
            assert emb is not None
            assert callable(emb)
            assert hasattr(emb, "vector_size")
            assert hasattr(emb, "backend")

    def test_build_embedder_chooses_ollama_when_probe_succeeds(self, monkeypatch):
        with _clean_env(monkeypatch):
            monkeypatch.setattr(emb_mod, "_probe_ollama_alive", lambda timeout=1.5: True)
            emb = emb_mod.build_embedder_from_env()
            assert emb.backend == "ollama"          # type: ignore[attr-defined]
            assert emb.vector_size == 1024           # type: ignore[attr-defined]

    def test_build_embedder_chooses_openai_when_only_key(self, monkeypatch):
        with _clean_env(monkeypatch, OPENAI_API_KEY="sk-openai"):
            monkeypatch.setattr(emb_mod, "_probe_ollama_alive", lambda timeout=1.5: False)
            emb = emb_mod.build_embedder_from_env()
            assert emb.backend == "openai"           # type: ignore[attr-defined]

    def test_build_embedder_falls_back_to_noop(self, monkeypatch):
        with _clean_env(monkeypatch):
            monkeypatch.setattr(emb_mod, "_probe_ollama_alive", lambda timeout=1.5: False)
            emb = emb_mod.build_embedder_from_env()
            assert emb.backend == "noop"             # type: ignore[attr-defined]
            assert emb.vector_size == 1024            # type: ignore[attr-defined]




class TestOpenRouterAssistantTextExtraction:
    """``_extract_openrouter_assistant_text`` — Qwen кладёт JSON в reasoning."""

    def test_prefers_non_empty_content(self):
        msg = {"content": '{"facts":[]}'}
        assert llm_mod._extract_openrouter_assistant_text(msg, json_mode=True) == '{"facts":[]}'

    def test_json_mode_falls_back_to_reasoning_string(self):
        msg = {"content": "", "reasoning": '{"facts":[{"key":"user.name","value":"Никита"}]}'}
        out = llm_mod._extract_openrouter_assistant_text(msg, json_mode=True)
        assert "Никита" in out

    def test_json_mode_falls_back_to_reasoning_details(self):
        msg = {
            "content": "",
            "reasoning_details": [
                {"type": "reasoning.text", "text": '{"facts":[]}'},
            ],
        }
        assert llm_mod._extract_openrouter_assistant_text(msg, json_mode=True) == '{"facts":[]}'

    def test_no_reasoning_leak_when_json_mode_off(self):
        msg = {"content": "", "reasoning": "secret"}
        assert llm_mod._extract_openrouter_assistant_text(msg, json_mode=False) == ""

    def test_index_token_content_falls_back_to_reasoning(self):
        msg = {"content": "[0]", "reasoning": '{"attach":false,"reply":"Ок, минуту."}'}
        out = llm_mod._extract_openrouter_assistant_text(msg, json_mode=True)
        assert "Ок, минуту." in out
        assert out != "[0]"

    def test_int_list_content_is_empty(self):
        assert llm_mod._normalize_openrouter_content([0]) == ""
        assert llm_mod._placeholder_openrouter_content("[0]") is True

