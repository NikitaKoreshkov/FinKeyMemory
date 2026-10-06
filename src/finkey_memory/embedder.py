# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Унифицированная фабрика embedder'а для семантического слоя памяти.

Идея — память должна работать **одинаково** через любой провайдер чата:
OpenRouter (cloud), Ollama (local), смешанный режим (cloud chat + local embed).

Поддерживаемые backend-ы (контракт один: ``(text: str) -> list[float]``,
плюс атрибут ``vector_size``):

  * ``ollama``      — Ollama ``/api/embed`` (``/api/embeddings`` для legacy).
                      Default: ``bge-m3`` (1024-dim, мультиязычный, 100+ языков).
                      Бесплатный, локальный, идеально для production.
  * ``openai``      — OpenAI ``/v1/embeddings``. Default: ``text-embedding-3-small`` (1536-dim).
                      Cloud fallback, если у юзера нет Ollama.
  * ``noop``        — детерминированный fallback (sha-based), если ни одного
                      provider'а недоступно. Качество семантики низкое, но
                      ai-core не падает и upsert/PG-пути остаются рабочими.

Auto-selection (когда ``FINKEY_EMBEDDER_BACKEND`` не задан):
  1. Ollama, если ``OLLAMA_BASE_URL`` достижим (или дефолтный 127.0.0.1:11434)
     **и** модель ``bge-m3`` (или явная) доступна.
  2. OpenAI, если есть ``OPENAI_API_KEY``.
  3. Noop fallback — никогда не падаем на старте.

Важно: ``OpenRouter`` НЕ предоставляет embeddings API. Поэтому даже когда
основной чат идёт через OpenRouter, embedder должен быть либо Ollama (рекомендуется),
либо OpenAI. Если у юзера ни того, ни другого — Qdrant просто не подключается
и память сохраняется только в PG/Redis (graceful degrade).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import ssl
import struct
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Optional


def _default_ssl_context() -> ssl.SSLContext:
    """Тот же helper, что в llm_completer.py — для OpenAI HTTPS на свежих Python."""
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL_CTX = _default_ssl_context()

logger = logging.getLogger("finkey.memory.embedder")


_session_lock = threading.Lock()
_session: object | None = None
_session_unavailable = False


def _get_session():
    """
    Разделяемая ``requests.Session`` с keep-alive (или None, если requests нет).

    Без пула каждый embed открывал новое TCP(+TLS) соединение: для локальной
    Ollama это лишние ~5-20 мс, для OpenAI — полный TLS handshake ~100-300 мс
    на каждый вызов.
    """
    global _session, _session_unavailable
    if _session_unavailable:
        return None
    if _session is None:
        with _session_lock:
            if _session is None and not _session_unavailable:
                try:
                    import requests  # type: ignore

                    s = requests.Session()
                    adapter = requests.adapters.HTTPAdapter(
                        pool_connections=8,
                        pool_maxsize=32,
                        max_retries=0,
                    )
                    s.mount("http://", adapter)
                    s.mount("https://", adapter)
                    _session = s
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Embedder keep-alive pool unavailable: %s", exc)
                    _session_unavailable = True
                    return None
    return _session


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    """
    POST JSON → dict. HTTP >=400 поднимается как ``urllib.error.HTTPError``,
    чтобы ветки обработки (например, 404 → legacy endpoint) работали одинаково
    и с пулом, и без него.
    """
    session = _get_session()
    body = json.dumps(payload).encode("utf-8")
    if session is None:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode())

    resp = session.post(url, data=body, headers=headers, timeout=timeout)
    if resp.status_code >= 400:
        import io

        raise urllib.error.HTTPError(
            url, resp.status_code, f"HTTP {resp.status_code}", resp.headers, io.BytesIO(resp.content)
        )
    return resp.json()



_DEFAULT_OLLAMA_EMBED_MODEL = "bge-m3"
_DEFAULT_OLLAMA_EMBED_DIM   = 1024

_DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
_DEFAULT_OPENAI_EMBED_DIM   = 1536

_NOOP_DIM_DEFAULT = 1024



EmbedFn = Callable[[str], list[float]]


def _embed_cache_size() -> int:
    try:
        return max(0, int(os.getenv("FINKEY_EMBED_CACHE_SIZE", "512")))
    except ValueError:
        return 512


class _InFlight:
    """Результат одного идущего embed-вызова, разделяемый между потоками."""

    __slots__ = ("event", "vec")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.vec: list[float] | None = None


def _with_embed_cache(embed: EmbedFn, dim: int, *, wait_timeout: float = 35.0) -> EmbedFn:
    """
    In-process LRU + single-flight поверх embed-вызова.

    За один ход пайплайн эмбеддит ОДИН И ТОТ ЖЕ текст сообщения ~6-8 раз
    (RAG-поиск, память разговоров, user-факты, 4 substrates) — каждый вызов
    это отдельный HTTP до Ollama (~0.4 с на CPU).

    LRU схлопывает их в один только для ПОСЛЕДОВАТЕЛЬНЫХ вызовов. Но
    ``MemoryManager.load_context`` раскладывает эти поиски по 12 параллельным
    воркерам, и тогда ни один из них ещё не успел положить вектор в кэш —
    все промахиваются и одновременно бьют в Ollama. Ollama обслуживает
    embed-запросы почти последовательно, поэтому N одинаковых запросов дают
    N×0.4 с прямо в TTFT.

    Поэтому здесь ещё и single-flight: первый поток для данного текста считает,
    остальные ждут его результат вместо собственного HTTP.
    Нулевые векторы (embed упал) не кэшируем, чтобы не залипать на ошибке.
    """
    from collections import OrderedDict

    max_size = _embed_cache_size()
    if max_size <= 0:
        return embed

    cache: "OrderedDict[str, tuple[float, ...]]" = OrderedDict()
    inflight: dict[str, _InFlight] = {}
    lock = threading.Lock()

    def cached_embed(text: str) -> list[float]:
        key = text[:8000]
        with lock:
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                return list(hit)
            pending = inflight.get(key)
            if pending is None:
                pending = _InFlight()
                inflight[key] = pending
                leader = True
            else:
                leader = False

        if not leader:
            # Ждём лидера. Если он не уложился в таймаут — считаем сами,
            # чтобы одна залипшая попытка не блокировала весь пайплайн.
            if pending.event.wait(timeout=wait_timeout) and pending.vec is not None:
                return list(pending.vec)
            return embed(key)

        vec: list[float] = []
        try:
            vec = embed(key)
        finally:
            pending.vec = list(vec) if vec else None
            with lock:
                if vec and any(vec):
                    cache[key] = tuple(vec)
                    if len(cache) > max_size:
                        cache.popitem(last=False)
                inflight.pop(key, None)
            pending.event.set()
        return vec

    for attr in ("vector_size", "backend", "model"):
        if hasattr(embed, attr):
            setattr(cached_embed, attr, getattr(embed, attr))
    return cached_embed




def make_ollama_embedder(
    *,
    base_url:  Optional[str] = None,
    model:     Optional[str] = None,
    timeout:   Optional[float] = None,
    vector_size: Optional[int] = None,
) -> EmbedFn:
    """
    Локальный мультиязычный embedder (``bge-m3`` по умолчанию).

    POST на ``{base}/api/embed`` (новый Ollama) с автоматическим fallback на
    легаси ``/api/embeddings``. Возвращает callable, у которого через атрибут
    ``vector_size`` сразу известна размерность вектора (полезно, чтобы
    ``SemanticMemory.ensure_collections(vector_size=...)`` сразу создал
    коллекции правильного размера).
    """
    base = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    mdl  = (model    or os.getenv("FINKEY_EMBEDDER_MODEL") or _DEFAULT_OLLAMA_EMBED_MODEL).strip()
    to   = float(timeout if timeout is not None else os.getenv("FINKEY_EMBEDDER_TIMEOUT_SEC", "30") or "30")
    try:
        dim = int(vector_size if vector_size is not None else (os.getenv("FINKEY_EMBEDDER_VECTOR_SIZE") or _DEFAULT_OLLAMA_EMBED_DIM))
    except ValueError:
        dim = _DEFAULT_OLLAMA_EMBED_DIM

    embed_url    = f"{base}/api/embed"
    legacy_url   = f"{base}/api/embeddings"

    # Ollama по умолчанию выгружает модель через 5 минут простоя, и следующий
    # запрос платит за её повторную загрузку (для bge-m3 это секунды, прямо в
    # TTFT). На проде с неравномерным трафиком это происходит постоянно, поэтому
    # держим модель резидентной. Выключить: FINKEY_EMBEDDER_KEEP_ALIVE="" .
    keep_alive = os.getenv("FINKEY_EMBEDDER_KEEP_ALIVE", "30m")
    if keep_alive is not None:
        keep_alive = keep_alive.strip()

    def embed(text: str) -> list[float]:
        payload: dict = {"model": mdl, "input": text[:8000]}
        if keep_alive:
            payload["keep_alive"] = keep_alive
        headers = {"Content-Type": "application/json"}
        try:
            data = _post_json(embed_url, payload, headers, to)
            if isinstance(data.get("embeddings"), list) and data["embeddings"]:
                vec = list(data["embeddings"][0])
                return vec
            if isinstance(data.get("embedding"), list):
                return list(data["embedding"])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                try:
                    data = _post_json(legacy_url, payload, headers, to)
                    if isinstance(data.get("embedding"), list):
                        return list(data["embedding"])
                except Exception as exc2:
                    logger.warning("Ollama legacy embed failed: %s", exc2)
            else:
                logger.warning("Ollama embed HTTP %s: %s", e.code, e.reason)
        except Exception as exc:
            logger.warning("Ollama embed failed (model=%s): %s", mdl, exc)
        return [0.0] * dim

    embed.vector_size = dim            # type: ignore[attr-defined]
    embed.backend     = "ollama"       # type: ignore[attr-defined]
    embed.model       = mdl            # type: ignore[attr-defined]
    return _with_embed_cache(embed, dim, wait_timeout=to + 5.0)




def make_openai_embedder(
    *,
    api_key:    Optional[str] = None,
    model:      Optional[str] = None,
    timeout:    Optional[float] = None,
    vector_size: Optional[int] = None,
) -> EmbedFn:
    """
    OpenAI Embeddings — fallback, если Ollama недоступна.

    Модель ``text-embedding-3-small`` (1536-dim) тоже работает мультиязычно
    (она тренирована на массиве 50+ языков, но не настолько хорошо в редких
    языках, как ``bge-m3``). Это хороший выбор для cloud-only пользователей.
    """
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    mdl = (model   or os.getenv("FINKEY_EMBEDDER_MODEL") or _DEFAULT_OPENAI_EMBED_MODEL).strip()
    to  = float(timeout if timeout is not None else os.getenv("FINKEY_EMBEDDER_TIMEOUT_SEC", "30") or "30")
    try:
        dim = int(vector_size if vector_size is not None else (os.getenv("FINKEY_EMBEDDER_VECTOR_SIZE") or _DEFAULT_OPENAI_EMBED_DIM))
    except ValueError:
        dim = _DEFAULT_OPENAI_EMBED_DIM

    def embed(text: str) -> list[float]:
        if not key:
            return [0.0] * dim
        try:
            data = _post_json(
                "https://api.openai.com/v1/embeddings",
                {"model": mdl, "input": text[:8000]},
                {
                    "Authorization": f"Bearer {key}",
                    "Content-Type":  "application/json",
                },
                to,
            )
            return list(data["data"][0]["embedding"])
        except Exception as exc:
            logger.warning("OpenAI embed failed (model=%s): %s", mdl, exc)
            return [0.0] * dim

    embed.vector_size = dim         # type: ignore[attr-defined]
    embed.backend     = "openai"    # type: ignore[attr-defined]
    embed.model       = mdl         # type: ignore[attr-defined]
    return _with_embed_cache(embed, dim, wait_timeout=to + 5.0)




def make_noop_embedder(*, vector_size: int = _NOOP_DIM_DEFAULT) -> EmbedFn:
    """
    Детерминированный sha-based «embedder» — НЕ для production.

    Возвращает воспроизводимый вектор фиксированной размерности (по умолчанию
    совпадает с Ollama bge-m3 — 1024), не зависящий от сети. Используется как
    safety-net, когда ни один реальный backend недоступен: позволяет коду
    upsert-ить точки в Qdrant и тесту работать, но **семантический поиск даст
    почти-ортогональные векторы и качество recall будет нулевым**.
    """
    dim = int(vector_size)

    def embed(text: str) -> list[float]:
        if not text:
            text = "<empty>"
        raw = b""
        counter = 0
        while len(raw) < dim * 2:
            h = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            raw += h
            counter += 1
        floats: list[float] = []
        for i in range(dim):
            word = struct.unpack(">H", raw[i * 2:i * 2 + 2])[0]
            floats.append(word / 65535.0 * 2.0 - 1.0)
        norm = math.sqrt(sum(x * x for x in floats)) or 1.0
        return [x / norm for x in floats]

    embed.vector_size = dim         # type: ignore[attr-defined]
    embed.backend     = "noop"      # type: ignore[attr-defined]
    embed.model       = "sha256"    # type: ignore[attr-defined]
    return embed




def _probe_ollama_alive(timeout: float = 1.5) -> bool:
    base = (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def detect_embedder_backend() -> str:
    """
    Возвращает выбранный backend:
      'ollama' | 'openai' | 'noop'

    Логика:
      1. ``FINKEY_EMBEDDER_BACKEND`` (explicit override).
      2. Legacy ``FINKEY_MEMORY_OPENAI_EMBED=1`` → openai.
      3. Ollama, если ``OLLAMA_BASE_URL`` отвечает на ``/api/tags``.
      4. OpenAI, если ``OPENAI_API_KEY`` задан.
      5. ``noop`` — никогда не падаем.
    """
    explicit = (os.getenv("FINKEY_EMBEDDER_BACKEND") or "").strip().lower()
    if explicit in ("ollama", "openai", "noop"):
        return explicit
    legacy_openai = (os.getenv("FINKEY_MEMORY_OPENAI_EMBED") or "").strip().lower() in ("1", "true", "yes", "on")
    if legacy_openai and os.getenv("OPENAI_API_KEY"):
        return "openai"
    if _probe_ollama_alive():
        return "ollama"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    return "noop"


def build_embedder_from_env() -> EmbedFn:
    """
    Единая точка входа: возвращает готовый embedder + лог о выбранном backend'е.

    Никогда не возвращает None: если ничего недоступно — даёт ``noop`` (см.
    ``make_noop_embedder``). Это позволяет коду выше не делать ``if embed is None``
    и упростить ветки.
    """
    backend = detect_embedder_backend()
    if backend == "ollama":
        emb = make_ollama_embedder()
    elif backend == "openai":
        emb = make_openai_embedder()
    else:
        emb = make_noop_embedder()
    logger.debug(
        "Embedder backend=%s model=%s dim=%d",
        getattr(emb, "backend", backend),
        getattr(emb, "model", "?"),
        getattr(emb, "vector_size", _NOOP_DIM_DEFAULT),
    )
    if backend == "ollama":
        _warm_embedder_background(emb)
    return emb


_shared_embedder: EmbedFn | None = None
_shared_embedder_lock = threading.Lock()


def get_shared_embedder() -> EmbedFn:
    """
    Один embedder на процесс — для всех рантайм-потребителей.

    Важно для latency: LRU и single-flight живут ВНУТРИ экземпляра. Если
    semantic cache, SemanticMemory и RagEngine каждый строят свой embedder,
    то один и тот же текст пользователя эмбеддится ими параллельно и Ollama
    выполняет 3 одинаковых запроса вместо одного. С общим экземпляром
    первый вызов считает, остальные получают готовый вектор.

    ``build_embedder_from_env()`` по-прежнему отдаёт свежий экземпляр —
    она нужна там, где backend должен переопределяться из env (тесты, CLI).
    """
    global _shared_embedder
    if _shared_embedder is None:
        with _shared_embedder_lock:
            if _shared_embedder is None:
                _shared_embedder = build_embedder_from_env()
    return _shared_embedder


def _warm_embedder_background(embed: EmbedFn) -> None:
    """
    Прогревает модель в фоне, чтобы первый живой запрос не платил за её загрузку.

    Выключить: ``FINKEY_EMBEDDER_WARMUP=0``.
    """
    v = (os.getenv("FINKEY_EMBEDDER_WARMUP", "1") or "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return

    def _warm() -> None:
        try:
            t0 = time.perf_counter()
            embed("finkey embedder warmup")
            logger.info(
                "Embedder warmup done in %.0f ms (model=%s)",
                (time.perf_counter() - t0) * 1000.0,
                getattr(embed, "model", "?"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Embedder warmup skipped: %s", exc)

    threading.Thread(target=_warm, name="finkey-embed-warmup", daemon=True).start()
