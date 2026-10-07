# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey MemoryManager — unified facade over Redis, optional PostgreSQL, Qdrant, and a
small in-process volatile store for cognition / semantic facts until product DDL ships.

Core operations:

  load_context(company_id, user_id, conv_id, message)
      → MemoryContext
      Redis emotional arc; PG history + impressions; Qdrant RAG + user memories;
      volatile facts / impressions / episode snapshot / traces / procedures (used for prompts).

  save_turn(…, cognition=)
      Persists messages (PG/Redis), optional cognition into volatile + Qdrant paths,
      bumps interaction count, and every FINKEY_MEMORY_MAINT_INTERVAL enqueues one
      (company_id, user_id) for lightweight merge / decay / canonical semantic re-embed.

Impressions: volatile merge-on-write; PG mirror when configured; Qdrant user_memory as before.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Optional

from finkey_memory.consciousness_state import EmotionType, UserProfile
from finkey_memory.cognition_snapshot import CognitionTurnSnapshot
from finkey_memory.consolidation import clip_fact_value
from finkey_memory.episode_codec import episode_from_snapshot, episode_to_plain_dict
from finkey_memory.decay import DecayScheduler
from finkey_memory.dream import DreamWorker
from finkey_memory.extractor_runner import AsyncExtractorRunner
from finkey_memory.fact_store import FactStore
from finkey_memory.long_term import LongTermMemory
from finkey_memory.maintenance import run_user_memory_maintenance
from finkey_memory.metrics import LOAD_CONTEXT_LATENCY
from finkey_memory.scene_persona import (
    SceneBlock,
    ScenePersonaService,
    persona_enabled,
    scenes_enabled,
)
from finkey_memory.schema import (
    ConversationTurn,
    MemoryContext,
    RedisKeys,
    SessionState,
    UserImpression,
)
from finkey_memory.semantic import SemanticMemory
from finkey_memory.session import SessionMemory
from finkey_memory.summary_worker import SummaryWorker
from finkey_memory.temporal import as_datetime
from finkey_memory.volatile_store import VolatileKnowledgeStore

logger = logging.getLogger(__name__)

SUMMARY_TRIGGER_EVERY = 20

_MAINT_INTERVAL = max(7, int(os.getenv("FINKEY_MEMORY_MAINT_INTERVAL", "33")))


def _turn_plain_text(content: Any) -> str:
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, str):
        return content
    return str(content or "")


def _entity_expand_enabled() -> bool:
    v = (os.getenv("FINKEY_MEMORY_ENTITY_EXPAND") or "1").strip().lower()
    return v in ("1", "true", "yes", "on", "")


def _entity_expand_limit() -> int:
    try:
        return max(0, int(os.getenv("FINKEY_MEMORY_ENTITY_EXPAND_LIMIT", "6")))
    except ValueError:
        return 6


def _pg_prune_every_n_turns() -> int:
    """
    How often (per-user interaction counter in ``save_turn``) to run PG message retention.

    Uses the same counter as ``FINKEY_MEMORY_MAINT_INTERVAL`` (``bump_interaction``), so
    prune fires on the same cadence family — default **12** is slightly more frequent than
    maintenance (33) so long threads get capped without waiting only on the maintenance queue.
    """
    try:
        return max(1, int(os.getenv("FINKEY_PG_MESSAGES_PRUNE_EVERY_N_TURNS", "12")))
    except ValueError:
        return 12


def _cognition_substrates_enabled() -> bool:
    """
    Флаг записи и отображения cognition-substrate'ов (episodic/semantic/emotional/procedural).

    Production-default: **включено**. Substrate'ы дают модели эпизодическую и
    процедурную опору поверх durable-фактов; при желании их можно временно
    отключить через ``FINKEY_COGNITION_SUBSTRATES_ENABLED=0`` (например, для
    A/B-тестов влияния на качество ответа).
    """
    v = (os.getenv("FINKEY_COGNITION_SUBSTRATES_ENABLED") or "").strip().lower()
    if not v:
        return True
    return v in ("1", "true", "yes", "on")


def _durable_facts_limit() -> int:
    try:
        return max(0, int(os.getenv("FINKEY_MEMORY_DURABLE_FACTS_LIMIT", "32")))
    except ValueError:
        return 32


def _fact_hits_limit() -> int:
    try:
        return max(0, int(os.getenv("FINKEY_MEMORY_FACT_HITS_LIMIT", "4")))
    except ValueError:
        return 4


def _scene_nav_top() -> int:
    """Сколько тем уходит в промпт: дальше список становится пересказом памяти."""
    try:
        return max(0, int(os.getenv("FINKEY_MEMORY_SCENE_NAV_TOP", "8")))
    except ValueError:
        return 8


# ── Read-time recency/frequency weighting (A1) ───────────────────────────────
# Свежесть даёт до +15% к score и линейно гаснет за 90 дней; частота — заметно
# меньше (+5%), иначе один переказ перебивает релевантность.
_FACT_RECENCY_BONUS    = 0.15
_FACT_RECENCY_WINDOW_S = 90.0 * 86400.0
_FACT_ACCESS_BONUS     = 0.05
_FACT_ACCESS_REFERENCE = 10.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _to_epoch(value: Any) -> Optional[float]:
    """unix float | ISO-строка | datetime → epoch; None, если поля нет вовсе."""
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass
    try:
        return as_datetime(value).timestamp()
    except Exception:  # noqa: BLE001 — поле чужого слоя, молча считаем «нет данных»
        return None


def _row_epoch(row: dict) -> float:
    for name in ("updated_at", "touched_at", "created_at"):
        epoch = _to_epoch(row.get(name))
        if epoch is not None:
            return epoch
    return 0.0


def _durable_sort_key(row: dict) -> tuple[float, float]:
    """(priority DESC, updated_at DESC) — знаки через отрицание; sorted стабилен."""
    return (-_as_float(row.get("priority"), 0.5), -_row_epoch(row))


def _fact_recency_bonus(hit: dict, now_ts: float) -> float:
    updated = _to_epoch(hit.get("updated_at"))
    if updated is None:
        return 0.0
    age = max(0.0, now_ts - updated)
    return _FACT_RECENCY_BONUS * max(0.0, 1.0 - age / _FACT_RECENCY_WINDOW_S)


def _fact_access_bonus(hit: dict) -> float:
    raw = hit.get("access_count")
    if raw is None:
        return 0.0
    count = _as_float(raw, 0.0)
    if count <= 0.0:
        return 0.0
    return _FACT_ACCESS_BONUS * min(1.0, count / _FACT_ACCESS_REFERENCE)


def _rerank_fact_hits(hits: list[dict]) -> list[dict]:
    """
    final = score * (1 + recency_bonus + access_bonus).

    Точки, записанные до A1, полей не имеют → бонус 0 и сырой score, порядок
    совпадает со старым. Исходный ``score`` не портим: добавляем
    ``score_effective``.
    """
    if not hits:
        return []
    now_ts = time.time()
    ranked: list[tuple[float, int, dict]] = []
    for idx, hit in enumerate(hits):
        raw = _as_float(hit.get("score"), 0.0)
        bonus = _fact_recency_bonus(hit, now_ts) + _fact_access_bonus(hit)
        final = raw * (1.0 + bonus)
        hit["score_effective"] = round(final, 6)
        ranked.append((final, -idx, hit))
    ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [t[2] for t in ranked]


def _load_context_workers() -> int:
    """Пул для параллельного load_context (14 независимых I/O-вызовов).

    Веток стало 14 (добавились persona и scenes из L2/L3): при старом
    default=12 две последние встали бы в очередь и добавились к TTFT суммарно,
    а не максимумом самой медленной.
    """
    try:
        return max(2, int(os.getenv("FINKEY_MEMORY_LOAD_WORKERS", "14")))
    except ValueError:
        return 14


def _load_context_timing_enabled() -> bool:
    """Пофазный лог load_context. Выключить: ``FINKEY_MEMORY_LOAD_TIMING=0``."""
    v = (os.getenv("FINKEY_MEMORY_LOAD_TIMING", "1") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


class MemoryManager:
    """
    Unified memory facade.

    Any of the three backends can be None — the manager degrades gracefully.
    This makes it safe to run without Redis or Qdrant during development.
    """

    def __init__(
        self,
        session:           Optional[SessionMemory]        = None,
        long_term:         Optional[LongTermMemory]       = None,
        semantic:          Optional[SemanticMemory]       = None,
        extractor_runner:  Optional[AsyncExtractorRunner] = None,
    ) -> None:
        self._session         = session
        self._long_term       = long_term
        self._semantic        = semantic
        self._volatile        = VolatileKnowledgeStore()
        self._extractor_runner = extractor_runner
        self._fact_store: Optional[FactStore] = None
        self._decay_scheduler: Optional[DecayScheduler] = None
        self._summary_worker: Optional[SummaryWorker] = None
        self._dream_worker: Optional[DreamWorker] = None
        self._scene_persona: Optional[ScenePersonaService] = None
        self._maint_jobs: deque[tuple[str, str]] = deque(maxlen=2048)
        self._extract_turn_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._extract_turn_lock = threading.Lock()
        self._rag_engine: Optional[object] = None


    def set_rag_engine(self, engine: object) -> None:
        self._rag_engine = engine

    @property
    def rag_engine(self) -> Optional[object]:
        return self._rag_engine


    @property
    def fact_store(self) -> Optional[FactStore]:
        if self._fact_store is not None:
            return self._fact_store
        if self._long_term is None:
            return None
        try:
            self._fact_store = FactStore(
                long_term    = self._long_term,
                volatile     = self._volatile,
                semantic     = self._semantic,
                redis_client = self._session.client if self._session else None,
            )
        except Exception as exc:
            logger.warning("FactStore construction failed: %s", exc)
            return None
        return self._fact_store

    @property
    def scene_persona(self) -> Optional[ScenePersonaService]:
        """
        L2 (сцены) + L3 (портрет) фасад. PG обязателен, Qdrant и LLM — нет.

        Completer берём тем же путём, что и экстрактор
        (``llm_completer.build_extractor_completer_from_env``): второго
        второго самописного LLM-клиента в памяти быть не должно. Если бэкенда реально
        нет (ни ключей, ни локального Ollama) — падение ловится на вызове, сервис
        продолжает работать без генерации и оставляет строки нетронутыми.
        """
        if self._long_term is None:
            return None
        if self._scene_persona is None:
            if not (scenes_enabled() or persona_enabled()):
                return None
            completer: Any = None
            try:
                from finkey_memory.llm_completer import build_extractor_completer_from_env

                completer = build_extractor_completer_from_env()
            except Exception as exc:  # noqa: BLE001 — LLM опционален для L2/L3
                logger.debug("scene/persona completer unavailable: %s", exc)
            try:
                self._scene_persona = ScenePersonaService(
                    long_term = self._long_term,
                    semantic  = self._semantic,
                    completer = completer,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ScenePersonaService construction failed: %s", exc)
                return None
        return self._scene_persona

    def resolve_user_id_for_vectors(self, company_id: str, user_id: str) -> str:
        """UUID пользователя для Qdrant payload (как в ``_search_user_facts``)."""
        if self._long_term is None:
            return user_id
        try:
            resolved = self._long_term._resolve_user_id(company_id, user_id)  # noqa: SLF001
            return resolved or user_id
        except Exception:
            return user_id

    def upsert_extracted_facts(
        self,
        *,
        company_id:      str,
        user_id:         str,
        conversation_id: Optional[str],
        facts,
    ):
        """
        Public facade: writes facts through ``FactStore`` (PG + vectors + L0).

        Without PostgreSQL the facts are still stored in the in-process L0
        volatile store (per tenant, visible to ``load_context``) — nothing is
        silently dropped; ``UpsertReport.volatile_ok`` counts those writes.
        Durable L1 history, as-of archive, scenes and persona need PG.
        """
        fs = self.fact_store
        if fs is None:
            from .fact_store import UpsertReport

            report = UpsertReport(accepted=len(facts))
            for fact in facts:
                try:
                    self._volatile.upsert_fact(
                        company_id, user_id,
                        fact.key_normalized,
                        fact.value,
                        confidence = float(fact.confidence),
                        source     = str(fact.source or "inferred"),
                        priority   = float(fact.priority),
                        expires_at = getattr(fact, "expires_at", None),
                    )
                    report.volatile_ok += 1
                    report.written_keys.append(fact.key_normalized)
                except Exception as exc:  # noqa: BLE001
                    report.errors.append(str(exc))
            return report
        report = fs.upsert_extracted_facts(
            company_id=company_id,
            user_id=user_id,
            conversation_id=conversation_id,
            facts=facts,
        )
        if report is not None and getattr(report, "pg_ok", 0) > 0:
            # Счётчик P4-триггера портрета: память растёт — портрет обязан
            # пересобираться не «когда-нибудь», а через interval фактов.
            self._note_persona_memory_growth(
                company_id = company_id,
                user_id    = user_id,
                delta      = int(report.pg_ok),
            )
            self.enqueue_dream(company_id, user_id)
        return report

    def _note_persona_memory_growth(self, company_id: str, user_id: str, delta: int) -> None:
        """``user_persona.memories_since_last += delta``. Один дешёвый UPSERT после
        успешной записи фактов; без PG/сервиса — no-op, никогда не будит LLM."""
        svc = self.scene_persona
        if svc is None or delta <= 0:
            return
        try:
            svc.persona.bump_memories(company_id, user_id, delta)
        except Exception as exc:  # noqa: BLE001 — счётчик не роняет запись памяти
            logger.debug("persona memories counter bump skipped: %s", exc)

    def set_extractor_runner(self, runner: Optional[AsyncExtractorRunner]) -> None:
        """Поздняя инъекция runner'a (composition root зовёт после построения mm)."""
        self._extractor_runner = runner


    @property
    def decay_scheduler(self) -> Optional[DecayScheduler]:
        return self._decay_scheduler

    def start_decay_scheduler(self) -> bool:
        """
        Лениво инициализирует и стартует ``DecayScheduler`` на базе ``fact_store``.
        Возвращает True, если scheduler стартовал в этом вызове.
        Без ``LongTermMemory`` decay не имеет смысла (некому хранить ``expires_at``).
        """
        fs = self.fact_store
        if fs is None:
            return False
        if self._decay_scheduler is None:
            self._decay_scheduler = DecayScheduler(fact_store=fs)
        return self._decay_scheduler.start()

    def stop_decay_scheduler(self, *, wait: bool = False, timeout: float = 5.0) -> None:
        if self._decay_scheduler is not None:
            try:
                self._decay_scheduler.stop(wait=wait, timeout=timeout)
            except Exception as exc:
                logger.warning("decay scheduler stop failed: %s", exc)

    @property
    def summary_worker(self) -> Optional[SummaryWorker]:
        return self._summary_worker

    def start_summary_worker(self, *, generate_fn=None) -> bool:
        """Start background SummaryWorker (claims PG summary_jobs → conv memories)."""
        if self._long_term is None:
            return False
        if self._summary_worker is None:
            self._summary_worker = SummaryWorker(
                long_term=self._long_term,
                semantic=self._semantic,
                generate_fn=generate_fn,
            )
        elif generate_fn is not None and self._summary_worker.generate_fn is None:
            self._summary_worker.generate_fn = generate_fn
        return self._summary_worker.start()

    def stop_summary_worker(self, *, wait: bool = False, timeout: float = 5.0) -> None:
        if self._summary_worker is not None:
            try:
                self._summary_worker.stop(wait=wait, timeout=timeout)
            except Exception as exc:
                logger.warning("summary worker stop failed: %s", exc)

    @property
    def dream_worker(self) -> Optional[DreamWorker]:
        return self._dream_worker

    def start_dream_worker(self, *, embed_fn=None, merge_fn=None) -> bool:
        """Start DreamWorker (importance prune + semantic dedupe)."""
        fs = self.fact_store
        if fs is None:
            return False
        if embed_fn is None and self._semantic is not None:
            try:
                emb = getattr(self._semantic, "_emb", None)
                if emb is not None and callable(emb):
                    embed_fn = emb
            except Exception:
                embed_fn = None
        if self._dream_worker is None:
            self._dream_worker = DreamWorker(
                fact_store    = fs,
                embed_fn      = embed_fn,
                semantic      = self._semantic,
                merge_fn      = merge_fn,
                #: L2/L3 живут в dream-цикле: там LLM-вызовы не стоят TTFT.
                scene_persona = self.scene_persona,
            )
        return self._dream_worker.start()

    def stop_dream_worker(self, *, wait: bool = False, timeout: float = 5.0) -> None:
        if self._dream_worker is not None:
            try:
                self._dream_worker.stop(wait=wait, timeout=timeout)
            except Exception as exc:
                logger.warning("dream worker stop failed: %s", exc)

    def enqueue_dream(self, company_id: str, user_id: str) -> None:
        if self._dream_worker is not None:
            try:
                self._dream_worker.enqueue(company_id, user_id)
            except Exception as exc:
                logger.debug("dream enqueue failed: %s", exc)


    def enqueue_memory_maintenance(self, company_id: str, user_id: str) -> None:
        self._maint_jobs.append((company_id, user_id))

    def flush_memory_maintenance(self, *, max_jobs: int = 4) -> int:
        """
        Drain queued maintenance — call from cron, worker tick, or after heavy writes.
        Returns number of user tenants processed.
        """
        processed = 0
        while self._maint_jobs and processed < max_jobs:
            cid, uid = self._maint_jobs.popleft()
            try:
                run_user_memory_maintenance(
                    store      = self._volatile,
                    company_id = cid,
                    user_id    = uid,
                    semantic   = self._semantic,
                )
                self.enqueue_dream(cid, uid)
                if self._long_term:
                    try:
                        pr = self._long_term.prune_user_chat_messages(cid, uid)
                        if pr.get("deleted_age") or pr.get("deleted_cap"):
                            logger.info(
                                "PG message retention (%s,%s): %s",
                                cid,
                                uid,
                                pr,
                            )
                    except Exception as exc:
                        logger.warning("PG message retention failed (%s,%s): %s", cid, uid, exc)
            except Exception as exc:
                logger.warning("Maintenance job failed (%s,%s): %s", cid, uid, exc)
            processed += 1
        return processed


    def load_company_kb_prompt_block(self, company_id: str, user_message: str) -> str:
        """
        RAG-only system block (company documents). Used when full ``load_context``
        is skipped (consciousness off, or no ``conversation_id``).
        """
        rag_docs = self._search_rag(company_id, user_message)
        if not rag_docs:
            return ""
        from finkey_memory.consciousness_state import UserProfile

        ctx = MemoryContext(
            company_id=company_id,
            user_id="",
            conversation_id="",
            user_profile=UserProfile(
                user_id="",
                user_name=None,
                role=None,
                company=company_id,
            ),
            session=SessionState(
                company_id=company_id,
                user_id="",
                conversation_id="",
            ),
            relevant_rag_docs=rag_docs,
        )
        return ctx.memory_prompt_block

    def load_company_kb_context_with_images(
        self,
        company_id: str,
        user_message: str,
    ) -> tuple[str, list[str]]:
        """
        Like ``load_company_kb_prompt_block`` but also returns deduplicated image
        data-URLs extracted from the retrieved document chunks.

        Returns ``(text_block, image_data_urls)``.  Both may be empty.
        """
        engine = self._rag_engine
        if engine is None or not (user_message or "").strip():
            return self.load_company_kb_prompt_block(company_id, user_message), []

        try:
            from finkey_memory.rag import RagRetrieveRequest, RagSource

            pack = engine.retrieve(  # type: ignore[attr-defined]
                RagRetrieveRequest(
                    query=user_message.strip(),
                    tenant_id=(company_id or "default").strip() or "default",
                    user_id=None,
                    sources=(RagSource.KNOWLEDGE,),
                ),
            )
            chunks = list(pack.chunks or ())
            if not chunks:
                return self.load_company_kb_prompt_block(company_id, user_message), []

            texts = [c.text for c in chunks if (c.text or "").strip()]
            seen: set[str] = set()
            images: list[str] = []
            for c in chunks:
                for url in getattr(c, "image_urls", ()):
                    if url and url not in seen:
                        seen.add(url)
                        images.append(url)

        except Exception as exc:
            logger.warning("load_company_kb_context_with_images failed: %s", exc)
            return self.load_company_kb_prompt_block(company_id, user_message), []

        if not texts:
            return "", images

        from finkey_memory.consciousness_state import UserProfile

        ctx = MemoryContext(
            company_id=company_id,
            user_id="",
            conversation_id="",
            user_profile=UserProfile(
                user_id="",
                user_name=None,
                role=None,
                company=company_id,
            ),
            session=SessionState(
                company_id=company_id,
                user_id="",
                conversation_id="",
            ),
            relevant_rag_docs=texts,
        )
        return ctx.memory_prompt_block, images

    def load_context(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
        user_message:    str,
    ) -> MemoryContext:
        """
        Assembles a full MemoryContext from all three layers.
        Missing layers degrade gracefully to empty data.
        """
        with LOAD_CONTEXT_LATENCY.time(company_id=company_id):
            return self._load_context_inner(company_id, user_id, conversation_id, user_message)

    def _load_context_inner(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
        user_message:    str,
    ) -> MemoryContext:
        # Все под-загрузки независимы (Redis / PG / Qdrant round-trips) —
        # выполняем их параллельно: последовательная цепочка из ~12 удалённых
        # вызовов давала 5-7 секунд к TTFT, параллельная — максимум самого
        # медленного (~1-2 с). Каждый воркер деградирует к пустым данным сам.
        from concurrent.futures import ThreadPoolExecutor

        substrates_enabled = _cognition_substrates_enabled()
        timing_on = _load_context_timing_enabled()
        timings: dict[str, float] = {}
        timings_lock = threading.Lock()

        def _timed(label: str, fn, *args):
            """Обёртка для замера каждой ветки: видно, кто именно держит TTFT."""
            if not timing_on:
                return fn(*args)
            t0 = time.perf_counter()
            try:
                return fn(*args)
            finally:
                with timings_lock:
                    timings[label] = (time.perf_counter() - t0) * 1000.0

        def _substrates() -> tuple[list[str], list[str], list[str], list[str]]:
            try:
                return self._load_four_substrates(company_id, user_id, user_message)
            except Exception as exc:
                logger.warning("Four-substrate memory load failed: %s", exc)
                return [], [], [], []

        wall0 = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=_load_context_workers(),
            thread_name_prefix="mem-load",
        ) as pool:
            fut_profile     = pool.submit(_timed, "profile", self._load_user_profile, company_id, user_id)
            fut_session     = pool.submit(_timed, "session", self._load_or_create_session, company_id, user_id, conversation_id)
            fut_arc         = pool.submit(_timed, "arc", self._load_arc, company_id, user_id, conversation_id)
            fut_dims        = pool.submit(_timed, "dims", self._load_dims, company_id, user_id, conversation_id)
            fut_history     = pool.submit(_timed, "history", self._load_history, company_id, user_id, conversation_id)
            fut_impressions = pool.submit(_timed, "impressions", self._load_impressions, company_id, user_id)
            fut_rag         = pool.submit(_timed, "rag", self._search_rag, company_id, user_message)
            fut_memories    = pool.submit(_timed, "memories", self._search_memories, company_id, user_id, user_message)
            fut_identity    = pool.submit(_timed, "identity", self._load_identity_pin, company_id, user_id)
            fut_durable     = pool.submit(_timed, "durable", self._load_durable_facts, company_id, user_id)
            fut_fact_hits   = pool.submit(_timed, "fact_hits", self._search_user_facts, company_id, user_id, user_message)
            fut_substrates  = pool.submit(_timed, "substrates", _substrates)
            fut_persona     = pool.submit(_timed, "persona", self._load_persona_body, company_id, user_id)
            fut_scenes      = pool.submit(_timed, "scenes", self._load_scene_blocks, company_id, user_id)

            user_profile       = fut_profile.result()
            session            = fut_session.result()
            arc                = fut_arc.result()
            dims               = fut_dims.result()
            history            = fut_history.result()
            impressions        = fut_impressions.result()
            rag_docs           = fut_rag.result()
            memories           = fut_memories.result()
            identity_pin       = fut_identity.result()
            durable_facts, fact_history = fut_durable.result()
            relevant_fact_hits = fut_fact_hits.result()
            epi, sem, emo_lines, proc = fut_substrates.result()
            persona_body       = fut_persona.result()
            scene_blocks       = fut_scenes.result()

        if timing_on:
            wall_ms = (time.perf_counter() - wall0) * 1000.0
            slowest = sorted(timings.items(), key=lambda kv: -kv[1])[:4]
            logger.info(
                "[finkey.memory] load_context wall=%.0fms slowest=%s",
                wall_ms,
                " ".join(f"{k}={v:.0f}ms" for k, v in slowest),
            )

        if _entity_expand_enabled() and user_message.strip():
            try:
                relevant_fact_hits = self._merge_entity_fact_hits(
                    company_id, user_id, user_message, durable_facts, relevant_fact_hits
                )
            except Exception as exc:
                logger.debug("entity fact expand skipped: %s", exc)

        ctx = MemoryContext(
            company_id          = company_id,
            user_id             = user_id,
            conversation_id     = conversation_id,
            user_profile        = user_profile,
            session             = session,
            emotional_arc       = arc,
            emotional_dims      = dims,
            recent_history      = history,
            relevant_rag_docs   = rag_docs,
            relevant_memories   = memories,
            user_impressions    = impressions,
            identity_pin        = identity_pin,
            durable_facts       = durable_facts,
            relevant_fact_hits  = relevant_fact_hits,
            fact_history        = fact_history,
            persona             = persona_body,
            scene_nav           = self._scene_nav_rows(scene_blocks),
            episodic_highlights = epi,
            semantic_facts      = sem,
            emotional_memory    = emo_lines,
            procedural_hints    = proc,
            cognition_substrates_enabled = substrates_enabled,
        )

        # Частотное воскрешение: пишем после сборки контекста, одним батчем.
        self._touch_recalled_facts(company_id, user_id, durable_facts, relevant_fact_hits)
        # Heat сцен — тот же принцип (после сборки, вне пула, один UPDATE).
        self._bump_scene_heat(
            company_id,
            user_id,
            scene_blocks,
            durable_facts,
            relevant_fact_hits,
            memories,
        )
        return ctx


    def save_turn(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
        user_turn:       ConversationTurn,
        assistant_turn:  ConversationTurn,
        updated_arc:     list[EmotionType],
        cognition:       Optional[CognitionTurnSnapshot] = None,
    ) -> None:
        """Persist everything from this turn to all relevant stores."""

        if self._session:
            try:
                self._session.push_message(company_id, user_id, conversation_id, user_turn)
                self._session.push_message(company_id, user_id, conversation_id, assistant_turn)
            except Exception as exc:
                logger.warning("Redis message push failed: %s", exc)

        if self._session:
            try:
                self._session.save_arc(company_id, user_id, conversation_id, updated_arc)
            except Exception as exc:
                logger.warning("Redis arc save failed: %s", exc)

        if self._session:
            try:
                self._session.touch_session(company_id, user_id, conversation_id)
            except Exception as exc:
                logger.warning("Redis touch failed: %s", exc)

        if self._long_term:
            try:
                self._long_term.save_message(company_id, conversation_id, user_id, user_turn)
                self._long_term.save_message(company_id, conversation_id, user_id, assistant_turn)
            except Exception as exc:
                logger.error("PostgreSQL message save failed: %s", exc)

        if self._long_term:
            try:
                dominant = updated_arc[-1].value if updated_arc else None
                self._long_term.update_conversation_stats(
                    company_id       = company_id,
                    conv_id          = conversation_id,
                    dominant_emotion = dominant,
                )
            except Exception as exc:
                logger.warning("PostgreSQL conv stats update failed: %s", exc)

        if self._long_term:
            try:
                session = self._load_or_create_session(company_id, user_id, conversation_id)
                if session.turn_number > 0 and session.turn_number % SUMMARY_TRIGGER_EVERY == 0:
                    self._long_term.enqueue_summary(company_id, conversation_id)
            except Exception as exc:
                logger.warning("Summary job enqueue failed: %s", exc)

        if cognition and _cognition_substrates_enabled():
            try:
                self._persist_four_substrates_after_turn(
                    company_id       = company_id,
                    user_id          = user_id,
                    conversation_id  = conversation_id,
                    user_text        = _turn_plain_text(user_turn.content),
                    assistant_text   = _turn_plain_text(assistant_turn.content),
                    cognition        = cognition,
                )
            except Exception as exc:
                logger.warning("Four-substrate persistence failed: %s", exc)

        n = self._volatile.bump_interaction(company_id, user_id)
        if self._long_term:
            try:
                pe = _pg_prune_every_n_turns()
                if pe > 0 and n > 0 and n % pe == 0:
                    pr = self._long_term.prune_user_chat_messages(company_id, user_id)
                    if pr.get("deleted_age") or pr.get("deleted_cap"):
                        logger.info("PG message retention (save_turn n=%s): %s", n, pr)
            except Exception as exc:
                logger.warning("PG message retention (save_turn): %s", exc)

        if _MAINT_INTERVAL > 0 and n % _MAINT_INTERVAL == 0:
            self.enqueue_memory_maintenance(company_id, user_id)
        if self._maint_jobs:
            self.flush_memory_maintenance(max_jobs=1)

        if self._extractor_runner is not None and self._extractor_runner.enabled:
            try:
                with self._extract_turn_lock:
                    self._extract_turn_counts[(company_id, conversation_id)] += 1
                    turn_idx = self._extract_turn_counts[(company_id, conversation_id)]
                self._extractor_runner.schedule(
                    company_id      = company_id,
                    user_id         = user_id,
                    conversation_id = conversation_id,
                    turn_index      = turn_idx,
                    fallback_recent_turns = [
                        {"role": user_turn.role,      "content": _turn_plain_text(user_turn.content)},
                        {"role": assistant_turn.role, "content": _turn_plain_text(assistant_turn.content)},
                    ],
                )
            except Exception as exc:
                logger.warning("MemoryExtractor schedule failed: %s", exc)

    def shutdown(self, *, wait: bool = False) -> None:
        """Корректно остановить background-runner-ы (для graceful shutdown)."""
        if self._extractor_runner is not None:
            try:
                self._extractor_runner.shutdown(wait=wait)
            except Exception as exc:
                logger.warning("extractor runner shutdown failed: %s", exc)
        self.stop_summary_worker(wait=wait)
        self.stop_dream_worker(wait=wait)
        self.stop_decay_scheduler(wait=wait)


    def save_dims(
        self,
        company_id:      str,
        user_id:         str,
        conversation_id: str,
        dims,
    ) -> None:
        if self._session:
            try:
                self._session.save_dims(company_id, user_id, conversation_id, dims)
            except Exception as exc:
                logger.warning("Redis dims save failed: %s", exc)


    def _load_four_substrates(
        self,
        company_id:   str,
        user_id:      str,
        user_message: str,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Episodic / facts / traces / playbook from volatile core + vectors."""
        episodic = self._volatile.recent_episode_lines(company_id, user_id, limit=8)
        facts = self._volatile.fact_lines_for_prompt(company_id, user_id, limit=16)
        emo_tr = self._volatile.recent_emotional_lines(company_id, user_id, limit=8)
        proc = self._volatile.procedural_lines(company_id, user_id, limit=6)

        query = user_message.strip()
        if self._semantic and query:
            # Четыре независимых векторных поиска: последовательно это 4 round-trip
            # в Qdrant внутри одного воркера load_context, и вся загрузка памяти
            # ждала именно эту ветку. Запрос к embedder один и тот же — single-flight
            # в embedder'е отдаёт его всем четырём без повторного HTTP.
            from concurrent.futures import ThreadPoolExecutor

            probes = (
                ("episodic", 4, 0.48),
                ("procedural", 3, 0.48),
                ("semantic", 3, 0.5),
                ("emotional", 3, 0.45),
            )

            def _probe(kind: str, top_k: int, score_min: float) -> list[str]:
                try:
                    return self._semantic.search_user_memories(
                        company_id, user_id, query,
                        top_k=top_k, score_min=score_min, deep_kinds=[kind],
                    )
                except Exception as exc:
                    logger.warning("Substrate probe %s failed: %s", kind, exc)
                    return []

            with ThreadPoolExecutor(
                max_workers=len(probes),
                thread_name_prefix="mem-substrate",
            ) as pool:
                futures = [pool.submit(_probe, k, n, s) for k, n, s in probes]
                extra_epi, extra_proc, extra_sem, extra_em = [f.result() for f in futures]

            for src, dst in (
                (extra_epi, episodic),
                (extra_proc, proc),
                (extra_sem, facts),
                (extra_em, emo_tr),
            ):
                for x in src:
                    xl = x.strip()
                    if xl and xl not in dst:
                        dst.insert(0, xl)

        return episodic[:8], facts[:14], emo_tr[:8], proc[:8]

    def _persist_four_substrates_after_turn(
        self,
        company_id:       str,
        user_id:          str,
        conversation_id:  str,
        user_text:        str,
        assistant_text:    str,
        cognition:       CognitionTurnSnapshot,
    ) -> None:
        ep = episode_from_snapshot(
            company_id, user_id, conversation_id or "",
            user_text, assistant_text, cognition,
        )

        self._volatile.save_memory_episode_snapshot(
            company_id, user_id,
            episode_to_plain_dict(ep),
            ep.id,
        )

        self._volatile.append_emotional_trace(
            company_id,
            user_id,
            user_emotion         = cognition.primary_emotion or "neutral",
            user_intensity       = float(cognition.emotion_intensity or 0.0),
            finkey_tenderness    = float(cognition.tenderness or 0.0),
            resonance            = min(
                1.0,
                float(cognition.tenderness or 0.0) * 0.65 + float(cognition.concern or 0.0) * 0.35,
            ),
            finkey_inner_feeling = cognition.inner_feeling or "",
        )

        self._volatile.upsert_fact(
            company_id, user_id,
            "dominant_emotion",
            cognition.primary_emotion or "neutral",
            confidence = float(0.45 + min(0.35, cognition.emotion_intensity * 0.35)),
            source     = "inferred",
            priority   = 0.36,
        )

        cn = cognition.core_need.strip()
        if len(cn) > 14:
            self._volatile.upsert_fact(
                company_id, user_id,
                "core_need_focus",
                clip_fact_value(cn, max_chars=900),
                confidence = 0.52,
                source     = "inferred",
                priority   = 0.78,
            )

        st = cognition.subtext_insight.strip()
        if len(st) > 20:
            self._volatile.upsert_fact(
                company_id, user_id,
                "subtext_read",
                clip_fact_value(st, max_chars=800),
                confidence = 0.42,
                source     = "inferred",
                priority   = 0.48,
            )

        if cognition.perception_literal.strip():
            lit = cognition.perception_literal.strip()
            self._volatile.upsert_fact(
                company_id, user_id,
                "surface_message_snapshot",
                clip_fact_value(lit, max_chars=600),
                confidence = 0.38,
                source     = "inferred",
                priority   = 0.22,
            )

        playbook = (
            f"Response tone: {cognition.tone_name}\n"
            f"Approach: {cognition.approach}\n"
            f"Validate first: {'yes' if cognition.validate_first else 'no'}\n"
        ).strip()

        self._volatile.upsert_procedural_memory(
            company_id, user_id,
            title   = "interaction_playbook",
            content = playbook[:3500],
            tags    = ["auto", "procedural", cognition.topic_type or "general"],
        )

        if not self._semantic:
            return

        try:
            self._semantic.upsert_user_memory(
                company_id     = company_id,
                user_id        = user_id,
                memory_text    = clip_fact_value(ep.to_text_summary(), max_chars=1200),
                memory_type    = "episodic_trace",
                source_conv_id = conversation_id if conversation_id else None,
                deep_kind      = "episodic",
            )
            self._semantic.upsert_user_memory(
                company_id     = company_id,
                user_id        = user_id,
                memory_text    = clip_fact_value(playbook, max_chars=950),
                memory_type    = "procedural_hint",
                deep_kind      = "procedural",
                source_conv_id = conversation_id if conversation_id else None,
            )

            compact_sem = clip_fact_value(
                " | ".join(
                    filter(
                        None,
                        (
                            f"emotion={cognition.primary_emotion}",
                            f"need={cn[:160]}" if cn else None,
                            f"subtext={st[:160]}" if st else None,
                        ),
                    )
                ),
                max_chars=900,
            )
            if len(compact_sem) > 26:
                self._semantic.upsert_user_memory(
                    company_id     = company_id,
                    user_id        = user_id,
                    memory_text    = compact_sem,
                    memory_type    = "semantic_bundle",
                    source_conv_id = conversation_id if conversation_id else None,
                    deep_kind      = "semantic",
                )

            emo_pack = (
                f"{cognition.primary_emotion or 'neutral'} (i={cognition.emotion_intensity:.2f}); "
                f"FinKey tenderness={cognition.tenderness:.2f}; concern={cognition.concern:.2f}; "
                f"{(cognition.inner_feeling or '')[:180]}"
            )
            self._semantic.upsert_user_memory(
                company_id     = company_id,
                user_id        = user_id,
                memory_text    = clip_fact_value(emo_pack, max_chars=800),
                memory_type    = "emotional_vector",
                source_conv_id = conversation_id if conversation_id else None,
                deep_kind      = "emotional",
            )
        except Exception as exc:
            logger.warning("Qdrant four-substrate vectors failed: %s", exc)

    def save_impression(self, impression: UserImpression) -> None:
        """Volatile merge first; optional Postgres mirror once your DDL exists."""
        try:
            self._volatile.upsert_impression(impression)
        except Exception as exc:
            logger.warning("Volatile impression save failed: %s", exc)

        if self._long_term:
            try:
                self._long_term.upsert_impression(impression)
            except Exception as exc:
                logger.error("Impression PostgreSQL save failed: %s", exc)

        if self._semantic:
            try:
                self._semantic.upsert_user_memory(
                    company_id     = impression.company_id,
                    user_id        = impression.user_id,
                    memory_text    = impression.content,
                    memory_type    = impression.impression_type,
                    source_conv_id = impression.source_conv_id,
                )
            except Exception as exc:
                logger.warning("Qdrant user memory save failed: %s", exc)


    def _load_user_profile(self, company_id: str, user_id: str) -> UserProfile:
        if self._long_term:
            try:
                row = self._long_term.get_user_profile(company_id, user_id)
                if row:
                    return UserProfile(
                        user_id   = user_id,
                        user_name = row.get("user_name"),
                        role      = row.get("role"),
                        company   = row.get("company_name"),
                    )
            except Exception as exc:
                logger.warning("Profile load failed: %s", exc)
        return UserProfile(user_id=user_id)

    def _load_or_create_session(
        self, company_id: str, user_id: str, conv_id: str,
    ) -> SessionState:
        if self._session:
            try:
                s = self._session.load_session(company_id, user_id, conv_id)
                if s:
                    return s
            except Exception as exc:
                logger.warning("Session load failed: %s", exc)
        return SessionState(
            company_id=company_id, user_id=user_id, conversation_id=conv_id,
        )

    def _load_arc(
        self, company_id: str, user_id: str, conv_id: str,
    ) -> list[EmotionType]:
        if self._session:
            try:
                return self._session.load_arc(company_id, user_id, conv_id)
            except Exception as exc:
                logger.warning("Arc load failed: %s", exc)
        return []

    def _load_dims(self, company_id: str, user_id: str, conv_id: str):
        if self._session:
            try:
                return self._session.load_dims(company_id, user_id, conv_id)
            except Exception as exc:
                logger.warning("Dims load failed: %s", exc)
        return None

    def _load_history(self, company_id: str, user_id: str, conv_id: str) -> list[ConversationTurn]:
        if self._session:
            try:
                return self._session.load_recent_messages(company_id, user_id, conv_id)
            except Exception as exc:
                logger.warning("History load failed: %s", exc)
        return []

    def _load_impressions(self, company_id: str, user_id: str) -> list[UserImpression]:
        volatile_sorted = []
        try:
            volatile_sorted = self._volatile.list_impressions(company_id, user_id)
        except Exception as exc:
            logger.warning("Volatile impressions failed: %s", exc)

        if not self._long_term:
            return volatile_sorted

        try:
            pg = self._long_term.get_user_impressions(company_id, user_id)
        except Exception as exc:
            logger.warning("Impressions load failed: %s", exc)
            return volatile_sorted

        by_type = {i.impression_type: i for i in pg}
        for lv in reversed(volatile_sorted):
            by_type[lv.impression_type] = lv
        return sorted(by_type.values(), key=lambda x: (-x.priority, -x.confidence))[:56]

    def _search_rag(self, company_id: str, query: str) -> list[str]:
        """
        Production path: ``RagEngine`` (hybrid dense+BM25 + RRF + rerank + cache).
        Fallback 1: in-process advanced pipeline over plain dense Qdrant.
        Fallback 2: legacy ``semantic.search_rag``.
        """
        engine = self._rag_engine
        if engine is not None and (query or "").strip():
            try:
                from finkey_memory.rag import RagRetrieveRequest, RagSource

                pack = engine.retrieve(  # type: ignore[attr-defined]
                    RagRetrieveRequest(
                        query=query.strip(),
                        tenant_id=company_id,
                        user_id=None,
                        sources=(RagSource.KNOWLEDGE,),
                    ),
                )
                chunks = list(pack.chunks or ())
                if chunks:
                    return [c.text for c in chunks if (c.text or "").strip()]
            except Exception as exc:
                logger.warning("RagEngine retrieve failed, falling back: %s", exc)

        if not self._semantic:
            return []
        try:
            from finkey_memory.rag.pipeline import retrieve_company_rag

            redis_client = None
            if self._session is not None:
                redis_client = getattr(self._session, "client", None)
            return retrieve_company_rag(
                semantic=self._semantic,
                company_id=company_id,
                query=query,
                redis_client=redis_client,
            )
        except Exception as exc:
            logger.warning("Advanced RAG pipeline failed, fallback search_rag: %s", exc)
            try:
                return self._semantic.search_rag(company_id, query, top_k=5)
            except Exception as exc2:
                logger.warning("RAG search failed: %s", exc2)
                return []

    def _search_memories(self, company_id: str, user_id: str, query: str) -> list[str]:
        if self._semantic:
            try:
                past = self._semantic.search_past_conversations(
                    company_id, user_id, query, top_k=3,
                )
                user_mems = self._semantic.search_user_memories(
                    company_id, user_id, query, top_k=3,
                )
                return past + user_mems
            except Exception as exc:
                logger.warning("Memory search failed: %s", exc)
        return []


    def get_identity_pin(self, company_id: str, user_id: str) -> str:
        """Public accessor — same Redis text block `_load_identity_pin` uses
        for the system prompt. Read-only, no side effects; safe to call from
        a stateless HTTP route (e.g. a recall-snapshot UI)."""
        return self._load_identity_pin(company_id, user_id)

    def get_session_snapshot(self, company_id: str, user_id: str) -> Optional[dict]:
        """Tone/topic of the user's most recent conversation, for callers that
        don't have a `conversation_id` on hand (e.g. a snapshot UI). Resolves
        "most recent" via `LongTermMemory.get_user_recent_conversations`, then
        reads that conversation's live Redis session state. Returns ``None``
        when long-term/session storage is unavailable or the user has no
        conversations yet — never fabricated content.
        """
        if self._long_term is None or self._session is None:
            return None
        try:
            recent = self._long_term.get_user_recent_conversations(
                company_id, user_id, limit=1,
            )
        except Exception as exc:
            logger.debug("get_session_snapshot: recent conversations lookup failed: %s", exc)
            return None
        if not recent:
            return None
        conv_id = str(recent[0].get("id") or "").strip()
        if not conv_id:
            return None
        try:
            state = self._session.load_session(company_id, user_id, conv_id)
        except Exception as exc:
            logger.debug("get_session_snapshot: load_session failed: %s", exc)
            return None
        if state is None:
            return None
        return {
            "conversation_id": conv_id,
            "last_topic": state.last_topic,
            "relationship_tone": state.relationship_tone,
            "turn_number": state.turn_number,
        }

    def get_semantic_snapshot(self, company_id: str, user_id: str, *, limit: int = 24) -> list[dict]:
        """List (not search) a user's Qdrant semantic impressions — see
        `SemanticMemory.scroll_user_memories`. `[]` when Qdrant isn't
        configured or the user has none yet."""
        if self._semantic is None:
            return []
        try:
            return self._semantic.scroll_user_memories(company_id, user_id, limit=limit)
        except Exception as exc:
            logger.debug("get_semantic_snapshot failed: %s", exc)
            return []

    def _load_identity_pin(self, company_id: str, user_id: str) -> str:
        """Готовый текстовый блок из Redis (см. ``RedisKeys.identity_pin``)."""
        if self._session is None:
            return ""
        try:
            client = self._session.client
            value = client.get(RedisKeys.identity_pin(company_id, user_id))
        except Exception as exc:
            logger.debug("identity_pin load failed: %s", exc)
            return ""
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="replace")
            except Exception:
                return ""
        return str(value).strip()

    def _load_durable_facts(
        self, company_id: str, user_id: str
    ) -> tuple[list[dict], dict[str, tuple[str, float]]]:
        """
        Top-N активных user_facts из PG + история вытесненных значений.

        Без PG — пустые коллекции (volatile отдаёт своё через _load_four_substrates).

        Порядок строк сортируем сами: ORDER BY в PG закрывает только
        priority/confidence/touched, а prompt берёт «первые N на группу» — без
        своей сортировки (priority DESC, updated_at DESC) в капы попадали не
        самые свежие факты. Сортировка стабильная, поэтому равенство полей
        сохраняет порядок PG.

        ``{key}__asof__{ts}`` — архив, в промпт он не идёт, но из него собирается
        history base_key → (старое value, ts ухода) для «was: … until …».
        """
        fs = self.fact_store
        if fs is None:
            return [], {}
        try:
            rows = fs.list_user_facts(
                company_id=company_id,
                user_id=user_id,
                limit=_durable_facts_limit() + 16,
            )

            active: list[dict] = []
            history: dict[str, tuple[str, float]] = {}
            for r in rows or []:
                kn = str(r.get("key_normalized") or "")
                if "__asof__" not in kn:
                    active.append(r)
                    continue
                base, _, ts_raw = kn.partition("__asof__")
                if not base:
                    continue
                try:
                    ts = float(ts_raw)
                except (TypeError, ValueError):
                    continue
                prev = history.get(base)
                if prev is None or ts > prev[1]:
                    history[base] = (str(r.get("value") or ""), ts)

            active.sort(key=_durable_sort_key)
            return active[: _durable_facts_limit()], history
        except Exception as exc:
            logger.warning("durable_facts load failed: %s", exc)
            return [], {}

    def _load_persona_body(self, company_id: str, user_id: str) -> str:
        """
        L3 working persona для промпта. '' — когда PG/флага/строки нет.

        Читается уже обрезанным (кап 2000 в ``scene_persona.clip_persona_body``),
        поэтому schema-рендер лишь вписывает его в общий бюджет блока.
        """
        svc = self.scene_persona
        if svc is None:
            return ""
        try:
            return svc.load_persona_body(company_id, user_id)
        except Exception as exc:  # noqa: BLE001 — портрет не держит TTFT
            logger.debug("persona load skipped: %s", exc)
            return ""

    def _load_scene_blocks(self, company_id: str, user_id: str) -> list[SceneBlock]:
        """L2 сцены (top-``_scene_nav_top`` по heat). Пусто — если слоёв нет/выключены."""
        svc = self.scene_persona
        if svc is None:
            return []
        try:
            return svc.load_scenes(company_id, user_id, limit=_scene_nav_top() * 3)
        except Exception as exc:  # noqa: BLE001
            logger.debug("scene load skipped: %s", exc)
            return []

    def _scene_nav_rows(self, scenes: list[SceneBlock]) -> list[tuple[str, str, int]]:
        """(key, summary, heat) горячих сцен — ровно то, что видит модель."""
        svc = self.scene_persona
        if svc is None or not scenes:
            return []
        try:
            return svc.scene_nav(scenes, top=_scene_nav_top())
        except Exception as exc:  # noqa: BLE001
            logger.debug("scene nav skipped: %s", exc)
            return []

    def _bump_scene_heat(
        self,
        company_id: str,
        user_id: str,
        scenes: list[SceneBlock],
        durable_facts: list[dict],
        fact_hits: list[dict],
        memory_lines: list[str],
    ) -> None:
        """
        Heat++ за попавшее в контекст: ключ факта из чьей-то сцены либо векторный
        hit саммари сцены. Один батчевый UPDATE, после сборки контекста.
        """
        if not scenes:
            return
        svc = self.scene_persona
        if svc is None:
            return
        keys: list[str] = []
        for row in durable_facts or []:
            kn = str(row.get("key_normalized") or "")
            if kn:
                keys.append(kn)
        for hit in fact_hits or []:
            kn = str(hit.get("key_normalized") or hit.get("key") or "")
            if kn:
                keys.append(kn)
        try:
            svc.bump_recall_heat(
                company_id,
                user_id,
                scenes,
                recalled_fact_keys    = keys,
                recalled_memory_lines = memory_lines,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("scene heat bump skipped: %s", exc)

    def _merge_entity_fact_hits(
        self,
        company_id: str,
        user_id: str,
        query: str,
        durable_facts: list[dict],
        hits: list[dict],
    ) -> list[dict]:
        """Multi-hop entity expand → extra fact rows into relevant_fact_hits."""
        from finkey_memory.entity_memory import (
            get_entity_index,
            load_entity_index_redis,
            rebuild_entity_index,
        )

        limit = _entity_expand_limit()
        if limit <= 0:
            return hits

        idx = None
        if self._session is not None:
            try:
                idx = load_entity_index_redis(self._session.client, company_id, user_id)
            except Exception:
                idx = None
        if idx is None or not idx.nodes:
            pairs = [
                (str(r.get("key_normalized") or ""), str(r.get("value") or ""))
                for r in (durable_facts or [])
                if r.get("key_normalized")
            ]
            if pairs:
                idx = rebuild_entity_index(company_id, user_id, pairs)
            else:
                idx = get_entity_index(company_id, user_id)
        keys = idx.expand_from_query(query, max_facts=limit)
        if not keys:
            return hits

        by_key = {
            str(r.get("key_normalized") or ""): r
            for r in (durable_facts or [])
            if r.get("key_normalized")
        }
        seen = {
            str(h.get("key_normalized") or h.get("key") or "")
            for h in (hits or [])
        }
        out = list(hits or [])
        for k in keys:
            if not k or k in seen:
                continue
            row = by_key.get(k)
            if row is None:
                fs = self.fact_store
                if fs is not None:
                    try:
                        row = fs.get_user_fact(company_id, user_id, k)
                    except Exception:
                        row = None
            if not row:
                continue
            out.append({
                "key_normalized": row.get("key_normalized") or k,
                "value": row.get("value"),
                "category": row.get("category"),
                "score": 0.62,
                "source": "entity_expand",
            })
            seen.add(k)
            if len(out) >= _fact_hits_limit() + limit:
                break
        return out

    def _search_user_facts(self, company_id: str, user_id: str, query: str) -> list[dict]:
        """Семантический recall по коллекции ``finkey_user_facts`` + read-time вес."""
        if self._semantic is None or not query.strip():
            return []
        uid = user_id
        if self._long_term is not None:
            try:
                resolved = self._long_term._resolve_user_id(company_id, user_id)  # noqa: SLF001
                if resolved:
                    uid = resolved
            except Exception:
                pass
        try:
            hits = self._semantic.search_user_facts(
                company_id=company_id,
                user_id=uid,
                query=query.strip(),
                top_k=_fact_hits_limit(),
                score_min=0.55,
            )
        except Exception as exc:
            logger.debug("search_user_facts failed: %s", exc)
            return []
        try:
            return _rerank_fact_hits(hits)
        except Exception as exc:  # noqa: BLE001 — весим, не роняем recall
            logger.debug("fact hit rerank skipped: %s", exc)
            return hits

    def _touch_recalled_facts(
        self,
        company_id: str,
        user_id: str,
        durable_facts: list[dict],
        fact_hits: list[dict],
    ) -> None:
        """
        Один batched access-count инкремент за ход — ПОСЛЕ сборки контекста и вне
        пула загрузки (единственный write в hot-path, он не должен стоить TTFT).

        Без PG / без колонки ``access_count`` FactStore.touch_facts сам
        деградирует в no-op с одним warning на процесс.
        """
        keys: list[str] = []
        for row in (durable_facts or []):
            kn = str(row.get("key_normalized") or "")
            if kn:
                keys.append(kn)
        for hit in (fact_hits or []):
            kn = str(hit.get("key_normalized") or hit.get("key") or "")
            if kn:
                keys.append(kn)
        if not keys:
            return
        fs = self.fact_store
        if fs is None:
            return
        try:
            fs.touch_facts(company_id=company_id, user_id=user_id, keys=keys)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fact access bump skipped: %s", exc)
