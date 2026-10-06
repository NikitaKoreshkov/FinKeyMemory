# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Conversation summary worker — episodic digests for long chats.

Claims ``summary_jobs`` from Postgres, builds a summary of recent messages,
writes ``conversations.summary`` and upserts into Qdrant ``finkey_conv_memories``.

Env:
  FINKEY_MEMORY_SUMMARY_ENABLED       default 1
  FINKEY_MEMORY_SUMMARY_INTERVAL_SEC  default 45
  FINKEY_MEMORY_SUMMARY_BATCH         default 3
  FINKEY_MEMORY_SUMMARY_MAX_MSGS      default 80
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

logger = logging.getLogger("finkey.memory.summary_worker")

if TYPE_CHECKING:
    from finkey_memory.long_term import LongTermMemory
    from finkey_memory.semantic import SemanticMemory


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


GenerateFn = Callable[[list[dict], int, float], str]


def heuristic_summarize(messages: list[dict], *, max_chars: int = 1200) -> str:
    """Zero-LLM fallback: compress roles into a readable episodic blurb."""
    parts: list[str] = []
    for m in messages[-40:]:
        role = str(m.get("role") or "?")
        content = re.sub(r"\s+", " ", str(m.get("content") or "")).strip()
        if not content:
            continue
        parts.append(f"{role}: {content[:220]}")
    blob = " | ".join(parts)
    if len(blob) <= max_chars:
        return blob or "(empty conversation)"
    return blob[: max_chars - 20] + " …"


def llm_summarize(messages: list[dict], generate_fn: GenerateFn) -> str:
    lines = []
    for m in messages[-60:]:
        role = str(m.get("role") or "?")
        content = str(m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content[:500]}")
    prompt = (
        "Summarize this conversation for long-term episodic memory. "
        "Include: user goals, decisions, preferences, open questions, tone. "
        "Max 180 words. Same language as the conversation.\n\n"
        + "\n".join(lines)
    )
    try:
        try:
            out = (generate_fn([{"role": "user", "content": prompt}], 400, 0.2) or "").strip()
        except TypeError:
            out = (generate_fn([{"role": "user", "content": prompt}], max_tokens=400, temperature=0.2) or "").strip()
        return out or heuristic_summarize(messages)
    except Exception as exc:
        logger.warning("LLM summarize failed: %s", exc)
        return heuristic_summarize(messages)


def extract_topics(summary: str, *, max_topics: int = 6) -> list[str]:
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]{4,}", summary or "")
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        k = w.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(w)
        if len(out) >= max_topics:
            break
    return out


@dataclass
class SummaryWorkerStats:
    ticks: int = 0
    jobs_ok: int = 0
    jobs_fail: int = 0
    qdrant_ok: int = 0
    last_run_ts: float = 0.0


@dataclass
class SummaryWorker:
    long_term: "LongTermMemory"
    semantic: Optional["SemanticMemory"] = None
    generate_fn: Optional[GenerateFn] = None
    interval_sec: int = field(default_factory=lambda: _env_int("FINKEY_MEMORY_SUMMARY_INTERVAL_SEC", 45))
    batch_size: int = field(default_factory=lambda: _env_int("FINKEY_MEMORY_SUMMARY_BATCH", 3))
    max_msgs: int = field(default_factory=lambda: _env_int("FINKEY_MEMORY_SUMMARY_MAX_MSGS", 80))
    enabled: bool = field(default_factory=lambda: _env_bool("FINKEY_MEMORY_SUMMARY_ENABLED", True))
    initial_delay: int = field(default_factory=lambda: _env_int("FINKEY_MEMORY_SUMMARY_INITIAL_SEC", 20))
    stats: SummaryWorkerStats = field(default_factory=SummaryWorkerStats)

    _thread: Optional[threading.Thread] = None
    _stop_evt: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> bool:
        with self._lock:
            if not self.enabled:
                logger.info("SummaryWorker disabled by env")
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._loop, name="finkey-summary-worker", daemon=True)
            self._thread.start()
            logger.info("SummaryWorker started interval=%ss batch=%s", self.interval_sec, self.batch_size)
            return True

    def stop(self, *, wait: bool = False, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        t = self._thread
        if wait and t is not None:
            t.join(timeout=timeout)

    def run_once(self, *, max_jobs: Optional[int] = None) -> int:
        """Process up to N jobs synchronously (tests / flush)."""
        n = max_jobs if max_jobs is not None else self.batch_size
        done = 0
        for _ in range(max(1, n)):
            if not self._process_one():
                break
            done += 1
        self.stats.ticks += 1
        self.stats.last_run_ts = time.time()
        return done

    def _loop(self) -> None:
        if self.initial_delay > 0:
            self._stop_evt.wait(self.initial_delay)
        while not self._stop_evt.is_set():
            try:
                self.run_once()
            except Exception as exc:
                logger.warning("SummaryWorker tick failed: %s", exc)
            self._stop_evt.wait(max(5, self.interval_sec))

    def _process_one(self) -> bool:
        job = self.long_term.claim_summary_job()
        if not job:
            return False
        job_id = str(job.get("id") or "")
        company_id = str(job.get("company_id") or "")
        conv_id = str(job.get("conversation_id") or "")
        try:
            msgs = self.long_term.get_conversation_messages(
                company_id, conv_id, limit=self.max_msgs, offset=0
            )
            if self.generate_fn is not None:
                summary = llm_summarize(msgs, self.generate_fn)
            else:
                summary = heuristic_summarize(msgs)
            self.long_term.save_conversation_summary(company_id, conv_id, summary)
            topics = extract_topics(summary)
            # Resolve user for Qdrant scope
            user_id = self._user_for_conv(company_id, conv_id, msgs)
            if self.semantic is not None and user_id and summary.strip():
                try:
                    self.semantic.upsert_conversation_memory(
                        company_id=company_id,
                        user_id=user_id,
                        conv_id=conv_id,
                        summary=summary,
                        topics=topics,
                    )
                    self.stats.qdrant_ok += 1
                except Exception as exc:
                    logger.warning("conv memory upsert failed: %s", exc)
            self.long_term.complete_summary_job(job_id, error=None)
            self.stats.jobs_ok += 1
            return True
        except Exception as exc:
            logger.warning("summary job %s failed: %s", job_id, exc)
            try:
                self.long_term.complete_summary_job(job_id, error=str(exc)[:500])
            except Exception:
                pass
            self.stats.jobs_fail += 1
            return True  # consumed

    def _user_for_conv(self, company_id: str, conv_id: str, msgs: list[dict]) -> str:
        # Prefer external id from conversation owner if available via recent conv API is heavy;
        # fall back to company-scoped synthetic from message rows (user_id uuid).
        try:
            row = self.long_term._fetchone(  # noqa: SLF001
                "SELECT user_id::text AS uid FROM conversations WHERE id=%s AND company_id=%s",
                (conv_id, company_id),
            )
            if row and row.get("uid"):
                return str(row["uid"])
        except Exception:
            pass
        return f"{company_id}:unknown"
