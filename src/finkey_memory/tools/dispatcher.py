# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Execute memory_* tool calls against ``MemoryManager`` / Qdrant / PG."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from finkey_memory.extractor import ExtractedFact, FactCategory
from finkey_memory.manager import MemoryManager
from finkey_memory.rag.pipeline import retrieve_company_rag

logger = logging.getLogger(__name__)


class MemoryToolError(Exception):
    """User-facing tool error (returned in JSON)."""


class MemoryToolDispatcher:
    """
    Server-side executor for ``MEMORY_TOOLS``.

    ``user_id`` / ``company_id`` come from the authenticated chat request — tools cannot
    override tenant scope.
    """

    def __init__(
        self,
        mem: MemoryManager,
        *,
        company_id: str,
        user_id: str,
        conversation_id: str,
    ) -> None:
        self._mem = mem
        self._company_id = company_id
        self._user_id = user_id
        self._conv_id = conversation_id or ""

    def _redis(self) -> Any:
        session = self._mem._session  # noqa: SLF001
        return getattr(session, "client", None) if session else None

    def execute(self, name: str, arguments: Optional[dict[str, Any]]) -> str:
        """Returns JSON string for the OpenAI ``tool`` message."""
        args = arguments if isinstance(arguments, dict) else {}
        try:
            payload = self._dispatch(name, args)
            return json.dumps({"ok": True, **payload}, ensure_ascii=False)
        except MemoryToolError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        except Exception as exc:
            logger.exception("memory tool %r failed", name)
            return json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )

    def _dispatch(self, name: str, args: dict[str, Any]) -> dict:
        if name == "memory_search":
            return {"result": self._memory_search(args)}
        if name == "memory_save_fact":
            return self._memory_save_fact(args)
        if name == "memory_forget_fact":
            return self._memory_forget_fact(args)
        if name == "memory_list_facts":
            return {"facts": self._memory_list_facts(args)}
        raise MemoryToolError(f"unknown_tool:{name}")

    def _memory_search(self, args: dict[str, Any]) -> dict:
        query = (args.get("query") or "").strip()
        if not query:
            raise MemoryToolError("query_required")
        top_k = args.get("top_k")
        try:
            tk = int(top_k) if top_k is not None else 8
        except (TypeError, ValueError):
            tk = 8
        tk = max(1, min(tk, 24))

        scopes = args.get("scopes")
        if not isinstance(scopes, list) or not scopes:
            scopes = [
                "facts",
                "company_kb",
                "conversation_history",
                "user_impressions",
            ]

        uid = self._mem.resolve_user_id_for_vectors(self._company_id, self._user_id)
        sem = self._mem._semantic  # noqa: SLF001
        out: dict[str, Any] = {}

        if "facts" in scopes and sem is not None:
            try:
                out["facts"] = sem.search_user_facts(
                    company_id=self._company_id,
                    user_id=uid,
                    query=query,
                    top_k=tk,
                    score_min=0.42,
                )
            except Exception as exc:
                out["facts_error"] = str(exc)

        if "company_kb" in scopes:
            engine = getattr(self._mem, "rag_engine", None)
            kb_payload: list[str] | None = None
            if engine is not None:
                try:
                    from finkey_memory.rag import RagRetrieveRequest, RagSource

                    pack = engine.retrieve(
                        RagRetrieveRequest(
                            query=query,
                            tenant_id=self._company_id,
                            user_id=self._user_id,
                            sources=(RagSource.KNOWLEDGE,),
                        ),
                    )
                    if pack.chunks:
                        kb_payload = [
                            {
                                "text": c.text[:1600],
                                "score": float(c.score),
                                "doc_id": c.metadata.get("doc_id"),
                                "chunk_index": c.metadata.get("chunk_index"),
                            }
                            for c in pack.chunks
                        ]
                except Exception as exc:
                    out["company_kb_error"] = f"engine:{exc}"

            if kb_payload is None and sem is not None:
                try:
                    kb_payload = retrieve_company_rag(
                        semantic=sem,
                        company_id=self._company_id,
                        query=query,
                        redis_client=self._redis(),
                    )
                except Exception as exc:
                    out["company_kb_error"] = str(exc)
            if kb_payload is not None:
                out["company_kb"] = kb_payload

        if "conversation_history" in scopes and sem is not None:
            try:
                out["conversation_history"] = sem.search_past_conversations(
                    self._company_id,
                    uid,
                    query,
                    top_k=max(2, tk // 2),
                    score_min=0.48,
                )
            except Exception as exc:
                out["conversation_history_error"] = str(exc)

        if "user_impressions" in scopes and sem is not None:
            try:
                out["user_impressions"] = sem.search_user_memories(
                    self._company_id,
                    uid,
                    query,
                    top_k=max(2, tk // 2),
                    score_min=0.48,
                )
            except Exception as exc:
                out["user_impressions_error"] = str(exc)

        return out

    def _memory_save_fact(self, args: dict[str, Any]) -> dict:
        key = (args.get("key_normalized") or "").strip()
        value = (args.get("value") or "").strip()
        if not key or not value:
            raise MemoryToolError("key_and_value_required")
        cat = FactCategory.parse(str(args.get("category") or "other"))
        try:
            conf = float(args.get("confidence", 0.85))
        except (TypeError, ValueError):
            conf = 0.85
        conf = max(0.0, min(conf, 1.0))
        ev = (args.get("evidence_snippet") or "").strip()
        if len(ev) > 1200:
            ev = ev[:1200]

        fact = ExtractedFact(
            key_normalized=key,
            value=value,
            category=cat,
            confidence=conf,
            evidence_snippet=ev,
            source="assistant_tool",
            source_conv_id=self._conv_id or None,
        )
        fs = self._mem.fact_store
        if fs is None:
            raise MemoryToolError("fact_store_unavailable")
        rep = fs.upsert_extracted_facts(
            company_id=self._company_id,
            user_id=self._user_id,
            conversation_id=self._conv_id or None,
            facts=[fact],
        )
        return {
            "saved": rep.pg_ok > 0,
            "pg_ok": rep.pg_ok,
            "written_keys": rep.written_keys,
            "errors": rep.errors[:6],
        }

    def _memory_forget_fact(self, args: dict[str, Any]) -> dict:
        key = (args.get("key_normalized") or "").strip()
        if not key:
            raise MemoryToolError("key_required")
        fs = self._mem.fact_store
        if fs is None:
            raise MemoryToolError("fact_store_unavailable")
        ok = fs.soft_delete_fact(
            company_id=self._company_id,
            user_id=self._user_id,
            key_normalized=key,
            reason="assistant_tool",
            by_user=True,
        )
        return {"deleted": bool(ok), "key_normalized": key}

    def _memory_list_facts(self, args: dict[str, Any]) -> list[dict]:
        fs = self._mem.fact_store
        if fs is None:
            raise MemoryToolError("fact_store_unavailable")
        cats = args.get("categories")
        categories = [str(c).strip() for c in cats] if isinstance(cats, list) else None
        try:
            lim = int(args.get("limit") or 40)
        except (TypeError, ValueError):
            lim = 40
        lim = max(1, min(lim, 80))
        rows = fs.list_user_facts(
            company_id=self._company_id,
            user_id=self._user_id,
            categories=categories,
            limit=lim,
        )
        slim: list[dict] = []
        for r in rows:
            slim.append(
                {
                    "key_normalized": r.get("key_normalized"),
                    "value": (r.get("value") or "")[:400],
                    "category": r.get("category"),
                    "confidence": r.get("confidence"),
                    "source": r.get("source"),
                },
            )
        return slim
