# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey semantic memory — Qdrant layer.

Stores and retrieves meaning, not just text.

Three collections, all multi-tenant (filtered by company_id payload):

  finkey_rag
      Company knowledge base — PDFs, policies, manuals, FAQs.
      RAG retrieval for work-related questions.
      Filtered by: company_id

  finkey_conv_memories
      Embedded summaries of past conversations.
      FinKey searches these to recall what she discussed with a user before.
      Filtered by: company_id + user_id

  finkey_user_memories
      FinKey's semantic impressions of users.
      "This person tends to ask about X under pressure."
      Not just text — embedded patterns. Used to enrich context.
      Filtered by: company_id + user_id

Multi-tenancy strategy:
  All collections use payload-based filtering (Qdrant supports this natively
  with indexed payload fields). One physical collection per type, isolated
  logically by company_id. This scales to thousands of companies.

  For enterprise isolation requirements, collections can be split per company
  by changing collection_name() to return company-specific names.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from finkey_memory.schema import QdrantCollections, QdrantPayloadFields

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def _now_epoch() -> float:
    return datetime.now(timezone.utc).timestamp()


def _as_float(value: object, default: float) -> float:
    """Payload numbers must always be present — the rerank guards None, not absence."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _vector_search(
    client,
    *,
    collection_name: str,
    vector:          list[float],
    limit:           int,
    query_filter,
    score_threshold: float,
) -> list:
    """
    Unified vector-search wrapper.

    qdrant-client ≥ 1.10 deprecated :meth:`.search` in favour of
    :meth:`.query_points`. We try the new API first and gracefully fall back
    to the legacy one — this keeps ai-core compatible with both old server
    deployments and modern (in-memory) clients.
    """
    if hasattr(client, "query_points"):
        resp = client.query_points(
            collection_name = collection_name,
            query           = vector,
            limit           = limit,
            query_filter    = query_filter,
            score_threshold = score_threshold,
            with_payload    = True,
        )
        return getattr(resp, "points", resp) or []
    return client.search(                                       # type: ignore[attr-defined]
        collection_name = collection_name,
        query_vector    = vector,
        limit           = limit,
        query_filter    = query_filter,
        score_threshold = score_threshold,
    )


class SemanticMemory:
    """
    Qdrant-backed semantic memory.

    Accepts a Qdrant client on init::

        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        mem = SemanticMemory(qdrant_client=client, embed_fn=my_embed_function)

    ``embed_fn`` is a callable: ``(text: str) -> list[float]``
    (e.g. OpenAI embeddings, local sentence-transformers, etc.)
    """

    def __init__(self, qdrant_client, embed_fn) -> None:
        self._q   = qdrant_client
        self._emb = embed_fn


    def ensure_collections(self, vector_size: int = 1536) -> None:
        """Create Qdrant collections if they don't exist yet."""
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

        for name in [
            QdrantCollections.RAG,
            QdrantCollections.CONV_MEMORIES,
            QdrantCollections.USER_MEMORIES,
            QdrantCollections.USER_FACTS,
        ]:
            existing = [c.name for c in self._q.get_collections().collections]
            if name not in existing:
                self._q.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection: %s", name)
            try:
                self._q.create_payload_index(
                    collection_name=name,
                    field_name=QdrantPayloadFields.COMPANY_ID,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                logger.debug("Payload index company_id on %s: %s", name, exc)
            if name != QdrantCollections.RAG:
                try:
                    self._q.create_payload_index(
                        collection_name=name,
                        field_name=QdrantPayloadFields.USER_ID,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception as exc:
                    logger.debug("Payload index user_id on %s: %s", name, exc)
            if name == QdrantCollections.USER_MEMORIES:
                try:
                    self._q.create_payload_index(
                        collection_name=name,
                        field_name="deep_kind",
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception as exc:
                    logger.debug("Payload index deep_kind on %s: %s", name, exc)
            if name == QdrantCollections.USER_FACTS:
                for fname in ("category", "key_normalized", "source"):
                    try:
                        self._q.create_payload_index(
                            collection_name=name,
                            field_name=fname,
                            field_schema=PayloadSchemaType.KEYWORD,
                        )
                    except Exception as exc:
                        logger.debug("Payload index %s on %s: %s", fname, name, exc)


    def upsert_rag_chunk(
        self,
        company_id:    str,
        doc_id:        str,
        chunk_index:   int,
        chunk_text:    str,
        content_type:  str = "document",
        metadata:      Optional[dict] = None,
    ) -> None:
        from qdrant_client.models import PointStruct
        vector    = self._emb(chunk_text)
        point_id  = _new_id()
        payload   = {
            QdrantPayloadFields.COMPANY_ID:   company_id,
            QdrantPayloadFields.CONTENT:      chunk_text,
            QdrantPayloadFields.MEMORY_TYPE:  "rag_doc",
            QdrantPayloadFields.CREATED_AT:   _now_iso(),
            "doc_id":      doc_id,
            "chunk_index": chunk_index,
            "content_type": content_type,
            **(metadata or {}),
        }
        self._q.upsert(
            collection_name=QdrantCollections.RAG,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def search_rag(
        self,
        company_id: str,
        query:      str,
        top_k:      int = 5,
        score_min:  float = 0.65,
    ) -> list[str]:
        """
        Search company knowledge base for chunks relevant to a query.
        Returns list of text chunks, ordered by relevance.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        vector = self._emb(query)
        results = _vector_search(
            self._q,
            collection_name = QdrantCollections.RAG,
            vector          = vector,
            limit           = top_k,
            query_filter    = Filter(
                must=[FieldCondition(
                    key=QdrantPayloadFields.COMPANY_ID,
                    match=MatchValue(value=company_id),
                )],
            ),
            score_threshold = score_min,
        )
        return [r.payload.get(QdrantPayloadFields.CONTENT, "") for r in results]

    def search_rag_scored(
        self,
        company_id: str,
        query: str,
        *,
        top_k: int = 12,
        score_min: float = 0.45,
    ) -> list[dict]:
        """
        Same as ``search_rag`` but returns scored rows for fusion / rerank pipelines.

        Each item: ``id``, ``score``, ``content``, ``doc_id``, ``chunk_index`` (optional).
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        vector = self._emb(query)
        results = _vector_search(
            self._q,
            collection_name=QdrantCollections.RAG,
            vector=vector,
            limit=top_k,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key=QdrantPayloadFields.COMPANY_ID,
                        match=MatchValue(value=company_id),
                    ),
                ],
            ),
            score_threshold=score_min,
        )
        out: list[dict] = []
        for r in results:
            pl = r.payload or {}
            out.append(
                {
                    "id": str(r.id),
                    "score": float(r.score),
                    "content": pl.get(QdrantPayloadFields.CONTENT, "") or "",
                    "doc_id": pl.get("doc_id"),
                    "chunk_index": pl.get("chunk_index"),
                },
            )
        return out

    def delete_rag_document(self, company_id: str, doc_id: str) -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self._q.delete(
            collection_name=QdrantCollections.RAG,
            points_selector=Filter(
                must=[
                    FieldCondition(key=QdrantPayloadFields.COMPANY_ID, match=MatchValue(value=company_id)),
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                ]
            ),
        )


    def upsert_conversation_memory(
        self,
        company_id:  str,
        user_id:     str,
        conv_id:     str,
        summary:     str,
        topics:      Optional[list[str]] = None,
        emotion:     Optional[str]       = None,
    ) -> None:
        """Embed and store a conversation summary for future semantic retrieval."""
        from qdrant_client.models import PointStruct
        vector   = self._emb(summary)
        point_id = _new_id()
        payload  = {
            QdrantPayloadFields.COMPANY_ID:      company_id,
            QdrantPayloadFields.USER_ID:         user_id,
            QdrantPayloadFields.CONV_ID:         conv_id,
            QdrantPayloadFields.CONTENT:         summary,
            QdrantPayloadFields.MEMORY_TYPE:     "conv_summary",
            QdrantPayloadFields.CREATED_AT:      _now_iso(),
            "topics":   topics or [],
            "emotion":  emotion,
        }
        self._q.upsert(
            collection_name=QdrantCollections.CONV_MEMORIES,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def search_past_conversations(
        self,
        company_id: str,
        user_id:    str,
        query:      str,
        top_k:      int = 3,
        score_min:  float = 0.60,
    ) -> list[str]:
        """
        Find past conversation summaries semantically relevant to the current query.
        Returns list of summary texts, ordered by relevance.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        vector = self._emb(query)
        results = _vector_search(
            self._q,
            collection_name = QdrantCollections.CONV_MEMORIES,
            vector          = vector,
            limit           = top_k,
            query_filter    = Filter(
                must=[
                    FieldCondition(key=QdrantPayloadFields.COMPANY_ID, match=MatchValue(value=company_id)),
                    FieldCondition(key=QdrantPayloadFields.USER_ID,    match=MatchValue(value=user_id)),
                ],
            ),
            score_threshold = score_min,
        )
        return [r.payload.get(QdrantPayloadFields.CONTENT, "") for r in results]

    def scroll_user_memories(
        self,
        company_id: str,
        user_id: str,
        *,
        limit: int = 24,
    ) -> list[dict]:
        """List a user's semantic impressions without a query embedding —
        a payload-filtered Qdrant ``scroll``, not a vector search. Used by
        recall-snapshot UIs that want "what's in there" rather than "what
        matches this query". Returns ``[]`` on any Qdrant error (collection
        missing, unreachable, etc.) — callers should treat that as "no
        semantic memory yet", not fabricate content."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        try:
            points, _ = self._q.scroll(
                collection_name=QdrantCollections.USER_MEMORIES,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key=QdrantPayloadFields.COMPANY_ID,
                            match=MatchValue(value=company_id),
                        ),
                        FieldCondition(
                            key=QdrantPayloadFields.USER_ID,
                            match=MatchValue(value=user_id),
                        ),
                    ],
                ),
                limit=max(1, min(200, limit)),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.debug("scroll_user_memories failed (%s/%s): %s", company_id, user_id, exc)
            return []

        out: list[dict] = []
        for p in points:
            pl = p.payload or {}
            out.append(
                {
                    "id": str(p.id),
                    "content": pl.get(QdrantPayloadFields.CONTENT, "") or "",
                    "memory_type": pl.get(QdrantPayloadFields.MEMORY_TYPE, "") or "",
                    "deep_kind": pl.get("deep_kind"),
                    "created_at": pl.get(QdrantPayloadFields.CREATED_AT),
                },
            )
        return out

    def upsert_user_memory(
        self,
        company_id:      str,
        user_id:         str,
        memory_text:     str,
        memory_type:     str,
        source_conv_id: Optional[str]       = None,
        deep_kind:      Optional[str]       = None,
        payload_extra: Optional[dict]     = None,
        point_id:       Optional[str]       = None,
    ) -> None:
        """
        Store a semantic impression of a user.
        E.g.: "Этот человек часто обращается под давлением с вопросами о приоритетах"

        ``point_id`` — внешний стабильный id точки. Без него каждая запись создаёт
        новый point, и повторная консолидация одной сцены плодила бы дубли в
        USER_MEMORIES; для deep_kind='scene' id выводим из (company, user, scene_key).
        """
        from qdrant_client.models import PointStruct
        vector   = self._emb(memory_text)
        point_id = (point_id or "").strip() or _new_id()
        payload  = {
            QdrantPayloadFields.COMPANY_ID:  company_id,
            QdrantPayloadFields.USER_ID:     user_id,
            QdrantPayloadFields.CONTENT:     memory_text,
            QdrantPayloadFields.MEMORY_TYPE: memory_type,
            QdrantPayloadFields.CREATED_AT:  _now_iso(),
            "source_conv_id": source_conv_id,
        }
        if deep_kind:
            payload["deep_kind"] = deep_kind
        if payload_extra:
            for k, v in payload_extra.items():
                payload[str(k)] = v
        self._q.upsert(
            collection_name=QdrantCollections.USER_MEMORIES,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def delete_user_memory_point(self, point_id: str) -> None:
        """
        Точка из USER_MEMORIES по стабильному id (scene-эмбединг при merge/удалении
        сцены). Без этого удалённая сцена навсегда оставалась бы recall-им.
        """
        from qdrant_client.models import PointIdsList
        pid = (point_id or "").strip()
        if not pid:
            return
        try:
            self._q.delete(
                collection_name = QdrantCollections.USER_MEMORIES,
                points_selector = PointIdsList(points=[pid]),
            )
        except Exception as exc:
            logger.warning("Qdrant delete_user_memory_point failed: %s", exc)


    def upsert_user_fact(
        self,
        *,
        company_id:     str,
        user_id:        str,
        point_id:       str,
        key_normalized: str,
        value:          str,
        category:       str,
        confidence:     float,
        source:         str,
        source_conv_id: Optional[str] = None,
        evidence:       Optional[str] = None,
        updated_at:     Optional[float] = None,
        priority:       Optional[float] = None,
        access_count:   Optional[int]   = None,
    ) -> None:
        """
        Индексирует value факта в Qdrant для семантического recall'a.
        point_id == PG ``user_facts.id`` — позволяет точечно удалить вместе с PG.

        ``updated_at`` / ``priority`` / ``access_count`` кладутся в payload, чтобы
        чтение (``search_user_facts`` → rerank в менеджере) умело взвешивать
        свежесть без лишнего round-trip в PG. Point id не меняется.
        """
        from qdrant_client.models import PointStruct
        text_for_embed = f"[{category}] {key_normalized}: {value}"
        vector = self._emb(text_for_embed)
        payload = {
            QdrantPayloadFields.COMPANY_ID:   company_id,
            QdrantPayloadFields.USER_ID:      user_id,
            QdrantPayloadFields.CONTENT:      value,
            QdrantPayloadFields.MEMORY_TYPE:  "user_fact",
            QdrantPayloadFields.CREATED_AT:   _now_iso(),
            "category":       category,
            "key_normalized": key_normalized,
            "confidence":     float(confidence),
            "source":         source,
            "source_conv_id": source_conv_id,
            "evidence":       (evidence or "")[:300],
            "updated_at":     _as_float(updated_at, _now_epoch()),
            "priority":       _as_float(priority, 0.5),
            "access_count":   int(_as_float(access_count, 0.0)),
        }
        self._q.upsert(
            collection_name=QdrantCollections.USER_FACTS,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def delete_user_fact(self, point_id: str) -> None:
        from qdrant_client.models import PointIdsList
        try:
            self._q.delete(
                collection_name=QdrantCollections.USER_FACTS,
                points_selector=PointIdsList(points=[point_id]),
            )
        except Exception as exc:
            logger.warning("Qdrant delete_user_fact failed: %s", exc)

    def search_user_facts(
        self,
        company_id: str,
        user_id:    str,
        query:      str,
        *,
        top_k:      int   = 5,
        score_min:  float = 0.55,
        categories: Optional[list[str]] = None,
    ) -> list[dict]:
        """Семантический поиск по фактам пользователя; возвращает payload-ы с score."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        vector = self._emb(query)
        must = [
            FieldCondition(key=QdrantPayloadFields.COMPANY_ID, match=MatchValue(value=company_id)),
            FieldCondition(key=QdrantPayloadFields.USER_ID,    match=MatchValue(value=user_id)),
        ]
        if categories:
            cat_should = [FieldCondition(key="category", match=MatchValue(value=c)) for c in categories if c]
            filt = Filter(must=must + [Filter(should=cat_should)])
        else:
            filt = Filter(must=must)
        results = _vector_search(
            self._q,
            collection_name = QdrantCollections.USER_FACTS,
            vector          = vector,
            limit           = top_k,
            query_filter    = filt,
            score_threshold = score_min,
        )
        out = []
        for r in results:
            out.append({
                "id":             str(r.id),
                "score":          float(r.score),
                "key_normalized": r.payload.get("key_normalized"),
                "value":          r.payload.get(QdrantPayloadFields.CONTENT),
                "category":       r.payload.get("category"),
                "confidence":     r.payload.get("confidence"),
                "source":         r.payload.get("source"),
                # Read-time rerank inputs; older points (pre-A1) simply lack them.
                "updated_at":     r.payload.get("updated_at"),
                "priority":       r.payload.get("priority"),
                "access_count":   r.payload.get("access_count"),
            })
        return out

    def delete_all_user_facts_vectors(self, company_id: str, user_id: str) -> None:
        """Удалить все Qdrant-точки durable-фактов пользователя (после memory.purge_user)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        try:
            self._q.delete(
                collection_name=QdrantCollections.USER_FACTS,
                points_selector=Filter(
                    must=[
                        FieldCondition(key=QdrantPayloadFields.COMPANY_ID, match=MatchValue(value=company_id)),
                        FieldCondition(key=QdrantPayloadFields.USER_ID,    match=MatchValue(value=user_id)),
                    ],
                ),
            )
        except Exception as exc:
            logger.warning("Qdrant delete_all_user_facts_vectors failed: %s", exc)

    def search_user_memories(
        self,
        company_id: str,
        user_id:    str,
        query:      str,
        top_k:      int = 5,
        score_min:  float = 0.55,
        deep_kinds: Optional[list[str]] = None,
    ) -> list[str]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        vector = self._emb(query)
        must = [
            FieldCondition(key=QdrantPayloadFields.COMPANY_ID, match=MatchValue(value=company_id)),
            FieldCondition(key=QdrantPayloadFields.USER_ID,    match=MatchValue(value=user_id)),
        ]
        filt: Filter
        if deep_kinds:
            dk_should = [
                FieldCondition(key="deep_kind", match=MatchValue(value=k))
                for k in deep_kinds
                if k
            ]
            filt = Filter(must=must + [Filter(should=dk_should)])
        else:
            filt = Filter(must=must)
        results = _vector_search(
            self._q,
            collection_name = QdrantCollections.USER_MEMORIES,
            vector          = vector,
            limit           = top_k,
            query_filter    = filt,
            score_threshold = score_min,
        )
        return [r.payload.get(QdrantPayloadFields.CONTENT, "") for r in results]
