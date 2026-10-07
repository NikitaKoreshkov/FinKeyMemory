# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey long-term memory — PostgreSQL adapter (канонический слой).

Работает поверх схемы, описанной в ``services/internal/migrations/``:

  * ``companies(id TEXT PK, name, account_type, plan, ...)`` — id == slug;
  * ``users(id UUID PK, company_id, external_id, email, name, ...)`` — внешний id фронтенда
    хранится в ``external_id``; адаптер сам резолвит «external_id ↔ UUID».
  * ``conversations(id TEXT PK, company_id, user_id UUID, ...)``;
  * ``messages(id UUID PK, conversation_id, company_id, user_id UUID, role, content JSONB, ...)``;
  * ``user_impressions``, ``summary_jobs``, ``audit_log`` — см. миграции.

Контракт совместим с предыдущей версией (тот же набор публичных методов).
Главное отличие — все записи под FK-ограничениями. Если callsite не вызвал
``product_store.ensure_tenant_user`` заранее, адаптер делает это сам (ленивый upsert):
``companies (id=company_id)`` и ``users (company_id, external_id)`` создаются,
если их ещё нет — благодаря этому старый external_id-flow вызывающий код продолжает
работать без явного auth-этапа.

Подключение: ``conn_factory`` — callable, возвращающий PEP 249 connection
(например, ``psycopg2.pool.ThreadedConnectionPool.getconn``). Соединение закрывается
после каждой операции (``.close()`` возвращает его в пул).
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from finkey_memory.consciousness_state import UserProfile  # noqa: F401 — re-exported for callers
from finkey_memory.schema import ConversationTurn, UserImpression

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _looks_like_uuid(s: str) -> bool:
    return bool(s) and bool(_UUID_RE.match(s))


#: Колонки SELECT по ``user_facts``. ``access_count`` появилась только в
#: миграции 026, а старую базу нельзя ломать: запрос с неизвестной колонкой
#: упал бы целиком, ``_fetchall`` вернул [] и память «исчезла» бы вместо
#: деградации. Поэтому колонку добавляем только когда схема проапгрейджена.
_FACTS_SELECT_COLUMNS = """
            id::text AS id, key_normalized, value, category,
                   confidence::float AS confidence, source, priority::float AS priority,
                   evidence_snippet, source_conv_id, source_message_id::text AS source_message_id,
                   expires_at, confirmed_at, touched_at, created_at, updated_at"""

_FACTS_SELECT_COLUMNS_WITH_ACCESS = _FACTS_SELECT_COLUMNS + """
                   , COALESCE(access_count, 0) AS access_count"""


def _content_to_jsonb(content: Any) -> str:
    """Превращает произвольный content (str/list/dict) в JSON-строку для JSONB-колонки."""
    import json as _json

    if isinstance(content, str):
        return _json.dumps({"text": content}, ensure_ascii=False)
    return _json.dumps(content if content is not None else {}, ensure_ascii=False)


def _turn_content_to_jsonb(turn: ConversationTurn) -> str:
    """Assistant/user message body + optional stream metadata in one JSONB blob."""
    import json as _json

    if isinstance(turn.content, dict):
        payload: dict[str, Any] = dict(turn.content)
    else:
        payload = {"text": str(turn.content or "")}

    reasoning = (turn.reasoning or "").strip()
    if reasoning:
        payload["reasoning"] = reasoning

    model_slug = (turn.model_slug or "").strip()
    if model_slug:
        payload["model_slug"] = model_slug

    if turn.specialist_route is not None:
        payload["specialist_route"] = bool(turn.specialist_route)

    if turn.thinking_ms is not None:
        payload["thinking_ms"] = int(turn.thinking_ms)

    return _json.dumps(payload, ensure_ascii=False)


class LongTermMemory:
    """
    PostgreSQL-backed long-term memory.

    Пример::

        import psycopg2.pool
        pool = psycopg2.pool.ThreadedConnectionPool(1, 10, dsn=DATABASE_URL)
        mem = LongTermMemory(conn_factory=pool.getconn)
    """

    def __init__(self, conn_factory) -> None:
        self._conn = conn_factory
        #: None = колонку ``user_facts.access_count`` ещё не проверяли (см. _facts_has_access_count).
        self._access_count_supported: Optional[bool] = None


    def get_company(self, company_id: str) -> Optional[dict]:
        return self._fetchone(
            "SELECT * FROM companies WHERE id = %s AND is_active = TRUE",
            (company_id,),
        )

    def get_company_by_slug(self, slug: str) -> Optional[dict]:
        return self.get_company(slug)

    def _ensure_company(self, company_id: str, name: Optional[str] = None) -> None:
        self._execute(
            """
            INSERT INTO companies (id, name)
            VALUES (%s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (company_id, name or company_id),
        )


    def get_or_create_user(
        self,
        company_id:  str,
        external_id: str,
        name:        Optional[str]  = None,
        role:        Optional[str]  = None,        # noqa: ARG002 — legacy signature
        department:  Optional[str]  = None,        # noqa: ARG002 — legacy signature
    ) -> dict:
        """
        Создать или вернуть пользователя по ``(company_id, external_id)``.

        ``role`` / ``department`` оставлены в сигнатуре для совместимости со старым
        кодом, но в каноне теперь это FK на ``roles`` / ``departments``.
        Установка делается отдельным auth-flow (Фаза 2).
        """
        self._ensure_company(company_id, name=None)
        row = self._fetchone_writing(
            """
            WITH new AS (
                INSERT INTO users (company_id, external_id, name, status)
                VALUES (%s, %s, %s, 'active')
                ON CONFLICT (company_id, external_id) DO UPDATE SET
                    name           = COALESCE(EXCLUDED.name, users.name),
                    last_active_at = NOW(),
                    updated_at     = NOW()
                RETURNING id::text AS id, external_id, name
            )
            SELECT id, external_id, name FROM new
            """,
            (company_id, external_id, name),
        )
        if not row:
            raise RuntimeError("Failed to upsert user (no row returned)")
        return {
            "id":          row["id"],
            "company_id":  company_id,
            "external_id": row["external_id"],
            "name":        row.get("name"),
        }

    def _facts_has_access_count(self) -> bool:
        """
        Есть ли в схеме ``user_facts.access_count`` (миграция 026).

        Один ``information_schema``-запрос на экземпляр, дальше — кэш. Probe
        падает/пустой ⇒ считаем «колонки нет» и собираем SELECT без неё:
        не-проапгрейджена база продолжает отдавать факты, а частотное
        воскрешение просто не работает (как и было до 026).
        """
        cached = self._access_count_supported
        if cached is not None:
            return cached
        row = self._fetchone(
            """
            SELECT 1 AS ok FROM information_schema.columns
            WHERE table_name = 'user_facts' AND column_name = 'access_count'
            LIMIT 1
            """,
            (),
        )
        supported = bool(row and row.get("ok"))
        self._access_count_supported = supported
        if not supported:
            logger.info(
                "user_facts.access_count absent — apply migration 026 to enable "
                "frequency revival (touch_user_facts will be a no-op)."
            )
        return supported

    def _facts_columns(self) -> str:
        return (
            _FACTS_SELECT_COLUMNS_WITH_ACCESS
            if self._facts_has_access_count()
            else _FACTS_SELECT_COLUMNS
        )

    def _facts_returning_suffix(self) -> str:
        """
        ``upsert_user_fact`` обязан вернуть ``access_count``: ``FactStore`` берёт из
        этой строки значение для Qdrant-payload, и без колонки он каждый раз пишет
        в вектор 0 — то есть обнуляет частоту revival ровно в момент, когда факт
        подтвердили повторно.
        """
        if self._facts_has_access_count():
            return ",\n                      COALESCE(access_count, 0) AS access_count"
        return ""


    def _resolve_user_id(self, company_id: str, user_id_or_external: str) -> Optional[str]:
        """
        Резолв «UUID ИЛИ external_id» → канонический UUID-строка.
        ``None`` — если такого пользователя нет (и не нужно создавать).
        """
        if not user_id_or_external:
            return None
        if _looks_like_uuid(user_id_or_external):
            row = self._fetchone(
                "SELECT id::text AS id FROM users WHERE id = %s::uuid AND company_id = %s",
                (user_id_or_external, company_id),
            )
            if row:
                return row["id"]
        row = self._fetchone(
            "SELECT id::text AS id FROM users WHERE external_id = %s AND company_id = %s",
            (user_id_or_external, company_id),
        )
        return row["id"] if row else None

    def _ensure_user(
        self,
        company_id: str,
        user_id_or_external: str,
        name: Optional[str] = None,
    ) -> str:
        """Гарантировать, что пользователь существует; вернуть его UUID."""
        uid = self._resolve_user_id(company_id, user_id_or_external)
        if uid:
            return uid
        return self.get_or_create_user(company_id, user_id_or_external, name=name)["id"]

    def get_user_profile(self, company_id: str, user_id_or_external: str) -> Optional[dict]:
        """Профиль из ``v_user_context``. Принимает UUID или external_id."""
        if _looks_like_uuid(user_id_or_external):
            row = self._fetchone(
                "SELECT * FROM v_user_context WHERE company_id = %s AND user_id = %s::uuid",
                (company_id, user_id_or_external),
            )
            if row:
                return row
        return self._fetchone(
            "SELECT * FROM v_user_context WHERE company_id = %s AND external_id = %s",
            (company_id, user_id_or_external),
        )

    def touch_user(self, company_id: str, user_id_or_external: str) -> None:
        uid = self._resolve_user_id(company_id, user_id_or_external)
        if not uid:
            return
        self._execute(
            "UPDATE users SET last_active_at = NOW() WHERE id = %s::uuid AND company_id = %s",
            (uid, company_id),
        )


    def create_conversation(
        self,
        company_id: str,
        user_id:    str,
        channel:    str = "chat",
    ) -> str:
        """Создать новый диалог; ``user_id`` — UUID или external_id."""
        uid = self._ensure_user(company_id, user_id)
        conv_id = str(uuid.uuid4())
        self._execute(
            """
            INSERT INTO conversations (id, company_id, user_id, channel)
            VALUES (%s, %s, %s::uuid, %s)
            """,
            (conv_id, company_id, uid, channel),
        )
        return conv_id

    def get_conversation(self, company_id: str, conv_id: str) -> Optional[dict]:
        return self._fetchone(
            "SELECT * FROM conversations WHERE id = %s AND company_id = %s",
            (conv_id, company_id),
        )

    def _ensure_conversation(
        self,
        company_id: str,
        conv_id:    str,
        user_uuid:  str,
        channel:    str = "chat",
    ) -> None:
        self._execute(
            """
            INSERT INTO conversations (id, company_id, user_id, channel)
            VALUES (%s, %s, %s::uuid, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (conv_id, company_id, user_uuid, channel),
        )

    def update_conversation_stats(
        self,
        company_id:      str,
        conv_id:         str,
        dominant_emotion: Optional[str]  = None,
        topics:          Optional[list[str]] = None,
        resolution:      Optional[str]  = None,
    ) -> None:
        fields, values = [], []
        fields.append("last_message_at = NOW()")
        fields.append("message_count = message_count + 1")
        fields.append("updated_at = NOW()")
        if dominant_emotion:
            fields.append("dominant_emotion = %s")
            values.append(dominant_emotion)
        if topics is not None:
            fields.append("topics = %s")
            values.append(topics)
        if resolution:
            fields.append("resolution_status = %s")
            values.append(resolution)
        values.extend([conv_id, company_id])
        self._execute(
            f"UPDATE conversations SET {', '.join(fields)} WHERE id = %s AND company_id = %s",
            tuple(values),
        )

    def save_conversation_summary(
        self,
        company_id: str,
        conv_id:    str,
        summary:    str,
    ) -> None:
        self._execute(
            """
            UPDATE conversations
            SET summary = %s, summary_generated_at = NOW(), updated_at = NOW()
            WHERE id = %s AND company_id = %s
            """,
            (summary, conv_id, company_id),
        )

    def get_user_recent_conversations(
        self,
        company_id: str,
        user_id_or_external: str,
        limit: int = 10,
    ) -> list[dict]:
        uid = self._resolve_user_id(company_id, user_id_or_external)
        if not uid:
            return []
        return self._fetchall(
            """
            SELECT id, started_at, last_message_at, message_count, summary,
                   dominant_emotion, topics, resolution_status
            FROM conversations
            WHERE company_id = %s AND user_id = %s::uuid
            ORDER BY last_message_at DESC
            LIMIT %s
            """,
            (company_id, uid, limit),
        )


    def save_message(
        self,
        company_id:      str,
        conversation_id: str,
        user_id:         str,
        turn:            ConversationTurn,
    ) -> Optional[str]:
        """
        Сохранить одно сообщение. Возвращает ``message.id`` (UUID-строка) или ``None``,
        если запись пропущена из-за неразрешимого user_id (логируется warning).

        Адаптер ленив: при необходимости создаёт ``companies`` / ``users`` / ``conversations``,
        чтобы внешний код мог писать сообщения без явных предварительных шагов.
        """
        if not (turn and (turn.role or "").strip()):
            return None
        try:
            self._ensure_company(company_id)
            uid = self._ensure_user(company_id, user_id, name=None)
            self._ensure_conversation(company_id, conversation_id, uid, channel="chat")
        except Exception as exc:
            logger.warning(
                "save_message: tenant ensure failed (company=%s user=%s conv=%s): %s",
                company_id, user_id, conversation_id, exc,
            )
            return None

        row = self._fetchone_writing(
            """
            INSERT INTO messages (
                conversation_id, company_id, user_id, role, content,
                emotion_detected, emotion_intensity, temperature_used,
                tone_used, inner_feeling
            ) VALUES (%s, %s, %s::uuid, %s, %s::jsonb, %s, %s, %s, %s, %s)
            RETURNING id::text AS id
            """,
            (
                conversation_id,
                company_id,
                uid,
                turn.role,
                _turn_content_to_jsonb(turn),
                turn.emotion_detected,
                turn.emotion_intensity,
                turn.temperature_used,
                None,
                turn.inner_feeling,
            ),
        )
        return row["id"] if row else None

    def get_conversation_messages(
        self,
        company_id:      str,
        conversation_id: str,
        limit:           int = 50,
        offset:          int = 0,
    ) -> list[dict]:
        return self._fetchall(
            """
            SELECT id, role, content, created_at, emotion_detected,
                   emotion_intensity, temperature_used
            FROM messages
            WHERE conversation_id = %s AND company_id = %s
            ORDER BY created_at ASC
            LIMIT %s OFFSET %s
            """,
            (conversation_id, company_id, limit, offset),
        )

    def prune_user_chat_messages(self, company_id: str, user_id_or_external: str) -> dict[str, int]:
        """
        Automatic retention for ``messages`` for this tenant user (production defaults).

        **Opt-out:** set ``FINKEY_PG_MESSAGES_RETENTION_OFF=1`` (or ``true``) to skip entirely
        (infinite retention, previous behaviour).

        Otherwise (defaults, override per env):

        * ``FINKEY_PG_MESSAGES_RETENTION_DAYS`` — default **365**; delete rows older than N days.
          Set to **0** to disable age-based deletion only.
        * ``FINKEY_PG_MESSAGES_MAX_PER_CONVERSATION`` — default **2500**; keep only the last N
          messages per ``conversation_id``. Set to **0** to disable per-thread cap only.

        After deletes, ``conversations.message_count`` is recomputed for that user's conversations.

        Called from ``MemoryManager.save_turn`` (every ``FINKEY_PG_MESSAGES_PRUNE_EVERY_N_TURNS``
        user interactions, default 12 — same family as ``bump_interaction``) and from
        ``flush_memory_maintenance``.
        """
        import os

        out: dict[str, int] = {"deleted_age": 0, "deleted_cap": 0, "convos_resynced": 0}

        raw_off = (os.getenv("FINKEY_PG_MESSAGES_RETENTION_OFF", "") or "").strip().lower()
        if raw_off in ("1", "true", "yes", "on"):
            return out

        uid = self._resolve_user_id(company_id, user_id_or_external)
        if not uid:
            return out

        try:
            ret_days = max(0, int(os.getenv("FINKEY_PG_MESSAGES_RETENTION_DAYS", "365")))
        except ValueError:
            ret_days = 365

        try:
            max_per = max(0, int(os.getenv("FINKEY_PG_MESSAGES_MAX_PER_CONVERSATION", "2500")))
        except ValueError:
            max_per = 2500

        if ret_days <= 0 and max_per <= 0:
            return out

        conn = self._conn()
        try:
            with conn.cursor() as cur:
                if ret_days > 0:
                    cur.execute(
                        """
                        DELETE FROM messages
                        WHERE company_id = %s AND user_id = %s::uuid
                          AND created_at < NOW() - (%s::integer * INTERVAL '1 day')
                        """,
                        (company_id, uid, ret_days),
                    )
                    out["deleted_age"] = int(cur.rowcount or 0)

                if max_per > 0:
                    cur.execute(
                        """
                        DELETE FROM messages AS m
                        USING (
                            SELECT id
                            FROM (
                                SELECT id,
                                       ROW_NUMBER() OVER (
                                           PARTITION BY conversation_id
                                           ORDER BY created_at DESC, id DESC
                                       ) AS rn
                                FROM messages
                                WHERE company_id = %s AND user_id = %s::uuid
                            ) ranked
                            WHERE ranked.rn > %s
                        ) doomed
                        WHERE m.id = doomed.id
                        """,
                        (company_id, uid, max_per),
                    )
                    out["deleted_cap"] = int(cur.rowcount or 0)

                if out["deleted_age"] or out["deleted_cap"]:
                    cur.execute(
                        """
                        UPDATE conversations AS c
                        SET message_count = sub.n,
                            updated_at = NOW()
                        FROM (
                            SELECT c2.id AS cid, COUNT(m.id)::integer AS n
                            FROM conversations c2
                            LEFT JOIN messages m
                              ON m.conversation_id = c2.id
                             AND m.company_id = c2.company_id
                            WHERE c2.company_id = %s AND c2.user_id = %s::uuid
                            GROUP BY c2.id
                        ) sub
                        WHERE c.id = sub.cid AND c.company_id = %s
                        """,
                        (company_id, uid, company_id),
                    )
                    out["convos_resynced"] = int(cur.rowcount or 0)

            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:  # pragma: no cover
                pass
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if out["deleted_age"] or out["deleted_cap"]:
            logger.info(
                "prune_user_chat_messages company=%s user=%s deleted_age=%s deleted_cap=%s resynced=%s",
                company_id,
                user_id_or_external,
                out["deleted_age"],
                out["deleted_cap"],
                out["convos_resynced"],
            )

        return out


    def get_user_impressions(
        self,
        company_id: str,
        user_id_or_external: str,
        types:      Optional[list[str]] = None,
    ) -> list[UserImpression]:
        uid = self._resolve_user_id(company_id, user_id_or_external)
        if not uid:
            return []
        if types:
            placeholders = ",".join(["%s"] * len(types))
            rows = self._fetchall(
                f"""
                SELECT * FROM user_impressions
                WHERE company_id = %s AND user_id = %s::uuid AND impression_type IN ({placeholders})
                  AND deleted_at IS NULL
                ORDER BY priority DESC, confidence DESC, updated_at DESC
                """,
                (company_id, uid, *types),
            )
        else:
            rows = self._fetchall(
                """
                SELECT * FROM user_impressions
                WHERE company_id = %s AND user_id = %s::uuid AND deleted_at IS NULL
                ORDER BY priority DESC, confidence DESC, updated_at DESC
                """,
                (company_id, uid),
            )
        return [
            UserImpression(
                user_id           = str(r["user_id"]),
                company_id        = str(r["company_id"]),
                impression_type   = r["impression_type"],
                content           = r["content"],
                confidence        = float(r.get("confidence") or 0.5),
                source_conv_id    = r.get("source_conv_id"),
                source            = str(r.get("source") or "inferred"),
                priority          = float(r.get("priority") or 0.5),
                expires_at        = r.get("expires_at"),
                last_confirmed_at = r.get("last_confirmed_at"),
            )
            for r in rows
        ]

    def upsert_impression(self, impression: UserImpression) -> None:
        """Идемпотентный upsert по ``(company_id, user_id, impression_type)``."""
        uid = self._ensure_user(impression.company_id, impression.user_id)
        self._execute(
            """
            INSERT INTO user_impressions
                (company_id, user_id, impression_type, content, confidence,
                 source, priority, source_conv_id, expires_at, last_confirmed_at)
            VALUES (%s, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (company_id, user_id, impression_type) DO UPDATE SET
                content           = EXCLUDED.content,
                confidence        = GREATEST(user_impressions.confidence, EXCLUDED.confidence),
                source            = EXCLUDED.source,
                priority          = GREATEST(user_impressions.priority, EXCLUDED.priority),
                source_conv_id    = COALESCE(EXCLUDED.source_conv_id, user_impressions.source_conv_id),
                expires_at        = COALESCE(EXCLUDED.expires_at, user_impressions.expires_at),
                last_confirmed_at = GREATEST(
                    COALESCE(user_impressions.last_confirmed_at, 'epoch'::timestamptz),
                    COALESCE(EXCLUDED.last_confirmed_at, 'epoch'::timestamptz)
                ),
                evidence_count    = user_impressions.evidence_count + 1,
                updated_at        = NOW(),
                deleted_at        = NULL
            """,
            (
                impression.company_id, uid, impression.impression_type,
                impression.content, impression.confidence,
                impression.source, impression.priority,
                impression.source_conv_id, impression.expires_at,
                impression.last_confirmed_at,
            ),
        )


    def upsert_user_fact(
        self,
        company_id:        str,
        user_id_or_external: str,
        *,
        key_normalized:    str,
        value:             str,
        category:          str,
        confidence:        float                     = 0.5,
        source:            str                       = "inferred",
        priority:          float                     = 0.5,
        evidence_snippet: Optional[str]             = None,
        source_conv_id:    Optional[str]             = None,
        source_message_id: Optional[str]             = None,
        expires_at:        Optional[datetime]        = None,
        confirmed_at:      Optional[datetime]        = None,
    ) -> Optional[dict]:
        """
        Идемпотентный upsert одного факта.

        Логика merge на стороне SQL через ``finkey_source_weight`` (миграция 006):
          * ``value/source`` — выигрывает более «тяжёлый» источник; при равенстве — большая ``confidence``.
          * ``confidence/priority`` — ``GREATEST``.
          * ``evidence_snippet/source_conv_id/source_message_id`` — обновляются, если пришло свежее.
          * ``expires_at`` — ``LEAST`` ненулевых (раньше истекает — раньше забыли).
          * ``confirmed_at`` — ``GREATEST``.
          * ``touched_at``/``updated_at`` — ``NOW()``; ``deleted_at`` сбрасываем (resurrect).

        Возвращает строку нового состояния факта (``id``, ``key_normalized``, ``value``, ...)
        или ``None``, если не удалось разрешить тенант / юзера.
        """
        if not (key_normalized or "").strip() or not (value or "").strip():
            return None
        try:
            uid = self._ensure_user(company_id, user_id_or_external)
        except Exception as exc:
            logger.warning("upsert_user_fact: ensure_user failed (%s)", exc)
            return None

        return self._fetchone_writing(
            """
            INSERT INTO user_facts (
                company_id, user_id, key_normalized, value, category,
                confidence, source, priority, evidence_snippet,
                source_conv_id, source_message_id, expires_at, confirmed_at,
                touched_at, updated_at, deleted_at
            ) VALUES (
                %s, %s::uuid, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                NOW(), NOW(), NULL
            )
            ON CONFLICT (company_id, user_id, key_normalized) WHERE deleted_at IS NULL
            DO UPDATE SET
                value = CASE
                    WHEN finkey_source_weight(EXCLUDED.source) > finkey_source_weight(user_facts.source)
                        THEN EXCLUDED.value
                    WHEN finkey_source_weight(EXCLUDED.source) = finkey_source_weight(user_facts.source)
                         AND EXCLUDED.confidence > user_facts.confidence
                        THEN EXCLUDED.value
                    ELSE user_facts.value
                END,
                category   = CASE
                    WHEN finkey_source_weight(EXCLUDED.source) > finkey_source_weight(user_facts.source)
                        THEN EXCLUDED.category
                    ELSE user_facts.category
                END,
                source     = CASE
                    WHEN finkey_source_weight(EXCLUDED.source) > finkey_source_weight(user_facts.source)
                        THEN EXCLUDED.source
                    WHEN finkey_source_weight(EXCLUDED.source) = finkey_source_weight(user_facts.source)
                         AND EXCLUDED.confidence > user_facts.confidence
                        THEN EXCLUDED.source
                    ELSE user_facts.source
                END,
                confidence = GREATEST(user_facts.confidence, EXCLUDED.confidence),
                priority   = GREATEST(user_facts.priority,   EXCLUDED.priority),
                evidence_snippet  = COALESCE(EXCLUDED.evidence_snippet,  user_facts.evidence_snippet),
                source_conv_id    = COALESCE(EXCLUDED.source_conv_id,    user_facts.source_conv_id),
                source_message_id = COALESCE(EXCLUDED.source_message_id, user_facts.source_message_id),
                expires_at = CASE
                    WHEN user_facts.expires_at IS NULL THEN EXCLUDED.expires_at
                    WHEN EXCLUDED.expires_at  IS NULL THEN user_facts.expires_at
                    ELSE LEAST(user_facts.expires_at, EXCLUDED.expires_at)
                END,
                confirmed_at = CASE
                    WHEN user_facts.confirmed_at IS NULL THEN EXCLUDED.confirmed_at
                    WHEN EXCLUDED.confirmed_at   IS NULL THEN user_facts.confirmed_at
                    ELSE GREATEST(user_facts.confirmed_at, EXCLUDED.confirmed_at)
                END,
                touched_at = NOW(),
                updated_at = NOW(),
                deleted_at = NULL
            RETURNING id::text AS id, key_normalized, value, category,
                      confidence, source, priority, evidence_snippet,
                      source_conv_id, source_message_id::text AS source_message_id,
                      expires_at, confirmed_at, touched_at, created_at, updated_at
            """ + self._facts_returning_suffix(),
            (
                company_id, uid, key_normalized, value, category,
                float(confidence), source, float(priority), evidence_snippet,
                source_conv_id, source_message_id, expires_at, confirmed_at,
            ),
        )

    def list_user_facts(
        self,
        company_id: str,
        user_id_or_external: str,
        *,
        categories:     Optional[list[str]] = None,
        min_confidence: Optional[float]     = None,
        limit:          int                 = 100,
    ) -> list[dict]:
        """Активные факты пользователя, отсортированные по убыванию priority/confidence/touched_at."""
        uid = self._resolve_user_id(company_id, user_id_or_external)
        if not uid:
            return []
        params: list = [company_id, uid]
        where = ["company_id = %s", "user_id = %s::uuid", "deleted_at IS NULL"]
        if categories:
            placeholders = ",".join(["%s"] * len(categories))
            where.append(f"category IN ({placeholders})")
            params.extend(categories)
        if min_confidence is not None:
            where.append("confidence >= %s")
            params.append(float(min_confidence))
        params.append(int(limit))
        return self._fetchall(
            f"""
            SELECT {self._facts_columns()}
            FROM user_facts
            WHERE {' AND '.join(where)}
            ORDER BY priority DESC, confidence DESC, touched_at DESC
            LIMIT %s
            """,
            tuple(params),
        )

    def list_user_fact_keys(self, company_id: str, user_id_or_external: str) -> set[str]:
        """Быстрая выборка только normalized-ключей — для known_keys экстрактора."""
        uid = self._resolve_user_id(company_id, user_id_or_external)
        if not uid:
            return set()
        rows = self._fetchall(
            """
            SELECT key_normalized FROM user_facts
            WHERE company_id = %s AND user_id = %s::uuid AND deleted_at IS NULL
            """,
            (company_id, uid),
        )
        return {r["key_normalized"] for r in rows if r.get("key_normalized")}

    def get_user_fact(
        self,
        company_id: str,
        user_id_or_external: str,
        key_normalized: str,
    ) -> Optional[dict]:
        uid = self._resolve_user_id(company_id, user_id_or_external)
        if not uid:
            return None
        return self._fetchone(
            f"""
            SELECT {self._facts_columns()}
            FROM user_facts
            WHERE company_id = %s AND user_id = %s::uuid
              AND key_normalized = %s AND deleted_at IS NULL
            """,
            (company_id, uid, key_normalized),
        )

    def soft_delete_user_fact(
        self,
        company_id: str,
        user_id_or_external: str,
        key_normalized: str,
    ) -> bool:
        """Помечаем deleted_at; партициальный unique-индекс освобождает ключ для повторной вставки."""
        uid = self._resolve_user_id(company_id, user_id_or_external)
        if not uid:
            return False
        row = self._fetchone_writing(
            """
            UPDATE user_facts
            SET deleted_at = NOW(), updated_at = NOW()
            WHERE company_id = %s AND user_id = %s::uuid
              AND key_normalized = %s AND deleted_at IS NULL
            RETURNING id::text AS id
            """,
            (company_id, uid, key_normalized),
        )
        return bool(row)


    def touch_user_facts(
        self,
        company_id: str,
        user_id_or_external: str,
        keys_normalized: Optional[list[str]] = None,
    ) -> int:
        """
        Батчевый ``access_count + 1`` по нормализованным ключам (частотное воскрешение).

        ``updated_at``/``touched_at`` не трогаем намеренно: read-time fresh-bonus
        (Phase A) взвешивает «когда человек это сказал», а инкремент от recall'а
        сделал бы каждый использованный факт бессрочно свежим и убил сигнал.

        Ищем по ``key_normalized``, а не по ``id``: ``FactStore.touch_facts``
        знает только ключи, а лишний round-trip в hot-path не нужен. Без
        колонки 026 — no-op (частотное воскрешение просто не работает).
        """
        keys = [str(k or "").strip() for k in (keys_normalized or []) if str(k or "").strip()]
        if not keys:
            return 0
        if not self._facts_has_access_count():
            return 0
        uid = self._resolve_user_id(company_id, user_id_or_external)
        if not uid:
            return 0
        rows = self._fetchall_writing(
            """
            UPDATE user_facts
               SET access_count = access_count + 1
             WHERE company_id = %s AND user_id = %s::uuid
               AND key_normalized = ANY(%s)
               AND deleted_at IS NULL
            RETURNING key_normalized
            """,
            (company_id, uid, keys),
        )
        return len(rows)

    def soft_delete_all_user_facts(self, company_id: str, user_id_or_external: str) -> int:
        """GDPR-friendly «забудь меня»: soft-delete всех активных фактов; вернуть count."""
        uid = self._resolve_user_id(company_id, user_id_or_external)
        if not uid:
            return 0
        rows = self._fetchall_writing(
            """
            UPDATE user_facts
            SET deleted_at = NOW(), updated_at = NOW()
            WHERE company_id = %s AND user_id = %s::uuid AND deleted_at IS NULL
            RETURNING id
            """,
            (company_id, uid),
        )
        return len(rows)

    def decay_expired_user_facts(
        self,
        *,
        company_id: Optional[str] = None,
        max_rows:   int           = 500,
        grace_sec:  int           = 0,
    ) -> list[dict]:
        """
        Soft-delete фактов, у которых ``expires_at`` уже прошёл (с опциональным
        запасом ``grace_sec``).

        Параметры
        ---------
        company_id
            Если задан — обработать только этого тенанта (нужно для пер-тенант
            расписаний). ``None`` — пройтись по всем тенантам в одной транзакции.
        max_rows
            Жёсткий потолок числа строк, помечаемых за один вызов. Защита от
            «убрали индекс — теперь decay съел половину кеша».
        grace_sec
            Льготный период после ``expires_at`` в секундах: факт реально
            гасится только если ``expires_at < now() - grace_sec``. Удобно,
            когда хочется чтобы факт жил ещё пару часов после формального
            истечения, на случай повторного появления.

        Возвращает список словарей с тем, что было помечено::

            [{"id": "...", "company_id": "...", "user_id": "...", "key_normalized": "..."}, ...]

        Вызывающая сторона (``FactStore.run_decay``) использует этот список,
        чтобы синхронизировать Qdrant / volatile / identity_pin.

        Реализация
        ----------
        Используем ``CTE`` со ``SELECT ... ORDER BY expires_at LIMIT ... FOR
        UPDATE SKIP LOCKED``, чтобы конкурирующие воркеры не наступали друг
        другу на пятки и чтобы partition-unique индекс по
        ``(company_id, user_id, key_normalized) WHERE deleted_at IS NULL``
        получил «вакансию» под повторный апсёрт.
        """
        where_extra = ""
        params: list = [int(max(0, grace_sec))]
        if company_id:
            where_extra = "AND company_id = %s"
            params.append(company_id)
        params.append(int(max(1, max_rows)))

        sql = f"""
            WITH expired AS (
                SELECT id
                FROM user_facts
                WHERE deleted_at IS NULL
                  AND expires_at IS NOT NULL
                  AND expires_at < NOW() - (%s || ' seconds')::interval
                  {where_extra}
                ORDER BY expires_at
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            )
            UPDATE user_facts uf
            SET deleted_at = NOW(), updated_at = NOW()
            FROM expired
            WHERE uf.id = expired.id
            RETURNING uf.id::text      AS id,
                      uf.company_id,
                      uf.user_id::text AS user_id,
                      uf.key_normalized,
                      uf.category,
                      uf.expires_at
        """
        return self._fetchall_writing(sql, tuple(params))


    # ── L2 scene blocks (миграция 026) ───────────────────────────────────────

    def _scene_user_id(self, company_id: str, user_id_or_external: str) -> Optional[str]:
        """
        Канонический ключ строк сцены/persona: ``users.id::text``, а при отсутствии
        пользователя — сам переданный идентификатор.

        Факты пишутся через ``_ensure_user`` (создаёт строку users), поэтому нормальный
        путь — резолв. Но read-path может прийти раньше любой записи (новый юзер,
        external_id из канала); падать нельзя, и TEXT-колонка это переживает.
        """
        uid = self._resolve_user_id(company_id, user_id_or_external)
        return uid or (str(user_id_or_external or "").strip() or None)

    def upsert_scene_block(
        self,
        company_id: str,
        user_id_or_external: str,
        *,
        scene_key:   str,
        summary:     str,
        content:     str,
        heat:        int                  = 1,
        facts_count: int                  = 0,
    ) -> Optional[dict]:
        """
        Идемпотентная запись одной сцены (``UNIQUE (company_id, user_id, scene_key)``).

        ``heat``/``summary`` приходят из кода (``scene_persona``), не из LLM-текста:
        в оригинале热度 — строка внутри markdown, которую модель переписывает сама,
        и она же решает, сколько ей add-нуть.
        """
        if not (scene_key or "").strip() or not (content or "").strip():
            return None
        uid = self._scene_user_id(company_id, user_id_or_external)
        if not uid:
            return None
        return self._fetchone_writing(
            """
            INSERT INTO user_scene_blocks
                (company_id, user_id, scene_key, summary, content, heat, facts_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (company_id, user_id, scene_key) DO UPDATE SET
                summary     = EXCLUDED.summary,
                content     = EXCLUDED.content,
                heat        = GREATEST(user_scene_blocks.heat, EXCLUDED.heat),
                facts_count = EXCLUDED.facts_count,
                updated_at  = NOW()
            RETURNING id, scene_key, summary, heat, facts_count, created_at, updated_at
            """,
            (
                company_id, uid, scene_key.strip(),
                (summary or "").strip()[:400],
                content, int(max(0, heat)), int(max(0, facts_count)),
            ),
        )

    def list_scene_blocks(
        self,
        company_id: str,
        user_id_or_external: str,
        *,
        limit: int = 24,
    ) -> list[dict]:
        """Сцены тенанта, горячие первые (индекс ``idx_user_scene_blocks_heat``)."""
        uid = self._scene_user_id(company_id, user_id_or_external)
        if not uid:
            return []
        return self._fetchall(
            """
            SELECT id, scene_key, summary, heat, facts_count,
                   left(content, 8000) AS content, created_at, updated_at
            FROM user_scene_blocks
            WHERE company_id = %s AND user_id = %s
            ORDER BY heat DESC, updated_at DESC
            LIMIT %s
            """,
            (company_id, uid, max(1, min(100, int(limit)))),
        )

    def bump_scene_heat(
        self,
        company_id: str,
        user_id_or_external: str,
        scene_keys: list[str],
    ) -> int:
        """Один UPDATE на все попавшие в recall сцены (``heat = heat + 1``)."""
        keys = [str(k or "").strip() for k in (scene_keys or []) if str(k or "").strip()]
        if not keys:
            return 0
        uid = self._scene_user_id(company_id, user_id_or_external)
        if not uid:
            return 0
        rows = self._fetchall_writing(
            """
            UPDATE user_scene_blocks
               SET heat = heat + 1, updated_at = NOW()
             WHERE company_id = %s AND user_id = %s AND scene_key = ANY(%s)
            RETURNING scene_key
            """,
            (company_id, uid, keys[:64]),
        )
        return len(rows)

    def delete_scene_block(
        self,
        company_id: str,
        user_id_or_external: str,
        scene_key: str,
    ) -> bool:
        """Сцены, растворённые в другой (merge), удаляем физически: квота по heat
        не должна считать мёртвые строки."""
        uid = self._scene_user_id(company_id, user_id_or_external)
        if not uid or not (scene_key or "").strip():
            return False
        row = self._fetchone_writing(
            """
            DELETE FROM user_scene_blocks
             WHERE company_id = %s AND user_id = %s AND scene_key = %s
            RETURNING id
            """,
            (company_id, uid, scene_key.strip()),
        )
        return bool(row)

    def count_scene_blocks(self, company_id: str, user_id_or_external: str) -> int:
        uid = self._scene_user_id(company_id, user_id_or_external)
        if not uid:
            return 0
        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM user_scene_blocks WHERE company_id = %s AND user_id = %s",
            (company_id, uid),
        )
        try:
            return int(row.get("n") or 0) if row else 0
        except (AttributeError, TypeError, ValueError):
            return 0


    # ── L3 persona (миграция 026) ────────────────────────────────────────────

    def get_persona(self, company_id: str, user_id_or_external: str) -> Optional[dict]:
        uid = self._scene_user_id(company_id, user_id_or_external)
        if not uid:
            return None
        return self._fetchone(
            """
            SELECT body, version, memories_since_last, pending_request,
                   pending_reason, last_generated_at
            FROM user_persona
            WHERE company_id = %s AND user_id = %s
            """,
            (company_id, uid),
        )

    def bump_persona_memories(
        self,
        company_id: str,
        user_id_or_external: str,
        delta: int = 1,
    ) -> int:
        """
        Счётчик P4-триггера: «столько памяти с последнего портрета».

        Строка создаётся с ``version = 0`` и пустым ``body`` — это честное
        «портрета ещё не было»: холодный старт (P2) отличается от восстановления
        испорченного ряда (P2.5, ``version >= 1`` при пустом ``body``).
        """
        step = int(delta) if delta else 1
        if step <= 0:
            return 0
        uid = self._scene_user_id(company_id, user_id_or_external)
        if not uid:
            return 0
        row = self._fetchone_writing(
            """
            INSERT INTO user_persona (company_id, user_id, body, version, memories_since_last)
            VALUES (%s, %s, '', 0, %s)
            ON CONFLICT (company_id, user_id) DO UPDATE SET
                memories_since_last = user_persona.memories_since_last + EXCLUDED.memories_since_last
            RETURNING memories_since_last
            """,
            (company_id, uid, step),
        )
        try:
            return int(row.get("memories_since_last") or 0) if row else 0
        except (AttributeError, TypeError, ValueError):
            return 0

    def upsert_persona(
        self,
        company_id: str,
        user_id_or_external: str,
        *,
        body: str,
    ) -> Optional[dict]:
        """
        Новый портрет: ``version + 1``, счётчик обнуляем, флаг явного запроса снимаем.

        Пустой ``body`` не пишем никогда — иначе следующий прогон сочтёт ряд
        испорченным и зациклит генерацию.
        """
        text = (body or "").strip()
        if not text:
            return None
        uid = self._scene_user_id(company_id, user_id_or_external)
        if not uid:
            return None
        return self._fetchone_writing(
            """
            INSERT INTO user_persona
                (company_id, user_id, body, version, memories_since_last,
                 pending_request, pending_reason, last_generated_at)
            VALUES (%s, %s, %s, 1, 0, FALSE, '', NOW())
            ON CONFLICT (company_id, user_id) DO UPDATE SET
                body                = EXCLUDED.body,
                version             = user_persona.version + 1,
                memories_since_last = 0,
                pending_request     = FALSE,
                pending_reason      = '',
                last_generated_at   = NOW()
            RETURNING body, version, memories_since_last, last_generated_at
            """,
            (company_id, uid, text[:4000]),
        )

    def request_persona_update(
        self,
        company_id: str,
        user_id_or_external: str,
        reason: str = "",
    ) -> int:
        """
        P1-флаг: «портрет нужно пересобрать немедленно».

        Ставится не регекспом по свободному тексту ответа модели (как в оригинале),
        а JSON-полем ``persona_update_request`` из структурированного ответа
        консолидации сцен — и читателем tool-вызова, если такой появится.
        """
        uid = self._scene_user_id(company_id, user_id_or_external)
        if not uid:
            return 0
        row = self._fetchone_writing(
            """
            INSERT INTO user_persona
                (company_id, user_id, body, version, pending_request, pending_reason)
            VALUES (%s, %s, '', 0, TRUE, %s)
            ON CONFLICT (company_id, user_id) DO UPDATE SET
                pending_request = TRUE,
                pending_reason  = EXCLUDED.pending_reason
            RETURNING version
            """,
            (company_id, uid, (reason or "").strip()[:200]),
        )
        return 1 if row else 0


    def enqueue_summary(self, company_id: str, conv_id: str) -> None:
        self._execute(
            """
            INSERT INTO summary_jobs (company_id, conversation_id, status)
            VALUES (%s, %s, 'pending')
            ON CONFLICT DO NOTHING
            """,
            (company_id, conv_id),
        )

    def claim_summary_job(self) -> Optional[dict]:
        """Claim one pending summary job for the background worker."""
        return self._fetchone_writing(
            """
            UPDATE summary_jobs
            SET status='processing', processed_at=NOW(), attempt_count = attempt_count + 1
            WHERE id = (
                SELECT id FROM summary_jobs
                WHERE status='pending'
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id::text AS id, company_id, conversation_id, attempt_count
            """,
            (),
        )

    def complete_summary_job(self, job_id: str, error: Optional[str] = None) -> None:
        status = "failed" if error else "done"
        self._execute(
            "UPDATE summary_jobs SET status=%s, error=%s, processed_at=NOW() WHERE id = %s::uuid",
            (status, error, job_id),
        )


    def audit(
        self,
        company_id:  str,
        action:      str,
        user_id:     Optional[str]      = None,
        entity_type: Optional[str]      = None,
        entity_id:   Optional[str]      = None,
        metadata:    Optional[dict]     = None,
        ip:          Optional[str]      = None,
    ) -> None:
        import json as _json

        uid: Optional[str] = None
        if user_id:
            uid = self._resolve_user_id(company_id, user_id)
        self._execute(
            """
            INSERT INTO audit_log (company_id, user_id, action, entity_type, entity_id, ip, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::inet, %s::jsonb)
            """,
            (
                company_id,
                uid,
                action,
                entity_type,
                entity_id,
                ip,
                _json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )


    def _execute(self, sql: str, params: tuple) -> None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:  # pragma: no cover
                pass
            logger.error("DB write failed: %s | SQL: %s", exc, sql[:160])
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _fetchone(self, sql: str, params: tuple) -> Optional[dict]:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))
        except Exception as exc:
            logger.error("DB read failed: %s", exc)
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _fetchone_writing(self, sql: str, params: tuple) -> Optional[dict]:
        """Как ``_fetchone``, но коммитит транзакцию (используется для RETURNING)."""
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                cols = [d[0] for d in cur.description] if cur.description else []
            conn.commit()
            if row is None:
                return None
            return dict(zip(cols, row))
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:  # pragma: no cover
                pass
            logger.error("DB write+read failed: %s | SQL: %s", exc, sql[:160])
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _fetchall(self, sql: str, params: tuple) -> list[dict]:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                if not rows:
                    return []
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in rows]
        except Exception as exc:
            logger.error("DB read failed: %s", exc)
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _fetchall_writing(self, sql: str, params: tuple) -> list[dict]:
        """Как ``_fetchall``, но коммитит — для ``UPDATE/DELETE ... RETURNING``."""
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() if cur.description else []
                cols = [d[0] for d in cur.description] if cur.description else []
            conn.commit()
            return [dict(zip(cols, row)) for row in rows]
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:  # pragma: no cover
                pass
            logger.error("DB bulk write+read failed: %s | SQL: %s", exc, sql[:160])
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass
