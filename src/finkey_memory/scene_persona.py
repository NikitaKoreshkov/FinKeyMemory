# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKeyMemory L2 «scene blocks» + L3 «working persona».

Два верхних слоя иерархии памяти, где надёжность зафиксирована в контракте,
а не в тексте промпта:

  * ``heat`` и ``summary`` сцен — колонки PG, их меняет только код; LLM
    возвращает **исключительно** назначение «факт → сцена» и одну строку
    саммари — модель не правит метаданные.
  * саммари каждой сцены эмбеддится в ``finkey_user_memories`` с
    ``deep_kind='scene'`` под стабильным point id, так что сцена попадает в
    обычный векторный поиск и может дать heat++ от самого recall'а.
  * персона — один JSON-ответ ``{"body": ...}`` с валидацией и жёстким
    лимитом 2000 символов в коде.
  * триггер-лестница портрета собрана из колонок ``user_persona``
    (один индексированный read).

Ничего не ломается при отсутствии PG / Qdrant / LLM: хранилища отдают пустое,
генерация пропускается, строки остаются нетронутыми.

Env:
  FINKEY_MEMORY_SCENES              default 1  (L2 консолидация)
  FINKEY_MEMORY_PERSONA             default 1  (L3 генерация)
  FINKEY_MEMORY_PERSONA_INTERVAL    default 30 (P4: сколько памяти между портретами)
  FINKEY_MEMORY_MAX_SCENES          default 15
  FINKEY_MEMORY_SCENE_MIN_FACTS     default 4  (не дёргать LLM ради 1-3 фактов)
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Optional

from finkey_memory.extractor import _try_parse_json, normalize_key  # noqa: SLF001

if TYPE_CHECKING:
    from finkey_memory.long_term import LongTermMemory
    from finkey_memory.semantic import SemanticMemory

logger = logging.getLogger("finkey.memory.scene_persona")

#: Контракт completer'а тот же, что у экстрактора (см. ``llm_completer``).
LLMCompleter = Callable[..., str]

# ── Капы: все в коде, ни один не зависит от доброй воли модели ────────────────
MAX_SCENES_PER_USER    = 15
MAX_SCENE_FACTS        = 40
MAX_UNASSIGNED_PER_RUN = 120
MAX_FACT_LINE_CHARS    = 240
MAX_SUMMARY_CHARS      = 160
PERSONA_BODY_MAX_CHARS = 2000
PERSONA_INPUT_CAP      = 4200
#: Больше этого в строке ключа — уже не ключ, а цитата: блокируем разбор блока.
MAX_FACT_KEY_CHARS     = 80
SCENE_HEAT_KEYS_CAP    = 64
#: Ниже этого портрет не считаем портретом («да, вижу» за 20 символов не спасает).
MIN_PERSONA_BODY_CHARS = 24
#: Брейкер LLM: N подряд отказов ⇒ не дёргаем бэкенд ещё N минут. Dream-цикл
#: иначе жёг бы таймаут на каждого тенанта каждого тика, когда ключей/Ollama нет.
LLM_BREAKER_THRESHOLD = 3
LLM_BREAKER_BACKOFF_SEC = 900

#: Сюда уезжает всё, для чего LLM не дал назначения. Факт обязан попасть в
#: какую-то сцену, иначе он вечно висит «не распределённым» и каждый прогон
#: заново платит LLM-вызовом за того же пациента.
MISC_SCENE_KEY = "misc.unsorted"
MISC_SCENE_SUMMARY = "Everything not yet placed into a topic scene."

META_START = "-----FINKEY-SCENE-----"
META_END   = "-----END-FINKEY-SCENE-----"
_META_KEYS = ("created", "updated", "summary", "heat", "facts")


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or "").strip() or default)
    except ValueError:
        return default


def scenes_enabled() -> bool:
    """L2-консолидация сцен. Снимается без последствий для нижних слоёв."""
    return _env_bool("FINKEY_MEMORY_SCENES", True)


def persona_enabled() -> bool:
    """L3-портрет. Флаг снимает и генерацию, и триггер-лестницу."""
    return _env_bool("FINKEY_MEMORY_PERSONA", True)


def persona_interval() -> int:
    """P4: сколько записанных фактов нужно между двумя портретами."""
    return max(1, _env_int("FINKEY_MEMORY_PERSONA_INTERVAL", 30))


def max_scenes_per_user() -> int:
    return max(3, _env_int("FINKEY_MEMORY_MAX_SCENES", MAX_SCENES_PER_USER))


def min_facts_for_consolidation() -> int:
    return max(1, _env_int("FINKEY_MEMORY_SCENE_MIN_FACTS", 4))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clip(text: object, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit].strip()


def _one_line(text: object, limit: int = MAX_SUMMARY_CHARS) -> str:
    """Сводим к одной строке: перенос сжёг бы machine-блок и навигацию."""
    return " ".join(str(text or "").replace("\r", "\n").split("\n"))[:limit].strip()


def is_safe_scene_key(key: str) -> bool:
    """scene_key/fact_key = вывод ``normalize_key``: [a-z0-9._-], без запятых."""
    if not key or len(key) > MAX_FACT_KEY_CHARS:
        return False
    if key != normalize_key(key):
        return False
    return not any(ch in key for ch in (",", ":", " ", "\n", "\t"))


def scene_point_id(company_id: str, user_id: str, scene_key: str) -> str:
    """Стабильный Qdrant point id: повторная консолидация перезаписывает точку."""
    seed = f"finkey-scene:{company_id}:{user_id}:{scene_key}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def scene_embed_text(scene: "SceneBlock") -> str:
    """
    Текст, который эмбеддится для сцены. Одна функция на запись и на сравнение:
    второй сигнал heat++ ловится точным совпадением этой строки с recall-hit'ом,
    и любое расхождение в форматировании развело бы их навсегда.
    """
    return _clip(f"{scene.scene_key}: {scene.summary}", 400)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SceneBlock:
    """
    Одна L2-сцена. ``content`` рендерит код (machine-блок + bullet-строки),
    поэтому ``source_fact_keys`` — точная учётная запись, а не догадка по markdown.
    """
    scene_key:        str
    summary:          str
    content:          str
    heat:             int          = 1
    facts_count:      int          = 0
    source_fact_keys: list[str]    = field(default_factory=list)
    created_at:       str          = ""
    updated_at:       str          = ""

    def __post_init__(self) -> None:
        key = normalize_key(str(self.scene_key or ""))
        if not is_safe_scene_key(key):
            raise ValueError(f"unsafe scene_key: {self.scene_key!r}")
        self.scene_key = key
        self.summary   = _one_line(self.summary)
        content        = str(self.content or "")
        if len(content) > 20000:
            raise ValueError(f"scene content too large: {key}")
        self.content = content
        try:
            self.heat = max(0, int(self.heat))
        except (TypeError, ValueError):
            self.heat = 1
        try:
            self.facts_count = max(0, int(self.facts_count))
        except (TypeError, ValueError):
            self.facts_count = len(self.source_fact_keys)
        clean: list[str] = []
        seen: set[str] = set()
        for raw in self.source_fact_keys or []:
            fact_key = str(raw or "").strip()
            if not fact_key or fact_key in seen or "__asof__" in fact_key:
                continue
            seen.add(fact_key)
            clean.append(fact_key[:MAX_FACT_KEY_CHARS])
            if len(clean) >= MAX_SCENE_FACTS:
                break
        self.source_fact_keys = clean


@dataclass
class PersonaDoc:
    """L3-портрет; ``version == 0`` означает «портрета ещё не было»."""
    body:                str           = ""
    version:             int           = 0
    memories_since_last: int           = 0
    pending_request:     bool          = False
    pending_reason:      str           = ""
    last_generated_at:   Optional[str] = None

    def __post_init__(self) -> None:
        self.body = str(self.body or "")
        for name in ("version", "memories_since_last"):
            try:
                setattr(self, name, max(0, int(getattr(self, name) or 0)))
            except (TypeError, ValueError):
                setattr(self, name, 0)
        self.pending_request = bool(self.pending_request)
        self.pending_reason  = _clip(self.pending_reason, 200)

    @property
    def has_body(self) -> bool:
        return bool(self.body.strip())

    @property
    def corrupt(self) -> bool:
        """Ряд уже генерировался, но тело пустое — это случай восстановления (P2.5)."""
        return self.version >= 1 and not self.has_body

    @classmethod
    def from_row(cls, row: Optional[dict]) -> "PersonaDoc":
        if not row:
            return cls()
        raw_ts = row.get("last_generated_at")
        return cls(
            body                 = str(row.get("body") or ""),
            version              = row.get("version"),
            memories_since_last  = row.get("memories_since_last"),
            pending_request      = bool(row.get("pending_request")),
            pending_reason       = str(row.get("pending_reason") or ""),
            last_generated_at    = (
                raw_ts.isoformat() if hasattr(raw_ts, "isoformat")
                else (str(raw_ts) if raw_ts else None)
            ),
        )


@dataclass
class SceneConsolidationReport:
    unassigned:     int  = 0
    scenes_before:  int  = 0
    scenes_after:   int  = 0
    created:        int  = 0
    updated:        int  = 0
    merged_away:    int  = 0
    facts_assigned: int  = 0
    embedded:       int  = 0
    llm_used:       bool = False
    skipped_reason: str  = ""

    def changed(self) -> bool:
        return bool(self.created or self.updated or self.merged_away)

    def to_dict(self) -> dict:
        return {
            "unassigned":     self.unassigned,
            "scenes_before":  self.scenes_before,
            "scenes_after":   self.scenes_after,
            "created":        self.created,
            "updated":        self.updated,
            "merged_away":    self.merged_away,
            "facts_assigned": self.facts_assigned,
            "embedded":       self.embedded,
            "llm_used":       self.llm_used,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class PersonaGenerationReport:
    generated:      bool = False
    reason:         str  = ""
    version:        int  = 0
    chars:          int  = 0
    skipped_reason: str  = ""

    def to_dict(self) -> dict:
        return {
            "generated":      self.generated,
            "reason":         self.reason,
            "version":        self.version,
            "chars":          self.chars,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class _SceneLLMAnswer:
    """Один структурированный ответ консолидации: назначение + out-of-band сигнал."""
    assignments:   list[dict] = field(default_factory=list)
    persona_signal: str       = ""


# ─────────────────────────────────────────────────────────────────────────────
# Рендер и разбор сцены (мы владеем форматом — отсюда «strict parse»)
# ─────────────────────────────────────────────────────────────────────────────

def render_scene_content(
    *,
    scene_key:   str,
    summary:     str,
    fact_lines:  list[tuple[str, str]],
    heat:        int,
    created_at:  str = "",
    updated_at:  str = "",
) -> str:
    """
    Machine-блок + маркированный список фактов. Свободного markdown от модели тут
    нет намеренно: разбор должен быть обратимым, а учёт «какие факты уже в сцене»
    — точным, иначе консолидация каждый раз видит их новыми.
    """
    keys = [k for k, _ in fact_lines[:MAX_SCENE_FACTS] if k]
    meta = [
        META_START,
        f"created: {created_at or _now_iso()}",
        f"updated: {updated_at or _now_iso()}",
        f"summary: {_one_line(summary)}",
        f"heat: {int(max(0, heat))}",
        f"facts: {','.join(keys)}",
        META_END,
    ]
    bullets: list[str] = []
    for key, value in fact_lines[:MAX_SCENE_FACTS]:
        text = _clip(value, MAX_FACT_LINE_CHARS)
        if text and key and key not in text:
            bullets.append(f"- {key}: {text}")
        else:
            bullets.append(f"- {_clip(text or key, MAX_FACT_LINE_CHARS)}")
    body = "\n".join(bullets) if bullets else "- (no facts yet)"
    return "\n".join(meta) + "\n\n" + body


def _recover_fact_keys_from_bullets(content: str) -> list[str]:
    """Запасной путь: bullet-строки рендерим мы, значит «- key: value» разбирается
    детерминированно даже когда machine-блок испорчен."""
    out: list[str] = []
    for line in (content or "").splitlines():
        text = line.strip()
        if not text.startswith("- "):
            continue
        head, sep, _rest = text[2:].partition(":")
        if not sep:
            continue
        key = head.strip()
        if is_safe_scene_key(key) and key not in out:
            out.append(key)
        if len(out) >= MAX_SCENE_FACTS:
            break
    return out


def _as_iso(value: object) -> str:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(timespec="seconds")  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return str(value)
    return str(value or "")


def parse_scene_row(row: dict) -> Optional[SceneBlock]:
    """
    Строка PG → ``SceneBlock``. Machine-блок разбирается строго: неизвестный ключ,
    перепутанные маркеры или не-числовой heat = блок нам не доверяется, включается
    восстановление по bullet-строкам. Колонки ``summary``/``heat`` остаются
    источником истины для навигации — блок нужен только для учёта фактов.
    """
    key = str(row.get("scene_key") or "").strip()
    content = str(row.get("content") or "")
    if not key or not content:
        return None
    fact_keys, strict_ok = _parse_meta_block(content)
    if not strict_ok:
        fact_keys = _recover_fact_keys_from_bullets(content)
        logger.debug("scene %s: malformed machine block, recovered %d keys", key, len(fact_keys))
    try:
        return SceneBlock(
            scene_key        = key,
            summary          = str(row.get("summary") or ""),
            content          = content,
            heat             = row.get("heat"),
            facts_count      = row.get("facts_count"),
            source_fact_keys = fact_keys,
            created_at       = _as_iso(row.get("created_at")),
            updated_at       = _as_iso(row.get("updated_at")),
        )
    except ValueError as exc:
        logger.info("scene row skipped (%s): %s", key, exc)
        return None


def _parse_meta_block(content: str) -> tuple[list[str], bool]:
    """
    → ``(fact_keys, strict_ok)``.

    ``strict_ok=False`` — маркеров нет, они перепутаны, внутрь блока попала
    неизвестная строка или heat не число. Молча «починить» такое нельзя: учёт
    фактов стал бы выдумкой, а невыдуманные факты — вечным аутсайдером.
    """
    start = content.find(META_START)
    end = content.find(META_END)
    if start == -1 or end == -1 or end <= start:
        return [], False
    raw = content[start + len(META_START):end].strip()
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        head, sep, tail = text.partition(":")
        if not sep:
            return [], False
        name = head.strip().lower()
        if name not in _META_KEYS or name in fields:
            return [], False
        fields[name] = tail.strip()
    if not {"summary", "heat", "facts"} <= set(fields):
        return [], False
    try:
        int(fields["heat"] or 0)
    except ValueError:
        return [], False
    keys: list[str] = []
    for part in fields["facts"].split(","):
        key = part.strip()
        if not key:
            continue
        if not is_safe_scene_key(key) or key in keys:
            return [], False
        keys.append(key)
        if len(keys) >= MAX_SCENE_FACTS:
            break
    return keys, True


# ─────────────────────────────────────────────────────────────────────────────
# Хранилища
# ─────────────────────────────────────────────────────────────────────────────

class SceneStore:
    """``user_scene_blocks``. Без PG — пустые чтения и ни одной записи."""

    def __init__(self, long_term: Optional["LongTermMemory"] = None) -> None:
        self._lt = long_term

    def list_scenes(self, company_id: str, user_id: str, *, limit: int = 24) -> list[SceneBlock]:
        if self._lt is None:
            return []
        try:
            rows = self._lt.list_scene_blocks(company_id, user_id, limit=limit)
        except Exception as exc:  # noqa: BLE001 — отсутствующая таблица ≠ сломанный recall
            logger.debug("list_scene_blocks failed: %s", exc)
            return []
        out: list[SceneBlock] = []
        for row in rows or []:
            block = parse_scene_row(row)
            if block is not None:
                out.append(block)
        return out

    def write(self, company_id: str, user_id: str, block: SceneBlock) -> bool:
        if self._lt is None:
            return False
        try:
            row = self._lt.upsert_scene_block(
                company_id          = company_id,
                user_id_or_external = user_id,
                scene_key           = block.scene_key,
                summary             = block.summary,
                content             = block.content,
                heat                = block.heat,
                facts_count         = len(block.source_fact_keys),
            )
            return bool(row)
        except Exception as exc:  # noqa: BLE001
            logger.debug("upsert_scene_block failed (%s): %s", block.scene_key, exc)
            return False

    def bump_heat(self, company_id: str, user_id: str, scene_keys: list[str]) -> int:
        if self._lt is None or not scene_keys:
            return 0
        try:
            return int(self._lt.bump_scene_heat(company_id, user_id, scene_keys[:SCENE_HEAT_KEYS_CAP]))
        except Exception as exc:  # noqa: BLE001
            logger.debug("bump_scene_heat failed: %s", exc)
            return 0

    def delete(self, company_id: str, user_id: str, scene_key: str) -> bool:
        if self._lt is None:
            return False
        try:
            return bool(self._lt.delete_scene_block(company_id, user_id, scene_key))
        except Exception as exc:  # noqa: BLE001
            logger.debug("delete_scene_block failed (%s): %s", scene_key, exc)
            return False

    def count(self, company_id: str, user_id: str) -> int:
        if self._lt is None:
            return 0
        try:
            return int(self._lt.count_scene_blocks(company_id, user_id))
        except Exception as exc:  # noqa: BLE001
            logger.debug("count_scene_blocks failed: %s", exc)
            return 0


class PersonaStore:
    """``user_persona``. Все методы деградируют в «нет портрета», не в исключение."""

    def __init__(self, long_term: Optional["LongTermMemory"] = None) -> None:
        self._lt = long_term

    def get(self, company_id: str, user_id: str) -> PersonaDoc:
        if self._lt is None:
            return PersonaDoc()
        try:
            row = self._lt.get_persona(company_id, user_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_persona failed: %s", exc)
            return PersonaDoc()
        return PersonaDoc.from_row(row)

    def bump_memories(self, company_id: str, user_id: str, delta: int = 1) -> int:
        if self._lt is None:
            return 0
        try:
            return int(self._lt.bump_persona_memories(company_id, user_id, delta))
        except Exception as exc:  # noqa: BLE001
            logger.debug("bump_persona_memories failed: %s", exc)
            return 0

    def save(self, company_id: str, user_id: str, body: str) -> Optional[dict]:
        if self._lt is None:
            return None
        try:
            return self._lt.upsert_persona(company_id=company_id, user_id_or_external=user_id, body=body)
        except Exception as exc:  # noqa: BLE001
            logger.debug("upsert_persona failed: %s", exc)
            return None

    def request_update(self, company_id: str, user_id: str, reason: str) -> bool:
        if self._lt is None:
            return False
        try:
            return bool(self._lt.request_persona_update(company_id, user_id, reason))
        except Exception as exc:  # noqa: BLE001
            logger.debug("request_persona_update failed: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Промпты (структура подсказана reference-промптами, текст — наш, terse EN)
# ─────────────────────────────────────────────────────────────────────────────

_SCENE_SYSTEM = (
    "You sort a person's durable memory facts into topic scenes.\n"
    "Answer with ONE JSON object only. No prose, no markdown fence.\n"
    "Rules:\n"
    "- Default is putting a fact into an EXISTING scene. A new scene needs a\n"
    "  clearly different topic; when in doubt reuse an existing scene.\n"
    "- scene_key: lowercase words joined by dots, max 60 chars, ASCII-ish\n"
    "  (example: backend.python.async). Keys already used must be kept as-is.\n"
    "- summary: one single line, 15-30 words, what this scene is about.\n"
    "- Every submitted fact key must appear in exactly one scene. Use the exact\n"
    "  keys given; never invent facts, values or keys.\n"
    "- Do not profile, diagnose or judge the person. Name topics and work.\n"
    "- Never put health, grief or intimacy wording into a summary."
)

_SCENE_USER_TEMPLATE = """Current scenes ({scene_count} / {max_scenes} allowed):
{scenes_block}

Unassigned facts to place ({fact_count}):
{facts_block}

Shape:
{{"scenes":[
  {{"scene_key":"backend.python","summary":"...","action":"merge",
    "fact_keys":["context.project.finkey"],"merge_from":["backend.python.flask"]}},
  {{"scene_key":"family.weekends","summary":"...","action":"new","fact_keys":["..."]}}
],
"persona_update_request":""}}

action: "new" = topic that does not exist yet; "merge" = write this scene, folding
in every existing scene listed in merge_from (leave merge_from empty to append the
facts to the existing scene of the same key).
persona_update_request: a short reason ONLY if the facts above genuinely contradict
or outgrow the current portrait; otherwise an empty string.
"""

_PERSONA_SYSTEM = (
    "You write a working portrait of one person from their topic scenes.\n"
    "Answer with ONE JSON object: {{\"body\": \"...\"}}. No prose outside it.\n"
    "Rules:\n"
    "- body: plain text, max {body_max} characters total, terse.\n"
    "- Only what the scenes state. No invention, no diagnosis, no guesses about\n"
    "  health, motives or feelings. Unwritten topic = leave it out.\n"
    "- Same language as the scene material.\n"
    "- Structure: who they are and what they work on; recurring topics;\n"
    "  standing preferences; open threads. Short lines, no bullet spam.\n"
    "- Prefer dropping a weak sentence over padding the portrait."
)

_PERSONA_USER_TEMPLATE = """Regenerating because: {reason}

Scenes ({scene_count}):
{scenes_block}

Current portrait (may be empty — then write the first one):
---
{existing}
---

Return {{"body": "..."}} within {body_max} characters.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Сервис
# ─────────────────────────────────────────────────────────────────────────────

class ScenePersonaService:
    """
    Единая точка входа L2/L3: чтение для ``load_context``, писанина для dream-цикла.

    ``completer`` — тот же gateway-путь, что у экстрактора
    (``llm_completer.build_extractor_completer_from_env``); ``None`` = LLM нет →
    генерация пропускается, хранилище остаётся нетронутым.
    """

    def __init__(
        self,
        *,
        long_term: Optional["LongTermMemory"] = None,
        semantic:  Optional["SemanticMemory"]  = None,
        completer: Optional[LLMCompleter]      = None,
    ) -> None:
        self._lt  = long_term
        self._sem = semantic
        self._llm = completer
        self.scenes  = SceneStore(long_term)
        self.persona = PersonaStore(long_term)
        self._llm_failures    = 0
        self._llm_blocked_until = 0.0

    @property
    def usable(self) -> bool:
        return self._lt is not None

    # ── чтение (hot path) ────────────────────────────────────────────────────

    def load_scenes(self, company_id: str, user_id: str, *, limit: int = 24) -> list[SceneBlock]:
        if not scenes_enabled() or self._lt is None:
            return []
        return self.scenes.list_scenes(company_id, user_id, limit=limit)

    def scene_nav(self, scenes: list[SceneBlock], *, top: int = 8) -> list[tuple[str, str, int]]:
        """(key, summary, heat) горячих сцен — то, что уходит в MemoryContext."""
        ordered = sorted(scenes, key=lambda s: (-s.heat, s.scene_key))[:max(0, top)]
        return [(s.scene_key, _clip(s.summary, 120), s.heat) for s in ordered]

    def load_persona_body(self, company_id: str, user_id: str) -> str:
        if not persona_enabled() or self._lt is None:
            return ""
        doc = self.persona.get(company_id, user_id)
        return clip_persona_body(doc.body) if doc.has_body else ""

    def bump_recall_heat(
        self,
        company_id: str,
        user_id: str,
        scenes: list[SceneBlock],
        *,
        recalled_fact_keys:      Optional[list[str]] = None,
        recalled_memory_lines:   Optional[list[str]] = None,
    ) -> int:
        """
        Heat++ по двум сигналам: факт из чьего-то fact-set попал в собранный
        контекст, либо векторный hit сцены уехал в ``relevant_memories``.

        Второй сигнал ловится сравнением строк — саммари сцены эмбеддится как
        ``content``, поэтому hit уже виден среди релевантных памятишек и отдельный
        запрос к Qdrant в hot-path не нужен.
        """
        if not scenes or self._lt is None:
            return 0
        hit_keys = {str(k or "").strip() for k in (recalled_fact_keys or []) if k}
        touched: list[str] = [
            s.scene_key for s in scenes if hit_keys & set(s.source_fact_keys)
        ]
        lines = {
            " ".join(str(m or "").split()).lower()
            for m in (recalled_memory_lines or []) if str(m or "").strip()
        }
        if lines:
            for scene in scenes:
                embed = scene_embed_text(scene).lower()
                if embed and embed in lines and scene.scene_key not in touched:
                    touched.append(scene.scene_key)
        if not touched:
            return 0
        return self.scenes.bump_heat(company_id, user_id, touched)

    # ── запись (dream cycle) ─────────────────────────────────────────────────

    def consolidate_scenes(self, company_id: str, user_id: str) -> SceneConsolidationReport:
        """
        Разложить ещё не разобранные по темам durable-факты по сценам.

        LLM даёт только назначение и саммари; контент, heat и учёт фактов — код.
        """
        report = SceneConsolidationReport()
        if not scenes_enabled() or self._lt is None:
            report.skipped_reason = "disabled_or_no_pg"
            return report

        scenes = self.scenes.list_scenes(company_id, user_id, limit=64)
        report.scenes_before = len(scenes)

        facts = self._active_facts(company_id, user_id)
        if not facts:
            report.skipped_reason = "no_facts"
            return report

        placed: set[str] = set()
        for scene in scenes:
            placed.update(scene.source_fact_keys)
        unassigned = [(k, v) for k, v in facts if k not in placed]
        report.unassigned = len(unassigned)
        if len(unassigned) < min_facts_for_consolidation():
            report.skipped_reason = "not_enough_new_facts"
            return report
        unassigned = unassigned[:MAX_UNASSIGNED_PER_RUN]
        if self._llm is None:
            # Без LLM назначение не выдумываем: свалить всё в misc — всё равно что
            # молча похоронить тему. Факты остаются доступными нижним слоям.
            report.skipped_reason = "no_llm"
            return report

        answer = self._ask_assignments(scenes, unassigned)
        report.llm_used = answer is not None
        if answer is None:
            report.skipped_reason = "llm_unavailable"
            return report
        if answer.persona_signal:
            self.persona.request_update(company_id, user_id, answer.persona_signal)
        if not answer.assignments:
            report.skipped_reason = "no_valid_assignments"
            return report

        by_key = dict(facts)
        plan, leftovers = self._validate_assignments(
            assignments=answer.assignments,
            scenes=scenes,
            submitted={k for k, _ in unassigned},
        )
        if not plan and not leftovers:
            report.skipped_reason = "nothing_after_validation"
            return report

        written = self._apply_plan(
            company_id = company_id,
            user_id    = user_id,
            scenes     = scenes,
            plan       = plan,
            by_key     = by_key,
            report     = report,
        )
        if leftovers:
            written.extend(self._fold_into_misc(
                company_id = company_id,
                user_id    = user_id,
                scenes     = scenes,
                keys       = leftovers,
                by_key     = by_key,
                report     = report,
            ))
        # Кап проверяем после всех записей: misc и merge сами могли перейти 15.
        written.extend(self._enforce_cap(company_id, user_id, by_key, report))

        report.embedded = self._embed_scenes(company_id, user_id, written)
        report.scenes_after = self.scenes.count(company_id, user_id)
        return report

    def persona_trigger_reason(
        self,
        company_id: str,
        user_id: str,
        *,
        scene_count: Optional[int] = None,
        doc:         Optional[PersonaDoc] = None,
    ) -> str:
        """
        Пять приоритетов лестницы:

        P1 явная просьба (колонка ``pending_request``) → P2 холодный старт →
        P2.5 восстановление испорченного ряда → P3 первый прирост после
        холодного старта → P4 порог новых памятишек. '' = не триггерить.
        """
        if not persona_enabled() or self._lt is None:
            return ""
        doc = doc or self.persona.get(company_id, user_id)
        count = scene_count if scene_count is not None else self.scenes.count(company_id, user_id)
        if count <= 0 and not doc.corrupt:
            return ""

        if doc.pending_request:
            return f"explicit request: {doc.pending_reason or 'persona update'}"
        if count > 0 and doc.version == 0 and not doc.has_body:
            return "cold start: scenes exist, no portrait yet"
        if doc.corrupt and count > 0:
            return "recovery: persona row lost its body"
        if count == 1 and doc.version == 1 and doc.memories_since_last > 0:
            return "first scene grew after the cold-start portrait"
        if doc.memories_since_last >= persona_interval():
            return f"threshold: {doc.memories_since_last} >= {persona_interval()}"
        return ""

    def maybe_regenerate_persona(self, company_id: str, user_id: str) -> PersonaGenerationReport:
        """Триггер-лестница + генерация. Триггера нет или LLM недоступен — no-op."""
        out = PersonaGenerationReport()
        if not persona_enabled() or self._lt is None:
            out.skipped_reason = "disabled_or_no_pg"
            return out
        scenes = self.scenes.list_scenes(company_id, user_id, limit=24)
        doc = self.persona.get(company_id, user_id)
        reason = self.persona_trigger_reason(
            company_id, user_id, scene_count=len(scenes), doc=doc,
        )
        out.reason  = reason
        out.version = doc.version
        if not reason:
            out.skipped_reason = "no_trigger"
            return out
        if self._llm is None:
            out.skipped_reason = "no_llm"
            return out
        if not scenes:
            out.skipped_reason = "no_scenes"
            return out

        body = self._ask_persona_body(scenes, doc, reason)
        if not body:
            out.skipped_reason = "llm_empty_or_invalid"
            return out
        row = self.persona.save(company_id, user_id, body)
        if not row:
            out.skipped_reason = "store_unavailable"
            return out
        out.generated = True
        out.version   = int(row.get("version") or (doc.version + 1))
        out.chars     = len(body)
        return out

    # ── внутренние шаги консолидации ─────────────────────────────────────────

    def _active_facts(self, company_id: str, user_id: str) -> list[tuple[str, str]]:
        """
        [(key, value)] активных фактов.

        Отсекаем: архивы ``__asof__`` (история, не тема), отражения ``pattern:*``
        (уже вывод dream'а, не слова человека) и ключи, которые ``normalize_key``
        переписал бы — machine-блок хранит канонические ключи, неканонический
        никогда бы не совпал при разборе и вечно считался бы нераспределённым.
        """
        try:
            rows = self._lt.list_user_facts(company_id, user_id, limit=400)
        except Exception as exc:  # noqa: BLE001
            logger.debug("consolidate: list_user_facts failed: %s", exc)
            return []
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for row in rows or []:
            key = str(row.get("key_normalized") or "").strip()
            value = str(row.get("value") or "").strip()
            if not key or not value or key in seen:
                continue
            if "__asof__" in key or key.startswith("pattern:") or key != normalize_key(key):
                continue
            seen.add(key)
            out.append((key, value))
        return out

    def _ask_assignments(
        self,
        scenes: list[SceneBlock],
        unassigned: list[tuple[str, str]],
    ) -> Optional[_SceneLLMAnswer]:
        scenes_block = "\n".join(
            f"- {s.scene_key} | heat {s.heat} | {_clip(s.summary, 120)}" for s in scenes
        ) or "(none yet)"
        facts_block = "\n".join(f"- {k}: {_clip(v, 180)}" for k, v in unassigned)
        messages = [
            {"role": "system", "content": _SCENE_SYSTEM},
            {"role": "user", "content": _SCENE_USER_TEMPLATE.format(
                scene_count  = len(scenes),
                max_scenes   = max_scenes_per_user(),
                scenes_block = scenes_block,
                fact_count   = len(unassigned),
                facts_block  = facts_block,
            )},
        ]
        raw = self._call_llm(messages, max_tokens=1400, temperature=0.0)
        if not raw:
            return None
        obj = _try_parse_json(raw)
        if not isinstance(obj, dict):
            logger.info("consolidate: response is not a JSON object — assignment refused")
            return _SceneLLMAnswer()
        items = obj.get("scenes")
        if not isinstance(items, list):
            return _SceneLLMAnswer()
        # Out-of-band «пересобери портрет»: у оригинала это регексп по свободному
        # тексту ответа модели, здесь — обычное поле той же JSON-структуры.
        return _SceneLLMAnswer(
            assignments    = items,
            persona_signal = _clip(obj.get("persona_update_request"), 200),
        )

    def _validate_assignments(
        self,
        *,
        assignments: list[dict],
        scenes:      list[SceneBlock],
        submitted:   set[str],
    ) -> tuple[list[dict], list[str]]:
        """
        Строго: нормализовать ключи, оставить только выданные факты, запретить
        двойное назначение факта и двойную запись одной сцены. Всё, что не прошло
        валидацию, — в leftovers (уезжают в misc), чтобы не предлагать их LLM
        снова и снова.
        """
        existing = {s.scene_key: s for s in scenes}
        taken: set[str] = set()
        by_key: dict[str, dict] = {}
        for item in assignments:
            if not isinstance(item, dict):
                continue
            key = normalize_key(str(item.get("scene_key") or ""))
            summary = _one_line(item.get("summary"))
            action = str(item.get("action") or "").strip().lower()
            if not is_safe_scene_key(key) or not summary or action not in ("new", "merge"):
                continue
            raw_keys = item.get("fact_keys") if isinstance(item.get("fact_keys"), list) else []
            picked: list[str] = []
            for raw in raw_keys:
                fact_key = str(raw or "").strip()
                if fact_key in submitted and fact_key not in taken:
                    taken.add(fact_key)
                    picked.append(fact_key)
                    if len(picked) >= MAX_SCENE_FACTS:
                        break
            merge_from: list[str] = []
            raw_from = item.get("merge_from") if isinstance(item.get("merge_from"), list) else []
            for raw in raw_from:
                other = normalize_key(str(raw or ""))
                if other in existing and other != key and other not in merge_from:
                    merge_from.append(other)
                    if len(merge_from) >= 5:
                        break
            if not picked and not merge_from:
                continue
            if key in existing:
                # Новая сцена под занятым ключом — это UPDATE, не CREATE.
                action = "merge"
            elif action == "merge":
                # merge без существующей цели и без merge_from = обычный new.
                action = "new"
            prev = by_key.get(key)
            if prev is not None:
                # Модель отдала одну тему двумя объектами: складываем в один,
                # иначе heat/отчёты посчитались бы дважды за одну запись.
                prev["fact_keys"]  = list(dict.fromkeys(prev["fact_keys"] + picked))[:MAX_SCENE_FACTS]
                prev["merge_from"] = list(dict.fromkeys(prev["merge_from"] + merge_from))[:5]
                if len(prev["summary"]) < len(summary):
                    prev["summary"] = summary
                continue
            by_key[key] = {
                "scene_key":  key,
                "summary":    summary,
                "action":     action,
                "fact_keys":  picked,
                "merge_from": merge_from,
            }

        # Сцена не может одновременно быть получателем и быть растворённой:
        # иначе её старые ключи окажутся в двух местах, а строка — удалённой и
        # тут же пересозданной. Такие merge_from снимаем.
        targets = set(by_key)
        plan = list(by_key.values())
        for item in plan:
            item["merge_from"] = [m for m in item["merge_from"] if m not in targets]

        leftovers = sorted(submitted - taken)[:MAX_UNASSIGNED_PER_RUN]
        return plan, leftovers

    def _apply_plan(
        self,
        *,
        company_id: str,
        user_id: str,
        scenes: list[SceneBlock],
        plan: list[dict],
        by_key: dict[str, str],
        report: SceneConsolidationReport,
    ) -> list[SceneBlock]:
        """Собрать и записать сцены по валидированному плану; merge-нутые — удалить."""
        existing = {s.scene_key: s for s in scenes}
        written: list[SceneBlock] = []

        for item in plan:
            key = item["scene_key"]
            current = existing.get(key)
            dissolved = [existing[m] for m in item["merge_from"] if m in existing]

            keys = list(dict.fromkeys(
                list(current.source_fact_keys if current else [])
                + [k for s in dissolved for k in s.source_fact_keys]
                + list(item["fact_keys"])
            ))[:MAX_SCENE_FACTS]
            report.facts_assigned += len(item["fact_keys"])

            # heat = сумма heat'ов всех участвующих + 1 (правило оригинала), но
            # считают его колонки, а не модель.
            heats = [s.heat for s in ([current] if current else []) + dissolved]
            heat = (sum(heats) + 1) if heats else 1
            created_at = (current.created_at if current else _now_iso()) or _now_iso()
            summary = item["summary"]
            try:
                block = SceneBlock(
                    scene_key        = key,
                    summary          = summary,
                    content          = render_scene_content(
                        scene_key  = key,
                        summary    = summary,
                        fact_lines = [(k, by_key.get(k, "")) for k in keys],
                        heat       = heat,
                        created_at = created_at,
                    ),
                    heat             = heat,
                    source_fact_keys = keys,
                    created_at       = created_at,
                )
            except ValueError as exc:
                logger.info("scene refused (%s/%s): %s", company_id, key, exc)
                continue
            if not self.scenes.write(company_id, user_id, block):
                continue
            written.append(block)
            if current is None:
                report.created += 1
            else:
                report.updated += 1
            for gone in dissolved:
                if self.scenes.delete(company_id, user_id, gone.scene_key):
                    report.merged_away += 1
                    self._drop_scene_point(company_id, user_id, gone.scene_key)
        return written

    def _enforce_cap(
        self,
        company_id: str,
        user_id: str,
        by_key: dict[str, str],
        report: SceneConsolidationReport,
    ) -> list[SceneBlock]:
        """
        Кап 15: пока сцен больше, сливаем ДВЕ САМЫЕ ХОЛОДНЫЕ (heat = сумма + 1).

        Буквальное «merge hottest into next» сожрало бы самую используемую тему
        вместе со второй — навигация теряет горячую сцену ровно тогда, когда она
        нужнее всего. Слияние хвоста сохраняет горячие ключи нетронутыми и
        совпадает с собственным приоритетом оригинала («merge the lowest-heat
        scenes» из их prompt-правила про превышение капа).
        """
        out: list[SceneBlock] = []
        cap = max_scenes_per_user()
        for _ in range(cap):  # жёсткий потолок итераций: цикл не должен жить вечно
            scenes = self.scenes.list_scenes(company_id, user_id, limit=64)
            if len(scenes) <= cap:
                break
            ordered = sorted(scenes, key=lambda s: (s.heat, s.scene_key))
            cold_a, cold_b = ordered[0], ordered[1]
            merged_keys = list(dict.fromkeys(
                list(cold_a.source_fact_keys) + list(cold_b.source_fact_keys)
            ))[:MAX_SCENE_FACTS]
            heat = cold_a.heat + cold_b.heat + 1
            summary = _one_line(f"{cold_a.summary or cold_a.scene_key} / {cold_b.summary or cold_b.scene_key}")
            try:
                block = SceneBlock(
                    scene_key        = cold_a.scene_key,
                    summary          = summary,
                    content          = render_scene_content(
                        scene_key  = cold_a.scene_key,
                        summary    = summary,
                        fact_lines = [(k, by_key.get(k, "")) for k in merged_keys],
                        heat       = heat,
                        created_at = cold_a.created_at or _now_iso(),
                    ),
                    heat             = heat,
                    source_fact_keys = merged_keys,
                    created_at       = cold_a.created_at,
                )
            except ValueError as exc:
                logger.info("cap merge refused (%s): %s", cold_a.scene_key, exc)
                break
            if not self.scenes.write(company_id, user_id, block):
                break
            out.append(block)
            report.updated += 1
            if self.scenes.delete(company_id, user_id, cold_b.scene_key):
                report.merged_away += 1
                self._drop_scene_point(company_id, user_id, cold_b.scene_key)
        return out

    def _fold_into_misc(
        self,
        *,
        company_id: str,
        user_id: str,
        scenes: list[SceneBlock],
        keys: list[str],
        by_key: dict[str, str],
        report: SceneConsolidationReport,
    ) -> list[SceneBlock]:
        """
        Нераспределённое не выбрасываем: оно находит ``misc.unsorted`` и перестаёт
        быть «новым» для следующего прогона. Иначе LLM каждый раз заново платит за
        то, что сам же проигнорировал.

        Misc capped by ``MAX_SCENE_FACTS`` и старые ключи идут первыми: когда он
        забит, давление ложится на модель — новые факты приходится размещать по
        темам, а не прятать.
        """
        misc = next((s for s in scenes if s.scene_key == MISC_SCENE_KEY), None)
        merged = list(dict.fromkeys(
            list(misc.source_fact_keys if misc else []) + keys
        ))[:MAX_SCENE_FACTS]
        heat = (misc.heat + 1) if misc else 1
        summary = _one_line(misc.summary if misc else MISC_SCENE_SUMMARY) or MISC_SCENE_SUMMARY
        created_at = misc.created_at if misc else _now_iso()
        try:
            block = SceneBlock(
                scene_key        = MISC_SCENE_KEY,
                summary          = summary,
                content          = render_scene_content(
                    scene_key  = MISC_SCENE_KEY,
                    summary    = summary,
                    fact_lines = [(k, by_key.get(k, "")) for k in merged],
                    heat       = heat,
                    created_at = created_at,
                ),
                heat             = heat,
                source_fact_keys = merged,
                created_at       = created_at,
            )
        except ValueError as exc:
            logger.info("misc scene refused (%s/%s): %s", company_id, user_id, exc)
            return []
        if not self.scenes.write(company_id, user_id, block):
            return []
        if misc:
            report.updated += 1
        else:
            report.created += 1
        report.facts_assigned += len(keys)
        return [block]

    def _ask_persona_body(
        self,
        scenes: list[SceneBlock],
        doc: PersonaDoc,
        reason: str,
    ) -> str:
        scenes_block = "\n".join(
            f"### {s.scene_key} (heat {s.heat})\n{s.content[:900]}" for s in scenes
        )[:PERSONA_INPUT_CAP]
        messages = [
            {"role": "system", "content": _PERSONA_SYSTEM.format(body_max=PERSONA_BODY_MAX_CHARS)},
            {"role": "user", "content": _PERSONA_USER_TEMPLATE.format(
                reason       = reason,
                scene_count  = len(scenes),
                scenes_block = scenes_block,
                existing     = (doc.body[:800] if doc.has_body else "(none yet)"),
                body_max     = PERSONA_BODY_MAX_CHARS,
            )},
        ]
        raw = self._call_llm(messages, max_tokens=900, temperature=0.2)
        if not raw:
            return ""
        obj = _try_parse_json(raw)
        if not isinstance(obj, dict):
            logger.info("persona: response is not a JSON object — portrait refused")
            return ""
        body = obj.get("body")
        if not isinstance(body, str):
            logger.info("persona: JSON has no string body — portrait refused")
            return ""
        text = body.strip()
        if len(text) < MIN_PERSONA_BODY_CHARS:
            logger.info("persona: body too short (%d chars) — portrait refused", len(text))
            return ""
        return clip_persona_body(text)

    def _call_llm(self, messages: list[dict], *, max_tokens: int, temperature: float) -> str:
        """
        Тот же контракт, что у экстрактора; наружу ничего не бросаем.

        После ``LLM_BREAKER_THRESHOLD`` подряд отказов бэкенд замораживается на
        ``LLM_BREAKER_BACKOFF_SEC``: авто-детект ``llm_completer`` всегда отдаёт
        какой-то клиент (fallback на Ollama), поэтому «LLM не настроен» приходит
        не на построении, а ошибкой соединения — и без брейкера дергался бы на
        каждом тенанте каждого dream-тика.
        """
        if self._llm is None:
            return ""
        now = time.time()
        if now < self._llm_blocked_until:
            logger.debug("scene/persona LLM blocked for %.0fs more", self._llm_blocked_until - now)
            return ""
        t0 = now
        try:
            raw = self._llm(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:  # noqa: BLE001 — память важнее одного LLM-шага
            self._llm_failures += 1
            logger.warning(
                "scene/persona LLM call failed (%s): %s", type(exc).__name__, str(exc)[:300]
            )
            if self._llm_failures >= LLM_BREAKER_THRESHOLD:
                self._llm_blocked_until = time.time() + LLM_BREAKER_BACKOFF_SEC
                self._llm_failures = 0
                logger.warning(
                    "scene/persona LLM paused for %ss after %d failures",
                    LLM_BREAKER_BACKOFF_SEC, LLM_BREAKER_THRESHOLD,
                )
            return ""
        self._llm_failures = 0
        logger.debug("scene/persona LLM elapsed_ms=%d", int((time.time() - t0) * 1000))
        return str(raw or "")

    # ── эмбеддинги сцен ───────────────────────────────────────────────────────

    def _embed_scenes(self, company_id: str, user_id: str, scenes: list[SceneBlock]) -> int:
        """
        Саммари сцены → ``finkey_user_memories`` с ``deep_kind='scene'``.

        В оригинале сцены не эмбеддились ничему: recall по теме сцены был невозможен,
        navigation — единственный путь. Point id детерминирован, поэтому
        переконсолидация перезаписывает точку, а не плодит дубль.
        """
        if self._sem is None or not scenes:
            return 0
        ok = 0
        for scene in scenes:
            text = scene_embed_text(scene)
            if not text:
                continue
            try:
                self._sem.upsert_user_memory(
                    company_id    = company_id,
                    user_id       = user_id,
                    memory_text   = text,
                    memory_type   = "scene",
                    deep_kind     = "scene",
                    point_id      = scene_point_id(company_id, user_id, scene.scene_key),
                    payload_extra = {"scene_key": scene.scene_key, "heat": scene.heat},
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001 — вектор сцены не критичен
                logger.debug("scene embed skipped (%s): %s", scene.scene_key, exc)
        return ok

    def _drop_scene_point(self, company_id: str, user_id: str, scene_key: str) -> None:
        if self._sem is None:
            return
        try:
            self._sem.delete_user_memory_point(scene_point_id(company_id, user_id, scene_key))
        except Exception as exc:  # noqa: BLE001
            logger.debug("scene point drop skipped (%s): %s", scene_key, exc)


def clip_persona_body(body: str, limit: int = PERSONA_BODY_MAX_CHARS) -> str:
    """
    ЖЁСТКИЙ кап портрета (2000 символов) — в коде, не в промпте.

    Режем по границе абзаца/строки, если она не слишком далеко от лимита, иначе
    обрываем поперёк: недописанная строка портрета читается как написанная.
    """
    text = str(body or "").strip()
    if len(text) <= limit:
        return text
    window = text[:limit]
    for boundary in ("\n\n", "\n", ". ", " "):
        cut = window.rfind(boundary)
        if cut > limit * 0.7:
            return window[:cut].strip()
    return window.strip()
