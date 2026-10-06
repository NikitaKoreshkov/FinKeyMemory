# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey memory schema — keys, dataclasses, Qdrant field names.

Partitions: ``company_id`` + ``user_id``. Persistent SQL schemas are defined in your
backend; ai-core carries a volatile fact store plus optional adapters.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from finkey_memory.consciousness_state import EmotionalDimensions, EmotionType, UserProfile
from finkey_memory.temporal import values_conflict


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)



class RedisKeys:
    """Generates namespaced Redis keys. Nothing writes a key without this class."""

    ROOT = "finkey"

    @classmethod
    def session(cls, company_id: str, user_id: str, conv_id: str) -> str:
        return f"{cls.ROOT}:{company_id}:user:{user_id}:session:{conv_id}"

    @classmethod
    def emotional_arc(cls, company_id: str, user_id: str, conv_id: str) -> str:
        return f"{cls.ROOT}:{company_id}:user:{user_id}:arc:{conv_id}"

    @classmethod
    def emotional_dims(cls, company_id: str, user_id: str, conv_id: str) -> str:
        return f"{cls.ROOT}:{company_id}:user:{user_id}:dims:{conv_id}"

    @classmethod
    def user_profile_cache(cls, company_id: str, user_id: str) -> str:
        return f"{cls.ROOT}:{company_id}:user:{user_id}:profile"

    @classmethod
    def conv_messages(cls, company_id: str, user_id: str, conv_id: str) -> str:
        """Last N messages in the conversation — for fast context window assembly.

        Включает ``user_id``: без него совпадение conversation_id у двух
        пользователей одной компании давало утечку чужой истории в контекст.
        """
        return f"{cls.ROOT}:{company_id}:user:{user_id}:conv:{conv_id}:messages"

    @classmethod
    def rate_limit(cls, company_id: str, user_id: str) -> str:
        return f"{cls.ROOT}:{company_id}:user:{user_id}:rate"

    @classmethod
    def identity_pin(cls, company_id: str, user_id: str) -> str:
        """Компактный текстовый блок идентичности для system-prompt (Фаза 4)."""
        return f"{cls.ROOT}:{company_id}:user:{user_id}:identity_pin"


TTL_SESSION          = 60 * 60 * 24
TTL_ARC              = 60 * 60 * 24
TTL_DIMS             = 60 * 60 * 24
TTL_PROFILE_CACHE    = 60 * 60
TTL_MESSAGES         = 60 * 60 * 24
TTL_IDENTITY_PIN     = 60 * 60 * 6



class QdrantCollections:
    CORE_V4         = "finkey_core_v4"         # global AI knowledge (no company_id)
    COMPANY_KB      = "finkey_company_kb"
    EMPLOYEE_CHATS  = "finkey_employee_chats"

    CONV_MEMORIES   = "finkey_conv_memories"
    USER_MEMORIES   = "finkey_user_memories"
    USER_FACTS      = "finkey_user_facts"

    RAG             = "finkey_rag"


class QdrantPayloadFields:
    COMPANY_ID      = "company_id"
    USER_ID         = "user_id"
    CONV_ID         = "conversation_id"
    CONTENT         = "content"
    MEMORY_TYPE     = "memory_type"
    CREATED_AT      = "created_at"
    RELEVANCE_SCORE = "relevance_score"
    SECTION         = "section"
    DOC_TYPE        = "doc_type"
    ACCESS_LEVEL    = "access_level"
    PROFESSION_SLUG = "profession_slug"
    IS_TRAINING     = "is_training"
    SOURCE_NAME     = "source_name"



@dataclass
class SessionState:
    """Everything Redis needs to hold for an active conversation."""
    company_id:         str
    user_id:            str
    conversation_id:    str
    turn_number:        int                   = 0
    emotional_arc:      list[str]             = field(default_factory=list)
    emotional_dims:     Optional[EmotionalDimensions] = None
    last_topic:         Optional[str]         = None
    relationship_tone:  str                   = "neutral"


@dataclass
class ConversationTurn:
    """A single message + FinKey's metadata for that turn."""
    role:              str
    content:           str
    emotion_detected:  Optional[str]   = None
    emotion_intensity: Optional[float] = None
    temperature_used:  Optional[float] = None
    inner_feeling:     Optional[str]   = None
    reasoning:         Optional[str]   = None
    model_slug:        Optional[str]   = None
    specialist_route:  Optional[bool]  = None
    thinking_ms:       Optional[int]   = None


@dataclass
class SemanticFactRecord:
    """
    Canonical keyed fact stored in-process (and mapped to rows in your DB later).

    Consolidation merges rows that share ``fact_key_normalized`` using source,
    confidence, priority and recency.
    """
    fact_key_normalized: str
    value:              str
    confidence:          float                       = 0.5
    source:             str                         = "inferred"
    priority:           float                       = 0.5
    expires_at:         Optional[datetime]          = None
    confirmed_at:       Optional[datetime]          = None
    created_at:         datetime                    = field(default_factory=_utc_now)
    touched_at:         datetime                    = field(default_factory=_utc_now)
    id:                 str                         = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class UserImpression:
    """FinKey's learned long-term impression of a user."""
    user_id:          str
    company_id:       str
    impression_type:  str
    content:          str
    confidence:       float = 0.5
    source_conv_id:   Optional[str] = None
    source:           str   = "inferred"
    priority:         float = 0.5
    expires_at:       Optional[datetime] = None
    last_confirmed_at: Optional[datetime] = None


_DURABLE_FACT_GROUPS: tuple[tuple[str, str], ...] = (
    ("identity",     "Identity"),
    ("affiliation",  "Affiliation"),
    ("context",      "Context / projects"),
    ("goal",         "Goals"),
    ("commitment",   "Commitments"),
    ("relationship", "Relationships"),
    ("preference",   "Preferences"),
    ("constraint",   "Constraints"),
    ("state",        "Current state"),
)

# How-you-speak contracts are standing orders, not optional trivia.
_VOICE_IMPRESSION_TYPES = frozenset({
    "communication_style",
    "tone",
    "response_language",
    "response_format",
    "register",
    "voice",
})
_VOICE_KEY_NEEDLES = (
    "tone",
    "language",
    "response_language",
    "response_format",
    "communication_style",
    "communication",
    "register",
    "how_to_talk",
    "how_you_speak",
    "preferred_language",
    "output_language",
)
_VOICE_LABEL_NEEDLES = (
    "tone",
    "response language",
    "response format",
    "язык ответа",
    "язык общения",
    "стиль",
    "тон",
    "как говорить",
    "как с вами",
    "register",
)
# Programming / project "language" is not a speaking contract.
_VOICE_KEY_EXCLUDES = (
    "programming",
    "code_language",
    "spoken_name",
    "native_language",
)


def _blob(*parts: object) -> str:
    return " ".join(str(p or "") for p in parts).lower().replace("_", " ").replace(".", " ")


def _is_voice_contract_fact(fact: dict) -> bool:
    """True for stored language/tone/register — not city, name, or project language."""
    cat = (fact.get("category") or "").strip().lower()
    kn = (fact.get("key_normalized") or fact.get("key") or "").strip().lower()
    label = (fact.get("label") or "").strip().lower()
    if any(ex in kn for ex in _VOICE_KEY_EXCLUDES):
        return False
    if any(n in kn for n in _VOICE_KEY_NEEDLES):
        return True
    if cat == "preference" and any(n in _blob(label, kn) for n in _VOICE_LABEL_NEEDLES):
        return True
    if any(n in label for n in _VOICE_LABEL_NEEDLES):
        return True
    return False


def _is_voice_impression(imp: "UserImpression") -> bool:
    return (imp.impression_type or "").strip().lower() in _VOICE_IMPRESSION_TYPES


def _voice_contract_label(key: str, label: str = "") -> str:
    raw = (label or "").strip()
    if raw:
        return raw
    kn = (key or "").lower()
    if "language" in kn or "locale" in kn or "язык" in kn:
        return "Language"
    if "format" in kn:
        return "Response format"
    if "tone" in kn or "тон" in kn or "style" in kn or "register" in kn:
        return "Tone"
    tail = kn.rsplit(".", 1)[-1].replace("_", " ").strip()
    return tail.title() if tail else "Voice"


def _collect_voice_contracts(
    *,
    durable_facts: list[dict],
    fact_hits: list[dict],
    impressions: list["UserImpression"],
) -> list[tuple[str, str]]:
    """Deduped (label, value) standing orders for how FinKey must speak."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(label: str, value: str) -> None:
        val = (value or "").strip()
        if not val:
            return
        sig = val.lower()
        if sig in seen:
            return
        seen.add(sig)
        rows.append((label, val))

    for fact in durable_facts or []:
        if not _is_voice_contract_fact(fact):
            continue
        kn = str(fact.get("key_normalized") or fact.get("key") or "")
        add(_voice_contract_label(kn, str(fact.get("label") or "")), str(fact.get("value") or ""))
    for hit in fact_hits or []:
        if not _is_voice_contract_fact(hit):
            continue
        kn = str(hit.get("key_normalized") or hit.get("key") or "")
        add(_voice_contract_label(kn, str(hit.get("label") or "")), str(hit.get("value") or ""))
    for imp in impressions or []:
        if not _is_voice_impression(imp):
            continue
        add(_voice_contract_label(imp.impression_type, ""), imp.content)
    return rows


@dataclass
class MemoryContext:
    """
    Everything the MindEngine needs from memory before generating a response.
    Assembled by MemoryManager.load_context().
    """
    company_id:          str
    user_id:             str
    conversation_id:     str
    user_profile:        UserProfile
    session:             SessionState

    emotional_arc:       list[EmotionType]   = field(default_factory=list)
    emotional_dims:      Optional[EmotionalDimensions] = None
    recent_history:      list[ConversationTurn] = field(default_factory=list)

    relevant_rag_docs:   list[str]           = field(default_factory=list)
    relevant_memories:   list[str]           = field(default_factory=list)

    user_impressions:    list[UserImpression] = field(default_factory=list)

    identity_pin:        str        = ""
    durable_facts:       list[dict] = field(default_factory=list)
    relevant_fact_hits:  list[dict] = field(default_factory=list)

    #: base_key → (архивное value, unix ts ухода) из ``{key}__asof__{ts}``-строк.
    #: Поле с default — старые вызовы MemoryContext(...) не ломаются.
    fact_history:        dict[str, tuple[str, float]] = field(default_factory=dict)

    #: L3 working persona (уже обрезан жёстким капом в scene_persona).
    persona:             str = ""
    #: L2 навигация по сценам: (scene_key, summary, heat), горячие первые, top-8.
    scene_nav:           list[tuple[str, str, int]] = field(default_factory=list)

    episodic_highlights: list[str] = field(default_factory=list)
    semantic_facts:      list[str] = field(default_factory=list)
    emotional_memory:    list[str] = field(default_factory=list)
    procedural_hints:    list[str] = field(default_factory=list)

    cognition_substrates_enabled: bool = False

    durable_facts_per_group: int = 4
    durable_facts_total_cap: int = 18
    fact_hits_cap:           int = 4


    @property
    def has_voice_contract(self) -> bool:
        """Language / tone / register stored for this person — must drive the reply."""
        return bool(self._voice_contracts())

    def _voice_contracts(self) -> list[tuple[str, str]]:
        return _collect_voice_contracts(
            durable_facts=self.durable_facts or [],
            fact_hits=self.relevant_fact_hits or [],
            impressions=self.user_impressions or [],
        )

    @property
    def memory_prompt_block(self) -> str:
        """
        Voice contract first (standing orders), then identity, ranked durable facts,
        remaining impressions, RAG. Cognition-substrates only if enabled.
        """
        parts: list[str] = []
        voice_rows = self._voice_contracts()
        voice_values = {value.lower() for _, value in voice_rows}

        if voice_rows:
            lines = "\n".join(f"  — {label}: {value}" for label, value in voice_rows)
            parts.append(
                "DIRECTIVE (how you speak — standing contract):\n"
                "The lines below are ORDERS for YOUR voice this turn and every turn "
                "until the person changes them. They outrank HOW I SPEAK defaults "
                "(warmth, mirroring a casual latest message, buddy register) and "
                "any personality-mode block. Speak this way in the answer itself.\n"
                "Do not confess a 'default voice' that contradicts this contract. "
                "Do not say the contract is optional, or that you are ignoring it. "
                "If they ask whether you follow it — the register of the reply IS "
                "the answer. An explicit request in THIS message may update the "
                "register for the turn; otherwise the contract stands.\n"
                f"{lines}"
            )

        pin = (self.identity_pin or "").strip()
        if pin:
            has_self_facts = any(
                (f.get("category") or "").lower() in ("identity", "affiliation", "context")
                and (f.get("value") or "").strip()
                for f in (self.durable_facts or [])
            )
            directive = ""
            if has_self_facts:
                directive = (
                    "\n\nDIRECTIVE (memory grounding):\n"
                    "The facts above are our source of truth about THIS user. "
                    "When the user asks anything about themselves — name, age, location, "
                    "company, role, current project, preferences, goals — answer "
                    "DIRECTLY from these facts, in the user's own language. "
                    "Never reply 'I don't know who you are' or 'you haven't told me' "
                    "if the answer is present above. If the user's new message "
                    "contradicts a stored fact, ask a short clarifying question "
                    "instead of falling back to 'I don't know'. "
                    "Use the name naturally — not in every message, only when it fits.\n"
                    "Do not dump personal biography (city, age, life story) into an "
                    "unrelated technical question; greetings → name is enough. "
                    "Working memory I stored (current project, current query, how we work) "
                    "still applies. Language, tone and register from the how-you-speak "
                    "contract still apply in every reply; if none is stored, the default "
                    "warm register stands.\n"
                    "Forbidden phrases (never): 'based on your profile/data/memories', "
                    "'I can see that…', 'looking at your information…', "
                    "'my records show…', 'according to your data…', or any meta about "
                    "retrieving memory — speak as if you simply know them."
                )
            parts.append(pin + directive)

        # L3 портрет + L2 навигация по темам — после контракта о голосе и
        # идентичности, ДО durable-фактов: портрет объясняет, зачем эти факты, а
        # список тем говорит модели, куда копать глубже.
        persona_block = _persona_scene_block(self.persona, self.scene_nav)
        if persona_block:
            parts.append(persona_block)

        # Durable-факты рендерим и при наличии pin: pin в Redis может быть
        # устаревшим (TTL 6ч), а PG — источник истины. Раньше был elif, и
        # свежие факты молча выпадали из промпта.
        if self.durable_facts:
            grouped = _group_durable_facts(self.durable_facts)
            chunks: list[str] = []
            shown_total = 0
            history_shown = 0
            for cat_key, cat_label in _DURABLE_FACT_GROUPS:
                rows = grouped.get(cat_key) or []
                if not rows:
                    continue
                take = rows[: self.durable_facts_per_group]
                if not take:
                    continue
                lines = []
                for r in take:
                    if _is_voice_contract_fact(r):
                        continue
                    kn  = r.get("key_normalized") or ""
                    val = (r.get("value") or "").strip()
                    if not val:
                        continue
                    suffix = ""
                    if history_shown < _FACT_HISTORY_MAX:
                        entry = (self.fact_history or {}).get(str(kn))
                        if entry is None:
                            # archive_key() обрезает базу до 60 символов.
                            entry = (self.fact_history or {}).get(str(kn)[:60])
                        suffix = _fact_history_annotation(entry, val)
                        if suffix:
                            history_shown += 1
                    if kn and kn not in val:
                        lines.append(f"  — {kn}: {val}{suffix}")
                    else:
                        lines.append(f"  — {val}{suffix}")
                    shown_total += 1
                    if shown_total >= self.durable_facts_total_cap:
                        break
                if lines:
                    chunks.append(f"{cat_label}:\n" + "\n".join(lines))
                if shown_total >= self.durable_facts_total_cap:
                    break
            if chunks:
                parts.append(
                    "Memory I keep and use (I stored this to act on it, not to ignore it):\n"
                    + "\n\n".join(chunks)
                )

        hits = (self.relevant_fact_hits or [])[: self.fact_hits_cap]
        if hits:
            seen: set[str] = set()
            hit_lines: list[str] = []
            for h in hits:
                if _is_voice_contract_fact(h):
                    continue
                val = (h.get("value") or "").strip()
                if not val or val in seen:
                    continue
                seen.add(val)
                hit_lines.append(f"  — {val}")
            if hit_lines:
                parts.append(
                    "Recalled for this message:\n" + "\n".join(hit_lines)
                )

        other_impressions = [
            i for i in (self.user_impressions or [])
            if not _is_voice_impression(i)
            and (i.content or "").strip().lower() not in voice_values
        ]
        if other_impressions:
            summaries = "\n".join(f"  — {i.content}" for i in other_impressions[:4])
            parts.append(f"Communication style and observations:\n{summaries}")

        if self.relevant_rag_docs:
            cap = 5
            try:
                import os
                cap = max(3, int(os.getenv("FINKEY_RAG_PROMPT_CHUNKS", "10") or "10"))
            except ValueError:
                cap = 5
            docs = "\n\n".join(self.relevant_rag_docs[:cap])
            grounded = (
                "\n\nDIRECTIVE (company knowledge grounding):\n"
                "The excerpts above are the company's knowledge base. "
                "When answering about company rules, products, processes, or policies, "
                "use ONLY these excerpts plus what the user said in this chat. "
                "If the answer is not in the excerpts, say you do not have that "
                "information in the company documents — do not invent. "
                "Reply in the user's language."
            )
            parts.append(f"Company knowledge excerpts:\n{docs}{grounded}")

        if self.relevant_memories:
            mems = "\n".join(f"  — {m}" for m in self.relevant_memories[:3])
            parts.append(f"From earlier conversations:\n{mems}")

        if self.cognition_substrates_enabled:
            if self.episodic_highlights:
                epi = "\n".join(f"  — {m}" for m in self.episodic_highlights[:4])
                parts.append(f"Episodic memory (meaningful beats):\n{epi}")
            if self.semantic_facts:
                sf = "\n".join(f"  — {f}" for f in self.semantic_facts[:8])
                parts.append(f"Semantic facts about context:\n{sf}")
            if self.emotional_memory:
                emos = "\n".join(f"  — {e}" for e in self.emotional_memory[:5])
                parts.append(f"Recent emotional trajectory:\n{emos}")
            if self.procedural_hints:
                proc = "\n".join(f"  — {p}" for p in self.procedural_hints[:4])
                parts.append(f"Procedural cues (how to steer the dialogue):\n{proc}")

        if parts:
            parts.append(_MEMORY_VOICE_DIRECTIVE)

        return "\n\n".join(parts) if parts else ""


_MEMORY_VOICE_DIRECTIVE = (
    "DIRECTIVE (memory voice):\n"
    "Everything I stored here is for USE. I do not treat it as optional trivia "
    "and I do not confess that I am ignoring what I wrote.\n"
    "How-you-speak (language, tone, register, response format): if stored — standing "
    "contract for this reply, including greetings and technical questions. "
    "If nothing is stored — the default warm/direct register in HOW I SPEAK stands.\n"
    "Working notes I stored (current task/query, project, how we work, format, tools, "
    "constraints, goals, commitments): act on them this turn; do not skip them "
    "because the latest message looks generic or technical.\n"
    "Biography (name, city, age, family): know it; greetings may use the name; "
    "do not dump city/life-story into an unrelated technical answer. "
    "If they ask who they are / where they live — answer from these facts. "
    "Do not surface sensitive life details (health, trauma, grief, conflicts) "
    "unless the user raised them now.\n"
    "Never narrate retrieval: no 'based on your profile', 'I can see…', "
    "'from your data/memories', 'my records show…'. "
    "If they ask what you remember — then speak about past conversations plainly."
)


# ── L3 persona + L2 scene navigation (бюджет на двоих, см. кап ниже) ─────────
_PERSONA_NAV_CAP_TOTAL   = 2600
_PERSONA_BODY_PROMPT_CAP = 1700
_SCENE_NAV_PROMPT_TOP    = 8
_SCENE_NAV_LINE_CAP      = 120
_PERSONA_HEADER = "Who this person is (working persona, regenerated as memory grows):"
_NAV_HEADER     = "Live topics:"


def _trim_at(text: str, limit: int) -> str:
    """Обрезка по границе строки/предложения, если она близко к лимиту."""
    body = str(text or "").strip()
    if len(body) <= limit or limit <= 0:
        return body if limit > 0 else ""
    window = body[:limit]
    for boundary in ("\n\n", "\n", ". ", " "):
        cut = window.rfind(boundary)
        if cut > limit * 0.7:
            return window[:cut].strip()
    return window.strip()


def _persona_scene_block(persona: str, scene_nav: list[tuple[str, str, int]]) -> str:
    """
    Один блок «кто этот человек» + «о чём мы сейчас говорим».

    Порядок внутри блока фиксирован: портрет, затем темы. Навигация дешёвая и
    почти всегда полезная, поэтому при превышении бюджета режем портрет, а не
    список тем. Heat в строку не печатаем: порядок уже по heat.
    """
    rows: list[tuple[str, str]] = []
    for item in (scene_nav or [])[:_SCENE_NAV_PROMPT_TOP]:
        try:
            key, summary, _heat = item
        except (TypeError, ValueError):
            continue
        key = str(key or "").strip()
        if not key:
            continue
        rows.append((key, _trim_at(summary, _SCENE_NAV_LINE_CAP)))

    parts: list[str] = []
    body = _trim_at(persona, _PERSONA_BODY_PROMPT_CAP)
    if body:
        parts.append(f"{_PERSONA_HEADER}\n{body}")
    if rows:
        lines = [f"  - {key}: {summary}" if summary else f"  - {key}" for key, summary in rows]
        parts.append(_NAV_HEADER + "\n" + "\n".join(lines))
    if not parts:
        return ""

    block = "\n\n".join(parts)
    if len(block) > _PERSONA_NAV_CAP_TOTAL and body:
        nav_len = len(parts[-1]) if rows else 0
        budget = _PERSONA_NAV_CAP_TOTAL - nav_len - len(_PERSONA_HEADER) - 4
        shorter = _trim_at(body, max(160, budget))
        parts[0] = f"{_PERSONA_HEADER}\n{shorter}"
        block = "\n\n".join(parts)
    return block[:_PERSONA_NAV_CAP_TOTAL]


def _group_durable_facts(facts: list[dict]) -> dict[str, list[dict]]:
    """Группировка durable-фактов по category с сохранением порядка (priority DESC, updated_at DESC)."""
    out: dict[str, list[dict]] = {}
    for f in facts:
        cat = (f.get("category") or "other").lower()
        out.setdefault(cat, []).append(f)
    return out


#: Больше четырёх «было раньше» — уже пересказ биографии, а не сигнал модели.
_FACT_HISTORY_MAX      = 4
_FACT_HISTORY_CLIP     = 80


def _asof_date_str(raw: object) -> str:
    try:
        return datetime.fromtimestamp(float(raw), timezone.utc).strftime("%Y-%m-%d")  # type: ignore[arg-type]
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _fact_history_annotation(entry: object, current_value: str) -> str:
    """
    `` (was: <old> until <YYYY-MM-DD>)`` для факта, у которого есть архив.

    Аннотация только когда старое значение отличается существенно
    (``values_conflict``: перестановка пробелов/регистр/вложение — не история).
    """
    if not entry:
        return ""
    try:
        old_value, old_ts = entry
    except (TypeError, ValueError):
        return ""
    old = str(old_value or "").strip().replace("\n", " ")
    if not old or not values_conflict(old, current_value):
        return ""
    when = _asof_date_str(old_ts)
    if not when:
        return ""
    if len(old) > _FACT_HISTORY_CLIP:
        old = old[: _FACT_HISTORY_CLIP - 1].rstrip() + "…"
    return f" (was: {old} until {when})"
