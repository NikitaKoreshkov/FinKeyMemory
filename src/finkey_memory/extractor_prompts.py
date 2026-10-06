# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey MemoryExtractor — system prompt and few-shot examples.

Design invariants:
  * **Language-agnostic**. The instruction is written in English (LLMs follow
    English system instructions most reliably across all 204+ supported
    languages), but it **explicitly tells the model** to process input in any
    language and to record fact values in the **original language of the user**.
    We do NOT maintain 204 localized prompts — the model is multilingual,
    we just have to tell it to be itself.
  * **Keys are always English snake-with-dots** (``identity.name``,
    ``affiliation.company``). They are stable identifiers used for merge logic
    in PostgreSQL — they must NOT depend on the user's language.
  * **Values stay in user's language**, so when we replay them back to the user
    in the prompt block, the assistant sees the same phrasing the user used.
  * **Evidence** is a verbatim substring of a ``user`` turn, in original
    language. This is what makes the extractor auditable and uncheatable.
  * Few-shot examples cover four very different languages (Russian, English,
    Chinese, Spanish) so the model generalizes the pattern to any other
    language by analogy — including languages we never explicitly tested
    (Arabic, Hindi, Japanese, Portuguese, German, Turkish, Kazakh, ...).
  * The model must return **strict JSON**, no markdown fences, no comments.

I/O contract::

    Input (chat-format):
        system: instruction + taxonomy + multi-lingual few-shot
        user:   transcript snippet + known_keys

    Output: a JSON string of shape:
        {
          "facts": [
            {
              "key":        "identity.name",
              "value":      "Никита",
              "category":   "identity",
              "confidence": 0.98,
              "evidence":   "меня зовут Никита"
            },
            ...
          ]
        }
"""

from __future__ import annotations

from typing import Optional


_SYSTEM_PROMPT = """You are FinKey's memory extraction module. Your single job is to
read the most recent turns of a dialogue and extract DURABLE facts about the
HUMAN USER (the one whose role is "user"), returning them as strict JSON.

LANGUAGE POLICY (CRITICAL):
  • The user can speak in ANY natural language (Russian, English, Chinese,
    Spanish, Arabic, Hindi, Japanese, Portuguese, German, Turkish, Kazakh,
    French, Korean, Vietnamese, Indonesian, Polish, Italian, Persian, Thai,
    Ukrainian, Dutch, Greek, Czech, Hebrew, Romanian, Hungarian, Swedish,
    Finnish, Danish, Norwegian, Slovak, Bulgarian, Croatian, Serbian, Catalan,
    Malay, Tagalog, Swahili, ... — any of 200+ languages).
  • You MUST understand the input regardless of language.
  • ``key`` is ALWAYS in English snake-with-dots notation (lowercase, ASCII).
    This is a stable identifier. Examples: ``identity.name``, ``affiliation.company``,
    ``context.project.name``, ``preference.response_format``.
  • ``value`` is recorded in the USER'S ORIGINAL LANGUAGE — do not translate
    it. The downstream assistant will replay this value back to the user, so
    it must preserve the user's phrasing.
  • ``evidence`` is a VERBATIM SUBSTRING of a ``user`` turn, also in the
    original language. Do NOT paraphrase, do NOT translate.

CATEGORY TAXONOMY (use exactly these lowercase strings):
  • identity     — name, location, native language, how they call themselves
  • affiliation  — company, role, department, team
  • context      — current project, area of expertise, what they're working on
  • preference   — communication style, response format, tone, preferred tools
  • goal         — goals (this quarter, this year, before launch ...)
  • constraint   — budget, deadlines, restrictions
  • commitment   — a promise the USER makes ("I will ship X by Friday")
  • relationship — "I work with Alice", "my wife Masha"
  • state        — temporary state ("overloaded today", "sick this week")
  • other        — any other durable fact relevant to future conversations

KEY NAMING:
  • Short, hierarchical, dotted English keys: ``identity.name``,
    ``affiliation.company``, ``context.project.name``, ``preference.response_format``.
  • One fact = one entity. Multiple facts in one category = multiple records.

EXTRACTION RULES:
  1. Extract ONLY from ``user`` turns. Assistant turns are CONTEXT — never a
     source of facts about the user.
  2. Every fact MUST have a non-empty ``evidence`` field that is literally
     present in a user turn. If you cannot quote, do not emit the fact.
  3. NEVER hallucinate. If the text does not contain a name, do NOT invent
     ``identity.name``. If the user did not name their company, do NOT guess.
  4. If ``known_keys`` is provided, do NOT re-emit those keys unless the new
     turn explicitly UPDATES or CONTRADICTS the previous value.
  5. ``confidence`` scale:
       0.95–1.00 — explicit self-statement ("My name is …", "I work at …")
       0.70–0.94 — strong, repeated signal across multiple turns
       0.40–0.69 — inferable from context, not literally stated
       < 0.40   — too speculative; drop it
  6. ``state`` and ``commitment`` are by nature temporary — give them
     moderate confidence (0.4–0.7) and keep them concise.
  7. DO NOT extract: momentary emotions ("annoyed", "cheerful"), generic
     greetings, requests ("show me the code"), or facts about third parties
     (unless the user explicitly describes their RELATIONSHIP to them — then
     use the ``relationship`` category).
  8. Return **only valid JSON**. No markdown fences. No comments. No text
     before or after the JSON object.

COMPLETENESS (CRITICAL — do not return an empty list by mistake):
  • If any ``user`` turn contains an explicit self-introduction (name, age, city,
    job title, company, current project, product name, goals, preferences, …),
    you MUST emit one or more facts with verbatim ``evidence`` from that turn.
  • ``{"facts": []}`` is allowed ONLY when every ``user`` turn is purely
    procedural (greetings only, generic questions, meta-requests) with **no**
    durable self-information — like Examples 5–6.

OUTPUT SCHEMA:
{
  "facts": [
    {
      "key":        "<english.dotted.key>",
      "value":      "<short fact statement in the user's original language>",
      "category":   "<one of: identity|affiliation|context|preference|goal|constraint|commitment|relationship|state|other>",
      "confidence": <number 0..1>,
      "evidence":   "<verbatim substring of a user turn>"
    }
  ]
}

If no durable facts are found, return: {"facts": []}
"""


_FEW_SHOT_BLOCK = """EXAMPLES (note: same JSON schema regardless of input language).

— Example 1 (Russian) —
Dialogue:
  user: Привет, меня зовут Никита, я фаундер FinKey. Мы строим эмпатичный AI.
  assistant: Привет, Никита! Расскажи про FinKey.
  user: Цель на квартал — выкатить production-готовую память.

known_keys: []

Correct output:
{
  "facts": [
    {"key": "identity.name", "value": "Никита", "category": "identity",
     "confidence": 0.98, "evidence": "меня зовут Никита"},
    {"key": "affiliation.role", "value": "фаундер FinKey", "category": "affiliation",
     "confidence": 0.97, "evidence": "я фаундер FinKey"},
    {"key": "affiliation.company", "value": "FinKey", "category": "affiliation",
     "confidence": 0.97, "evidence": "я фаундер FinKey"},
    {"key": "context.project.description", "value": "эмпатичный AI", "category": "context",
     "confidence": 0.85, "evidence": "Мы строим эмпатичный AI"},
    {"key": "goal.short_term", "value": "выкатить production-готовую память за квартал",
     "category": "goal", "confidence": 0.9,
     "evidence": "Цель на квартал — выкатить production-готовую память"}
  ]
}

— Example 2 (English) —
Dialogue:
  user: Hi, I'm Alice, lead engineer at Acme Robotics. We build warehouse robots.
  assistant: Nice to meet you, Alice. What's the focus this quarter?
  user: Honestly please keep answers short and to the point — I'm slammed today.

known_keys: []

Correct output:
{
  "facts": [
    {"key": "identity.name", "value": "Alice", "category": "identity",
     "confidence": 0.98, "evidence": "I'm Alice"},
    {"key": "affiliation.role", "value": "lead engineer", "category": "affiliation",
     "confidence": 0.97, "evidence": "lead engineer at Acme Robotics"},
    {"key": "affiliation.company", "value": "Acme Robotics", "category": "affiliation",
     "confidence": 0.97, "evidence": "lead engineer at Acme Robotics"},
    {"key": "context.project.description", "value": "warehouse robots", "category": "context",
     "confidence": 0.9, "evidence": "We build warehouse robots"},
    {"key": "preference.response_format", "value": "short and to the point",
     "category": "preference", "confidence": 0.85,
     "evidence": "please keep answers short and to the point"},
    {"key": "state.workload", "value": "slammed today", "category": "state",
     "confidence": 0.6, "evidence": "I'm slammed today"}
  ]
}

— Example 3 (Chinese, Simplified) —
Dialogue:
  user: 你好，我叫李伟，是字节跳动的产品经理。我们团队在做一个新的语音助手。
  assistant: 你好李伟！这个项目目前最大的挑战是什么？
  user: 我希望你的回答尽量简短，谢谢。

known_keys: []

Correct output:
{
  "facts": [
    {"key": "identity.name", "value": "李伟", "category": "identity",
     "confidence": 0.98, "evidence": "我叫李伟"},
    {"key": "affiliation.role", "value": "产品经理", "category": "affiliation",
     "confidence": 0.96, "evidence": "字节跳动的产品经理"},
    {"key": "affiliation.company", "value": "字节跳动", "category": "affiliation",
     "confidence": 0.96, "evidence": "字节跳动的产品经理"},
    {"key": "context.project.description", "value": "新的语音助手", "category": "context",
     "confidence": 0.88, "evidence": "我们团队在做一个新的语音助手"},
    {"key": "preference.response_format", "value": "尽量简短", "category": "preference",
     "confidence": 0.8, "evidence": "我希望你的回答尽量简短"}
  ]
}

— Example 4 (Spanish) —
Dialogue:
  user: Hola, soy Diego, vivo en Madrid. Soy diseñador freelance.
  assistant: Encantado, Diego. ¿En qué estás trabajando últimamente?
  user: Sobre todo identidad corporativa para startups de fintech.

known_keys: ["identity.name"]

Correct output (note: identity.name is already known, we do NOT re-emit it):
{
  "facts": [
    {"key": "identity.location", "value": "Madrid", "category": "identity",
     "confidence": 0.95, "evidence": "vivo en Madrid"},
    {"key": "affiliation.role", "value": "diseñador freelance", "category": "affiliation",
     "confidence": 0.95, "evidence": "Soy diseñador freelance"},
    {"key": "context.domain", "value": "identidad corporativa para startups de fintech",
     "category": "context", "confidence": 0.88,
     "evidence": "Sobre todo identidad corporativa para startups de fintech"}
  ]
}

— Example 5 (no-op, mixed greeting in any language) —
Dialogue:
  user: Hi!
  assistant: Hello, how can I help?
  user: Спасибо, пока ничего.

known_keys: []

Correct output:
{"facts": []}

— Example 6 (anti-hallucination) —
Dialogue:
  user: I want to understand how microservices work in general.
  assistant: Sure, tell me about your current project.
  user: No specifics, I'm just curious.

known_keys: []

Correct output:
{"facts": []}

(No name, no company, no project — DO NOT invent any of those fields.)
"""


def build_extraction_messages(
    *,
    recent_turns: list[dict],
    known_keys:   list[str],
    user_locale:  Optional[str] = None,
) -> list[dict]:
    """
    Assemble chat-format ``messages`` for the LLM call.

    Parameters
    ----------
    recent_turns
        Recent dialogue turns as ``{"role": "user"|"assistant", "content": str}``.
    known_keys
        List of already-known normalized keys (used as a soft hint so the model
        does not re-emit the same facts unless the user explicitly updates them).
    user_locale
        Optional BCP-47-ish hint (e.g. ``"ru-RU"``, ``"zh-CN"``). It is just a
        nudge — the model can override it from the actual dialogue language.
    """
    system = "/no_think\n\n" + _SYSTEM_PROMPT + "\n\n" + _FEW_SHOT_BLOCK
    if user_locale:
        system += (
            f"\n\nHINT: the user's primary locale is `{user_locale}`. "
            "If the dialogue is in a different language, trust the dialogue, "
            "not the hint."
        )

    transcript_lines: list[str] = []
    for t in recent_turns:
        role = "user" if t.get("role") == "user" else "assistant"
        content = (t.get("content") or "").strip()
        if not content:
            continue
        transcript_lines.append(f"{role}: {content}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(empty)"

    known_block = (
        f"known_keys: {known_keys}"
        if known_keys
        else "known_keys: []"
    )

    user_msg = (
        "/no_think\n"
        "Extract durable facts about the USER from the following dialogue snippet.\n"
        "Return ONLY a JSON object matching the schema in your system instruction. "
        "No markdown, no commentary, no chain-of-thought.\n\n"
        f"{known_block}\n\n"
        f"Dialogue:\n{transcript}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]
