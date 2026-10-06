# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Live L2/L3 memory checks against the REAL gateway LLM (OpenRouter).

Skipped unless ``OPENROUTER_API_KEY`` is present in the environment or in
``services/integrations/llm-gateway/.env`` (read key-name-only; values never
touch stdout). Run explicitly:

    python3 -m pytest tests/live/test_scene_persona_live.py -q -s
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from finkey_memory.scene_persona import (
    PERSONA_BODY_MAX_CHARS,
    ScenePersonaService,
    is_safe_scene_key,
    parse_scene_row,
)

GW_ENV = Path(__file__).resolve().parents[3] / "services/integrations/llm-gateway/.env"


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.replace("export ", "").strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_env(GW_ENV)
_HAS_KEY = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _HAS_KEY, reason="OPENROUTER_API_KEY not available — live check skipped"
    ),
]

FACTS = [
    ("career.works_at.acme", "data engineer at Acme"),
    ("context.lives_in.lisbon", "lives in Lisbon"),
    ("context.kid.mira.age7", "daughter Mira, 7 years old"),
    ("preference.language.ru", "prefers Russian"),
    ("goal.dashboard.deadline.oct15", "FinKey dashboard by 2026-10-15"),
    ("interest.trail.running.saturday", "trail running every Saturday"),
    ("career.reports_to.pavel", "reports to Pavel"),
    ("preference.diet.vegetarian", "vegetarian"),
    ("goal.tokyo.nov.talk", "conference talk in Tokyo in November"),
    ("context.tools.linear.figma", "uses Linear and Figma daily"),
    ("preference.style.short", "wants short answers"),
    ("context.budget.4200eur", "monthly budget 4200 EUR"),
]


class FakeLT:
    """Dict-backed store implementing only what SceneStore/PersonaStore call."""

    def __init__(self) -> None:
        self.scenes: dict = {}
        self.persona: dict = {}
        self.facts = []

    def list_user_facts(self, company_id, user_id, limit=200):
        return list(self.facts[:limit])

    def upsert_scene_block(self, company_id, user_id_or_external, scene_key, summary,
                           content, heat, facts_count):
        self.scenes[(company_id, user_id_or_external, scene_key)] = {
            "scene_key": scene_key, "summary": summary, "content": content,
            "heat": heat, "facts_count": facts_count,
        }
        return True

    def list_scene_blocks(self, company_id, user_id, limit=24):
        return [r for (c, u, _), r in self.scenes.items()
                if c == company_id and u == user_id][:limit]

    def bump_scene_heat(self, company_id, user_id, scene_keys):
        n = 0
        for k in scene_keys:
            row = self.scenes.get((company_id, user_id, k))
            if row:
                row["heat"] = int(row["heat"]) + 1
                n += 1
        return n

    def delete_scene_block(self, company_id, user_id, scene_key):
        return self.scenes.pop((company_id, user_id, scene_key), None) is not None

    def count_scene_blocks(self, company_id, user_id):
        return sum(1 for (c, _u, _k) in self.scenes if c == company_id)

    def get_persona(self, company_id, user_id):
        return self.persona.get((company_id, user_id))

    def bump_persona_memories(self, company_id, user_id, delta):
        row = self.persona.setdefault((company_id, user_id), {
            "body": "", "version": 0, "memories_since_last": 0, "pending_request": False,
        })
        row["memories_since_last"] = int(row.get("memories_since_last", 0)) + delta
        return row["memories_since_last"]

    def upsert_persona(self, company_id, user_id_or_external, body):
        prev = self.persona.get((company_id, user_id_or_external), {})
        self.persona[(company_id, user_id_or_external)] = {
            "body": body, "version": int(prev.get("version", 0)) + 1,
            "memories_since_last": 0, "pending_request": False,
        }
        return self.persona[(company_id, user_id_or_external)]

    def request_persona_update(self, company_id, user_id, reason):
        row = self.persona.setdefault((company_id, user_id), {
            "body": "", "version": 0, "memories_since_last": 0,
            "pending_request": True, "pending_reason": reason,
        })
        row["pending_request"] = True
        return True


@pytest.fixture(scope="module")
def svc_and_lt():
    os.environ.setdefault("FINKEY_MEMORY_EXTRACTOR_BACKEND", "openrouter")
    from finkey_memory.llm_completer import build_extractor_completer_from_env

    completer = build_extractor_completer_from_env()
    assert callable(completer), "gateway completer factory returned non-callable"
    lt = FakeLT()
    for i, (key, val) in enumerate(FACTS):
        lt.facts.append({
            "key_normalized": key, "value": val, "category": "context",
            "priority": 0.8, "source": "test",
            "updated_at": time.time() - i * 3600, "access_count": 0,
        })
    svc = ScenePersonaService(long_term=lt, semantic=None, completer=completer)
    return svc, lt


def test_a_consolidation_creates_valid_scenes(svc_and_lt):
    svc, lt = svc_and_lt
    t0 = time.time()
    rep = svc.consolidate_scenes("c1", "u1")
    print(f"\nconsolidation {rep.to_dict()} in {time.time()-t0:.1f}s")
    assert lt.scenes, "real LLM produced zero scenes from 12 topical facts"
    assert all(is_safe_scene_key(k[2]) for k in lt.scenes)
    assert all(parse_scene_row(r) is not None for r in lt.scenes.values())
    assert len(lt.scenes) <= 15


def test_b_second_pass_is_stable(svc_and_lt):
    svc, lt = svc_and_lt
    rep2 = svc.consolidate_scenes("c1", "u1")
    print(f"second pass {rep2.to_dict()}")
    assert len(lt.scenes) <= 15


def test_c_persona_generated_and_capped(svc_and_lt):
    svc, lt = svc_and_lt
    lt.bump_persona_memories("c1", "u1", 30)
    gen = svc.maybe_regenerate_persona("c1", "u1")
    print(f"persona gen {gen.to_dict()}")
    assert gen.generated, f"persona not generated: {gen.skipped_reason}"
    body = (lt.get_persona("c1", "u1") or {}).get("body", "")
    assert body.strip()
    assert len(body) <= PERSONA_BODY_MAX_CHARS
    assert len(body.split()) >= 5


def test_d_trigger_ladder_no_re_fire_then_explicit(svc_and_lt):
    svc, lt = svc_and_lt
    assert svc.persona_trigger_reason("c1", "u1") == ""
    svc.persona.request_update("c1", "u1", "user asked")
    assert svc.persona_trigger_reason("c1", "u1").startswith("explicit request")


def test_f_russian_facts_pipeline(svc_and_lt):
    """RU corpus on a second tenant: language-agnostic scene/persona must hold."""
    svc, _lt = svc_and_lt
    lt = FakeLT()
    ru = [
        ("family.daughter.mira7", "дочке Мира 7 лет, ходит в танцы"),
        ("context.city.kazan", "живёт в Казани с 2024"),
        ("preference.diet.vegetarian", "строгий вегетарианец"),
        ("hobby.trailrun.saturday", "каждую субботу трейлраннинг"),
        ("career.acme.dataeng", "дата-инженер в Acme, отчётность к 15 октября"),
        ("preference.language.ru", "все ответы по-русски, кратко"),
        ("context.budget.eur4200", "месячный бюджет 4200 евро"),
        ("travel.tokyo.november", "конференция в Токио в ноябре"),
    ]
    for i, (k, v) in enumerate(ru):
        lt.facts.append({
            "key_normalized": k, "value": v, "category": "context",
            "priority": 0.8, "source": "test",
            "updated_at": time.time() - i * 3600, "access_count": 0,
        })
    svc2 = ScenePersonaService(
        long_term=lt, semantic=None, completer=svc._llm,
    )
    rep = svc2.consolidate_scenes("c2", "u2")
    assert rep.llm_used and lt.scenes, rep.to_dict()
    assert all(parse_scene_row(r) is not None for r in lt.scenes.values())
    lt.bump_persona_memories("c2", "u2", 30)
    gen = svc2.maybe_regenerate_persona("c2", "u2")
    assert gen.generated, gen.skipped_reason
    body = (lt.get_persona("c2", "u2") or {}).get("body", "")
    assert len(body) <= PERSONA_BODY_MAX_CHARS
    assert any(ord(ch) > 1000 for ch in body), "RU persona body lost Cyrillic entirely"


def test_e_recall_heat_and_prompt_surfaces(svc_and_lt):
    svc, lt = svc_and_lt
    scenes = svc.load_scenes("c1", "u1")
    assert scenes
    first = scenes[0]
    key = ("c1", "u1", first.scene_key)
    h0 = lt.scenes[key]["heat"]
    fk = list(first.source_fact_keys[:1])
    if fk:
        bumped = svc.bump_recall_heat("c1", "u1", scenes, recalled_fact_keys=fk)
        assert bumped >= 1 and lt.scenes[key]["heat"] >= h0
    nav = svc.scene_nav(scenes)
    assert all(len(k) <= 80 and len(s) <= 120 for k, s, _h in nav)
    body = svc.load_persona_body("c1", "u1")
    assert body
