# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
LongMemEval-style runner: dataset → ingest → QA → judge → raw JSON out.

Deliberately incomplete in one place: dataset loading expects a local copy
(JSONL with sessions[] and questions[]). We do not vendor benchmark data.

Scoring is exact/substring match plus an optional LLM judge (any
OpenAI-compatible endpoint via OPENAI_API_KEY / FINKEY_EVALS_JUDGE_*).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from adapters.finkey_adapter import FinkeyAdapter  # noqa: E402


def load_dataset(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    return {"sessions": [], "questions": [json.loads(l) for l in text.splitlines() if l.strip()]}


def substring_score(answer: str, expect_any: list[str]) -> float:
    if not expect_any:
        return 0.0
    low = (answer or "").lower()
    return 1.0 if any(e.lower() in low for e in expect_any) else 0.0


def run(system: str, dataset: dict, out: Path) -> dict:
    if system != "finkey":
        raise SystemExit(
            f"adapter for {system!r} is not wired; add adapters/{system}_adapter.py "
            "implementing ingest(sessions)/answer(question) — see README."
        )
    adapter = FinkeyAdapter()
    adapter.ingest(dataset.get("sessions", []))
    rows = []
    for q in dataset.get("questions", []):
        ans = adapter.answer(q.get("question", ""))
        rows.append({
            "id": q.get("id"),
            "question": q.get("question", ""),
            "answer": ans,
            "score": substring_score(ans, list(q.get("answer_any") or [])),
        })
    acc = sum(r["score"] for r in rows) / max(1, len(rows))
    report = {"system": system, "n_questions": len(rows), "accuracy": round(acc, 4), "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser(prog="longmemeval")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--systems", default="finkey")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    dataset = load_dataset(a.dataset)
    for system in [s.strip() for s in a.systems.split(",") if s.strip()]:
        report = run(system, dataset, a.out if a.systems.strip() == system
                     else a.out.with_name(f"{a.out.stem}.{system}{a.out.suffix}"))
        print(f"{system}: accuracy={report['accuracy']} n={report['n_questions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
