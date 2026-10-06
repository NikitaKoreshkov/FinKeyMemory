# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Unit-тесты ``DecayScheduler`` — фон-поток, периодически чистящий истёкшие user_facts.

Тесты используют **fake FactStore**, чтобы:
  * не зависеть от живого PostgreSQL (PG-уровневый тест лежит в
    services/internal/tests/test_long_term_decay.py);
  * проверить именно scheduling-логику (lifecycle, kick, stop, batch, грейс).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from finkey_memory.decay import DecayScheduler, DecaySchedulerStats
from finkey_memory.fact_store import DecayReport




@dataclass
class _FakeFactStore:
    """Минимальный stub: считает вызовы run_decay и возвращает запланированные отчёты."""
    decay_calls: list[dict] = field(default_factory=list)
    reports:     list[DecayReport] = field(default_factory=list)
    raise_on:    int = -1
    lock:        threading.Lock = field(default_factory=threading.Lock)

    def run_decay(self, *, company_id, max_rows, grace_sec) -> DecayReport:
        with self.lock:
            self.decay_calls.append({
                "company_id": company_id,
                "max_rows":   max_rows,
                "grace_sec":  grace_sec,
            })
            n = len(self.decay_calls)
            if self.raise_on == n:
                raise RuntimeError("planned failure")
            if self.reports:
                return self.reports.pop(0)
            return DecayReport(pg_decayed=0)




def test_scheduler_tick_once_calls_factstore():
    fs = _FakeFactStore(reports=[DecayReport(pg_decayed=3, qdrant_ok=3)])
    sch = DecayScheduler(
        fact_store=fs, interval_sec=60, batch_size=42, grace_sec=10,
        initial_delay=0, enabled=True,
    )
    report = sch.tick_once()
    assert isinstance(report, DecayReport)
    assert report.pg_decayed == 3
    assert len(fs.decay_calls) == 1
    assert fs.decay_calls[0]["max_rows"]  == 42
    assert fs.decay_calls[0]["grace_sec"] == 10
    assert fs.decay_calls[0]["company_id"] is None


def test_scheduler_stats_accumulate_across_ticks():
    fs = _FakeFactStore(reports=[
        DecayReport(pg_decayed=2, qdrant_ok=2),
        DecayReport(pg_decayed=5, qdrant_ok=4, qdrant_fail=1),
        DecayReport(pg_decayed=0),
    ])
    sch = DecayScheduler(fact_store=fs, interval_sec=60, enabled=True)
    sch.tick_once()
    sch.tick_once()
    sch.tick_once()
    assert sch.stats.ticks == 3
    assert sch.stats.decayed_total == 7
    assert sch.stats.qdrant_ok == 6
    assert sch.stats.qdrant_fail == 1
    assert isinstance(sch.stats.last_report, DecayReport)


def test_scheduler_disabled_does_not_start():
    fs = _FakeFactStore()
    sch = DecayScheduler(fact_store=fs, enabled=False)
    assert sch.start() is False


def test_scheduler_start_and_stop_lifecycle():
    fs = _FakeFactStore(reports=[
        DecayReport(pg_decayed=1),
        DecayReport(pg_decayed=2),
    ])
    sch = DecayScheduler(
        fact_store=fs, interval_sec=600, initial_delay=0, enabled=True,
    )
    assert sch.start() is True
    deadline = time.time() + 2.0
    while time.time() < deadline and sch.stats.ticks < 1:
        time.sleep(0.01)
    assert sch.stats.ticks >= 1, "expected at least one initial tick"
    assert sch.start() is False
    sch.stop(wait=True, timeout=2.0)


def test_scheduler_kick_wakes_up_early():
    """kick() должен разбудить scheduler досрочно (без ожидания interval_sec)."""
    fs = _FakeFactStore(reports=[
        DecayReport(pg_decayed=0),
        DecayReport(pg_decayed=99),
        DecayReport(pg_decayed=0),
        DecayReport(pg_decayed=0),
    ])
    sch = DecayScheduler(
        fact_store=fs,
        interval_sec=3600,
        initial_delay=0,
        enabled=True,
    )
    sch.start()
    t0 = time.time()
    while time.time() - t0 < 1.5 and sch.stats.ticks < 1:
        time.sleep(0.01)
    assert sch.stats.ticks >= 1

    sch.kick()
    t1 = time.time()
    while time.time() - t1 < 1.5 and sch.stats.ticks < 2:
        time.sleep(0.01)
    assert sch.stats.ticks >= 2

    sch.stop(wait=True, timeout=2.0)
    assert sch.stats.decayed_total >= 99


def test_scheduler_tick_failure_does_not_kill_loop():
    """Падение run_decay не должно убивать фон-поток — следующий тик пробует снова."""
    fs = _FakeFactStore(
        reports=[
            DecayReport(pg_decayed=0),
            DecayReport(pg_decayed=42),
        ],
        raise_on=2,
    )
    sch = DecayScheduler(
        fact_store=fs, interval_sec=3600, initial_delay=0, enabled=True,
    )
    sch.start()
    t0 = time.time()
    while time.time() - t0 < 1.5 and sch.stats.ticks < 1:
        time.sleep(0.01)
    sch.kick()
    t1 = time.time()
    while time.time() - t1 < 1.5 and sch.stats.errors < 1:
        time.sleep(0.01)
    sch.kick()
    t2 = time.time()
    while time.time() - t2 < 1.5 and sch.stats.decayed_total < 42:
        time.sleep(0.01)

    sch.stop(wait=True, timeout=2.0)
    assert sch.stats.errors >= 1
    assert sch.stats.decayed_total == 42


def test_scheduler_default_stats_fields():
    s = DecaySchedulerStats()
    assert s.ticks == 0
    assert s.decayed_total == 0
    assert s.errors == 0
    assert s.last_report is None
