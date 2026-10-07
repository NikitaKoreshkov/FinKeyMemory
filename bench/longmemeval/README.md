# LongMemEval-style harness (skeleton)

Goal: durable-conversation recall measured on a public dataset, same questions,
same judge, one adapter per system. **No numbers are published from this folder
until it has actually been run end-to-end and the raw JSON outputs are committed
next to this file.**

Layout:

```
run_eval.py     # dataset → per-system ingest → QA pass → judge scoring
adapters/       # finkey.py (done), others are stubs that raise until wired
results/        # committed raw run JSONs go here (one file per system per run)
```

Datasets are long-conversation QA suites (LongMemEval, LoCoMo — download and
licence terms are yours; we deliberately do not vendor them).

```bash
pip install -e ".[test]"
python bench/longmemeval/run_eval.py --dataset ./datasets/longmemeval \
    --systems finkey --out results/finkey-run1.json
```

Competitor adapters are opt-in: install their package in your own venv and wire
the thin adapter interface (`ingest(sessions)`, `answer(question)`). Judge
requires an API key by design — the heuristic judge tier (`finkey-evals`) can
score a subset offline.
