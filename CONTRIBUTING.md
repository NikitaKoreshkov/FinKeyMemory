# Contributing to FinKeyMemory

Thanks for your interest!

## Contributor licence terms

By opening a pull request you agree that copyright of your contribution is
**assigned to FinKey**, and that FinKey may relicense it (including commercially)
while keeping the community edition under AGPL-3.0. This is what lets the project
stay open *and* fundable. You keep the right to use your contribution elsewhere.

If you cannot agree to assignment, open an issue instead of a PR — we'll work it out.

## Ground rules

- Python 3.10+, no new required dependencies without a strong reason.
- Storage-touching changes must keep the degradation contract: missing PG / Redis /
  Qdrant / embedder must disable the layer, never crash the manager.
- Every new behaviour ships with a test in `tests/` (pure-Python first; anything
  requiring live backends goes to `tests/live/`).
- The LLM must never write metadata directly (heat, validity windows, persona caps) —
  propose an assignment, let code compute state.

## Running the suite

```bash
pip install -e ".[test]"
pytest
```
