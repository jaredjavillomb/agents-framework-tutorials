# Research agent contract

The spec is `SPEC.md`. Data: `data/search_fixtures.json` (30 fixed articles about a fictional company;
there is no live web) and `questions.json` (10 questions with expected facts). Code: `tools.py`
(`web_search`, `calculator`, the shared tool-call budget), `schema.py` (the `Report`) and
`scoring.py` (deterministic scoring).
