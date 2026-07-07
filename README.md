# Coding Prep Agent (MVP)

Personal placement-prep agent: onboarding → adaptive plan → DSA tracker → agent chat → readiness + a simple mock loop. Single FastAPI process, SQLite, vanilla-JS dashboard. No external services required.

## Run
```
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000 → sign up → fill Profile → "Generate Plan".

Optional LLM for open-ended agent chat: set `OPENAI_API_KEY`. Without it, chat uses deterministic rule-based routing (plan / recommend / readiness).

## Test
```
python test_placement.py
```

## What it does (maps to the docs)
- **Onboarding / Profile** (PRD F1) — role, companies, interview date, daily minutes, weak areas.
- **Adaptive plan** (F2) — rule-based, backwards from interview date, weakest topics first. Weak = declared weak areas + DSA topics with no solved problems.
- **DSA tracker** (F3) — add/list solved problems with tags; solved tags feed weak-topic ranking.
- **Agent chat** — intent-routed over *your own data* (plan / recommend / readiness), LLM fallback if a key is set.
- **Mock loop** (F4, simplified) — question → heuristic-scored feedback.
- **Readiness** — weighted: solved problems (40) + tasks done (30) + mock avg (30).

## Deliberately skipped (add when needed)
| Skipped | Add when |
|---|---|
| Postgres / Redis | multi-user scale; SQLite + in-proc state is enough for one user |
| pgvector + RAG (M6) | you upload notes/JD/resume to *retrieve* — data access is one module to swap |
| Resume tailoring (F5/M8) | you need JD-vs-resume bullets; `documents` seam is in the schema |
| LangGraph / service split | rule-based routing covers the P0 loop; `answer()` is the seam |
| Real LLM mock scoring | swap `score_answer()` / `llm_answer()` when a key + budget exist |

## Files
- `main.py` — API + planner + agent + auth (PBKDF2 + HMAC token, stdlib only).
- `static/index.html` — dashboard, profile, DSA, mock, chat (single file, no build step).
- `test_placement.py` — self-checks for planner/readiness/scoring/auth.
