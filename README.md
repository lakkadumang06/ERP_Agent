# ERP-Agent

LangChain + LangGraph agent that reads invoices and purchase orders, validates them against an ERP database with tool calls, and saves approved orders after a human-in-the-loop check. Built phase by phase.

- Phase 1: settings, model-agnostic LLM layer, Pydantic schemas, SQLAlchemy models, seed data
- Phase 2: sample documents, document loading + structured extraction, deterministic validation
- Phase 3: ERP tools and the LangGraph workflow (extract → validate → human review → save)
- Phase 4: FastAPI + web UI + Docker
- Phase 5: provider evals
- Phase 6: full README and architecture docs
- Phase 7: MCP server

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                 # add a free Groq / Gemini key
python -m app.seed
pytest -q
```

Author: Umang Lakkad
