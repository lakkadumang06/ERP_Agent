# Architecture

ERP-Agent is a LangGraph `StateGraph`. Every node reads and writes one shared
state object; a checkpointer (`MemorySaver`) saves the state after each node
so a run can pause at `human_review` and resume later by `thread_id`.

## Graph

```mermaid
flowchart TD
    S([START]) --> A[load_document<br/>.txt / .pdf → raw_text]
    A --> B[extract<br/>get_llm().with_structured_output(PurchaseDocument)<br/>retry once on schema error]
    B --> C[rule_check<br/>validate.py: required fields,<br/>line totals, document total]
    C --> D[erp_check<br/>ERP lookups + LLM agent bound to<br/>find_customer · check_stock · get_product_price]
    D -->|route: no issues| F[save_order<br/>create_order → stock reduced]
    D -->|route: issues| E[human_review<br/>interrupt payload: document + issues]
    D -->|route: extraction failed| X([END · status = error])
    E -->|Command resume approve=true,<br/>ERP unchanged since the approver's view| F
    E -->|Command resume approve=false| Y([END · status = rejected])
    E -->|approve, but ERP drifted or blocks| E
    F --> Z([END · status = saved])

    classDef llm fill:#eef3ff,stroke:#4c6ef5;
    classDef human fill:#fff4e6,stroke:#f08c00;
    classDef db fill:#e6fcf5,stroke:#0ca678;
    class B,D llm;
    class E human;
    class F db;
```

Blue nodes call the LLM, the orange node waits for a person, the green node
writes to the database.

## State

| Field | Type | Written by |
|---|---|---|
| `file_path`, `raw_text` | `str` | caller / `load_document` |
| `provider` | `"openai" \| "anthropic" \| None` | caller |
| `document` | `PurchaseDocument \| None` | `extract` |
| `issues` | `list[Issue]` | `rule_check`, `erp_check`, `save_order` |
| `customer_id` | `int \| None` | `erp_check` |
| `needs_human` | `bool` | `erp_check` |
| `approved` | `bool \| None` | `human_review` |
| `order_id` | `int \| None` | `save_order` |
| `erp_snapshot` | `dict` (`ErpSnapshot`) | `erp_check` |
| `recheck` | `dict` (`RecheckResult`) | `human_review`, `save_order` |
| `approved_drift_hash` | `str \| None` | `human_review` |
| `status` | `saved \| needs_review \| needs_re_review \| rejected \| error` | several nodes |
| `log` | `list[str]` (additive reducer) | every node |

## Three layers of validation

1. **Rules** (`app/validate.py`): pure Python, exact, unit-tested. Line totals,
   document total, required fields.
2. **ERP lookups** (`erp_lookups` in `app/graph.py`): the same three read-only
   tools the agent uses, called deterministically. Unknown customer, unknown
   SKU, insufficient stock, price deviation.
3. **LLM agent** (`run_erp_agent`): the model is bound only to the three
   read-only tools, loops until it stops calling tools (max 8 iterations) and
   then returns structured `Issue` objects plus the matched customer id. Its
   findings are merged with layers 1 and 2; it can add issues, never remove them.

Any issue, including a warning, routes the document to `human_review`. The
save node calls `create_order`, which is atomic: approval cannot write an
order with an unknown SKU or insufficient stock.

## Approval re-check (v1.1)

Human approval has its own timing gap: a proposal can wait in the queue for an
hour while prices, stock or the customer record change. `app/recheck.py`
closes it without any LLM involvement:

1. `erp_check` calls `take_snapshot(document)` and stores the result in the
   state (`erp_snapshot`: price and stock per SKU, matched customer, UTC time).
2. `human_review` calls `compare_with_current(document, snapshot)` before it
   interrupts, so the payload (and `GET /documents/{thread_id}`) carries
   `snapshot`, `current`, `drift`, `drift_hash` and `blocked`.
3. The approver resumes with `{"approve": true, "seen_drift_hash": ...}`. The
   node compares again. If the drift now hashes differently from what the
   approver saw, or any item has severity `block`, the node returns status
   `needs_re_review` and the conditional edge routes back to `human_review`,
   which interrupts again with the fresh view.
4. `save_order` compares one last time against the approved hash right before
   `create_order`; the stock check inside the transaction stays the final guard.

## Runtime pieces

| Piece | Where | Notes |
|---|---|---|
| LLM factory | `app/llm.py` | only module importing `langchain_openai` / `langchain_anthropic`; `OPENAI_BASE_URL` enables Groq, Gemini, Ollama |
| Tools | `app/tools.py` | `@tool` functions over SQLAlchemy; `create_order` never exposed to the model |
| API | `app/api.py` | one compiled graph per process so `/documents/{thread_id}/review` finds the paused thread |
| Tracing | LangSmith | every run carries tags `erp-agent`, `provider:<name>` and metadata `{provider, model}` |
| Evals | `evals/run_evals.py` | same graph, fresh database per provider, auto-reject instead of interrupt |
