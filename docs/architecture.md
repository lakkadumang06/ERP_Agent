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
    E -->|Command resume approve v<sub>n</sub>| F
    E -->|Command resume reject| Y([END · status = rejected])
    E -->|refresh, expired or ERP changed → proposal v<sub>n+1</sub>| C
    E -->|approval names an old version| E
    F -->|saved| Z([END · status = saved])
    F -->|ERP changed inside the save transaction → stale| C

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
| `erp_snapshot` | `dict` (`ErpSnapshot`: values with source + time, version, hash) | `erp_check` |
| `proposal_version` | `int` (0 before `erp_check`, then 1, 2, ...) | `erp_check` |
| `stale_reason`, `refresh_pending` | `str \| None`, `bool` | `human_review`, `save_order` |
| `recheck` | `dict` (`RecheckResult`) | `human_review`, `save_order` |
| `approver` | `str \| None` | `human_review` |
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

## Approval tied to a proposal version (v1.2)

Human approval has its own timing gap: a proposal can wait in the queue for an
hour while prices, stock or the customer record change. The graph closes it
without any LLM involvement:

```text
Propose (snapshot v1) → Review (at proposal vs now) → Approve v1 → Save re-checks price, customer and stock in one transaction
```

1. `erp_check` calls `take_snapshot(document, version=n)` (`app/recheck.py`,
   built on `app/snapshot.py`) and stores it in the state: every value the
   decision depends on with `source` and `read_at`, a version number and a
   hash of the decision-relevant values (prices, customer data; stock is
   informational). An audit row `proposed` is written.
2. `human_review` calls `compare_with_current()` before it interrupts, so the
   payload (and `GET /documents/{thread_id}`) carries `proposal` and
   `recheck.values` (`at_proposal`, `now`, `source`, `changed`) plus `stale`,
   `blocked` and `expired` (`PROPOSAL_TTL_HOURS`, default 24).
3. The approver resumes with `{"action": "approve", "proposal_version", "proposal_hash", "actor"}`.
   An approval for another version is refused and the current one re-shown.
   An expired or changed proposal is marked stale (audit `expired` / `stale`)
   and the edge routes back to `rule_check`, so `erp_check` runs again and
   produces version n+1, which always waits for a human (`needs_re_review`).
   `{"action": "refresh"}` does the same on request.
4. `save_order` calls `create_order(..., snapshot, thread_id, proposal_version, approved_by)`.
   Inside the transaction the customer and product rows are locked
   (`with_for_update`, effective on PostgreSQL) and re-read, compared with
   the snapshot, and stock is checked as before. Any difference raises
   `StaleProposalError`: nothing is written, the version is marked stale and
   the run goes back to `rule_check` for the next version. The unique
   constraint `orders(thread_id, proposal_version)` and a per-thread lock in
   the API make a repeated approve return the existing order.
5. `audit_log` records `proposed`, `approved`, `saved`, `stale`, `expired`,
   `refreshed`, `rejected` and `replayed` with actor, reason and the values
   at proposal time and at decision time.

## Runtime pieces

| Piece | Where | Notes |
|---|---|---|
| LLM factory | `app/llm.py` | only module importing `langchain_openai` / `langchain_anthropic`; `OPENAI_BASE_URL` enables Groq, Gemini, Ollama |
| Tools | `app/tools.py` | `@tool` functions over SQLAlchemy; `create_order` never exposed to the model |
| API | `app/api.py` | one compiled graph per process so `/documents/{thread_id}/review` finds the paused thread |
| Tracing | LangSmith | every run carries tags `erp-agent`, `provider:<name>` and metadata `{provider, model}` |
| Evals | `evals/run_evals.py` | same graph, fresh database per provider, auto-reject instead of interrupt |
