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
    E -->|Command resume approve=true| F
    E -->|Command resume approve=false| Y([END · status = rejected])
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
| `status` | `saved \| needs_review \| rejected \| error` | several nodes |
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

## Runtime pieces

| Piece | Where | Notes |
|---|---|---|
| LLM factory | `app/llm.py` | only module importing `langchain_openai` / `langchain_anthropic`; `OPENAI_BASE_URL` enables Groq, Gemini, Ollama |
| Tools | `app/tools.py` | `@tool` functions over SQLAlchemy; `create_order` never exposed to the model |
| API | `app/api.py` | one compiled graph per process so `/documents/{thread_id}/review` finds the paused thread |
| Tracing | LangSmith | every run carries tags `erp-agent`, `provider:<name>` and metadata `{provider, model}` |
| Evals | `evals/run_evals.py` | same graph, fresh database per provider, auto-reject instead of interrupt |
