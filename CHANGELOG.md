# Changelog

All notable changes to ERP-Agent. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - 2026-11-10

### Added

- **Approval tied to a proposal version** (closes the stale-approval gap for price and customer data; stock was already re-checked in the save transaction).
  - `app/snapshot.py`: the proposal snapshot now records every decision-relevant value (catalogue price per line, customer status, address, credit limit, payment terms) with its source (`products.unit_price`, `customers.credit_limit`, ...) and the UTC time it was read, plus a version number and a hash of those values.
  - `GET /documents/{thread_id}` returns `proposal` (version, hash, created/expires) and `recheck.values`: per value `at_proposal`, `now`, `source`, `changed`; plus `stale`, `blocked`, `expired`.
  - Web UI: value table with a source column and highlighted changed rows; when anything decision-relevant changed, the proposal expired or a change blocks, a warning and **Refresh proposal** replace **Approve**. Approver name field (stored in the audit log).
  - `POST /documents/{thread_id}/review` requires `proposal_version` and `proposal_hash` for approvals. `create_order` locks the customer and product rows, re-reads price and customer data inside the save transaction, compares with the snapshot and keeps the stock check; on any difference it raises `StaleProposalError`, nothing is written, the version is marked stale, the rule check and the ERP check run again and the next version is created for the approver.
  - Idempotent approve: per-thread lock in the API, unique constraint `orders(thread_id, proposal_version)`, replay of the saved result on a repeated request.
  - `POST /documents/{thread_id}/refresh` creates the next version on request. Proposals older than `PROPOSAL_TTL_HOURS` (default 24) must be refreshed before approval.
  - Audit log (`audit_log` table, `GET /documents/{thread_id}/audit`): `proposed`, `approved`, `saved`, `stale`, `expired`, `refreshed`, `rejected`, `replayed`, with actor, reason, order id, and the values at proposal time and at decision time.
  - Customer master data: `address`, `status`, `credit_limit`, `payment_terms`, `row_version` (also `row_version` on products). `erp_check` flags `customer_blocked` and `credit_limit_exceeded`.
  - Tests: price change → stale + v2; credit limit change → same; nothing changes → saved exactly once; double approve (sequential and concurrent) → one order; expired → must refresh; stock drop → still blocked; change between pre-check and transaction → caught in the transaction.

### Changed

- `seen_drift_hash` is gone; approvals name `proposal_version` + `proposal_hash`. `resume_document` takes `action="approve" | "reject" | "refresh"`.
- The database schema changed (new customer columns, `orders.thread_id`/`proposal_version`/`approved_by`, `audit_log`): run `python -m app.seed` to recreate `data/erp.db`.
- Version bumped to 1.2.0.

## [1.1.0] - 2026-11-10

### Added

- **Approval re-check** (stale-proposal protection). A proposal can wait in the review queue for an hour; by then a price, a stock level or the customer record may have changed, and the approver would sign off on data that was only right when the proposal was drafted. Now:
  - `erp_check` stores an `erp_snapshot` (catalogue price and stock per SKU, matched customer, UTC timestamp) in the graph state.
  - `app/recheck.py`: `compare_with_current(document, snapshot)` re-reads the ERP through the existing read-only tools and returns `Drift` items (`field`, `sku`, `value_at_proposal`, `value_now`, `severity`). `block` = stock now below the quantity, customer missing or different, SKU removed. `warn` = catalogue price moved by more than `RECHECK_PRICE_TOLERANCE` (default 0.5 %) or stock dropped but still sufficient.
  - The interrupt payload and the new `GET /documents/{thread_id}` return `recheck` with `snapshot`, `current`, `drift`, `drift_hash` and `blocked`.
  - `POST /documents/{thread_id}/review` accepts `seen_drift_hash`. The ERP is compared again at approval time: new drift the approver did not see returns `status="needs_re_review"` (nothing saved, the run waits for a new look); blocking drift is never saved.
  - `save_order` runs the re-check once more right before `create_order`; the stock check inside the transaction remains the final guard. Every decision is logged with `thread_id`, `snapshot_at` and a drift summary.
  - Web UI: "At proposal vs Now" table with changed cells highlighted, a drift banner, a **Re-check now** button, and approvals that carry the hash of the view on screen. The CLI prints the drift before asking "Approve?".
  - Tests for every path (price changed → `needs_re_review`; approved after seeing the drift → saved; stock dropped below quantity → blocked; no drift → saved as before).
- GitHub Actions CI (Python 3.11, ruff, pytest) and a CI badge in the README.

### Changed

- `ProcessStatus` gains `needs_re_review`; `ProcessResult` gains `recheck`.
- Version bumped to 1.1.0.

Inspired by a discussion on LinkedIn about the timing gap in human approval.

## [1.0.0] - 2026-11-05

Initial public release: structured extraction (`with_structured_output`), deterministic rule checks, ERP lookups plus a tool-calling validation agent bound to read-only tools, LangGraph `interrupt()` for human review, atomic `create_order`, FastAPI with a web UI, provider evals, Docker and Render deployment, optional MCP server.
