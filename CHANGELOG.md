# Changelog

All notable changes to ERP-Agent. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
