"""Audit log for proposals: who approved which version, and why one went stale.

Every write here is deterministic Python called from the graph nodes or the
save transaction; the LLM never touches it. Events:

* ``proposed``   a proposal version was created (values at proposal time)
* ``approved``   an approver accepted a version (values at proposal vs now)
* ``saved``      the order was written (``order_id``)
* ``stale``      a version was marked "stale - needs re-review" (``reason``)
* ``expired``    a version was too old to approve
* ``refreshed``  the approver asked for a new version
* ``rejected``   the approver rejected the proposal
* ``replayed``   an approve request was repeated and answered with the existing order
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from app.db import AuditLog, get_session
from app.schemas import AuditEntryOut

logger = logging.getLogger(__name__)


def record(
    event: str,
    *,
    thread_id: str,
    proposal_version: int,
    proposal_hash: str,
    actor: str | None = None,
    reason: str | None = None,
    order_id: int | None = None,
    at_proposal: dict[str, Any] | None = None,
    current: dict[str, Any] | None = None,
) -> None:
    """Append one audit row in its own short transaction."""
    with get_session() as session:
        session.add(
            AuditLog(
                thread_id=thread_id,
                proposal_version=proposal_version,
                proposal_hash=proposal_hash,
                event=event,
                actor=actor,
                reason=(reason or None) and reason[:500],
                order_id=order_id,
                at_proposal=at_proposal,
                current=current,
            )
        )
    logger.info(
        "audit thread=%s v%s %s event=%s actor=%s%s%s",
        thread_id,
        proposal_version,
        proposal_hash,
        event,
        actor or "-",
        f" order={order_id}" if order_id is not None else "",
        f" reason={reason}" if reason else "",
    )


def entries_for(thread_id: str) -> list[AuditEntryOut]:
    """All audit rows of one thread, oldest first."""
    with get_session() as session:
        rows = session.scalars(
            select(AuditLog).where(AuditLog.thread_id == thread_id).order_by(AuditLog.id)
        )
        return [
            AuditEntryOut(
                id=row.id,
                created_at=row.created_at.isoformat(timespec="seconds"),
                thread_id=row.thread_id,
                proposal_version=row.proposal_version,
                proposal_hash=row.proposal_hash,
                event=row.event,
                actor=row.actor,
                reason=row.reason,
                order_id=row.order_id,
                at_proposal=row.at_proposal,
                current=row.current,
            )
            for row in rows
        ]
