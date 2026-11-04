"""FastAPI surface for ERP-Agent.

    GET  /health
    POST /documents                      multipart file OR JSON {text, provider}
    POST /documents/{thread_id}/review   body {"approve": bool}
    GET  /orders
    GET  /products
    GET  /providers                      configured LLM providers (feeds the UI dropdown)
    GET  /                               web UI (app/static)
    GET  /samples/{name}                 sample documents for the UI

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app import __version__
from app.config import PROVIDERS, configure_logging, get_settings
from app.db import Order, OrderLine, Product, get_session, init_db
from app.extract import SUPPORTED_SUFFIXES, load_document
from app.graph import build_graph, resume_document, run_document
from app.llm import available_providers, model_name_for
from app.schemas import (
    OrderLineOut,
    OrderOut,
    ProcessResult,
    ProcessTextRequest,
    ProductOut,
    ProviderOut,
    ReviewRequest,
)

logger = logging.getLogger(__name__)

# One compiled graph (and so one MemorySaver) per process: /documents/{id}/review
# must find the thread that POST /documents paused.
_graph: CompiledStateGraph | None = None


def get_graph() -> CompiledStateGraph:
    """Return the process-wide compiled graph, building it on first use."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def set_graph(graph: CompiledStateGraph | None) -> None:
    """Replace the process-wide graph (tests inject one with a fake LLM)."""
    global _graph
    _graph = graph


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    init_db()
    settings = get_settings()
    logger.info(
        "ERP-Agent API ready (provider=%s, tracing=%s)",
        settings.llm_provider,
        settings.langsmith_tracing,
    )
    yield


app = FastAPI(
    title="ERP-Agent",
    version=__version__,
    description=(
        "LangChain + LangGraph agent that turns invoices and purchase orders "
        "into validated ERP orders."
    ),
    lifespan=lifespan,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_docs"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Single-page web UI. Never cached, so UI updates show on a normal reload."""
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/samples/{name}", include_in_schema=False)
def sample_document(name: str) -> FileResponse:
    """Serve one of the bundled sample documents (used by the UI's sample picker)."""
    path = SAMPLES_DIR / Path(name).name  # Path(name).name blocks directory traversal
    if not path.is_file():
        raise HTTPException(status_code=404, detail="unknown sample")
    return FileResponse(path)


@app.get("/providers", response_model=list[ProviderOut], tags=["ops"])
def list_providers() -> list[ProviderOut]:
    """All known LLM providers with their model and whether a key is configured."""
    return [ProviderOut(**p) for p in available_providers()]


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe with the active provider and model."""
    settings = get_settings()
    return {"status": "ok", "provider": settings.llm_provider, "model": model_name_for()}


@app.post(
    "/documents",
    response_model=ProcessResult,
    tags=["agent"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {"schema": ProcessTextRequest.model_json_schema()},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "provider": {"type": "string", "enum": list(PROVIDERS)},
                        },
                        "required": ["file"],
                    }
                },
            }
        }
    },
)
async def process_document(request: Request) -> ProcessResult:
    """Run the agent on a document.

    Send ``multipart/form-data`` with a ``file`` (PDF or text) or
    ``application/json`` ``{"text": "...", "provider": "openai"}``. Returns
    ``status="saved"`` with an ``order_id``, or ``status="needs_review"`` with
    a ``thread_id`` for ``POST /documents/{thread_id}/review``.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "filename"):
            raise HTTPException(status_code=422, detail="multipart body needs a 'file' field")
        filename = upload.filename or "upload.txt"
        if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=415, detail=f"unsupported file type {Path(filename).suffix!r}"
            )
        try:
            text = await run_in_threadpool(load_document, await upload.read(), filename=filename)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        provider = str(form.get("provider")) if form.get("provider") else None
    else:
        try:
            body = ProcessTextRequest.model_validate(await request.json())
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid JSON body: {exc}") from exc
        text, provider = body.text, body.provider

    if provider is not None and provider not in PROVIDERS:
        raise HTTPException(
            status_code=422, detail=f"provider must be one of {', '.join(PROVIDERS)}"
        )
    if provider is not None and not get_settings().has_api_key(provider):
        raise HTTPException(
            status_code=400, detail=f"{provider.upper()}_API_KEY is not configured on the server"
        )

    result, _ = await run_in_threadpool(
        run_document, None, provider, str(uuid.uuid4()), text=text, graph=get_graph()
    )
    return result


@app.post("/documents/{thread_id}/review", response_model=ProcessResult, tags=["agent"])
async def review_document(thread_id: str, body: ReviewRequest) -> ProcessResult:
    """Resume a run that is waiting for human approval."""
    graph = get_graph()
    snapshot = await run_in_threadpool(graph.get_state, {"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"unknown thread_id {thread_id}")
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="this run is not waiting for review")
    result, _ = await run_in_threadpool(resume_document, graph, thread_id, body.approve)
    return result


@app.get("/orders", response_model=list[OrderOut], tags=["erp"])
def list_orders(limit: int = 50) -> list[OrderOut]:
    """Saved orders with their lines, newest first."""
    with get_session() as session:
        stmt = (
            select(Order)
            .options(
                selectinload(Order.lines).selectinload(OrderLine.product),
                selectinload(Order.customer),
            )
            .order_by(Order.id.desc())
            .limit(limit)
        )
        return [
            OrderOut(
                id=o.id,
                customer_id=o.customer_id,
                customer_name=o.customer.name,
                document_number=o.document_number,
                total=o.total,
                status=o.status,
                lines=[
                    OrderLineOut(
                        sku=line.product.sku, quantity=line.quantity, unit_price=line.unit_price
                    )
                    for line in o.lines
                ],
            )
            for o in session.scalars(stmt)
        ]


@app.get("/products", response_model=list[ProductOut], tags=["erp"])
def list_products() -> list[ProductOut]:
    """Product catalogue with current stock levels."""
    with get_session() as session:
        return [
            ProductOut(
                sku=p.sku, name=p.name, category=p.category, unit_price=p.unit_price, stock=p.stock
            )
            for p in session.scalars(select(Product).order_by(Product.sku))
        ]
