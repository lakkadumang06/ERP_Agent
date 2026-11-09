"""API tests with FastAPI's TestClient and the fake LLM."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import api
from app.graph import build_graph
from app.schemas import PurchaseDocument
from tests.conftest import FakeERPModel


@pytest.fixture
def client_factory():
    def _make(model: FakeERPModel) -> TestClient:
        api.set_graph(build_graph(llm=model))
        return TestClient(api.app)

    yield _make
    api.set_graph(None)


def test_health_and_products(client_factory, clean_doc: PurchaseDocument) -> None:
    with client_factory(FakeERPModel(extracted=clean_doc)) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["provider"] == "openai"
        providers = client.get("/providers").json()
        assert [p["name"] for p in providers][:3] == ["openai", "anthropic", "groq"]
        assert next(p for p in providers if p["name"] == "openai")["configured"] is True
        assert next(p for p in providers if p["name"] == "groq")["configured"] is False
        products = client.get("/products").json()
    assert len(products) == 25
    assert products[0] == {
        "sku": "JWL-001",
        "name": "Gold Plated Bangle Set",
        "category": "jewellery",
        "unit_price": 24.5,
        "stock": 40,
    }


def test_json_document_is_saved_and_listed(client_factory, clean_doc: PurchaseDocument) -> None:
    with client_factory(FakeERPModel(extracted=clean_doc)) as client:
        response = client.post("/documents", json={"text": "PO text", "provider": "openai"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "saved" and body["order_id"] is not None
        assert body["document"]["document_number"] == "PO-TEST-001"
        orders = client.get("/orders").json()
        stock = {p["sku"]: p["stock"] for p in client.get("/products").json()}
    assert orders[0]["id"] == body["order_id"]
    assert orders[0]["customer_name"] == "Shree Ganesh Jewels"
    assert {line["sku"] for line in orders[0]["lines"]} == {"JWL-001", "JWL-004"}
    assert stock["JWL-001"] == 30


def test_upload_then_review_flow(client_factory, wrong_total_doc: PurchaseDocument) -> None:
    with client_factory(FakeERPModel(extracted=wrong_total_doc)) as client:
        files = {"file": ("po.txt", b"Bestellung ... Summe 595,00", "text/plain")}
        response = client.post("/documents", files=files)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "needs_review"
        assert body["issues"][0]["code"] == "document_total_mismatch"
        thread_id = body["thread_id"]

        resumed = client.post(f"/documents/{thread_id}/review", json={"approve": True})
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["status"] == "saved"

        assert (
            client.post(f"/documents/{thread_id}/review", json={"approve": True}).status_code == 409
        )
        assert client.post("/documents/nope/review", json={"approve": True}).status_code == 404


def test_reject_via_api(client_factory, wrong_total_doc: PurchaseDocument) -> None:
    with client_factory(FakeERPModel(extracted=wrong_total_doc)) as client:
        thread_id = client.post("/documents", json={"text": "x"}).json()["thread_id"]
        rejected = client.post(f"/documents/{thread_id}/review", json={"approve": False}).json()
    assert rejected["status"] == "rejected" and rejected["order_id"] is None


def test_pdf_upload_is_read(client_factory, clean_doc: PurchaseDocument) -> None:
    with client_factory(FakeERPModel(extracted=clean_doc)) as client:
        with open("data/sample_docs/po_001.pdf", "rb") as fh:
            response = client.post(
                "/documents", files={"file": ("po_001.pdf", fh, "application/pdf")}
            )
    assert response.status_code == 200 and response.json()["status"] == "saved"


def test_validation_errors(client_factory, clean_doc: PurchaseDocument) -> None:
    with client_factory(FakeERPModel(extracted=clean_doc)) as client:
        assert client.post("/documents", json={"text": ""}).status_code == 422
        assert client.post("/documents", json={"text": "x", "provider": "grok"}).status_code == 422
        assert (
            client.post("/documents", json={"text": "x", "provider": "gemini"}).status_code == 400
        )
        bad = client.post(
            "/documents", files={"file": ("x.docx", b"...", "application/octet-stream")}
        )
        assert bad.status_code == 415


def test_web_ui_and_samples_are_served(client_factory, clean_doc: PurchaseDocument) -> None:
    with client_factory(FakeERPModel(extracted=clean_doc)) as client:
        page = client.get("/")
        assert page.status_code == 200 and "ERP-Agent" in page.text
        assert client.get("/static/app.js").status_code == 200
        assert "JWL-001" in client.get("/samples/po_001.txt").text
        assert client.get("/samples/../../.env").status_code == 404
        assert client.get("/samples/nope.txt").status_code == 404


def test_review_shows_then_vs_now_and_refuses_unseen_drift(
    client_factory, wrong_total_doc: PurchaseDocument
) -> None:
    from sqlalchemy import select

    from app.db import Product, get_session

    with client_factory(FakeERPModel(extracted=wrong_total_doc)) as client:
        body = client.post("/documents", json={"text": "x"}).json()
        thread_id = body["thread_id"]
        assert body["status"] == "needs_review" and body["recheck"]["drift"] == []
        assert body["recheck"]["snapshot"]["lines"][0] == {
            "sku": "JWL-001",
            "found": True,
            "unit_price_catalogue": 24.5,
            "stock": 40,
        }

        with get_session() as session:  # price changes while the order waits
            session.scalar(select(Product).where(Product.sku == "JWL-001")).unit_price = 30.0

        view = client.get(f"/documents/{thread_id}").json()
        assert view["status"] == "needs_review"
        drift = view["recheck"]["drift"]
        assert len(drift) == 1 and drift[0]["field"] == "price" and drift[0]["value_now"] == 30.0
        assert view["recheck"]["current"]["lines"][0]["unit_price_catalogue"] == 30.0

        # Approving without the hash = "I saw no drift": refused, nothing saved.
        refused = client.post(f"/documents/{thread_id}/review", json={"approve": True}).json()
        assert refused["status"] == "needs_re_review" and refused["order_id"] is None
        assert len(refused["recheck"]["drift"]) == 1
        assert client.get("/orders").json() == []

        # Approving against the version actually seen: saved.
        seen = refused["recheck"]["drift_hash"]
        saved = client.post(
            f"/documents/{thread_id}/review", json={"approve": True, "seen_drift_hash": seen}
        ).json()
        assert saved["status"] == "saved" and saved["order_id"] is not None
        assert client.get(f"/documents/{thread_id}").json()["status"] == "saved"
        assert client.get("/documents/nope").status_code == 404
