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

        proposal = body["proposal"]
        assert proposal["version"] == 1 and len(proposal["hash"]) == 16

        # approve=true without the version/hash is refused.
        assert (
            client.post(f"/documents/{thread_id}/review", json={"approve": True}).status_code == 422
        )

        decision = {
            "approve": True,
            "proposal_version": proposal["version"],
            "proposal_hash": proposal["hash"],
            "approver": "alice",
        }
        resumed = client.post(f"/documents/{thread_id}/review", json=decision)
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["status"] == "saved"
        order_id = resumed.json()["order_id"]

        # Double click / retry: same answer, still one order.
        again = client.post(f"/documents/{thread_id}/review", json=decision)
        assert again.status_code == 200 and again.json()["order_id"] == order_id
        orders = client.get("/orders").json()
        assert len(orders) == 1
        assert (
            orders[0]["thread_id"],
            orders[0]["proposal_version"],
            orders[0]["approved_by"],
        ) == (
            thread_id,
            1,
            "alice",
        )
        # A different decision on a finished run is a conflict.
        other = {**decision, "proposal_hash": "0000000000000000"}
        assert client.post(f"/documents/{thread_id}/review", json=other).status_code == 409
        assert client.post(f"/documents/{thread_id}/refresh").status_code == 409
        assert client.post("/documents/nope/review", json=decision).status_code == 404

        audit = client.get(f"/documents/{thread_id}/audit").json()
        assert [(e["proposal_version"], e["event"]) for e in audit] == [
            (1, "proposed"),
            (1, "approved"),
            (1, "saved"),
            (1, "replayed"),
        ]
        assert audit[1]["actor"] == "alice" and audit[2]["order_id"] == order_id
        assert client.get("/documents/nope/audit").status_code == 404


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


def test_review_shows_at_proposal_vs_now_and_requires_refresh(
    client_factory, wrong_total_doc: PurchaseDocument
) -> None:
    from sqlalchemy import select

    from app.db import Product, get_session

    with client_factory(FakeERPModel(extracted=wrong_total_doc)) as client:
        body = client.post("/documents", json={"text": "x"}).json()
        thread_id = body["thread_id"]
        assert body["status"] == "needs_review" and not body["recheck"]["stale"]
        values = {v["key"]: v for v in body["recheck"]["values"]}
        assert values["price:JWL-001"] == {
            "key": "price:JWL-001",
            "label": "Catalogue price",
            "sku": "JWL-001",
            "at_proposal": 24.5,
            "now": 24.5,
            "source": "products.unit_price",
            "read_at": body["proposal"]["created_at"],
            "changed": False,
            "decision_relevant": True,
        }
        assert values["customer.credit_limit"]["source"] == "customers.credit_limit"

        with get_session() as session:  # price changes while the order waits
            session.scalar(select(Product).where(Product.sku == "JWL-001")).unit_price = 30.0

        view = client.get(f"/documents/{thread_id}").json()
        assert view["status"] == "needs_review" and view["recheck"]["stale"]
        row = next(v for v in view["recheck"]["values"] if v["key"] == "price:JWL-001")
        assert (row["at_proposal"], row["now"], row["changed"]) == (24.5, 30.0, True)
        assert view["proposal"]["version"] == 1 and not view["proposal"]["expired"]

        # Approving v1 anyway: nothing saved, v2 is created for the approver.
        decision = {
            "approve": True,
            "proposal_version": 1,
            "proposal_hash": view["proposal"]["hash"],
            "approver": "alice",
        }
        refused = client.post(f"/documents/{thread_id}/review", json=decision).json()
        assert refused["status"] == "needs_re_review" and refused["order_id"] is None
        assert (
            refused["proposal"]["version"] == 2
            and "ERP changed" in refused["proposal"]["stale_reason"]
        )
        assert client.get("/orders").json() == []

        # The approver refreshes once more (nothing changed, v3) and approves v3.
        refreshed = client.post(f"/documents/{thread_id}/refresh", json={"actor": "alice"}).json()
        assert refreshed["status"] == "needs_re_review" and refreshed["proposal"]["version"] == 3
        saved = client.post(
            f"/documents/{thread_id}/review",
            json={
                **decision,
                "proposal_version": 3,
                "proposal_hash": refreshed["proposal"]["hash"],
            },
        ).json()
        assert saved["status"] == "saved" and saved["order_id"] is not None
        assert client.get(f"/documents/{thread_id}").json()["status"] == "saved"
        assert client.get("/documents/nope").status_code == 404
        events = [
            (e["proposal_version"], e["event"])
            for e in client.get(f"/documents/{thread_id}/audit").json()
        ]
        assert events == [
            (1, "proposed"),
            (1, "stale"),
            (2, "proposed"),
            (2, "refreshed"),
            (3, "proposed"),
            (3, "approved"),
            (3, "saved"),
        ]


def test_concurrent_double_approve_creates_one_order(
    client_factory, wrong_total_doc: PurchaseDocument
) -> None:
    """Two approve requests at the same time: the per-thread lock serialises them."""
    from concurrent.futures import ThreadPoolExecutor

    with client_factory(FakeERPModel(extracted=wrong_total_doc)) as client:
        body = client.post("/documents", json={"text": "x"}).json()
        thread_id, proposal = body["thread_id"], body["proposal"]
        decision = {
            "approve": True,
            "proposal_version": proposal["version"],
            "proposal_hash": proposal["hash"],
        }
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(
                    lambda _: client.post(f"/documents/{thread_id}/review", json=decision), range(2)
                )
            )
        assert [r.status_code for r in responses] == [200, 200]
        assert len({r.json()["order_id"] for r in responses}) == 1
        assert len(client.get("/orders").json()) == 1
