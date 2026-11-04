"""Shared pytest fixtures.

No API keys are needed: the LLM is replaced by :class:`FakeERPModel`, which
implements just enough of ``BaseChatModel`` for the graph (``bind_tools``,
``with_structured_output`` and a tool-calling first turn).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr

# Tests must not depend on the developer's .env: pin the database, the default
# provider and dummy keys (the LLM is always the fake model below).
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["ANTHROPIC_API_KEY"] = "test-key"
for free_provider in ("GROQ", "GEMINI", "OPENROUTER"):
    os.environ[f"{free_provider}_API_KEY"] = ""  # not configured, whatever .env says
os.environ.pop("LANGSMITH_TRACING", None)

from app.config import reset_settings_cache  # noqa: E402
from app.db import configure_engine  # noqa: E402
from app.schemas import AgentVerdict, OrderLine, PurchaseDocument  # noqa: E402
from app.seed import seed  # noqa: E402

reset_settings_cache()


class FakeERPModel(BaseChatModel):
    """Deterministic stand-in for ChatOpenAI / ChatAnthropic.

    * ``with_structured_output(PurchaseDocument)`` returns ``extracted`` after
      ``fail_extraction_times`` simulated parse failures.
    * ``with_structured_output(AgentVerdict)`` returns ``verdict``.
    * In the tool-calling loop the first turn issues ``tool_calls_to_make``
      and the next turn ends the loop with a plain text message.
    """

    extracted: PurchaseDocument | None = None
    verdict: AgentVerdict = AgentVerdict(customer_id=None, issues=[], summary="looks fine")
    tool_calls_to_make: list[dict[str, Any]] = []
    fail_extraction_times: int = 0
    _extraction_attempts: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "fake-erp-model"

    @property
    def extraction_attempts(self) -> int:
        return self._extraction_attempts

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
        if self.tool_calls_to_make and not has_tool_results:
            calls = [
                {"name": c["name"], "args": c["args"], "id": f"call_{i}", "type": "tool_call"}
                for i, c in enumerate(self.tool_calls_to_make)
            ]
            message = AIMessage(content="", tool_calls=calls)
        else:
            message = AIMessage(content="Checked the ERP; see verdict.")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeERPModel:  # type: ignore[override]
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:  # type: ignore[override]
        include_raw = bool(kwargs.get("include_raw"))

        def _run(_messages: Any) -> Any:
            if schema is PurchaseDocument:
                self._extraction_attempts += 1
                if self._extraction_attempts <= self.fail_extraction_times:
                    error = ValueError("simulated schema violation")
                    if include_raw:
                        return {
                            "raw": AIMessage(content="{}"),
                            "parsed": None,
                            "parsing_error": error,
                        }
                    raise error
                if include_raw:
                    return {
                        "raw": AIMessage(content="{}"),
                        "parsed": self.extracted,
                        "parsing_error": None,
                    }
                return self.extracted
            if schema is AgentVerdict:
                return self.verdict
            raise AssertionError(f"unexpected schema {schema}")

        return RunnableLambda(_run)


@pytest.fixture(autouse=True)
def erp_db(tmp_path) -> Iterator[None]:
    """Fresh seeded SQLite database for every test."""
    configure_engine(f"sqlite:///{tmp_path}/test_erp.db")
    seed()
    yield


@pytest.fixture
def clean_doc() -> PurchaseDocument:
    """A purchase order that matches the seed data exactly."""
    return PurchaseDocument(
        doc_type="purchase_order",
        document_number="PO-TEST-001",
        document_date="2026-01-13",
        customer_name="Shree Ganesh Jewels",
        currency="EUR",
        lines=[
            OrderLine(
                sku="JWL-001",
                description="Gold Plated Bangle Set",
                quantity=10,
                unit_price=24.50,
                line_total=245.00,
            ),
            OrderLine(
                sku="JWL-004",
                description="Pearl Drop Earrings",
                quantity=20,
                unit_price=12.50,
                line_total=250.00,
            ),
        ],
        stated_total=495.00,
    )


@pytest.fixture
def wrong_total_doc(clean_doc: PurchaseDocument) -> PurchaseDocument:
    return clean_doc.model_copy(update={"document_number": "PO-TEST-002", "stated_total": 595.00})


@pytest.fixture
def unknown_sku_doc(clean_doc: PurchaseDocument) -> PurchaseDocument:
    extra = OrderLine(
        sku="JWL-999", description="Mystery Item", quantity=1, unit_price=5.0, line_total=5.0
    )
    return clean_doc.model_copy(
        update={
            "document_number": "PO-TEST-003",
            "lines": [*clean_doc.lines, extra],
            "stated_total": 500.00,
        }
    )
