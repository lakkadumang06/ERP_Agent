"""Document loading and LLM-based structured extraction.

``load_document`` turns a PDF or text file into plain text and
``extract_document`` asks the model to fill the :class:`PurchaseDocument`
schema, retrying once if the first answer does not validate.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pypdf import PdfReader

from app.llm import get_llm
from app.schemas import PurchaseDocument

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

SYSTEM_PROMPT = """\
You are an accounts assistant at Nova Trade Hub GmbH, a wholesaler of
jewellery, Indian snacks and textiles. You read purchase orders (Bestellung)
and invoices (Rechnung) in English or German and fill the given JSON schema.

Rules:
1. Extract exactly what is written. Never invent, correct or complete values.
2. The customer is the company BUYING from Nova Trade Hub (the party that
   places the order or is billed). It is never Nova Trade Hub itself.
3. Copy SKUs / article numbers (Art.-Nr.) exactly as printed. If a line has no
   code, use null. Never invent a SKU.
4. Numbers must be numbers, not strings. German decimal commas (1.234,50)
   mean 1234.50. Do not recompute totals; copy the printed values.
5. German vocabulary: Menge = quantity, Einzelpreis = unit price,
   Gesamt / Betrag = line total, Gesamtsumme / Summe = grand total,
   Kunde = customer, Datum = date, Bestellnummer = order number,
   Lieferschein = delivery note (not the invoice number).
6. Dates as ISO YYYY-MM-DD (27.07.2026 -> 2026-07-27). Use null if missing.
"""

RETRY_SUFFIX = """

Your previous answer did not match the schema: {error}
Answer again and follow the schema exactly."""


def _pdf_to_text(data: bytes, name: str) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(f"no extractable text in {name} (scanned PDF? OCR is not supported)")
    logger.info("Loaded PDF %s (%d pages, %d chars)", name, len(pages), len(text))
    return text


def load_document(source: str | Path | bytes, *, filename: str | None = None) -> str:
    """Return the text of a ``.pdf``, ``.txt`` or ``.md`` document.

    Args:
        source: A file path, or the raw bytes of a file (e.g. an upload).
        filename: Required when ``source`` is bytes, to know the file type.

    Raises:
        FileNotFoundError: If a path does not exist.
        ValueError: Unsupported extension, or a PDF with no extractable text.
    """
    if isinstance(source, bytes):
        if not filename:
            raise ValueError("filename is required when loading from bytes")
        suffix = Path(filename).suffix.lower()
        data, name = source, filename
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"document not found: {path}")
        suffix, name = path.suffix.lower(), path.name
        data = path.read_bytes()

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported document type {suffix!r}; use .pdf or .txt")
    if suffix == ".pdf":
        return _pdf_to_text(data, name)
    text = data.decode("utf-8").strip()
    logger.info("Loaded text document %s (%d chars)", name, len(text))
    return text


def extract_document(
    text: str,
    provider: str | None = None,
    *,
    llm: BaseChatModel | None = None,
    max_attempts: int = 2,
) -> PurchaseDocument:
    """Extract a :class:`PurchaseDocument` from ``text``.

    Uses the provider's native structured-output mode (``json_schema``). If the
    answer fails Pydantic validation, the error is appended to the prompt and
    the call is repeated once. Both attempts are logged.

    Args:
        text: Plain text of the document.
        provider: ``"openai"`` / ``"anthropic"``; ``None`` uses ``LLM_PROVIDER``.
        llm: Explicit model (tests inject a fake); overrides ``provider``.
        max_attempts: Total attempts, including the first one.

    Raises:
        ValueError: If no valid structured output was obtained.
    """
    model = llm if llm is not None else get_llm(provider)
    structured = model.with_structured_output(
        PurchaseDocument, method="json_schema", include_raw=True
    )
    system_prompt = SYSTEM_PROMPT
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]
        try:
            result: Any = structured.invoke(messages)
        except Exception as exc:  # network / auth / provider errors
            last_error = exc
            logger.warning("Extraction attempt %d/%d failed: %s", attempt, max_attempts, exc)
            continue

        parsed = result.get("parsed") if isinstance(result, dict) else result
        if isinstance(parsed, PurchaseDocument):
            logger.info(
                "Extraction attempt %d/%d succeeded: %s %s with %d lines",
                attempt,
                max_attempts,
                parsed.doc_type,
                parsed.document_number,
                len(parsed.lines),
            )
            return parsed

        parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
        last_error = parsing_error or ValueError("model returned no parsable output")
        logger.warning(
            "Extraction attempt %d/%d returned invalid output: %s",
            attempt,
            max_attempts,
            last_error,
        )
        system_prompt = SYSTEM_PROMPT + RETRY_SUFFIX.format(error=last_error)

    raise ValueError(f"structured extraction failed after {max_attempts} attempts: {last_error}")
