#!/usr/bin/env python
"""Generate PDF versions of three sample documents.

    python scripts/make_pdfs.py

Reads ``po_001.txt``, ``inv_001.txt`` and ``err_wrong_total.txt`` from
``data/sample_docs`` and writes a ``.pdf`` next to each, so the PDF text is
guaranteed to match the text version. The PDFs are committed, so this only
needs re-running when the text documents change.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_docs"
SOURCES = ["po_001.txt", "inv_001.txt", "err_wrong_total.txt"]


def write_pdf(text: str, target: Path) -> None:
    """Render ``text`` line by line in a monospace font onto an A4 page."""
    pdf = canvas.Canvas(str(target), pagesize=A4)
    _, height = A4
    pdf.setFont("Courier", 10)
    y = height - 60
    for line in text.splitlines():
        if y < 50:  # new page if a document ever grows long
            pdf.showPage()
            pdf.setFont("Courier", 10)
            y = height - 60
        pdf.drawString(50, y, line)
        y -= 14
    pdf.save()


def main() -> None:
    for name in SOURCES:
        source = DOCS_DIR / name
        target = source.with_suffix(".pdf")
        write_pdf(source.read_text(encoding="utf-8"), target)
        print(f"wrote {target.relative_to(DOCS_DIR.parents[1])}")  # noqa: T201


if __name__ == "__main__":
    main()
