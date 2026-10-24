"""
Unit Tests for Parsing (PDF & HTML)
=====================================
Validates PyMuPDF extraction, BeautifulSoup HTML parsing, table detection,
metadata tagging, and error handling.
"""

import os
import pytest
from ingestion.parser import (
    Document,
    parse_pdf,
    parse_html,
    ingest_bid_folder,
    classify_document,
    clean_text,
)


def test_classify_document():
    doc_type, add_num = classify_document("Addendum 2 RFP JA-207652.pdf")
    assert doc_type == "addendum"
    assert add_num == 2

    doc_type, add_num = classify_document("Mercury_Affidavit.pdf")
    assert doc_type == "affidavit"
    assert add_num is None

    doc_type, add_num = classify_document("Dell_Laptop_Specs.pdf")
    assert doc_type == "specs"

    doc_type, add_num = classify_document("Bid Information - BidNet Direct.html")
    assert doc_type == "bid_page"


def test_clean_text():
    dirty = "Dallas ISD rev 05.2024 Page 1 of 50\nThis is the solicit-\nation text.    "
    cleaned = clean_text(dirty)
    assert "solicitation" in cleaned
    assert "Page 1 of 50" not in cleaned


def test_parse_html_file():
    html_path = os.path.join(
        "Bid1",
        "Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html",
    )
    if os.path.exists(html_path):
        docs = parse_html(html_path, bid_id="Bid1")
        assert len(docs) == 1
        doc = docs[0]
        assert isinstance(doc, Document)
        assert doc.bid_id == "Bid1"
        assert doc.doc_type == "bid_page"
        assert "Dallas" in doc.text or "168884" in doc.text


def test_parse_pdf_table_extraction():
    pdf_path = os.path.join("Bid2", "PORFP_-_Dell_Laptop_Final.pdf")
    if os.path.exists(pdf_path):
        docs = parse_pdf(pdf_path, bid_id="Bid2")
        assert len(docs) >= 3
        # Check if tables were extracted
        all_tables = [p.tables for p in docs if p.tables]
        assert len(all_tables) > 0
        combined_text = " ".join(p.text for p in docs)
        assert "Dell" in combined_text


def test_ingest_nonexistent_folder():
    with pytest.raises(ValueError):
        ingest_bid_folder("non_existent_folder_xyz_123")
