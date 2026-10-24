"""
Unit Tests for Chunker
========================
Validates section header detection, table preservation, chunk overlap,
and breadcrumb header enrichment.
"""

from ingestion.parser import Document
from ingestion.chunker import RFPChunker, Chunk


def test_chunker_initialization():
    chunker = RFPChunker(target_chunk_size=500, chunk_overlap=100)
    assert chunker.target_chunk_size == 500
    assert chunker.chunk_overlap == 100


def test_section_header_detection():
    chunker = RFPChunker()
    text = "Section 2 - Scope and Specifications\nThe district invites proposals for devices."
    header = chunker.detect_section_header(text)
    assert header is not None
    assert "Section 2" in header


def test_chunk_document_basic():
    chunker = RFPChunker(target_chunk_size=200, chunk_overlap=50)
    doc = Document(
        bid_id="TestBid",
        file_name="test_rfp.pdf",
        doc_type="rfp",
        page_number=1,
        text="Section 1: General Info\n" + ("This is a repetitive line of text to test chunking splitting logic. " * 20),
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].bid_id == "TestBid"
    assert chunks[0].page_number == 1
    # Check that breadcrumbs were prepended
    assert "[TestBid | test_rfp.pdf | Page 1" in chunks[0].text


def test_table_preservation():
    chunker = RFPChunker()
    doc = Document(
        bid_id="Bid2",
        file_name="table_doc.pdf",
        doc_type="rfp",
        page_number=3,
        text="Introduction to pricing.\n### Extracted Tables:\n| Item | Qty | Part # |\n| Laptop | 25 | 210-BLYZ |\n| Dock | 25 | WD22TB4 |",
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    # Verify table marker preserved
    table_chunk = next((c for c in chunks if c.is_table or "Extracted Tables" in c.text), None)
    assert table_chunk is not None
    assert "WD22TB4" in table_chunk.text
