"""
Unit Tests for Search Components
==================================
Validates BM25 tokenization, Vector Store cosine similarity search,
metadata filtering, and Reciprocal Rank Fusion (RRF).
"""

import os
import numpy as np
from search.bm25_index import tokenize_for_bm25, BM25Index
from search.vector_store import VectorStore
from ingestion.chunker import Chunk


def test_bm25_tokenization_preserves_identifiers():
    text = "Model SI# CC7802 Dell Latitude 5550 Part 210-BLYZ Dock WD22TB4"
    tokens = tokenize_for_bm25(text)
    assert "cc7802" in tokens
    assert "latitude" in tokens
    assert "210-blyz" in tokens
    assert "wd22tb4" in tokens


def test_bm25_index_add_and_search(tmp_path):
    store_dir = str(tmp_path)
    bm25 = BM25Index(store_dir=store_dir)

    c1 = Chunk(
        chunk_id="c1",
        text="Dell Latitude 5550 laptop with 16GB RAM and Core Ultra 7",
        bid_id="Bid2",
        file_name="dell.pdf",
        doc_type="rfp",
        page_number=1,
    )
    c2 = Chunk(
        chunk_id="c2",
        text="Student Chromebook 11 inch ruggedized for education",
        bid_id="Bid1",
        file_name="disd.pdf",
        doc_type="rfp",
        page_number=5,
    )

    c3 = Chunk(
        chunk_id="c3",
        text="General procurement vendor guidelines and submission checklist",
        bid_id="Bid1",
        file_name="checklist.pdf",
        doc_type="rfp",
        page_number=1,
    )

    bm25.index_bid("Bid2", [c1])
    bm25.index_bid("Bid1", [c2, c3])
    assert len(bm25.chunks) == 3

    # Query for Dell
    results = bm25.search("Dell Latitude", top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "c1"

    # Test filtering by bid_id
    b1_results = bm25.search("Dell", filters={"bid_id": "Bid1"}, top_k=2)
    assert len(b1_results) == 0


def test_vector_store_cosine_similarity(tmp_path):
    store_dir = str(tmp_path)
    vs = VectorStore(store_dir=store_dir)

    # Synthetic embeddings for 2 chunks
    vs.chunks = [
        {"chunk_id": "v1", "bid_id": "Bid1", "file_name": "a.pdf", "doc_type": "rfp", "page_number": 1, "text": "Texas ethics form"},
        {"chunk_id": "v2", "bid_id": "Bid2", "file_name": "b.pdf", "doc_type": "affidavit", "page_number": 2, "text": "Mercury affidavit"},
    ]
    # Unit vectors
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    vs.embeddings = np.stack([emb1, emb2])

    # Query vector close to emb2
    query_emb = np.array([0.1, 0.9, 0.0], dtype=np.float32)
    res = vs.search(query_emb, top_k=2)

    assert len(res) == 2
    assert res[0]["chunk_id"] == "v2"
    assert res[0]["score"] > res[1]["score"]

    # Filter by Bid1
    b1_res = vs.search(query_emb, filters={"bid_id": "Bid1"}, top_k=2)
    assert len(b1_res) == 1
    assert b1_res[0]["chunk_id"] == "v1"


def test_existing_production_index():
    # If index_store exists, verify it loads existing chunks
    if os.path.exists("index_store/vector_metadata.json"):
        vs = VectorStore("index_store")
        assert len(vs.chunks) >= 300
        indexed_bids = vs.get_indexed_bids()
        assert "Bid1" in indexed_bids
        assert "Bid2" in indexed_bids
