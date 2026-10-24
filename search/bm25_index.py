"""
BM25 Keyword Search Index for Exact Value Retrieval.
Preserves alphanumeric identifiers, model numbers, and part codes (e.g. JA-207652, 210-BLYZ).
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from ingestion.chunker import Chunk

logger = logging.getLogger("rfp_bm25")


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenizes text while retaining bid identifiers, part numbers, and alphanumeric codes.
    Example: 'SI# CC7802 Dell Latitude 5550 210-BLYZ' -> ['si', 'cc7802', 'dell', 'latitude', '5550', '210-blyz']
    """
    cleaned = re.sub(r'[\r\n\t]+', ' ', text)
    # Match alphanumeric sequences including dashes, underscores, and slashes
    tokens = re.findall(r'[a-zA-Z0-9]+(?:[-_/.][a-zA-Z0-9]+)*', cleaned)
    return [t.lower() for t in tokens if len(t) > 1]


class BM25Index:
    def __init__(self, store_dir: str = "index_store"):
        self.store_dir = store_dir
        self.data_file = os.path.join(store_dir, "bm25_data.json")
        self.chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load()

    def _load(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", [])
                    self.tokenized_corpus = data.get("tokenized_corpus", [])
                    if self.tokenized_corpus:
                        self.bm25 = BM25Okapi(self.tokenized_corpus)
                logger.info(f"Loaded existing BM25 index with {len(self.chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")
                self.chunks = []
                self.tokenized_corpus = []
                self.bm25 = None

    def save(self):
        os.makedirs(self.store_dir, exist_ok=True)
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump({
                    "chunks": self.chunks,
                    "tokenized_corpus": self.tokenized_corpus,
                }, f, indent=2)
            logger.info(f"Saved BM25 index to '{self.store_dir}'")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")

    def index_bid(self, bid_id: str, new_chunks: List[Chunk], force_reindex: bool = False):
        """Incrementally adds chunks to BM25 index."""
        existing_indices = [i for i, c in enumerate(self.chunks) if c["bid_id"] == bid_id]

        if existing_indices and not force_reindex:
            logger.info(f"Bid '{bid_id}' already has {len(existing_indices)} BM25 entries. Skipping.")
            return

        if existing_indices and force_reindex:
            keep_indices = [i for i, c in enumerate(self.chunks) if c["bid_id"] != bid_id]
            self.chunks = [self.chunks[i] for i in keep_indices]
            self.tokenized_corpus = [self.tokenized_corpus[i] for i in keep_indices]

        for c in new_chunks:
            c_dict = c.to_dict()
            tokens = tokenize_for_bm25(c.text)
            self.chunks.append(c_dict)
            self.tokenized_corpus.append(tokens)

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self.save()
            logger.info(f"Updated BM25 index with bid '{bid_id}'. Total chunks: {len(self.chunks)}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Performs BM25 search with metadata filtering."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)

        # Apply metadata filters and gather candidates
        scored_candidates = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, raw_scores)):
            if score <= 0:
                continue

            match = True
            if filters:
                if "bid_id" in filters and filters["bid_id"] and chunk["bid_id"] != filters["bid_id"]:
                    match = False
                if "doc_type" in filters and filters["doc_type"] and chunk["doc_type"] != filters["doc_type"]:
                    match = False
                if "addendum_number" in filters and filters["addendum_number"] is not None:
                    if chunk.get("addendum_number") != filters["addendum_number"]:
                        match = False
            if match:
                scored_candidates.append((score, chunk))

        # Sort descending by BM25 score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_matches = scored_candidates[:top_k]

        results = []
        for score, c in top_matches:
            results.append({
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "score": float(score),
                "bid_id": c["bid_id"],
                "file_name": c["file_name"],
                "doc_type": c["doc_type"],
                "page_number": c["page_number"],
                "addendum_number": c.get("addendum_number"),
                "document_date": c.get("document_date"),
                "is_table": c.get("is_table", False),
                "citation": {
                    "file": c["file_name"],
                    "page": c["page_number"],
                    "doc_type": c["doc_type"],
                    "addendum_number": c.get("addendum_number"),
                },
            })

        return results
