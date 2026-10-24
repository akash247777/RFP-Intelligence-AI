"""
Hybrid Search Engine combining Dense Vector Retrieval with BM25 Keyword Matching
via Reciprocal Rank Fusion (RRF) and optional LLM-based Re-ranking.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from ingestion.parser import ingest_bid_folder
from ingestion.chunker import RFPChunker
from search.embeddings import EmbeddingEngine
from search.vector_store import VectorStore
from search.bm25_index import BM25Index
from search.reranker import QueryExpander, LLMReranker

logger = logging.getLogger("rfp_search_engine")


class HybridSearchEngine:
    """
    RAG Search Engine for RFPs:
    1. Query Expansion & Filter Parsing.
    2. Dense Vector Semantic Retrieval.
    3. BM25 Keyword Search (critical for IDs, part numbers, specs).
    4. Reciprocal Rank Fusion (RRF) with configurable weights.
    5. Candidate Re-ranking with LLM / Cross-Encoder.
    6. Complete Citations (source file, page, chunk text).
    """

    def __init__(self, store_dir: str = "index_store", api_key: Optional[str] = None):
        self.store_dir = store_dir
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.embedder = EmbeddingEngine(api_key=self.api_key)
        self.vector_store = VectorStore(store_dir=store_dir)
        self.bm25_index = BM25Index(store_dir=store_dir)
        self.expander = QueryExpander()
        self.reranker = LLMReranker(api_key=self.api_key)
        self.chunker = RFPChunker()

    def index_bid_folder(self, folder_path: str, bid_id: Optional[str] = None, force_reindex: bool = False):
        """Indexes all HTML and PDF files in a bid folder incrementally."""
        target_bid_id = bid_id or os.path.basename(os.path.normpath(folder_path))
        logger.info(f"Indexing bid folder '{folder_path}' (bid_id: {target_bid_id})...")

        docs = ingest_bid_folder(folder_path, target_bid_id)
        chunks = self.chunker.chunk_documents(docs)

        # Index into vector store
        self.vector_store.index_bid(target_bid_id, chunks, self.embedder, force_reindex=force_reindex)
        # Index into BM25
        self.bm25_index.index_bid(target_bid_id, chunks, force_reindex=force_reindex)
        logger.info(f"Completed indexing for '{target_bid_id}' ({len(chunks)} chunks).")

    def search(
        self,
        query: str,
        bid_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        top_k: int = 5,
        mode: str = "hybrid_rerank",  # vector_only, bm25_only, hybrid, hybrid_rerank
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """
        Executes search using requested retrieval mode.
        Returns cited passages with full provenance metadata.
        """
        expanded_query, auto_filters = self.expander.expand_query(query)

        # Merge explicit filters with auto-detected filters
        filters: Dict[str, Any] = {}
        if bid_id:
            filters["bid_id"] = bid_id
        elif "bid_id" in auto_filters:
            filters["bid_id"] = auto_filters["bid_id"]

        if doc_type:
            filters["doc_type"] = doc_type
        elif "doc_type" in auto_filters:
            filters["doc_type"] = auto_filters["doc_type"]

        if "addendum_number" in auto_filters:
            filters["addendum_number"] = auto_filters["addendum_number"]

        # 1. Mode: Vector Only
        if mode == "vector_only":
            q_vec = self.embedder.embed_text(query)
            return self.vector_store.search(q_vec, top_k=top_k, filters=filters)

        # 2. Mode: BM25 Only
        if mode == "bm25_only":
            return self.bm25_index.search(query, top_k=top_k, filters=filters)

        # 3. Hybrid Modes (hybrid or hybrid_rerank)
        # Dense search with expanded query
        q_vec = self.embedder.embed_text(expanded_query)
        dense_results = self.vector_store.search(q_vec, top_k=top_k * 3, filters=filters)

        # Keyword search with expanded query + exact query
        bm25_results = self.bm25_index.search(f"{query} {expanded_query}", top_k=top_k * 3, filters=filters)

        # Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[str, float] = {}
        candidate_lookup: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            candidate_lookup[cid] = item
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (dense_weight / (rrf_k + rank + 1))
            item["dense_rank"] = rank + 1
            item["dense_score"] = item["score"]

        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            if cid not in candidate_lookup:
                candidate_lookup[cid] = item
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (bm25_weight / (rrf_k + rank + 1))
            candidate_lookup[cid]["bm25_rank"] = rank + 1
            candidate_lookup[cid]["bm25_score"] = item["score"]

        # Sort by RRF score
        sorted_cids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        fused_candidates = []
        for cid in sorted_cids:
            cand = candidate_lookup[cid]
            cand["rrf_score"] = float(fused_scores[cid])
            fused_candidates.append(cand)

        # 4. Re-ranking
        if mode == "hybrid_rerank" and len(fused_candidates) > 1:
            return self.reranker.rerank(query, fused_candidates[:top_k * 2], top_k=top_k)

        return fused_candidates[:top_k]
