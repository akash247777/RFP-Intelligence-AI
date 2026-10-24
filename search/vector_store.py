"""
Persistent Vector Store supporting Incremental Indexing, Metadata Filtering,
and High-Precision Cosine Similarity Search.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from ingestion.chunker import Chunk
from search.embeddings import EmbeddingEngine

logger = logging.getLogger("rfp_vector_store")


class VectorStore:
    def __init__(self, store_dir: str = "index_store"):
        self.store_dir = store_dir
        self.meta_file = os.path.join(store_dir, "vector_metadata.json")
        self.vec_file = os.path.join(store_dir, "vector_embeddings.npy")
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        if os.path.exists(self.meta_file) and os.path.exists(self.vec_file):
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.embeddings = np.load(self.vec_file)
                logger.info(f"Loaded existing vector index: {len(self.chunks)} chunks, shape {self.embeddings.shape}")
            except Exception as e:
                logger.warning(f"Failed to load vector store from disk: {e}")
                self.chunks = []
                self.embeddings = None

    def save(self):
        os.makedirs(self.store_dir, exist_ok=True)
        try:
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, indent=2)
            if self.embeddings is not None:
                np.save(self.vec_file, self.embeddings)
            logger.info(f"Saved vector index to '{self.store_dir}' ({len(self.chunks)} chunks)")
        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")

    def get_indexed_bids(self) -> List[str]:
        return sorted(list(set(c["bid_id"] for c in self.chunks)))

    def index_bid(self, bid_id: str, new_chunks: List[Chunk], embedder: EmbeddingEngine, force_reindex: bool = False):
        """
        Incrementally indexes a bid folder.
        If already indexed and not force_reindex, skips or updates changed chunks.
        """
        existing_indices = [i for i, c in enumerate(self.chunks) if c["bid_id"] == bid_id]

        if existing_indices and not force_reindex:
            logger.info(f"Bid '{bid_id}' already has {len(existing_indices)} indexed chunks. Skipping incremental duplicate.")
            return

        if existing_indices and force_reindex:
            logger.info(f"Force re-indexing bid '{bid_id}'. Removing {len(existing_indices)} existing chunks.")
            keep_indices = [i for i, c in enumerate(self.chunks) if c["bid_id"] != bid_id]
            self.chunks = [self.chunks[i] for i in keep_indices]
            if self.embeddings is not None and len(keep_indices) > 0:
                self.embeddings = self.embeddings[keep_indices]
            else:
                self.embeddings = None

        if not new_chunks:
            return

        chunk_texts = [c.text for c in new_chunks]
        new_embeds = embedder.embed_documents(chunk_texts)
        chunk_dicts = [c.to_dict() for c in new_chunks]

        if self.embeddings is None or len(self.chunks) == 0:
            self.embeddings = new_embeds
            self.chunks = chunk_dicts
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeds])
            self.chunks.extend(chunk_dicts)

        self.save()
        logger.info(f"Successfully indexed bid '{bid_id}' with {len(new_chunks)} chunks. Total index size: {len(self.chunks)}")

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs dense vector similarity search with metadata filtering.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Apply metadata filters to obtain candidate mask
        candidate_indices = []
        for idx, chunk in enumerate(self.chunks):
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
                candidate_indices.append(idx)

        if not candidate_indices:
            return []

        cand_embeds = self.embeddings[candidate_indices]
        # Cosine similarity
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        embed_norms = np.linalg.norm(cand_embeds, axis=1)
        embed_norms[embed_norms == 0] = 1e-10

        scores = np.dot(cand_embeds, query_vec) / (embed_norms * query_norm)

        # Get top-k
        sorted_pos = np.argsort(scores)[::-1][:top_k]
        results = []
        for p in sorted_pos:
            actual_idx = candidate_indices[p]
            c = self.chunks[actual_idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "score": float(scores[p]),
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
