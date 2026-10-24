"""
Embeddings Generator with Caching and Retry Logic.
Uses Google Gemini Embedding model (models/gemini-embedding-001) with local disk caching.
"""

import os
import json
import time
import hashlib
import logging
import threading
from typing import List, Optional
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger("rfp_embeddings")


class EmbeddingEngine:
    """Generates dense vector embeddings using Google Gemini API with local disk caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/gemini-embedding-001",
        cache_path: str = "index_store/embeddings_cache.json",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")

        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")
        return {}

    def _save_cache(self):
        with self._lock:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            try:
                # Copy cache dict snapshot to prevent concurrent modification
                cache_copy = dict(self._cache)
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_copy, f)
            except Exception as e:
                logger.warning(f"Failed to save embedding cache: {e}")

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single query or text string."""
        h = self._hash_text(text)
        if h in self._cache:
            return np.array(self._cache[h], dtype=np.float32)

        for attempt in range(4):
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_query",
                )
                vec = result["embedding"]
                self._cache[h] = vec
                self._save_cache()
                return np.array(vec, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"Failed to embed text after retries: {text[:80]}...")

    def embed_documents(self, texts: List[str], batch_size: int = 40) -> np.ndarray:
        """Embeds a list of document chunks with caching and batching."""
        embeddings: List[List[float]] = [None] * len(texts)  # type: ignore
        to_fetch_indices = []
        to_fetch_texts = []

        for idx, text in enumerate(texts):
            h = self._hash_text(text)
            if h in self._cache:
                embeddings[idx] = self._cache[h]
            else:
                to_fetch_indices.append(idx)
                to_fetch_texts.append(text)

        if to_fetch_texts:
            logger.info(f"Generating embeddings for {len(to_fetch_texts)} new chunks (cached: {len(texts) - len(to_fetch_texts)})...")
            for i in range(0, len(to_fetch_texts), batch_size):
                batch = to_fetch_texts[i : i + batch_size]
                batch_indices = to_fetch_indices[i : i + batch_size]

                for attempt in range(4):
                    try:
                        res = genai.embed_content(
                            model=self.model_name,
                            content=batch,
                            task_type="retrieval_document",
                        )
                        batch_embeds = res["embedding"]
                        for b_idx, orig_idx in enumerate(batch_indices):
                            vec = batch_embeds[b_idx]
                            embeddings[orig_idx] = vec
                            self._cache[self._hash_text(texts[orig_idx])] = vec
                        break
                    except Exception as e:
                        logger.warning(f"Batch embedding attempt {attempt+1} failed: {e}. Retrying...")
                        time.sleep(2.0 * (attempt + 1))
                time.sleep(0.4)  # respect rate limits

            self._save_cache()

        return np.array(embeddings, dtype=np.float32)
