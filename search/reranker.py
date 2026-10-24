"""
Query Understanding, Expansion, and LLM-based Re-ranker.
Enhances retrieval recall via synonym expansion and precision via LLM cross-scoring.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger("rfp_reranker")


class QueryExpander:
    """
    Understands and rewrites RFP queries:
    1. Expands terms into solicitation domain equivalents (e.g. deadline -> closing date).
    2. Detects explicit bid filters (Bid1, Bid2, addendum numbers, affidavits).
    """

    EXPANSION_RULES = {
        r'\b(?:deadline|due date|closing)\b': "due date submission deadline closing date proposal due solicitation due",
        r'\b(?:bond|bid bond|guarantee)\b': "bid bond requirement bid security surety bond 5% guarantee",
        r'\b(?:meeting|pre bid|conference)\b': "pre-bid meeting pre-proposal conference vendor meeting TEAMS conference",
        r'\b(?:warranty|guarantee)\b': "extended warranty 3 years limited hardware warranty repair replacement",
        r'\b(?:affidavit|affidavits|certificate)\b': "contract affidavit mercury affidavit required forms certifications",
        r'\b(?:payment|terms)\b': "payment terms Net 30 invoicing instructions 10 days",
        r'\b(?:spec|specs|specifications|hardware)\b': "processor CPU RAM storage SSD display screen memory Copilot",
        r'\b(?:model|part number|sku)\b': "model number part number SKU SI# CC7802 WD22TB4 210-BLYZ",
        r'\b(?:cooperative|contract vehicle|master contract)\b': "master contract 060B5400007 state contract cooperative vehicle",
        r'\b(?:reseller|mfg|registration|authorized)\b': "manufacturer registration authorized reseller Dell Letter of Authorization LOA",
        r'\b(?:delivery|shipment|lead time)\b': "delivery date delivery window 45 days after award lead time",
        r'\b(?:installation|imaging|deployment)\b': "installation imaging deployment setup staging services",
        r'\b(?:addendum|change|update)\b': "addendum amendment modification extension revision new due date",
    }

    def expand_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (expanded_query_string, extracted_filters).
        """
        extracted_filters: Dict[str, Any] = {}
        lower_q = query.lower()

        # Bid ID extraction
        if "bid1" in lower_q or "bid 1" in lower_q or "dallas" in lower_q:
            extracted_filters["bid_id"] = "Bid1"
        elif "bid2" in lower_q or "bid 2" in lower_q or "maryland" in lower_q or "dell laptop" in lower_q:
            extracted_filters["bid_id"] = "Bid2"

        # Doc type / Addendum extraction
        add_match = re.search(r'addendum\s*(?:no\.?|#)?\s*(\d+)', lower_q)
        if add_match:
            extracted_filters["doc_type"] = "addendum"
            extracted_filters["addendum_number"] = int(add_match.group(1))
        elif "addendum" in lower_q or "addenda" in lower_q:
            extracted_filters["doc_type"] = "addendum"

        # Add synonyms
        expanded_terms = [query]
        for pattern, replacement in self.EXPANSION_RULES.items():
            if re.search(pattern, lower_q):
                expanded_terms.append(replacement)

        full_expanded = " ".join(expanded_terms)
        return full_expanded, extracted_filters


class LLMReranker:
    """
    Cross-evaluates top retrieval candidates using Gemini 2.5 Flash
    for exact passage-level relevance scoring and de-duplication.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "models/gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model_name = model_name

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-ranks candidates based on direct relevance to query.
        """
        if not candidates or len(candidates) <= 1 or not self.api_key:
            return candidates[:top_k]

        # Format candidates for LLM scoring prompt
        items_prompt = []
        for i, c in enumerate(candidates[:12]):
            items_prompt.append(
                f"[Candidate {i}]\nFile: {c['file_name']} (Page {c['page_number']})\nSnippet: {c['text'][:400]}\n"
            )

        prompt = f"""You are an expert RFP retrieval re-ranker.
Given the User Query and candidate text passages, score each candidate's relevance from 0 to 10.
High score (8-10): Passage directly answers or contains key facts for the query.
Medium score (4-7): Passage contains related context.
Low score (0-3): Passage is irrelevant boilerplate or generic text.

User Query: "{query}"

Candidates:
{"".join(items_prompt)}

Respond ONLY with a JSON array of objects with keys "candidate_index" and "score" (0-10), sorted descending by score:
[
  {{"candidate_index": 0, "score": 9.5}},
  ...
]"""

        try:
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={"response_mime_type": "application/json", "temperature": 0.0},
            )
            resp = model.generate_content(prompt)
            scores_data = json.loads(resp.text)

            score_map = {}
            for item in scores_data:
                idx = item.get("candidate_index")
                scr = float(item.get("score", 0))
                if idx is not None and 0 <= idx < len(candidates):
                    score_map[idx] = scr

            # Attach rerank score and sort
            reranked = []
            for i, c in enumerate(candidates[:12]):
                c_copy = dict(c)
                c_copy["rerank_score"] = score_map.get(i, c.get("score", 0.0))
                reranked.append(c_copy)

            reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return reranked[:top_k]

        except Exception as e:
            logger.warning(f"LLM reranking encountered error: {e}. Falling back to original hybrid ranking.")
            return candidates[:top_k]
