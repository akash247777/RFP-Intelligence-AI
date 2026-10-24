"""
Q&A and Report Agent.
Synthesizes free-form natural language questions across one or multiple bids,
producing rigorous cited responses and human-readable comparison reports.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

from agents.state import SourceCitation, AgentTraceStep
from search.hybrid_search import HybridSearchEngine

load_dotenv()
logger = logging.getLogger("rfp_qa_agent")


class QAReportAgent:
    def __init__(self, search_engine: HybridSearchEngine, model_name: str = "models/gemini-2.5-flash"):
        self.name = "Q&A / Report Agent"
        self.search_engine = search_engine
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def answer_question(
        self,
        question: str,
        bid_id: Optional[str] = None,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        """
        Answers a user question grounded strictly in retrieved passages with exact citations.
        """
        t0 = time.time()
        logger.info(f"QAReportAgent answering: '{question}' (bid_id filter: {bid_id})")

        # Retrieve relevant passages using Hybrid Search + Re-ranker
        passages = self.search_engine.search(
            query=question,
            bid_id=bid_id,
            top_k=top_k,
            mode="hybrid_rerank",
        )

        if not passages:
            return {
                "question": question,
                "answer": "Not found in documents. No matching evidence passages could be retrieved.",
                "citations": [],
                "confidence": 0.0,
                "latency_ms": (time.time() - t0) * 1000,
            }

        context_blocks = []
        citations_list = []
        for idx, p in enumerate(passages):
            cit = {
                "file": p["file_name"],
                "page": p["page_number"],
                "doc_type": p["doc_type"],
                "addendum_number": p.get("addendum_number"),
            }
            citations_list.append(cit)
            context_blocks.append(
                f"[Evidence {idx+1}] File: {p['file_name']}, Page: {p['page_number']}, DocType: {p['doc_type']}\n"
                f"{p['text']}\n"
            )

        prompt = f"""You are an AI RFP Intelligence Specialist.
Answer the following user question using ONLY the provided Evidence Passages.

CRITICAL INSTRUCTIONS:
1. GROUNDING: If the answer cannot be determined from the passages, state clearly: "Not found in documents." Do not invent details.
2. CITATIONS: In your answer, explicitly reference the source file and page for every factual assertion (e.g. "[Addendum 2 RFP JA-207652..., Page 1]").
3. ADDENDUMS: If an addendum modifies or overrides an earlier base document, state the latest reconciled value and mention the addendum change.
4. ACCURACY: Provide specific numbers, dates, times, part codes, and dollar/percentage amounts where mentioned.

User Question: "{question}"

Evidence Passages:
{"".join(context_blocks)}

Respond with a JSON object:
{{
  "answer": "Comprehensive, factual answer with in-text citation mentions",
  "confidence": 0.95,
  "key_findings": ["Bullet point 1", "Bullet point 2"],
  "sources_used": [
    {{"file": "source.pdf", "page": 1, "note": "brief note on why this source was used"}}
  ]
}}"""

        try:
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={"response_mime_type": "application/json", "temperature": 0.1},
            )
            resp = model.generate_content(prompt)
            data = json.loads(resp.text)
            latency = (time.time() - t0) * 1000

            return {
                "question": question,
                "answer": data.get("answer", "No answer generated."),
                "key_findings": data.get("key_findings", []),
                "citations": data.get("sources_used", citations_list),
                "retrieved_passages": passages,
                "confidence": float(data.get("confidence", 0.9)),
                "latency_ms": latency,
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": f"Error during synthesis: {str(e)}",
                "citations": citations_list,
                "retrieved_passages": passages,
                "confidence": 0.4,
                "latency_ms": (time.time() - t0) * 1000,
            }
