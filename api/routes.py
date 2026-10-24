"""
FastAPI REST Interface for the RFP Intelligence Platform.
Exposes endpoints for indexing, hybrid search, question answering, and multi-agent extraction.
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from search.hybrid_search import HybridSearchEngine
from agents.orchestrator import OrchestratorAgent
from agents.qa_agent import QAReportAgent

app = FastAPI(
    title="RFP Intelligence Platform API",
    description="RAG Search Engine and Multi-Agent RFP Analysis System",
    version="1.0.0",
)

# Global engines
search_engine = HybridSearchEngine()
orchestrator = OrchestratorAgent(search_engine=search_engine)
qa_agent = QAReportAgent(search_engine=search_engine)


class IndexRequest(BaseModel):
    bid_id: str = Field(..., description="Unique bid identifier (e.g. Bid1, Bid2)")
    folder_path: str = Field(..., description="Path to bid directory containing HTML/PDF files")
    force_reindex: bool = Field(False, description="Whether to re-parse and overwrite existing index")


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question about RFPs")
    bid_id: Optional[str] = Field(None, description="Optional bid_id filter")
    top_k: int = Field(5, description="Number of evidence chunks to retrieve")


class ExtractRequest(BaseModel):
    bid_id: str = Field(..., description="Bid ID to extract structured records for")
    folder_path: Optional[str] = Field(None, description="Optional path to bid folder if not already indexed")


@app.get("/")
def root():
    return {
        "platform": "RFP Intelligence Platform",
        "status": "online",
        "indexed_bids": search_engine.vector_store.get_indexed_bids(),
        "endpoints": [
            "POST /index",
            "GET /search?q=...&bid_id=...&top_k=5",
            "POST /ask",
            "POST /extract",
            "GET /bids",
            "GET /compare",
        ],
    }


@app.get("/bids")
def get_indexed_bids():
    """Lists all currently indexed bids."""
    return {"indexed_bids": search_engine.vector_store.get_indexed_bids()}


@app.post("/index")
def index_bid_endpoint(req: IndexRequest):
    """Indexes a bid folder incrementally into the hybrid search engine."""
    if not os.path.isdir(req.folder_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {req.folder_path}")
    try:
        search_engine.index_bid_folder(req.folder_path, req.bid_id, force_reindex=req.force_reindex)
        return {
            "status": "success",
            "bid_id": req.bid_id,
            "message": f"Successfully indexed '{req.folder_path}'",
            "total_indexed_bids": search_engine.vector_store.get_indexed_bids(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
def search_endpoint(
    q: str = Query(..., description="Search query string"),
    bid_id: Optional[str] = Query(None, description="Filter by bid_id"),
    doc_type: Optional[str] = Query(None, description="Filter by doc_type"),
    top_k: int = Query(5, description="Number of top cited passages to return"),
    mode: str = Query("hybrid_rerank", description="Search mode: vector_only, bm25_only, hybrid, hybrid_rerank"),
):
    """Executes search and returns cited passages with provenance."""
    try:
        results = search_engine.search(
            query=q,
            bid_id=bid_id,
            doc_type=doc_type,
            top_k=top_k,
            mode=mode,
        )
        return {
            "query": q,
            "mode": mode,
            "results_count": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    """Answers a natural language question grounded in RFP evidence passages."""
    try:
        ans = qa_agent.answer_question(
            question=req.question,
            bid_id=req.bid_id,
            top_k=req.top_k,
        )
        return ans
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
def extract_endpoint(req: ExtractRequest):
    """Triggers the full multi-agent extraction pipeline for an RFP."""
    try:
        record = orchestrator.extract_bid(req.bid_id, req.folder_path)
        return record.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare")
def compare_bids_endpoint(bid1: str = "Bid1", bid2: str = "Bid2"):
    """Returns side-by-side comparison of structured records for multiple bids."""
    out_file1 = os.path.join("output", f"{bid1.lower()}_extracted.json")
    out_file2 = os.path.join("output", f"{bid2.lower()}_extracted.json")

    import json
    data1 = {}
    data2 = {}
    if os.path.exists(out_file1):
        with open(out_file1, "r", encoding="utf-8") as f:
            data1 = json.load(f)
    if os.path.exists(out_file2):
        with open(out_file2, "r", encoding="utf-8") as f:
            data2 = json.load(f)

    comparison = {}
    all_fields = sorted(list(set(list(data1.get("fields", {}).keys()) + list(data2.get("fields", {}).keys()))))
    for f in all_fields:
        comparison[f] = {
            bid1: data1.get("fields", {}).get(f, {}).get("value"),
            bid2: data2.get("fields", {}).get(f, {}).get("value"),
        }

    return {
        "bids": [bid1, bid2],
        "field_comparison": comparison,
    }
