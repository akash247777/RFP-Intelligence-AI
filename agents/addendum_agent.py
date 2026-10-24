"""
Addendum Reconciliation Agent.
Compares base RFP specifications and deadlines with all addendums,
detects overrides/extensions, updates fields with citations, and generates a change log.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

from agents.state import ExtractedField, SourceCitation, AddendumChange, AgentTraceStep
from search.hybrid_search import HybridSearchEngine

load_dotenv()
logger = logging.getLogger("rfp_addendum_agent")


class AddendumReconciliationAgent:
    def __init__(self, search_engine: HybridSearchEngine, model_name: str = "models/gemini-2.5-flash"):
        self.name = "Addendum Reconciliation Agent"
        self.search_engine = search_engine
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def reconcile(
        self,
        bid_id: str,
        current_fields: Dict[str, ExtractedField],
    ) -> Tuple[Dict[str, ExtractedField], List[AddendumChange], AgentTraceStep]:
        t0 = time.time()
        logger.info(f"AddendumReconciliationAgent checking addendums for '{bid_id}'...")

        # Search specifically for addendum documents
        addendum_chunks = self.search_engine.search(
            query=f"{bid_id} addendum purpose new due date questions amendment",
            bid_id=bid_id,
            doc_type="addendum",
            top_k=10,
            mode="hybrid",
        )

        if not addendum_chunks:
            logger.info(f"No addendum documents found for bid '{bid_id}'. Preserving base fields.")
            trace = AgentTraceStep(
                step_id=4,
                timestamp=datetime.now().isoformat(),
                agent_name=self.name,
                action="reconcile_addendums",
                inputs={"bid_id": bid_id, "addendum_count": 0},
                outputs={"reconciled_changes": 0, "message": "No addendums present"},
                latency_ms=(time.time() - t0) * 1000,
            )
            return current_fields, [], trace

        # Format addendum context
        addendum_text_blocks = []
        for idx, chunk in enumerate(addendum_chunks):
            add_num = chunk.get("addendum_number") or "Unknown"
            addendum_text_blocks.append(
                f"[Addendum Source {idx+1}] File: {chunk['file_name']}, Page: {chunk['page_number']}, Addendum #: {add_num}\n"
                f"{chunk['text']}\n"
            )

        # Current values summary
        current_summary = {k: v.value for k, v in current_fields.items() if v.value is not None}

        prompt = f"""You are the Addendum Reconciliation Agent for an RFP intelligence platform.
Your job is to examine all addendum passages for bid '{bid_id}' and identify if ANY addendum changes, extends, or updates any base RFP field.
Pay SPECIAL ATTENTION to:
1. Due Date (deadlines extended by addendums, e.g., Addendum 2 extending due date to July 9, 2024 at 2:00 PM CST).
2. Product Specifications (e.g. Q&A clarifications on ports, memory, processors, USB requirements).
3. Pre Bid Meeting or Award dates.

Current Field Values Before Addendums:
{json.dumps(current_summary, indent=2)}

Addendum Passages:
{"".join(addendum_text_blocks)}

Respond ONLY with a JSON object containing:
1. "changes": List of change objects:
   - "addendum_number": integer
   - "file_name": string
   - "field_affected": exact field name matching one of the keys
   - "original_value": previous value string
   - "new_value": new reconciled value
   - "description": concise explanation (e.g., "Extended by Addendum 2")
   - "source_page": integer page number
2. "updated_fields": Object containing updated field objects for fields that changed:
   {{
     "FieldName": {{
       "value": "new value",
       "sources": [{{"file": "filename.pdf", "page": 1, "doc_type": "addendum", "addendum_number": 2}}],
       "confidence": 0.95,
       "notes": "Extended by Addendum 2 (original date: ...)"
     }}
   }}
"""

        changes: List[AddendumChange] = []
        updated_fields = dict(current_fields)

        try:
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={"response_mime_type": "application/json", "temperature": 0.0},
            )
            resp = model.generate_content(prompt)
            data = json.loads(resp.text)

            for c in data.get("changes", []):
                change_obj = AddendumChange(
                    addendum_number=int(c.get("addendum_number", 1)),
                    file_name=c.get("file_name", ""),
                    field_affected=c.get("field_affected", ""),
                    original_value=str(c.get("original_value", "")),
                    new_value=str(c.get("new_value", "")),
                    description=c.get("description", ""),
                    source_page=int(c.get("source_page", 1)),
                )
                changes.append(change_obj)

            for fname, fval in data.get("updated_fields", {}).items():
                if fname in updated_fields and fval.get("value") is not None:
                    sources = [
                        SourceCitation(
                            file=s.get("file", ""),
                            page=int(s.get("page", 1)),
                            doc_type="addendum",
                            addendum_number=s.get("addendum_number"),
                        )
                        for s in fval.get("sources", [])
                    ]
                    updated_fields[fname] = ExtractedField(
                        value=fval.get("value"),
                        sources=sources,
                        confidence=float(fval.get("confidence", 0.95)),
                        notes=fval.get("notes"),
                        status="validated",
                    )
                    logger.info(f"Reconciled field '{fname}' via Addendum: {fval.get('value')}")

        except Exception as e:
            logger.error(f"Error in addendum reconciliation: {e}")

        trace = AgentTraceStep(
            step_id=4,
            timestamp=datetime.now().isoformat(),
            agent_name=self.name,
            action="reconcile_addendums",
            inputs={"bid_id": bid_id, "addendum_count": len(addendum_chunks)},
            outputs={"changes_count": len(changes), "updated_fields": list(data.get("updated_fields", {}).keys()) if 'data' in locals() else []},
            latency_ms=(time.time() - t0) * 1000,
        )
        return updated_fields, changes, trace
