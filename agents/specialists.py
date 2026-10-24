"""
Extraction Specialists:
1. Dates & Logistics Specialist
2. Commercial & Legal Specialist
3. Product & Specs Specialist
Each specialist queries the search engine for targeted evidence and extracts structured fields.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

from agents.state import ExtractedField, SourceCitation, AgentTraceStep
from search.hybrid_search import HybridSearchEngine

load_dotenv()
logger = logging.getLogger("rfp_specialists")


class BaseSpecialistAgent:
    def __init__(self, name: str, search_engine: HybridSearchEngine, model_name: str = "models/gemini-2.5-flash"):
        self.name = name
        self.search_engine = search_engine
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def _call_llm_for_fields(
        self,
        field_definitions: Dict[str, str],
        evidence_passages: List[Dict[str, Any]],
        bid_id: str,
    ) -> Dict[str, ExtractedField]:
        """
        Sends evidence passages and field definitions to Gemini for structured extraction.
        """
        context_blocks = []
        for idx, p in enumerate(evidence_passages):
            context_blocks.append(
                f"[Source {idx+1}] File: {p['file_name']}, Page: {p['page_number']}, DocType: {p['doc_type']}\n"
                f"Content: {p['text']}\n"
            )

        fields_desc = "\n".join([f"- **{k}**: {v}" for k, v in field_definitions.items()])

        prompt = f"""You are the {self.name} for an RFP Intelligence Platform.
Your task is to extract structured values for the specified fields for RFP bid: '{bid_id}' using ONLY the provided evidence passages.

CRITICAL RULES:
1. GROUNDING: Every field must be strictly grounded in the evidence passages. If a field is NOT explicitly mentioned or cannot be inferred from the documents, set "value": null, "notes": "Not found in documents", "confidence": 0.85, and "sources": [].
2. CITATIONS: Every non-null value MUST include the exact source citation (file name and page number) from the provided [Source X] blocks.
3. DUE DATE: Extract the proposal submission deadline including time and time zone if stated.
4. SPECIFICATIONS: Include concrete processor, RAM, storage, display, warranty details where applicable.
5. NO HALLUCINATIONS: Never fabricate dates, bond requirements, or model numbers.

Fields to Extract:
{fields_desc}

Evidence Passages:
{"".join(context_blocks)}

Respond ONLY with a JSON object where keys match the field names:
{{
  "FieldName": {{
    "value": "extracted value or null",
    "sources": [
      {{"file": "source_filename.pdf", "page": 1, "doc_type": "rfp"}}
    ],
    "confidence": 0.95,
    "notes": "explanation or null"
  }}
}}"""

        try:
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={"response_mime_type": "application/json", "temperature": 0.1},
            )
            response = model.generate_content(prompt)
            data = json.loads(response.text)

            extracted_fields: Dict[str, ExtractedField] = {}
            for field_name in field_definitions.keys():
                field_data = data.get(field_name, {})
                val = field_data.get("value")
                notes = field_data.get("notes")
                conf = float(field_data.get("confidence", 0.9 if val is not None else 0.8))

                sources = []
                for s in field_data.get("sources", []):
                    sources.append(
                        SourceCitation(
                            file=s.get("file", ""),
                            page=int(s.get("page", 1)),
                            doc_type=s.get("doc_type"),
                        )
                    )

                if val is None or val == "" or str(val).lower() == "null":
                    val = None
                    if not notes:
                        notes = "Not found in documents"
                    status = "not_found"
                else:
                    status = "draft"

                extracted_fields[field_name] = ExtractedField(
                    value=val,
                    sources=sources,
                    confidence=conf,
                    notes=notes,
                    status=status,
                )

            return extracted_fields

        except Exception as e:
            logger.error(f"{self.name} extraction error: {e}")
            # Fallback
            fallback: Dict[str, ExtractedField] = {}
            for field_name in field_definitions.keys():
                fallback[field_name] = ExtractedField(
                    value=None,
                    sources=[],
                    confidence=0.5,
                    notes=f"Extraction failure: {str(e)}",
                    status="rejected",
                )
            return fallback


class DatesLogisticsSpecialist(BaseSpecialistAgent):
    """Extracts Due Date, Bid Submission Type, Term of Bid, Pre Bid Meeting, Delivery Date, Installation."""

    FIELD_DEFINITIONS = {
        "Due Date": "Final submission deadline, including time and time zone. Must reflect any addendum changes.",
        "Bid Submission Type": "How the bid is submitted (electronic portal like eMMA or BidNet, sealed paper, email, etc.).",
        "Term of Bid": "Contract duration and any renewal options (e.g. 1 year, 3 years, etc.).",
        "Pre Bid Meeting": "Date, time, location and whether attendance is mandatory, or 'None'.",
        "Delivery Date": "Required delivery date or delivery window after award (e.g. within 45 days of award).",
        "Installation": "Whether installation / deployment / imaging services are required (Yes/No or description).",
    }

    def __init__(self, search_engine: HybridSearchEngine):
        super().__init__("Dates & Logistics Specialist", search_engine)

    def extract(self, bid_id: str) -> tuple[Dict[str, ExtractedField], AgentTraceStep]:
        t0 = time.time()
        queries = [
            f"{bid_id} proposal due date and time closing date submission deadline 06/10/2024",
            f"{bid_id} portal basic information closing date publication",
            f"{bid_id} bid submission instructions electronic portal sealed paper eMMA BidNet",
            f"{bid_id} pre-bid meeting conference date attendance mandatory",
            f"{bid_id} delivery date window 45 days after award term of contract renewal",
            f"{bid_id} installation deployment imaging services required",
        ]

        evidence = []
        seen_ids = set()
        for q in queries:
            results = self.search_engine.search(q, bid_id=bid_id, top_k=3, mode="hybrid")
            for r in results:
                if r["chunk_id"] not in seen_ids:
                    seen_ids.add(r["chunk_id"])
                    evidence.append(r)

        extracted = self._call_llm_for_fields(self.FIELD_DEFINITIONS, evidence, bid_id)
        latency = (time.time() - t0) * 1000

        trace = AgentTraceStep(
            step_id=1,
            timestamp=datetime.now().isoformat(),
            agent_name=self.name,
            action="extract_dates_logistics",
            inputs={"bid_id": bid_id, "queries": queries},
            tool_calls=["search_engine.search"] * len(queries),
            outputs={k: str(v.value) for k, v in extracted.items()},
            latency_ms=latency,
        )
        return extracted, trace


class CommercialLegalSpecialist(BaseSpecialistAgent):
    """Extracts Bid Bond Requirement, Payment Terms, Additional Documentation, MFG Registration, Contract/Cooperative, contact_info, company_name."""

    FIELD_DEFINITIONS = {
        "Bid Bond Requirement": "Required bid bond or security, amount or percentage, or 'Not required'.",
        "Payment Terms": "Payment terms (e.g. Net 30), invoicing conditions.",
        "Any Additional Documentation Required": "Affidavits, forms, certificates, insurance, W-9, Mercury Affidavit, Contract Affidavit, etc.",
        "MFG for Registration": "Manufacturer(s) the bidder must be registered / authorized with (e.g. Dell authorized reseller).",
        "Contract or Cooperative to use": "Referenced cooperative / state contract vehicle, if any (e.g. Master Contract 060B5400007).",
        "contact_info": "Procurement contact: name, email, phone.",
        "company_name": "Issuing organization / agency name.",
    }

    def __init__(self, search_engine: HybridSearchEngine):
        super().__init__("Commercial & Legal Specialist", search_engine)

    def extract(self, bid_id: str) -> tuple[Dict[str, ExtractedField], AgentTraceStep]:
        t0 = time.time()
        queries = [
            f"{bid_id} bid bond requirement security guarantee percentage",
            f"{bid_id} payment terms Net 30 invoicing instructions billing",
            f"{bid_id} required affidavits forms certificates mercury affidavit contract affidavit W-9",
            f"{bid_id} manufacturer registration authorized reseller Dell Letter of Authorization LOA",
            f"{bid_id} master contract cooperative vehicle state contract number 060B5400007",
            f"{bid_id} procurement contact officer name email phone buyer",
            f"{bid_id} issuing agency organization company name",
        ]

        evidence = []
        seen_ids = set()
        for q in queries:
            results = self.search_engine.search(q, bid_id=bid_id, top_k=3, mode="hybrid")
            for r in results:
                if r["chunk_id"] not in seen_ids:
                    seen_ids.add(r["chunk_id"])
                    evidence.append(r)

        extracted = self._call_llm_for_fields(self.FIELD_DEFINITIONS, evidence, bid_id)
        latency = (time.time() - t0) * 1000

        trace = AgentTraceStep(
            step_id=2,
            timestamp=datetime.now().isoformat(),
            agent_name=self.name,
            action="extract_commercial_legal",
            inputs={"bid_id": bid_id, "queries": queries},
            tool_calls=["search_engine.search"] * len(queries),
            outputs={k: str(v.value) for k, v in extracted.items()},
            latency_ms=latency,
        )
        return extracted, trace


class ProductSpecsSpecialist(BaseSpecialistAgent):
    """Extracts Bid Number, Title, Model_no, Part_no, Product, Bid Summary, Product Specification."""

    FIELD_DEFINITIONS = {
        "Bid Number": "Official solicitation / RFP identifier (e.g. JA-207652 or #E20P4600040 / BPM044557).",
        "Title": "Full title of the bid or RFP.",
        "Model_no": "Model number(s) of the requested product(s) (e.g. SI# CC7802 Dell Latitude 5550, WD22TB4).",
        "Part_no": "Part / SKU number(s) of the requested product(s) (e.g. 210-BLYZ, 379-BFNZ, etc.).",
        "Product": "Product name(s) and quantity (e.g. 30 Dell Latitude 5550 laptops, 30 WD22TB4 docks).",
        "Bid Summary": "A concise (3–6 sentence) summary of the bid, generated by the system.",
        "Product Specification": "Key technical specifications (CPU, RAM, storage, display, warranty, Copilot ready, etc.).",
    }

    def __init__(self, search_engine: HybridSearchEngine):
        super().__init__("Product & Specs Specialist", search_engine)

    def extract(self, bid_id: str) -> tuple[Dict[str, ExtractedField], AgentTraceStep]:
        t0 = time.time()
        queries = [
            f"{bid_id} solicitation number bid identifier RFP number reference",
            f"{bid_id} full title of bid request for proposal",
            f"{bid_id} model number SKU part number SI# CC7802 WD22TB4",
            f"{bid_id} requested products quantity laptops desktops monitors docks",
            f"{bid_id} technical specifications processor CPU RAM storage screen display warranty",
            f"{bid_id} scope and specifications purpose of request for proposal",
        ]

        evidence = []
        seen_ids = set()
        for q in queries:
            results = self.search_engine.search(q, bid_id=bid_id, top_k=4, mode="hybrid")
            for r in results:
                if r["chunk_id"] not in seen_ids:
                    seen_ids.add(r["chunk_id"])
                    evidence.append(r)

        extracted = self._call_llm_for_fields(self.FIELD_DEFINITIONS, evidence, bid_id)
        latency = (time.time() - t0) * 1000

        trace = AgentTraceStep(
            step_id=3,
            timestamp=datetime.now().isoformat(),
            agent_name=self.name,
            action="extract_product_specs",
            inputs={"bid_id": bid_id, "queries": queries},
            tool_calls=["search_engine.search"] * len(queries),
            outputs={k: str(v.value) for k, v in extracted.items()},
            latency_ms=latency,
        )
        return extracted, trace
