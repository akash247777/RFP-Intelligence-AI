"""
Multi-Agent Orchestrator and Planner.
Coordinates parallel specialist extraction, addendum reconciliation,
validation, feedback retry loops, shared state management, and trace observability.
"""

import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, Optional

from agents.state import AgentState, BidExtractionRecord, ExtractedField, SourceCitation, AgentTraceStep
from search.hybrid_search import HybridSearchEngine
from agents.specialists import DatesLogisticsSpecialist, CommercialLegalSpecialist, ProductSpecsSpecialist
from agents.addendum_agent import AddendumReconciliationAgent
from agents.validator_agent import ValidatorAgent
from agents.qa_agent import QAReportAgent

logger = logging.getLogger("rfp_orchestrator")


class OrchestratorAgent:
    def __init__(self, search_engine: Optional[HybridSearchEngine] = None, max_retries: int = 2):
        self.search_engine = search_engine or HybridSearchEngine()
        self.max_retries = max_retries
        self.dates_agent = DatesLogisticsSpecialist(self.search_engine)
        self.commercial_agent = CommercialLegalSpecialist(self.search_engine)
        self.product_agent = ProductSpecsSpecialist(self.search_engine)
        self.addendum_agent = AddendumReconciliationAgent(self.search_engine)
        self.validator_agent = ValidatorAgent()
        self.qa_agent = QAReportAgent(self.search_engine)

    def extract_bid(self, bid_id: str, folder_path: Optional[str] = None) -> BidExtractionRecord:
        """
        Executes full multi-agent extraction pipeline:
        1. Ingestion check / indexing if needed.
        2. Parallel execution of specialist agents.
        3. Addendum reconciliation and override application.
        4. Validation check.
        5. Feedback loop for rejected fields (up to max_retries).
        6. Production of final JSON conforming to Section 8.1.
        """
        logger.info(f"--- Starting Orchestrated Extraction for Bid: '{bid_id}' ---")
        t_start = time.time()

        state = AgentState(
            bid_id=bid_id,
            folder_path=folder_path,
            plan=[
                "1. Verify ingestion & search index",
                "2. Dispatch parallel specialist extraction agents",
                "3. Reconcile addendum overrides and date extensions",
                "4. Critic validation and schema verification",
                "5. Execute feedback retry loop for rejected fields if required",
                "6. Compile final JSON and export observability trace",
            ],
        )

        # Step 1: Ensure indexed
        indexed_bids = self.search_engine.vector_store.get_indexed_bids()
        if bid_id not in indexed_bids:
            if folder_path and os.path.isdir(folder_path):
                logger.info(f"Bid '{bid_id}' not yet indexed. Running Ingestion Agent on '{folder_path}'...")
                self.search_engine.index_bid_folder(folder_path, bid_id)
            else:
                logger.warning(f"Bid '{bid_id}' not indexed and folder_path not provided. Continuing with available index.")

        # Step 2: Parallel Specialist Extraction
        logger.info("Dispatching Specialist Agents concurrently via ThreadPoolExecutor...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_dates = executor.submit(self.dates_agent.extract, bid_id)
            fut_comm = executor.submit(self.commercial_agent.extract, bid_id)
            fut_prod = executor.submit(self.product_agent.extract, bid_id)

            dates_res, trace_dates = fut_dates.result()
            comm_res, trace_comm = fut_comm.result()
            prod_res, trace_prod = fut_prod.result()

        state.step_traces.extend([trace_dates, trace_comm, trace_prod])

        # Merge draft fields
        state.draft_fields.update(dates_res)
        state.draft_fields.update(comm_res)
        state.draft_fields.update(prod_res)
        logger.info(f"Specialist extraction complete in {(time.time()-t0):.2f}s. Extracted {len(state.draft_fields)} draft fields.")

        # Step 3: Addendum Reconciliation
        logger.info("Executing Addendum Reconciliation Agent...")
        reconciled_fields, addendum_changes, trace_add = self.addendum_agent.reconcile(bid_id, state.draft_fields)
        state.draft_fields = reconciled_fields
        state.addendum_changes = addendum_changes
        state.step_traces.append(trace_add)

        # Step 4: Validation
        logger.info("Executing Validator / Critic Agent...")
        val_report, failed_fields, trace_val = self.validator_agent.validate(bid_id, state.draft_fields)
        state.validation_report = val_report
        state.step_traces.append(trace_val)

        # Step 5: Feedback Loop for Rejected Fields
        retry_num = 0
        while failed_fields and retry_num < self.max_retries:
            retry_num += 1
            logger.warning(f"Feedback Loop Triggered! Retrying {len(failed_fields)} rejected fields (Attempt {retry_num}/{self.max_retries}): {failed_fields}")
            t_retry = time.time()

            for fname in failed_fields:
                state.retry_counts[fname] = state.retry_counts.get(fname, 0) + 1
                # Execute focused search with relaxed parameters
                retry_evidence = self.search_engine.search(
                    f"{bid_id} {fname} specific value details",
                    bid_id=bid_id,
                    top_k=5,
                    mode="hybrid_rerank",
                )
                # Call targeted extraction
                if fname in self.dates_agent.FIELD_DEFINITIONS:
                    sub_dict = {fname: self.dates_agent.FIELD_DEFINITIONS[fname]}
                    retried = self.dates_agent._call_llm_for_fields(sub_dict, retry_evidence, bid_id)
                elif fname in self.commercial_agent.FIELD_DEFINITIONS:
                    sub_dict = {fname: self.commercial_agent.FIELD_DEFINITIONS[fname]}
                    retried = self.commercial_agent._call_llm_for_fields(sub_dict, retry_evidence, bid_id)
                else:
                    sub_dict = {fname: self.product_agent.FIELD_DEFINITIONS.get(fname, fname)}
                    retried = self.product_agent._call_llm_for_fields(sub_dict, retry_evidence, bid_id)

                state.draft_fields.update(retried)

            # Re-validate after retry
            val_report, failed_fields, trace_reval = self.validator_agent.validate(bid_id, state.draft_fields)
            state.validation_report = val_report
            trace_reval.inputs["retry_attempt"] = retry_num
            state.step_traces.append(trace_reval)

        state.is_complete = True
        logger.info(f"Extraction pipeline complete in {(time.time() - t_start):.2f}s. Passed: {state.validation_report.passed}, Not Found: {state.validation_report.not_found}, Failed: {state.validation_report.failed}")

        # Assemble Final Record
        record = BidExtractionRecord(
            bid_id=bid_id,
            fields=state.draft_fields,
            addendum_changes=state.addendum_changes,
            validation=state.validation_report,
        )

        # Save to output file
        os.makedirs("output", exist_ok=True)
        out_file = os.path.join("output", f"{bid_id.lower()}_extracted.json")
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(record.model_dump_json(indent=2))
            logger.info(f"Saved extraction record to '{out_file}'")
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")

        # Save trace
        trace_file = os.path.join("output", "agent_trace.json")
        try:
            traces_json = [t.model_dump() for t in state.step_traces]
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump({"bid_id": bid_id, "total_duration_ms": (time.time() - t_start) * 1000, "steps": traces_json}, f, indent=2)
            logger.info(f"Saved agent execution trace to '{trace_file}'")
        except Exception as e:
            logger.error(f"Failed to write trace file: {e}")

        return record
