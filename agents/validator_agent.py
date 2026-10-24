"""
Validator / Critic Agent.
Performs verification across evidence support, date/format standards,
and anti-hallucination checks. Triggers orchestrator feedback loop for rejected fields.
"""

import time
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
from agents.state import ExtractedField, ValidationReport, AgentTraceStep

logger = logging.getLogger("rfp_validator")


class ValidatorAgent:
    def __init__(self):
        self.name = "Validator / Critic Agent"

    def validate(
        self,
        bid_id: str,
        fields: Dict[str, ExtractedField],
    ) -> Tuple[ValidationReport, List[str], AgentTraceStep]:
        t0 = time.time()
        passed_count = 0
        failed_count = 0
        not_found_count = 0
        feedback_notes: Dict[str, str] = {}
        failed_fields: List[str] = []

        for field_name, field_obj in fields.items():
            val = field_obj.value

            # Case 1: Field not found in documents
            if val is None or str(val).strip().lower() in ["null", "none", "n/a", "not found"]:
                field_obj.value = None
                if not field_obj.notes or "not found" not in field_obj.notes.lower():
                    field_obj.notes = "Not found in documents"
                field_obj.status = "not_found"
                not_found_count += 1
                continue

            # Case 2: Field has value -> Check citations
            if not field_obj.sources or len(field_obj.sources) == 0:
                # Guardrail: every non-null value MUST have at least one citation
                field_obj.status = "rejected"
                failed_count += 1
                feedback_notes[field_name] = "Rejected: Value provided without source citation."
                failed_fields.append(field_name)
                continue

            # Case 3: Format and consistency checks
            val_str = str(val).strip()

            # Due Date check
            if field_name == "Due Date":
                # Ensure year is present and reasonable
                if not any(year in val_str for year in ["2023", "2024", "2025", "2026", "2027"]):
                    field_obj.status = "rejected"
                    failed_count += 1
                    feedback_notes[field_name] = f"Rejected: Due date '{val_str}' missing valid year or recognizable date format."
                    failed_fields.append(field_name)
                    continue

            # Bid Number check
            if field_name == "Bid Number":
                if len(val_str) < 3 or val_str.lower() in ["rfp", "bid"]:
                    field_obj.status = "rejected"
                    failed_count += 1
                    feedback_notes[field_name] = f"Rejected: Ambiguous or generic solicitation identifier '{val_str}'."
                    failed_fields.append(field_name)
                    continue

            # All checks passed
            field_obj.status = "validated"
            passed_count += 1

        report = ValidationReport(
            passed=passed_count,
            failed=failed_count,
            not_found=not_found_count,
            feedback_notes=feedback_notes,
        )

        trace = AgentTraceStep(
            step_id=5,
            timestamp=datetime.now().isoformat(),
            agent_name=self.name,
            action="validate_extracted_fields",
            inputs={"bid_id": bid_id, "total_fields": len(fields)},
            outputs={
                "passed": passed_count,
                "failed": failed_count,
                "not_found": not_found_count,
                "failed_fields": failed_fields,
            },
            latency_ms=(time.time() - t0) * 1000,
        )

        logger.info(f"Validation for '{bid_id}': Passed={passed_count}, Not Found={not_found_count}, Failed={failed_count}")
        return report, failed_fields, trace
