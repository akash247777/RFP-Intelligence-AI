"""
Shared Multi-Agent State and Pydantic Schemas.
Defines structured schemas for extracted fields, citations, addendum reconciliation,
and pipeline observability traces.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    file: str
    page: int
    doc_type: Optional[str] = None
    addendum_number: Optional[int] = None
    chunk_id: Optional[str] = None
    snippet: Optional[str] = None


class ExtractedField(BaseModel):
    value: Optional[Any] = None
    sources: List[SourceCitation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: Optional[str] = None
    status: str = Field(default="draft")  # draft, validated, rejected, not_found


class AddendumChange(BaseModel):
    addendum_number: int
    file_name: str
    field_affected: str
    original_value: Optional[str] = None
    new_value: str
    description: str
    source_page: int


class ValidationReport(BaseModel):
    passed: int = 0
    failed: int = 0
    not_found: int = 0
    feedback_notes: Dict[str, str] = Field(default_factory=dict)


class AgentTraceStep(BaseModel):
    step_id: int
    timestamp: str
    agent_name: str
    action: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: List[str] = Field(default_factory=list)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    status: str = "success"


class BidExtractionRecord(BaseModel):
    bid_id: str
    fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    addendum_changes: List[AddendumChange] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)


class AgentState(BaseModel):
    bid_id: str
    folder_path: Optional[str] = None
    plan: List[str] = Field(default_factory=list)
    retrieved_evidence: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    draft_fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    addendum_changes: List[AddendumChange] = Field(default_factory=list)
    validation_report: ValidationReport = Field(default_factory=ValidationReport)
    step_traces: List[AgentTraceStep] = Field(default_factory=list)
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    is_complete: bool = False
