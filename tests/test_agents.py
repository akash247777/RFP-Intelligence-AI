"""
Unit Tests for Agents and Shared State
=======================================
Validates Pydantic schema serialization, Validator/Critic citation enforcement,
and anti-hallucination guardrails.
"""

from agents.state import (
    SourceCitation,
    ExtractedField,
    AddendumChange,
    ValidationReport,
    BidExtractionRecord,
)
from agents.validator_agent import ValidatorAgent


def test_state_serialization():
    citation = SourceCitation(
        file="Addendum 2.pdf",
        page=1,
        doc_type="addendum",
        addendum_number=2,
    )
    field = ExtractedField(
        value="July 9, 2024 at 2:00 PM CST",
        sources=[citation],
        confidence=1.0,
        notes="Extended by Addendum 2",
        status="validated",
    )
    assert field.value == "July 9, 2024 at 2:00 PM CST"
    assert len(field.sources) == 1
    assert field.sources[0].page == 1

    d = field.model_dump()
    assert d["value"] == "July 9, 2024 at 2:00 PM CST"
    assert d["status"] == "validated"


def test_validator_enforces_citations():
    validator = ValidatorAgent()

    # Field with value but NO citations -> must be rejected
    f_uncited = ExtractedField(
        value="5% Bid Bond Required",
        sources=[],
        confidence=0.9,
    )

    # Field marked as not found
    f_missing = ExtractedField(
        value=None,
        sources=[],
    )

    # Field with valid citation
    f_valid = ExtractedField(
        value="July 9, 2024 at 2:00 PM CST",
        sources=[SourceCitation(file="Addendum2.pdf", page=1, doc_type="addendum")],
        confidence=1.0,
    )

    fields = {
        "Bid Bond": f_uncited,
        "Pre Bid Meeting": f_missing,
        "Due Date": f_valid,
    }

    report, failed_fields, step = validator.validate("TestBid", fields)

    assert report.failed == 1
    assert report.not_found == 1
    assert report.passed == 1
    assert "Bid Bond" in failed_fields
    assert f_uncited.status == "rejected"
    assert f_missing.status == "not_found"
    assert f_valid.status == "validated"


def test_validator_date_sanity():
    validator = ValidatorAgent()

    # Field with invalid year
    f_bad_date = ExtractedField(
        value="October 15, 1992",
        sources=[SourceCitation(file="doc.pdf", page=1)],
    )

    fields = {"Due Date": f_bad_date}
    report, failed_fields, step = validator.validate("TestBid", fields)
    assert "Due Date" in failed_fields
    assert f_bad_date.status == "rejected"
