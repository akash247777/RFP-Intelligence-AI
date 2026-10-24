"""
Generates the Sample Q&A Log deliverable with 12 cited questions and answers across Bid1 and Bid2.
"""

import os
import json
import logging
from agents.qa_agent import QAReportAgent
from search.hybrid_search import HybridSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_qa_log")

SAMPLE_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the submission deadline for Bid1 after all addendums?",
        "bid_id": "Bid1",
    },
    {
        "id": 2,
        "question": "Which affidavits are required for the Dell laptop bid?",
        "bid_id": "Bid2",
    },
    {
        "id": 3,
        "question": "Compare the warranty requirements of both bids.",
        "bid_id": None,
    },
    {
        "id": 4,
        "question": "Is a bid bond required, and if so, how much?",
        "bid_id": "Bid1",
    },
    {
        "id": 5,
        "question": "What changed in Addendum 2 compared to the original RFP?",
        "bid_id": "Bid1",
    },
    {
        "id": 6,
        "question": "What are the core technical specifications (processor, RAM, storage) for the Dell laptops in Bid2?",
        "bid_id": "Bid2",
    },
    {
        "id": 7,
        "question": "When and how will the pre-proposal conference for Dallas ISD take place?",
        "bid_id": "Bid1",
    },
    {
        "id": 8,
        "question": "What docking stations are requested in the Maryland PORFP?",
        "bid_id": "Bid2",
    },
    {
        "id": 9,
        "question": "What are the payment and retainage terms for Dallas ISD subcontractors?",
        "bid_id": "Bid1",
    },
    {
        "id": 10,
        "question": "What manufacturer certifications or reseller authorizations are required for Bid2?",
        "bid_id": "Bid2",
    },
    {
        "id": 11,
        "question": "Can proposing vendors submit higher USB specifications than requested for Bid1 display monitors?",
        "bid_id": "Bid1",
    },
    {
        "id": 12,
        "question": "What is the delivery window specified after contract award for the Dell laptops?",
        "bid_id": "Bid2",
    },
]


def generate_qa_log():
    engine = HybridSearchEngine()
    qa_agent = QAReportAgent(engine)

    qa_results = []
    for item in SAMPLE_QUESTIONS:
        logger.info(f"Answering Q{item['id']}: {item['question']}")
        ans_data = qa_agent.answer_question(
            question=item["question"],
            bid_id=item["bid_id"],
            top_k=5,
        )
        qa_results.append({
            "id": item["id"],
            "bid_scope": item["bid_id"] or "Cross-Bid",
            "question": item["question"],
            "answer": ans_data.get("answer"),
            "key_findings": ans_data.get("key_findings", []),
            "citations": ans_data.get("citations", []),
            "confidence": ans_data.get("confidence", 0.9),
            "latency_ms": ans_data.get("latency_ms", 0.0),
        })

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "sample_qa_log.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa_results, f, indent=2)

    logger.info(f"Saved {len(qa_results)} questions & answers to '{out_path}'")
    return qa_results


if __name__ == "__main__":
    generate_qa_log()
