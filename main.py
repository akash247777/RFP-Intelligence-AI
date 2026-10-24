"""
Main Command Line Interface (CLI) for the RFP Intelligence Platform.
Supports:
- One-command Bid Extraction: python main.py --bid ./Bid1
- Natural Language Q&A: python main.py --ask "What is the submission deadline for Bid1 after all addendums?"
- Hybrid Search: python main.py --search "docking station" --bid Bid2
- Benchmark Evaluation: python main.py --eval
- REST API Server: python main.py --serve
- Streamlit Web Dashboard: python main.py --ui
"""

import os
import sys
import json
import argparse
import subprocess
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="RFP Intelligence Platform: RAG Search Engine & Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bid",
        type=str,
        help="Path to bid directory (e.g. ./Bid1 or ./Bid2) to extract structured JSON record",
    )
    parser.add_argument(
        "--ask",
        type=str,
        help="Ask a natural-language question about the bids (returns answer with citations)",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search query to run directly against the hybrid RAG search engine",
    )
    parser.add_argument(
        "--bid-filter",
        type=str,
        default=None,
        help="Optional bid ID filter (Bid1 or Bid2) for search or Q&A",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the retrieval evaluation benchmark and generate output/evaluation_report.md",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the FastAPI REST API server at http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive Streamlit Web UI",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for REST API (default: 8000)",
    )

    args = parser.parse_args()

    # 1. UI mode
    if args.ui:
        print("Launching Streamlit Web UI...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        return

    # 2. REST API Server mode
    if args.serve:
        print(f"Starting FastAPI server on http://127.0.0.1:{args.port}...")
        import uvicorn
        uvicorn.run("api.routes:app", host="0.0.0.0", port=args.port, reload=False)
        return

    # 3. Retrieval Evaluation mode
    if args.eval:
        print("Running Retrieval Evaluation Benchmark...")
        from eval.benchmark import run_retrieval_benchmark
        report = run_retrieval_benchmark()
        print(report)
        return

    # 4. Hybrid Search mode
    if args.search:
        from search.hybrid_search import HybridSearchEngine
        print(f"Searching for: '{args.search}' (filter: {args.bid_filter})...")
        engine = HybridSearchEngine()
        results = engine.search(args.search, bid_id=args.bid_filter, top_k=5)
        print(f"\nFound {len(results)} cited passages:\n" + "=" * 60)
        for i, r in enumerate(results, 1):
            cit = r.get("citation", {})
            print(f"[{i}] File: {cit.get('file')} | Page: {cit.get('page')} | DocType: {cit.get('doc_type')}")
            print(f"Score (RRF): {r.get('rrf_score', 0):.4f}")
            print(f"Text Snippet:\n{r['text'][:300]}...\n" + "-" * 60)
        return

    # 5. Question-Answering mode
    if args.ask:
        from search.hybrid_search import HybridSearchEngine
        from agents.qa_agent import QAReportAgent
        print(f"Question: '{args.ask}'...")
        engine = HybridSearchEngine()
        qa_agent = QAReportAgent(engine)
        ans = qa_agent.answer_question(args.ask, bid_id=args.bid_filter)
        print("\n" + "=" * 60 + "\nANSWER:\n" + "=" * 60)
        print(ans.get("answer"))
        if ans.get("key_findings"):
            print("\nKey Findings:")
            for kf in ans["key_findings"]:
                print(f"  • {kf}")
        print("\nCitations Used:")
        for c in ans.get("citations", []):
            print(f"  - File: {c.get('file')}, Page: {c.get('page')}")
        print(f"\nConfidence: {ans.get('confidence', 0.9):.2f} | Latency: {ans.get('latency_ms', 0):.0f}ms\n" + "=" * 60)
        return

    # 6. Extraction Mode: given a bid folder
    if args.bid:
        folder_path = args.bid
        bid_name = os.path.basename(os.path.normpath(folder_path))
        print(f"Extracting structured records for: '{bid_name}' from folder: '{folder_path}'...")

        from agents.orchestrator import OrchestratorAgent
        orch = OrchestratorAgent()
        record = orch.extract_bid(bid_name, folder_path)

        print("\n" + "=" * 60)
        print(f"SUCCESSFULLY EXTRACTED BID: {bid_name}")
        print("=" * 60)
        val = record.validation
        print(f"Validation: Passed={val.passed}, Not Found={val.not_found}, Failed={val.failed}")
        print(f"Addendum Changes Reconciled: {len(record.addendum_changes)}")
        print("\nKey Fields Summary:")
        for field_name in ["Bid Number", "Title", "Due Date", "Bid Submission Type", "Product", "company_name"]:
            if field_name in record.fields:
                f = record.fields[field_name]
                print(f"  • {field_name}: {f.value}")
        out_file = os.path.join("output", f"{bid_name.lower()}_extracted.json")
        print(f"\nComplete JSON record written to: {out_file}")
        return

    # Default: Show help
    parser.print_help()


if __name__ == "__main__":
    main()
