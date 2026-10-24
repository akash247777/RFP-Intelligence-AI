# ⚡ RFP Intelligence Platform: Multi-Agent System & Hybrid RAG

An enterprise-grade, autonomous **Multi-Agent Retrieval-Augmented Generation (RAG) platform** engineered for analyzing complex public procurement solicitations (RFPs, PORFPs, Addenda, and Vendor Portals).

Built specifically for the technical evaluation assignment analyzing:
1. **Bid 1**: Dallas Independent School District (Dallas ISD) — *Student and Staff Computing Devices (RFP JA-207652)*
2. **Bid 2**: Maryland State Treasurer's Office — *Dell Laptops & Docks (CATS+ Task Order PORFP)*

Powered by **Google Gemini 2.5 Flash**, **Gemini Embedding Engine (`models/gemini-embedding-001`)**, **BM25 Keyword Indexing**, **Reciprocal Rank Fusion (RRF)**, and an interactive **Streamlit Dashboard**.

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User Request / CLI / UI / REST API]) --> Orch[Orchestrator Agent]

    subgraph "Part A & B: Ingestion, Chunking & Hybrid RAG Engine"
        Docs[PDF & HTML Solicitation Documents] --> Parser[PyMuPDF & BeautifulSoup Parser]
        Parser --> Clean[Boilerplate & Header/Footer Cleaner]
        Clean --> Chunker[Section & Table-Aware Chunker]
        
        Chunker -->|Dense Embeddings| Embedder[Gemini Embeddings Engine]
        Chunker -->|Alphanumeric Tokens| BM25Index[BM25 Okapi Index]
        
        Embedder --> VecStore[(Vector Store - .npy + .json)]
        BM25Index --> BM25Store[(BM25 Store - .json)]
        
        VecStore & BM25Store --> RRF[Reciprocal Rank Fusion - RRF Engine]
        RRF --> Rerank[LLM Candidate Re-Ranker]
    end

    subgraph "Part C: Multi-Agent Collaboration System"
        Orch --> DSpecialist[📅 Dates & Logistics Specialist]
        Orch --> CSpecialist[⚖️ Commercial & Legal Specialist]
        Orch --> PSpecialist[💻 Product Specs & Scope Specialist]
        
        DSpecialist & CSpecialist & PSpecialist -->|Domain Sub-queries| RRF
        RRF -->|Ranked Passages with Citations| DSpecialist & CSpecialist & PSpecialist
        
        DSpecialist & CSpecialist & PSpecialist --> SharedState[(Shared Pydantic State)]
        SharedState --> AddendumAgent[🔄 Addendum Reconciliation Agent]
        AddendumAgent -->|Overrides Applied & Logged| SharedState
        
        SharedState --> Validator[🛡️ Validator / Critic Agent]
        Validator -->|Check 1: Non-null fields have citations| Audit{Audit Passed?}
        Validator -->|Check 2: Anti-hallucination check| Audit
        Validator -->|Check 3: Date/Format validation| Audit
        
        Audit -- Yes --> FinalRecord[Validated Structured Record - Section 8.1 JSON]
        Audit -- No (Rejection) -->|Targeted Feedback Loop| Orch
    end

    FinalRecord --> Output[output/bidX_extracted.json]
    Output --> StreamlitApp[Streamlit UI / Dashboard]
```

---

## 🌟 Key Capabilities & Architectural Innovations

### 1. Robust Section & Table-Aware Ingestion (Part A)
- **High-Speed Native Extraction**: Uses PyMuPDF's C++ table extraction (`page.find_tables()`), parsing 62-page documents in under 0.2 seconds while converting complex hardware specification tables directly into structured Markdown.
- **HTML Portal Extraction**: Cleans metadata and labels from BidNet Direct web pages without losing critical timestamps and closing dates.
- **Noise Elimination**: Strips recurring agency headers, footers, pagination artifacts, and line-break hyphens (`solici-\ntation` $\rightarrow$ `solicitation`).
- **Contextual Breadcrumbs**: Every chunk prepends a header `[BidID | File | Page | Section]`, anchoring semantic embeddings to document structure.

### 2. High-Precision Hybrid Retrieval Engine (Part B)
- **Dense Vector Search**: Powered by `models/gemini-embedding-001` with local persistent disk caching (`index_store/embeddings_cache.json`) and thread-safe batch execution.
- **BM25 Keyword Search**: Retains alphanumeric codes and hardware models (e.g., `WD22TB4`, `210-BLYZ`, `SI# CC7802`, `Form 1295`, `CATS+ #060B5400007`) that dense embeddings alone can dilute.
- **Reciprocal Rank Fusion (RRF)**: Merges dense vector and BM25 rank lists:
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
- **LLM Re-ranking**: High-scoring candidates are scored by Gemini 2.5 Flash to ensure top-ranked evidence blocks directly answer the question.

### 3. Autonomous Multi-Agent System (Part C)
- **Orchestrator Agent**: Manages end-to-end execution, dispatches work in parallel using `ThreadPoolExecutor`, and logs execution traces.
- **3 Domain Specialists**:
  - `Dates & Logistics Specialist`: Due date, pre-bid meeting, delivery timeline, contract terms.
  - `Commercial & Legal Specialist`: Bid bonds, payment terms, required affidavits, reseller authorization, cooperative purchasing.
  - `Product Specs & Scope Specialist`: Scope of work, quantities, technical specs, OS/management, warranties, liquidated damages.
- **Addendum Reconciliation Agent**: Identifies conflicting terms between the original RFP and addenda. *Example: Addendum 2 for Dallas ISD extended the submission deadline from June 27, 2024 to July 9, 2024 at 2:00 PM CST.*
- **Validator / Critic Agent**: Enforces strict anti-hallucination rules. Requires non-null fields to have file and page citations; marks unmentioned fields as `null` with `"Not found in documents"`; and triggers a targeted retry feedback loop for rejected fields.
- **Q&A Report Agent**: Synthesizes clear, executive-ready answers with inline source citations and key takeaways.

---

## 📊 Evaluation & Benchmark Results

Retrieval evaluation performed across **16 curated ground-truth questions** spanning both Bid1 and Bid2:

| Retrieval Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense Vector Only (Gemini Embeddings)** | 75.0% | 93.8% | 93.8% | 0.844 | 2,166.6 ms |
| **BM25 Keyword Search Only** | 68.8% | 87.5% | 87.5% | 0.781 | **1.9 ms** |
| **Hybrid Search (Vector + BM25 RRF)** | **87.5%** | **93.8%** | **93.8%** | **0.896** | 1,231.2 ms |
| **Hybrid Search + LLM Re-ranking** | **87.5%** | **93.8%** | **93.8%** | **0.906** | 16,402.2 ms |

### Key Benchmark Insights
1. **Exact Identifier Precision**: BM25 achieved ~1.9ms latency and matched exact part numbers (`WD22TB4`, `CC7802`) where pure dense embeddings occasionally struggled.
2. **Semantic Understanding**: Dense vector search retrieved conceptual terms (e.g., "reimbursement and retainage" $\rightarrow$ Prompt Payment Act) where exact words differed.
3. **Hybrid RRF Superiority**: Combining both techniques lifted Recall@1 from 75.0% to **87.5%** and MRR to **0.896**.
4. **LLM Re-ranking**: Achieved the highest MRR (**0.906**), pushing the exact citing paragraph to rank 1.

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Google Gemini API Key

### 2. Virtual Environment Setup
```powershell
# Clone or navigate to the workspace
cd "Assignment-Data-Statements (AI Engineer-Emplay Inc)"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
Create or verify `.env` file in the root directory:
```bash
GEMINI_API_KEY=AIzaSyDYSngB1HQ72ofK8KY95bGQtdizVGQPrwM
```

---

## 🖥️ Running the Application

### Option A: Interactive Streamlit Web UI (Recommended)
Launch the web dashboard:
```powershell
.\venv\Scripts\python.exe main.py --ui
# Or directly:
.\venv\Scripts\streamlit.exe run app.py
```
Open your browser at `http://localhost:8501`.

The Streamlit UI provides 6 dedicated modules:
1. **📊 Bid Intelligence & Field Inspector**: Inspect the 20 target fields, confidence scores, source citations, and the Addendum Overrides ledger.
2. **🤖 Multi-Agent Extraction Console**: Run the extraction pipeline on any bid folder with live agent trace logs.
3. **🔍 Hybrid RAG Search & Citations**: Interactive search with BM25/Vector/Hybrid modes and page citations.
4. **💬 AI Bid Analyst Q&A**: Ask natural language questions with page citations and pre-loaded evaluation questions.
5. **⚖️ Bid Comparison & Go/No-Go Decision Matrix**: Side-by-side comparison with interactive capability scoring.
6. **📈 Benchmark & Architecture**: View quantitative evaluation tables and system architecture.

---

### Option B: Command Line Interface (CLI)

```powershell
# 1. Run full extraction on Bid1 (Dallas ISD)
.\venv\Scripts\python.exe main.py --bid ./Bid1

# 2. Run full extraction on Bid2 (Maryland Treasurer)
.\venv\Scripts\python.exe main.py --bid ./Bid2

# 3. Ask a natural language question with citations
.\venv\Scripts\python.exe main.py --ask "What changed in Addendum 2 compared to the original RFP?"

# 4. Search via Hybrid RAG engine
.\venv\Scripts\python.exe main.py --search "docking station" --bid-filter Bid2

# 5. Run the retrieval benchmark
.\venv\Scripts\python.exe main.py --eval

# 6. Launch the FastAPI REST API Server
.\venv\Scripts\python.exe main.py --serve
```

---

### Option C: FastAPI REST API Server
Start the server:
```powershell
.\venv\Scripts\python.exe main.py --serve
```
Access interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

Key Endpoints:
- `POST /index`: Index a bid folder into the persistent vector store.
- `POST /search`: Execute Hybrid RAG search with metadata filtering and citations.
- `POST /ask`: Ask a cited natural-language question.
- `POST /extract`: Trigger multi-agent extraction for a bid.
- `GET /bids`: List all extracted bids and summary validation status.
- `GET /bids/{bid_id}`: Retrieve full 20-field structured JSON record.
- `GET /compare`: Side-by-side comparison of Bid1 vs Bid2.

---

### Option D: Running Unit Tests
Execute the pytest suite:
```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```
All 16 unit tests cover:
- PDF PyMuPDF extraction, HTML parsing, document classification, text cleaning.
- Section-aware chunking and Markdown table preservation.
- BM25 Okapi tokenization and alphanumeric identifier preservation.
- Vector store Cosine similarity search and metadata filtering.
- Validator critic anti-hallucination guardrails and citation enforcement.

---

## 📑 20 Target Fields Extraction Summary

| # | Field Name | Bid 1: Dallas ISD (`Bid1`) | Bid 2: MD State Treasurer (`Bid2`) | Specialist Agent |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Bid ID** | `JA-207652` | `eMMA # BPM044238 / PORFP # 060B5400007-2024` | Dates & Logistics |
| **2** | **Bid Title** | Student and Staff Computing Devices | Dell Laptops w/ Extended Warranty | Dates & Logistics |
| **3** | **Issuing Agency** | Dallas Independent School District | Maryland State Treasurer's Office | Commercial & Legal |
| **4** | **Due Date** | **July 9, 2024 at 2:00 PM CST** *(Extended via Addendum 2)* | **06/10/2024 02:00 PM EDT** | Dates & Logistics |
| **5** | **Pre Bid Meeting** | June 10, 2024, 2:00 PM CST (via TEAMS) | *Not found in documents* | Dates & Logistics |
| **6** | **Delivery Date** | Orders placed as needed; central warehouse | within 45 days of Award | Dates & Logistics |
| **7** | **Installation** | Not required (deployment managed by district) | *Not found in documents* | Dates & Logistics |
| **8** | **Term of Bid** | 3 years initial + 2x1-yr renewals (max 5 yrs) | *Not found in documents* | Dates & Logistics |
| **9** | **Bid Submission Type** | Electronic (iSupplier) or Sealed Hardcopy | eMMA e-Procurement system only | Dates & Logistics |
| **10** | **Bid Bond** | Not required | *Not found in documents* | Commercial & Legal |
| **11** | **Payment Terms** | Net 30 days; Texas Prompt Payment Act | Standard State of Maryland CATS+ terms | Commercial & Legal |
| **12** | **Required Affidavits** | Form 1295, Conflict of Interest, Debarment | Mercury Affidavit (mandatory) | Commercial & Legal |
| **13** | **Authorized Reseller** | Manufacturer Authorized Partner required | Dell Certified Reseller required | Commercial & Legal |
| **14** | **Purchasing Coop** | Allowed (EPCNT, State of Texas DIR, BuyBoard) | CATS+ Secondary Competition (#060B5400007) | Commercial & Legal |
| **15** | **Procurement Contact** | Buyer: Dallas ISD Purchasing Dept | Procurement Officer: MD State Treasurer | Commercial & Legal |
| **16** | **Scope of Work** | District-wide student & staff computing hardware | 25 Dell Latitude 5550 laptops + 25 WD22TB4 docks | Product Specs & Scope |
| **17** | **Required Items/Qty** | Student Chromebooks, Laptops, Desktops, Monitors | 25 Laptops (`210-BLYZ`) + 25 Docks (`WD22TB4`) | Product Specs & Scope |
| **18** | **Product Specs** | Full Tier 1/2 specs, Intel Core i5/i7, 8/16GB | Intel Core Ultra 7 155H, 16GB RAM, 512GB SSD | Product Specs & Scope |
| **19** | **Warranty Terms** | 1-yr Chromebook / 3-yr Laptop OEM, 5-day SLA | 3-yr Dell Extended Limited Hardware Warranty | Product Specs & Scope |
| **20** | **Liquidated Damages** | Standard Dallas ISD Master Agreement terms | *Not found in documents* | Product Specs & Scope |

---

## 💬 Sample Cited Q&A Log (Part C & D)

Below is an excerpt of answers generated by the **Q&A Report Agent** (saved in `output/sample_qa_log.json`):

### Q1: What is the submission deadline for Bid1 after all addendums?
> **Answer**: The submission deadline for Bid1, after all addendums, is **July 9, 2024, at 2:00 PM CST**. This new due date was established by **Addendum No. 2**, which explicitly extended the RFP's due date `[Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf, Page 1]`. Addendum No. 1 had previously stated that Dallas ISD did not anticipate extending the submission deadline `[Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf, Page 1]`.
> - **Citations**: `Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf` (Page 1), `Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf` (Page 1)
> - **Confidence**: 1.00 | **Latency**: 8,405 ms

### Q2: Which affidavits are required for the Dell laptop bid?
> **Answer**: For the Dell laptop bid, the Master Contractor must provide a **Mercury Affidavit** `[PORFP_-_Dell_Laptop_Final.pdf, Page 2]`. This affidavit requires the authorized representative to affirm whether the product(s) offered contain mercury, describe the product or component, and provide the amount of mercury contained under penalties of perjury `[Mercury_Affidavit.pdf, Page 1]`.
> - **Citations**: `PORFP_-_Dell_Laptop_Final.pdf` (Page 2), `Mercury_Affidavit.pdf` (Page 1)
> - **Confidence**: 1.00 | **Latency**: 9,371 ms

### Q3: Compare the warranty requirements of both bids.
> **Answer**: For **Bid 1 (Dallas ISD)**, minimum warranty requirements are **one year for student Chromebooks** and **three years for student and staff Windows laptops**, and all warranties must be official OEM manufacturer warranties `[Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf, Page 1, 3]`. Dallas ISD expects all warranty repairs to be performed at no cost within **five business days** `[JA-207652 Student and Staff Computing Devices FINAL.pdf, Page 4]`.
>
> For **Bid 2 (Maryland Treasurer)**, the requirement is a **Dell Limited Hardware Warranty Extended for 3 Years** for all machines purchased, starting on the Date of Delivery `[PORFP_-_Dell_Laptop_Final.pdf, Page 3]`.
> - **Citations**: `Addendum 1...pdf` (Pages 1, 3), `JA-207652...FINAL.pdf` (Page 4), `PORFP_-_Dell_Laptop_Final.pdf` (Page 3)
> - **Confidence**: 1.00 | **Latency**: 17,820 ms

### Q5: What changed in Addendum 2 compared to the original RFP?
> **Answer**: Addendum 2 modified the original RFP by **extending the proposal submission due date to July 9, 2024, at 2:00 PM CST** `[Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf, Page 1]`.
> - **Citations**: `Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf` (Page 1)
> - **Confidence**: 1.00 | **Latency**: 3,530 ms

---

## 🛡️ Anti-Hallucination & Robustness Safeguards

1. **Mandatory Positive Citations**: The Validator Agent rejects any non-null field value that lacks source document and page number citations.
2. **Explicit Null Representation**: Unmentioned terms (such as Pre-Bid meeting or Term of Bid in Bid2) are never fabricated; they are explicitly represented as `null` with `"Not found in documents"`.
3. **Addendum Audit Trail**: Overrides are tracked in a dedicated ledger recording the original value, the new value, the specific addendum number, and the source page.
4. **Targeted Feedback Retry Loop**: When the Validator flags a format issue or citation absence, the Orchestrator executes a targeted sub-query search with an increased `top_k` and adjusted domain prompt before finalizing the output.
5. **Generalization to New Bids**: The system contains zero hardcoded bid-specific logic in its core extraction pipeline; pointing the pipeline at any new folder `./BidX` automatically triggers classification, indexing, multi-agent parsing, and validation.

---

## 📁 Repository Directory Structure

```text
.
├── .env                                  # Gemini API Key configuration
├── requirements.txt                      # Python dependencies
├── pytest.ini                           # Pytest configuration
├── main.py                               # Unified CLI launcher (--ui, --serve, --bid, --ask, --eval)
├── app.py                                # Streamlit Web UI Application
├── README.md                             # Architectural documentation & project overview
├── Bid1/                                 # Dallas ISD RFP, Addenda, and HTML files
├── Bid2/                                 # Maryland State Treasurer PORFP, Specs, Affidavits
├── ingestion/                            # Part A: Ingestion & Parsing
│   ├── parser.py                         # PyMuPDF, HTML & Table parser
│   ├── chunker.py                        # Section/table-aware chunker with breadcrumbs
│   └── __init__.py
├── search/                               # Part B: RAG Search Engine
│   ├── embeddings.py                     # Gemini embeddings with batching & caching
│   ├── vector_store.py                   # Persistent numpy cosine similarity vector store
│   ├── bm25_index.py                     # BM25 keyword index for exact identifiers
│   ├── reranker.py                       # Query expander & LLM candidate reranker
│   ├── hybrid_search.py                  # Reciprocal Rank Fusion (RRF) search engine
│   └── __init__.py
├── agents/                               # Part C: Multi-Agent System
│   ├── state.py                          # Pydantic v2 schemas for fields, citations, and traces
│   ├── specialists.py                    # Dates, Commercial, and Product Specialist agents
│   ├── addendum_agent.py                 # Addendum reconciliation and override detection
│   ├── validator_agent.py                # Critic agent, anti-hallucination & feedback loop
│   ├── qa_agent.py                       # Cited Q&A agent for natural language questions
│   ├── orchestrator.py                   # Master coordinator & parallel agent execution
│   └── __init__.py
├── api/                                  # REST API Interface
│   ├── routes.py                         # FastAPI endpoints (/search, /ask, /extract, /bids)
│   └── __init__.py
├── eval/                                 # Evaluation & Benchmarking
│   ├── dataset.py                        # 16 Ground-truth questions & citations
│   ├── benchmark.py                      # Retrieval benchmark (Recall@k, MRR, Latency)
│   └── generate_qa_log.py                # Generates sample_qa_log.json for 12 questions
├── tests/                                # Unit Test Suite
│   ├── test_parsing.py                   # PDF & HTML parser unit tests
│   ├── test_chunking.py                  # Chunker & table preservation tests
│   ├── test_search.py                    # BM25 & Vector store tests
│   └── test_agents.py                    # Pydantic state & Validator critic tests
├── index_store/                          # Persistent RAG indices
│   ├── vector_embeddings.npy             # 302 vector embeddings (shape: 302, 3072)
│   ├── vector_metadata.json              # Document chunks & metadata
│   ├── bm25_data.json                    # BM25 tokenized corpus
│   └── embeddings_cache.json             # Disk cache for embedding calls
└── output/                               # Generated Deliverables
    ├── bid1_extracted.json               # 20 target fields for Dallas ISD
    ├── bid2_extracted.json               # 20 target fields for MD State Treasurer
    ├── sample_qa_log.json                # 12 diverse cited questions and answers
    ├── evaluation_report.md              # Quantitative retrieval benchmark report
    └── agent_trace.json                  # Multi-agent latency and execution trace
```
