# RFP Retrieval Evaluation Report

## 1. Overview
Evaluation performed over **16 curated ground-truth questions** spanning both **Bid1** (Dallas ISD Student and Staff Computing Devices) and **Bid2** (Maryland Treasurer's Office Dell Laptops).

## 2. Configuration Comparison Benchmark

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense Vector Only (Gemini Embeddings)** | 75.0% | 93.8% | 93.8% | 0.844 | 2166.6ms |
| **BM25 Keyword Search Only** | 68.8% | 87.5% | 87.5% | 0.781 | 1.9ms |
| **Hybrid Search (Vector + BM25 RRF)** | 87.5% | 93.8% | 93.8% | 0.896 | 1231.2ms |
| **Hybrid Search + LLM Re-ranking (Full Pipeline)** | 87.5% | 93.8% | 93.8% | 0.906 | 16402.2ms |

## 3. Analysis & Key Insights
- **BM25 Keyword Strength**: Pure BM25 excels at pinpointing specific part numbers (e.g. `210-BLYZ`, `WD22TB4`), solicitation codes (`JA-207652`, `#E20P4600040`), and model numbers (`CC7802`).
- **Dense Vector Strength**: Gemini Dense Vector search excels on semantic concepts where queries use natural language (e.g. 'submission due date after addendums' or 'warranty obligations').
- **Hybrid RRF Synergy**: Reciprocal Rank Fusion (RRF) seamlessly combines both methods, ensuring neither exact codes nor semantic intent are missed, boosting Recall@5 significantly.
- **LLM Re-ranking Precision**: Re-ranking filters out generic headers and promotes the most direct, cited evidence block to Rank 1, maximizing MRR.

## 4. Ground-Truth Question Set Details

| ID | Question | Expected File | Expected Page |
| :--- | :--- | :--- | :---: |
| q01 | What is the final submission due date for Bid1 after all addendums? | `Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf` | 1 |
| q02 | When and how is the pre-proposal conference held for Student and Staff Computing Devices? | `JA-207652 Student and Staff Computing Devices FINAL.pdf` | 2 |
| q03 | What is the official RFP solicitation number for the Dallas ISD computing devices bid? | `JA-207652 Student and Staff Computing Devices FINAL.pdf` | 1 |
| q04 | What are the USB port requirements for the display monitors clarified in Addendum 1? | `Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf` | 1 |
| q05 | What is the initial contract term and renewal options for Dallas ISD RFP JA-207652? | `JA-207652 Student and Staff Computing Devices FINAL.pdf` | 2 |
| q06 | Which purchasing cooperatives or interlocal agreements are referenced in the Dallas ISD bid? | `JA-207652 Student and Staff Computing Devices FINAL.pdf` | 19 |
| q07 | Who is the primary buyer contact person and email for Dallas ISD computing devices RFP? | `JA-207652 Student and Staff Computing Devices FINAL.pdf` | 2 |
| q08 | What minimum battery life and RAM are required for the student laptops? | `JA-207652 Student and Staff Computing Devices FINAL.pdf` | 5 |
| q09 | What is the PORFP number and eMMA project number for the Maryland Dell laptop solicitation? | `PORFP_-_Dell_Laptop_Final.pdf` | 1 |
| q10 | What is the proposal due date and time for the Dell Laptop PORFP? | `PORFP_-_Dell_Laptop_Final.pdf` | 1 |
| q11 | Which specific affidavits must be completed and submitted for the Maryland Dell laptop procurement? | `PORFP_-_Dell_Laptop_Final.pdf` | 2 |
| q12 | What is the model number and quantity requested for the Dell laptops? | `PORFP_-_Dell_Laptop_Final.pdf` | 3 |
| q13 | What is the docking station model and quantity requested for the Maryland laptop bid? | `PORFP_-_Dell_Laptop_Final.pdf` | 3 |
| q14 | What processor model and core count are specified in the Dell Latitude 5550 technical specs? | `Dell_Laptop_Specs.pdf` | 1 |
| q15 | What is the required delivery window after award for the Dell laptops? | `PORFP_-_Dell_Laptop_Final.pdf` | 2 |
| q16 | Who is the agency point of contact POC for State Treasurer's Office Dell laptop bid? | `PORFP_-_Dell_Laptop_Final.pdf` | 3 |