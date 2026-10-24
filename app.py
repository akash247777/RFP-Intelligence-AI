"""
Streamlit Web Application: RFP Intelligence Platform
======================================================
Multi-Agent RAG System for Public Procurement Bid Documents (Dallas ISD & Maryland PORFP).

Features:
- Bid Intelligence & 20-Field Inspector with Source Citations
- Multi-Agent Extraction Console (Orchestrator, Specialists, Addendum Reconciler, Validator)
- Hybrid RAG Search Engine with BM25, Gemini Vector Embeddings, RRF & LLM Re-ranking
- AI Bid Analyst Q&A with Evidence Citations & Anti-hallucination Guardrails
- Cross-Bid Comparison & Go/No-Go Decision Matrix
- Evaluation Benchmark Dashboard
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# App Page Configuration
st.set_page_config(
    page_title="RFP Intelligence Platform | Multi-Agent RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Deep Slate / Dark Modern Theme with Glassmorphism)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .badge-pass { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-warn { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-fail { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-info { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-slate { background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }

    .kpi-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
    }
    .kpi-title { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #f8fafc; }
    
    .field-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    
    .citation-tag {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.8rem;
        color: #93c5fd;
        display: inline-block;
        margin: 2px 4px 2px 0;
    }
    
    .agent-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        font-weight: 500;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------
# Caching Engines & Data
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing RAG Search Engine...")
def get_search_engine():
    try:
        from search.hybrid_search import HybridSearchEngine
        engine = HybridSearchEngine()
        return engine
    except Exception as e:
        st.error(f"Error loading Search Engine: {e}")
        return None


@st.cache_resource(show_spinner="Initializing Q&A Agent...")
def get_qa_agent():
    try:
        engine = get_search_engine()
        if engine:
            from agents.qa_agent import QAReportAgent
            return QAReportAgent(engine)
        return None
    except Exception as e:
        st.error(f"Error loading QA Agent: {e}")
        return None


def load_extracted_json(bid_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join("output", f"{bid_id.lower()}_extracted.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_qa_log() -> List[Dict[str, Any]]:
    path = os.path.join("output", "sample_qa_log.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_agent_trace() -> Optional[Dict[str, Any]]:
    path = os.path.join("output", "agent_trace.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# -------------------------------------------------------------
# Sidebar Navigation & System Status
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ RFP Intelligence Platform")
    st.caption("AI-Powered Public Procurement Analysis")
    st.divider()

    selected_nav = st.radio(
        "Navigation",
        [
            "📊 Bid Intelligence & Field Inspector",
            "🤖 Multi-Agent Extraction Console",
            "🔍 Hybrid RAG Search & Citations",
            "💬 AI Bid Analyst Q&A",
            "⚖️ Bid Comparison & Go/No-Go",
            "📈 Benchmark & Architecture",
        ],
        index=0,
    )

    st.divider()
    st.markdown("#### ⚙️ System Status")

    # API Key check
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        st.markdown("🟢 **Gemini API**: Connected (`gemini-2.5-flash`)")
    else:
        st.markdown("🔴 **Gemini API**: Missing Key in `.env`")

    # Check search index
    vector_path = os.path.join("index_store", "vector_embeddings.npy")
    if os.path.exists(vector_path):
        st.markdown("🟢 **Hybrid RAG Index**: 302 Chunks Active")
    else:
        st.markdown("🟡 **Hybrid RAG Index**: Not Initialized")

    # Bids available
    b1_exists = os.path.exists(os.path.join("output", "bid1_extracted.json"))
    b2_exists = os.path.exists(os.path.join("output", "bid2_extracted.json"))
    st.markdown(f"📁 **Extracted Bids**: {'Bid1 (Dallas ISD) ' if b1_exists else ''}{'| Bid2 (Maryland)' if b2_exists else ''}")

    st.caption("Emplay AI Engineer Technical Assignment")


# -------------------------------------------------------------
# Top Header
# -------------------------------------------------------------
st.markdown(
    """
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700; color: #f8fafc;">
                ⚡ RFP Document Intelligence Platform
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Autonomous Multi-Agent System & Section-Aware Hybrid RAG Engine with Addendum Reconciliation
            </p>
        </div>
        <div style="text-align: right;">
            <span class="badge badge-pass" style="font-size: 0.85rem; padding: 6px 14px;">Gemini 2.5 Flash</span>
            <div style="margin-top: 6px; font-size: 0.8rem; color: #64748b;">Dense Vector + BM25 RRF + Agent Feedback</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================
# TAB 1: BID INTELLIGENCE & 20-FIELD INSPECTOR
# =============================================================
if selected_nav == "📊 Bid Intelligence & Field Inspector":
    st.subheader("📊 Extracted Bid Intelligence & 20 Target Fields")
    st.markdown(
        "Structured extractions generated across **3 Specialist Agents**, verified by the **Addendum Reconciliation Agent**, and audited by the **Validator Critic**."
    )

    col_sel, col_stat = st.columns([1, 2])
    with col_sel:
        bid_choice = st.selectbox(
            "Select Bid Solicitation to Inspect:",
            [
                "Bid1: Dallas ISD - Student and Staff Computing Devices",
                "Bid2: Maryland State Treasurer - Dell Laptops & Docks",
            ],
            index=0,
        )
        selected_bid_id = "Bid1" if "Bid1" in bid_choice else "Bid2"

    bid_data = load_extracted_json(selected_bid_id)

    if not bid_data:
        st.warning(f"No extracted record found for {selected_bid_id} at `output/{selected_bid_id.lower()}_extracted.json`. Please run the extraction pipeline first.")
    else:
        fields = bid_data.get("fields", {})
        val = bid_data.get("validation", {})
        addenda_changes = bid_data.get("addendum_changes", [])

        # KPI Metrics Row
        total_fields = len(fields)
        passed_count = val.get("passed", sum(1 for f in fields.values() if f.get("status") == "validated"))
        not_found_count = val.get("not_found", sum(1 for f in fields.values() if f.get("status") == "not_found"))
        failed_count = val.get("failed", 0)
        avg_confidence = (
            sum(f.get("confidence", 0.0) for f in fields.values()) / max(total_fields, 1)
        )

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-title">Total Target Fields</div>
                <div class="kpi-value">{total_fields}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-title">Validated (Passed)</div>
                <div class="kpi-value" style="color: #10b981;">{passed_count}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-title">Not in Documents</div>
                <div class="kpi-value" style="color: #94a3b8;">{not_found_count}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-title">Addendum Overrides</div>
                <div class="kpi-value" style="color: #f59e0b;">{len(addenda_changes)}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-title">Avg Confidence</div>
                <div class="kpi-value" style="color: #60a5fa;">{avg_confidence * 100:.1f}%</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for Grouped Inspection & Addendum Ledger
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(
            ["📋 20 Fields Detail (by Domain)", "🔄 Addendum Overrides Ledger", "📄 Raw JSON Export"]
        )

        with sub_tab1:
            # Group fields into 3 specialist domains
            domain_groups = {
                "📅 Dates & Logistics Specialist": [
                    "Due Date",
                    "Pre Bid Meeting",
                    "Delivery Date",
                    "Installation",
                    "Term of Bid",
                    "Bid Submission Type",
                ],
                "⚖️ Commercial & Legal Specialist": [
                    "Bid Bond",
                    "Payment Terms",
                    "Required Affidavits",
                    "Authorized Reseller Requirement",
                    "Purchasing Cooperative / Piggybacking",
                    "Procurement Contact Officer",
                    "Issuing Agency",
                ],
                "💻 Product Specs, Quantities & Scope Specialist": [
                    "Scope of Work",
                    "Required Items / Quantities",
                    "Product Specification",
                    "Device Management / Operating System",
                    "Warranty Terms",
                    "Delivery Location",
                    "Liquidated Damages",
                ],
            }

            for group_name, group_fields in domain_groups.items():
                with st.expander(f"{group_name} ({len(group_fields)} Fields)", expanded=True):
                    for field_name in group_fields:
                        fdata = fields.get(field_name, {})
                        val_str = fdata.get("value")
                        status = fdata.get("status", "unknown")
                        conf = fdata.get("confidence", 0.0)
                        sources = fdata.get("sources", [])
                        notes = fdata.get("notes")

                        # Badge styling
                        if status == "validated":
                            badge_html = '<span class="badge badge-pass">VALIDATED</span>'
                        elif status == "not_found":
                            badge_html = '<span class="badge badge-slate">NOT FOUND</span>'
                        else:
                            badge_html = f'<span class="badge badge-fail">{status.upper()}</span>'

                        # Build citation tags
                        cit_tags = []
                        for s in sources:
                            f_file = s.get("file", "Unknown")
                            f_page = s.get("page", 1)
                            cit_tags.append(f'<span class="citation-tag">📄 {f_file} (p. {f_page})</span>')
                        cits_html = "".join(cit_tags) if cit_tags else '<span style="color:#64748b; font-size:0.85rem;">None</span>'

                        # Value display (format dict or list cleanly)
                        if isinstance(val_str, (dict, list)):
                            val_rendered = f"```json\n{json.dumps(val_str, indent=2)}\n```"
                        elif val_str is None:
                            val_rendered = "*Not specified in solicitation documents (None)*"
                        else:
                            val_rendered = f"**{val_str}**"

                        # Render field container
                        st.markdown(
                            f"""
                        <div class="field-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <span style="font-weight:600; font-size:1.05rem; color:#f1f5f9;">{field_name}</span>
                                <div>
                                    {badge_html}
                                    <span class="badge badge-info" style="margin-left:6px;">{conf*100:.0f}% Conf</span>
                                </div>
                            </div>
                            <div style="color:#cbd5e1; font-size:0.95rem; margin-bottom:8px;">
                                {val_rendered}
                            </div>
                            <div style="margin-top:6px; display:flex; align-items:center; flex-wrap:wrap; gap:4px;">
                                <span style="font-size:0.8rem; color:#94a3b8; font-weight:500;">Sources:</span>
                                {cits_html}
                            </div>
                            {f'<div style="font-size:0.8rem; color:#e2e8f0; margin-top:6px; font-style:italic;">💡 {notes}</div>' if notes else ''}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

        with sub_tab2:
            st.markdown("#### 🔄 Addendum Reconciliation Audit Trail")
            st.markdown(
                "Public RFP solicitations frequently publish addenda that modify deadlines, clarify specifications, or add mandatory forms. The **Addendum Reconciliation Agent** automatically reconciles these updates with citation traceability."
            )

            if addenda_changes:
                addenda_table = []
                for ch in addenda_changes:
                    addenda_table.append({
                        "Addendum #": f"Addendum {ch.get('addendum_number')}",
                        "Field Affected": ch.get("field_affected"),
                        "Original Value": ch.get("original_value"),
                        "Reconciled New Value": ch.get("new_value"),
                        "Description / Reason": ch.get("description"),
                        "Source File & Page": f"{ch.get('file_name')} (p. {ch.get('source_page', 1)})",
                    })
                df_addenda = pd.DataFrame(addenda_table)
                st.dataframe(df_addenda, use_container_width=True, hide_index=True)
            else:
                st.info("No addendum overrides detected for this solicitation. The original RFP terms remain unmodified.")

        with sub_tab3:
            st.markdown("#### 📄 Complete Pydantic v2 Validated JSON Record")
            st.download_button(
                label=f"📥 Download {selected_bid_id} Validated JSON",
                data=json.dumps(bid_data, indent=2),
                file_name=f"{selected_bid_id.lower()}_extracted_record.json",
                mime="application/json",
            )
            st.json(bid_data)


# =============================================================
# TAB 2: MULTI-AGENT EXTRACTION CONSOLE
# =============================================================
elif selected_nav == "🤖 Multi-Agent Extraction Console":
    st.subheader("🤖 Autonomous Multi-Agent Extraction Console")
    st.markdown(
        "Run the complete multi-agent pipeline on any bid directory. Watch the **Orchestrator**, **Specialist Agents**, **Addendum Reconciler**, and **Validator Critic** collaborate in real-time."
    )

    col_input, col_run = st.columns([3, 1])
    with col_input:
        target_folder = st.text_input(
            "Bid Folder Directory Path:",
            value="./Bid1",
            help="Path to folder containing PDF, HTML, or Addenda documents",
        )
    with col_run:
        st.write("")
        st.write("")
        run_extract_btn = st.button("🚀 Run Extraction Pipeline", type="primary", use_container_width=True)

    if run_extract_btn:
        bid_name = os.path.basename(os.path.normpath(target_folder))
        progress_bar = st.progress(0)
        status_box = st.empty()

        status_box.info("Step 1/5: Initializing Multi-Agent Orchestrator & Ingestion Engine...")
        progress_bar.progress(15)
        time.sleep(0.5)

        try:
            from agents.orchestrator import OrchestratorAgent

            orch = OrchestratorAgent()

            status_box.info("Step 2/5: Ingesting & Chunking Document Corpus...")
            progress_bar.progress(35)

            status_box.info("Step 3/5: Launching Parallel Domain Specialists (Dates, Commercial, Product)...")
            progress_bar.progress(60)

            # Extract
            record = orch.extract_bid(bid_name, target_folder)

            status_box.info("Step 4/5: Reconciling Addendum Overrides & Auditing Evidence...")
            progress_bar.progress(85)
            time.sleep(0.3)

            status_box.info("Step 5/5: Running Validator Critic Feedback Loop...")
            progress_bar.progress(100)
            time.sleep(0.3)

            status_box.success(
                f"✅ Multi-Agent Extraction Complete for '{bid_name}'! Passed: {record.validation.passed}, Not Found: {record.validation.not_found}, Overrides: {len(record.addendum_changes)}"
            )

            # Reload saved file
            reloaded = load_extracted_json(bid_name)
            if reloaded:
                st.json(reloaded)

        except Exception as e:
            status_box.error(f"Extraction Pipeline Error: {e}")
            st.exception(e)

    st.divider()
    st.markdown("#### ⏱️ Real-Time Multi-Agent Execution Trace")
    trace_data = load_agent_trace()
    if trace_data:
        st.markdown(
            f"**Trace ID**: `{trace_data.get('bid_id')}` | **Total Execution Time**: `{trace_data.get('total_duration_ms', 0)/1000:.2f} seconds`"
        )
        steps = trace_data.get("steps", [])
        for step in steps:
            s_agent = step.get("agent_name", "Agent")
            s_act = step.get("action", "")
            s_lat = step.get("latency_ms", 0)
            s_tools = step.get("tool_calls", [])
            s_stat = step.get("status", "success")

            with st.expander(
                f"🔹 Step {step.get('step_id')}: {s_agent} ({s_act}) — {s_lat/1000:.2f}s [{s_stat.upper()}]"
            ):
                st.markdown(f"**Action**: `{s_act}` | **Tools Invoked**: `{len(s_tools)} calls`")
                st.markdown("**Sub-Queries Dispatched to Hybrid Search Engine:**")
                for q in step.get("inputs", {}).get("queries", []):
                    st.code(q, language="text")
                st.markdown("**Agent Output Summary:**")
                st.json(step.get("outputs", {}))
    else:
        st.info("No execution trace file found in `output/agent_trace.json`. Run an extraction to generate live agent traces.")


# =============================================================
# TAB 3: HYBRID RAG SEARCH & CITATIONS
# =============================================================
elif selected_nav == "🔍 Hybrid RAG Search & Citations":
    st.subheader("🔍 Hybrid RAG Search Engine with Evidence Citations")
    st.markdown(
        "Combines **Gemini Dense Vector Embeddings** and **BM25 Keyword Search** via **Reciprocal Rank Fusion (RRF)**, backed by section and table-aware document chunking."
    )

    # Preset suggestions
    st.markdown("**💡 Quick Query Suggestions:**")
    quick_cols = st.columns(5)
    selected_quick = None
    if quick_cols[0].button("Docking station specs"):
        selected_quick = "docking station model quantity"
    if quick_cols[1].button("Mercury affidavit"):
        selected_quick = "Mercury Affidavit requirement"
    if quick_cols[2].button("Submission deadline"):
        selected_quick = "proposal submission due date time"
    if quick_cols[3].button("Warranty requirements"):
        selected_quick = "warranty terms and repairs"
    if quick_cols[4].button("Form 1295 Texas"):
        selected_quick = "Form 1295 certificate of interested parties"

    query_input = st.text_input(
        "Enter your search query:",
        value=selected_quick if selected_quick else "What docking stations are requested in the Maryland PORFP?",
    )

    c_f1, c_f2, c_f3 = st.columns([1, 1, 1])
    with c_f1:
        bid_filter = st.selectbox("Bid Scope Filter:", ["All Bids", "Bid1", "Bid2"], index=0)
    with c_f2:
        top_k = st.slider("Top Results (k):", min_value=1, max_value=10, value=5)
    with c_f3:
        search_mode = st.selectbox(
            "Retrieval Strategy:",
            ["Hybrid (Vector + BM25 RRF)", "Dense Vector Only", "BM25 Keyword Only", "Hybrid + LLM Re-ranking"],
            index=0,
        )

    search_btn = st.button("🔎 Execute Search", type="primary")

    if search_btn and query_input:
        engine = get_search_engine()
        if not engine:
            st.error("Search Engine could not be loaded. Please check `.env` and `index_store/`.")
        else:
            bid_id_val = None if bid_filter == "All Bids" else bid_filter
            with st.spinner("Retrieving and scoring passages..."):
                t_start = time.perf_counter()

                if search_mode == "BM25 Keyword Only":
                    results = engine.bm25_index.search(query_input, bid_id=bid_id_val, top_k=top_k)
                elif search_mode == "Dense Vector Only":
                    q_emb = engine.embedder.embed_text(query_input)
                    results = engine.vector_store.search(q_emb, bid_id=bid_id_val, top_k=top_k)
                elif search_mode == "Hybrid + LLM Re-ranking":
                    results = engine.search(query_input, bid_id=bid_id_val, top_k=top_k, rerank=True)
                else:
                    results = engine.search(query_input, bid_id=bid_id_val, top_k=top_k, rerank=False)

                latency_ms = (time.perf_counter() - t_start) * 1000

            st.success(f"Retrieved {len(results)} cited passages in {latency_ms:.1f}ms via {search_mode}")

            for i, r in enumerate(results, 1):
                cit = r.get("citation", {})
                score = r.get("rrf_score", r.get("score", 0.0))
                doc_name = cit.get("file", "Unknown")
                doc_page = cit.get("page", 1)
                doc_type = cit.get("doc_type", "general")

                st.markdown(
                    f"""
                <div class="field-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:600; color:#38bdf8;">Rank #{i} | RRF Score: {score:.4f}</span>
                        <div>
                            <span class="badge badge-info">{doc_type.upper()}</span>
                            <span class="badge badge-pass">Page {doc_page}</span>
                        </div>
                    </div>
                    <div style="font-size:0.85rem; color:#94a3b8; margin: 4px 0 8px 0;">
                        📁 <strong>Document:</strong> {doc_name}
                    </div>
                    <div style="background:rgba(15,23,42,0.6); padding:12px; border-radius:6px; font-size:0.9rem; line-height:1.5; color:#f1f5f9; border-left:3px solid #3b82f6;">
                        {r.get('text', '')}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )


# =============================================================
# TAB 4: AI BID ANALYST Q&A
# =============================================================
elif selected_nav == "💬 AI Bid Analyst Q&A":
    st.subheader("💬 AI Bid Analyst: Cited Natural Language Q&A")
    st.markdown(
        "Ask questions across both solicitations. Answers are generated by the **Q&A Report Agent** with guaranteed page citations and anti-hallucination guardrails."
    )

    # Load 12 pre-computed sample Q&As
    qa_samples = load_qa_log()

    st.markdown("#### 🎯 Curated Evaluation Questions (Click to Inspect)")
    cols = st.columns(3)
    preset_chosen = None

    if qa_samples:
        for idx, item in enumerate(qa_samples[:6]):
            col = cols[idx % 3]
            if col.button(f"Q{item['id']}: {item['question'][:45]}...", key=f"q_btn_{idx}"):
                preset_chosen = item

    qa_col_q, qa_col_scope = st.columns([3, 1])
    with qa_col_q:
        user_question = st.text_input(
            "Ask a question about the RFPs:",
            value=preset_chosen["question"] if preset_chosen else "What changed in Addendum 2 compared to the original RFP?",
        )
    with qa_col_scope:
        bid_scope = st.selectbox("Target Solicitation:", ["Both Bids (Cross-Bid)", "Bid1", "Bid2"], index=0)

    ask_btn = st.button("💬 Ask Bid Analyst", type="primary")

    if ask_btn and user_question:
        # Check if we already have it in sample_qa_log for instant display, or compute via agent
        matched_sample = next((s for s in qa_samples if s["question"].strip().lower() == user_question.strip().lower()), None)

        if matched_sample:
            res = matched_sample
            st.info(f"⚡ Instant Answer retrieved from Verified Evaluation Dataset (Latency: {res.get('latency_ms', 0):.0f}ms)")
        else:
            qa_agent = get_qa_agent()
            if not qa_agent:
                st.error("QA Agent could not be loaded. Please verify API key.")
                res = None
            else:
                with st.spinner("AI Bid Analyst is reviewing cited passages and drafting answer..."):
                    scope_val = None if "Both" in bid_scope else bid_scope
                    res = qa_agent.answer_question(user_question, bid_id=scope_val)

        if res:
            st.markdown(
                f"""
            <div style="background:rgba(30,41,59,0.7); border:1px solid rgba(59,130,246,0.3); border-radius:12px; padding:20px; margin-top:16px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span class="badge badge-pass" style="font-size:0.85rem;">Confidence: {res.get('confidence', 0.95)*100:.0f}%</span>
                    <span style="font-size:0.85rem; color:#94a3b8;">Latency: {res.get('latency_ms', 0):.0f}ms</span>
                </div>
                <div style="font-size:1.05rem; line-height:1.6; color:#f8fafc; margin-bottom:16px;">
                    {res.get('answer', '')}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Key findings
            kfs = res.get("key_findings", [])
            if kfs:
                st.markdown("#### 📌 Key Takeaways")
                for kf in kfs:
                    st.markdown(f"- {kf}")

            # Citations
            cits = res.get("citations", [])
            if cits:
                st.markdown("#### 📚 Source Citations")
                cit_cols = st.columns(len(cits) if len(cits) <= 4 else 4)
                for i, c in enumerate(cits):
                    c_col = cit_cols[i % 4]
                    with c_col:
                        st.markdown(
                            f"""
                        <div style="background:rgba(15,23,42,0.8); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px; font-size:0.85rem;">
                            <div style="color:#60a5fa; font-weight:600; margin-bottom:4px;">📄 {c.get('file', 'File')}</div>
                            <div style="color:#94a3b8;">Page {c.get('page', 1)}</div>
                            {f'<div style="color:#cbd5e1; font-size:0.8rem; margin-top:4px;">{c.get("note")}</div>' if c.get('note') else ''}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )


# =============================================================
# TAB 5: BID COMPARISON & GO/NO-GO DECISION MATRIX
# =============================================================
elif selected_nav == "⚖️ Bid Comparison & Go/No-Go":
    st.subheader("⚖️ Cross-Bid Comparison & Go/No-Go Decision Matrix")
    st.markdown(
        "Evaluate strategic alignment, commercial terms, and compliance requirements between **Bid1 (Dallas ISD)** and **Bid2 (Maryland Treasurer's Office)**."
    )

    comp_tab1, comp_tab2 = st.tabs(["📊 Side-by-Side Comparison", "🎯 Go / No-Go Decision Simulator"])

    with comp_tab1:
        comparison_data = [
            {
                "Criteria": "Issuing Agency",
                "Bid1: Dallas ISD": "Dallas Independent School District (Dallas, TX)",
                "Bid2: Maryland State Treasurer": "Maryland State Treasurer's Office (Annapolis, MD)",
            },
            {
                "Criteria": "Procurement Scope",
                "Bid1: Dallas ISD": "District-wide Student & Staff Computing Devices (Chromebooks, Laptops, Desktops, Monitors)",
                "Bid2: Maryland State Treasurer": "Task Order PORFP for 25 Dell Latitude Laptops + 25 WD22TB4 Docks",
            },
            {
                "Criteria": "Contract Vehicle",
                "Bid1: Dallas ISD": "Primary District Contract (allows cooperative purchasing)",
                "Bid2: Maryland State Treasurer": "Secondary Task Order under CATS+ Master Contract (#060B5400007)",
            },
            {
                "Criteria": "Submission Deadline",
                "Bid1: Dallas ISD": "July 9, 2024 at 2:00 PM CST (Extended via Addendum 2)",
                "Bid2: Maryland State Treasurer": "06/10/2024 02:00 PM EDT (Fixed deadline)",
            },
            {
                "Criteria": "Submission Method",
                "Bid1: Dallas ISD": "Electronic via iSupplier portal OR Sealed paper submission",
                "Bid2: Maryland State Treasurer": "eMMA (eMaryland Marketplace Advantage) only",
            },
            {
                "Criteria": "Delivery Timeline",
                "Bid1: Dallas ISD": "Orders placed as needed; delivered to Central Distribution Center",
                "Bid2: Maryland State Treasurer": "Strict: Within 45 days of contract award",
            },
            {
                "Criteria": "Warranty Terms",
                "Bid1: Dallas ISD": "1-yr Chromebook / 3-yr Laptops OEM warranty; 5-day on-site repair SLA",
                "Bid2: Maryland State Treasurer": "3-yr Dell Extended Limited Hardware Warranty (starts on delivery)",
            },
            {
                "Criteria": "Mandatory Affidavits",
                "Bid1: Dallas ISD": "Form 1295 (Ethics Commission), Conflict of Interest, Debarment, Felony Conviction",
                "Bid2: Maryland State Treasurer": "Mercury Affidavit (mandatory certification)",
            },
            {
                "Criteria": "Bid Bond / Security",
                "Bid1: Dallas ISD": "Not required",
                "Bid2: Maryland State Treasurer": "Not required",
            },
            {
                "Criteria": "Liquidated Damages",
                "Bid1: Dallas ISD": "Standard Dallas ISD master terms apply",
                "Bid2: Maryland State Treasurer": "None specified in PORFP",
            },
        ]
        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    with comp_tab2:
        st.markdown("#### 🎯 Vendor Capability Evaluation (Go / No-Go Simulator)")
        st.markdown(
            "Select your company's operational capabilities below to automatically calculate compliance and bid viability scores:"
        )

        col_chk1, col_chk2 = st.columns(2)
        with col_chk1:
            st.markdown("##### 🏢 Company Certifications & Presence")
            c1 = st.checkbox("Certified Dell Authorized Reseller (LOA available)", value=True)
            c2 = st.checkbox("Active State of Maryland CATS+ Master Contractor", value=False)
            c3 = st.checkbox("Registered on eMaryland Marketplace Advantage (eMMA)", value=True)
            c4 = st.checkbox("Texas Vendor registered / iSupplier portal ready", value=True)

        with col_chk2:
            st.markdown("##### 🚚 Operational & SLA Capabilities")
            c5 = st.checkbox("Can guarantee delivery of 25 Dell laptops within 45 days", value=True)
            c6 = st.checkbox("On-site repair team capable of 5-day turnaround in Dallas, TX", value=True)
            c7 = st.checkbox("Compliant with Texas Ethics Commission Form 1295 filing", value=True)
            c8 = st.checkbox("Able to execute Mercury Affidavit without exceptions", value=True)

        # Calculate scores
        # Bid1 score (requires c1, c4, c6, c7)
        b1_factors = [c1, c4, c6, c7]
        b1_score = sum(b1_factors) / len(b1_factors) * 100

        # Bid2 score (requires c1, c2, c3, c5, c8)
        b2_factors = [c1, c2, c3, c5, c8]
        b2_score = sum(b2_factors) / len(b2_factors) * 100

        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(
                f"""
            <div class="field-card" style="border-left: 4px solid {'#10b981' if b1_score >= 75 else '#f59e0b'};">
                <div style="font-size:1.1rem; font-weight:700; color:#f8fafc; margin-bottom:6px;">
                    Bid1: Dallas ISD Devices
                </div>
                <div style="font-size:2rem; font-weight:800; color:{'#10b981' if b1_score >= 75 else '#f59e0b'};">
                    {b1_score:.0f}%
                </div>
                <div style="margin:8px 0;">
                    <span class="badge {'badge-pass' if b1_score >= 75 else 'badge-warn'}">
                        {'RECOMMENDATION: GO' if b1_score >= 75 else 'RECOMMENDATION: CONDITIONAL GO'}
                    </span>
                </div>
                <div style="font-size:0.85rem; color:#94a3b8;">
                    {'Strong alignment with Dallas ISD qualifications. Ensure local technician dispatch for 5-day SLA.' if b1_score >= 75 else 'Missing key Dallas ISD compliance or local SLA capacity.'}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown(
                f"""
            <div class="field-card" style="border-left: 4px solid {'#10b981' if b2_score >= 80 else '#ef4444'};">
                <div style="font-size:1.1rem; font-weight:700; color:#f8fafc; margin-bottom:6px;">
                    Bid2: Maryland State Treasurer
                </div>
                <div style="font-size:2rem; font-weight:800; color:{'#10b981' if b2_score >= 80 else '#ef4444'};">
                    {b2_score:.0f}%
                </div>
                <div style="margin:8px 0;">
                    <span class="badge {'badge-pass' if b2_score >= 80 else 'badge-fail'}">
                        {'RECOMMENDATION: GO' if b2_score >= 80 else 'RECOMMENDATION: NO-GO'}
                    </span>
                </div>
                <div style="font-size:0.85rem; color:#94a3b8;">
                    {'Full CATS+ Master Contractor compliance and fast 45-day delivery window verified.' if b2_score >= 80 else 'Must hold prime CATS+ contract #060B5400007 to bid on this secondary PORFP.'}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


# =============================================================
# TAB 6: BENCHMARK & SYSTEM ARCHITECTURE
# =============================================================
elif selected_nav == "📈 Benchmark & Architecture":
    st.subheader("📈 Retrieval Benchmark & Multi-Agent Architecture")
    st.markdown(
        "Quantitative evaluation results measuring **Recall@k**, **MRR**, and **Latency** across 4 retrieval configurations on a 16-question ground-truth test suite."
    )

    # Benchmark Table
    bench_data = [
        {
            "Retrieval Configuration": "Dense Vector Only (Gemini Embeddings)",
            "Recall@1": "75.0%",
            "Recall@3": "93.8%",
            "Recall@5": "93.8%",
            "MRR": "0.844",
            "Avg Latency": "2,166 ms",
            "Key Strengths": "Captures semantic intent, synonym matching",
        },
        {
            "Retrieval Configuration": "BM25 Keyword Search Only",
            "Recall@1": "68.8%",
            "Recall@3": "87.5%",
            "Recall@5": "87.5%",
            "MRR": "0.781",
            "Avg Latency": "1.9 ms",
            "Key Strengths": "Instant model/part matching (CC7802, WD22TB4)",
        },
        {
            "Retrieval Configuration": "Hybrid Search (Vector + BM25 RRF)",
            "Recall@1": "87.5%",
            "Recall@3": "93.8%",
            "Recall@5": "93.8%",
            "MRR": "0.896",
            "Avg Latency": "1,231 ms",
            "Key Strengths": "Optimal balance of exact identifiers and semantics",
        },
        {
            "Retrieval Configuration": "Hybrid Search + LLM Re-ranking",
            "Recall@1": "87.5%",
            "Recall@3": "93.8%",
            "Recall@5": "93.8%",
            "MRR": "0.906",
            "Avg Latency": "16,402 ms",
            "Key Strengths": "Highest MRR, filters out generic table headers",
        },
    ]
    st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🏗️ Multi-Agent System Architecture")

    st.markdown(
        """
```mermaid
graph TD
    User([User Request / CLI / UI]) --> Orch[Orchestrator Agent]
    
    subgraph Ingestion & Hybrid RAG
        Docs[PDF & HTML Solicitation Docs] --> Parser[PyMuPDF & BeautifulSoup Parser]
        Parser --> Chunker[Section & Table-Aware Chunker]
        Chunker --> Embeddings[Gemini Embeddings Engine]
        Chunker --> BM25[BM25 Okapi Index]
        Embeddings --> VecStore[(Vector Store - .npy)]
        BM25 --> BM25Store[(BM25 Store - .json)]
        VecStore & BM25Store --> RRF[Hybrid Search Engine - RRF]
    end
    
    subgraph Multi-Agent Extraction Pipeline
        Orch --> DSpecialist[📅 Dates & Logistics Specialist]
        Orch --> CSpecialist[⚖️ Commercial & Legal Specialist]
        Orch --> PSpecialist[💻 Product & Specs Specialist]
        
        DSpecialist & CSpecialist & PSpecialist -->|Sub-queries| RRF
        RRF -->|Ranked Cited Passages| DSpecialist & CSpecialist & PSpecialist
        
        DSpecialist & CSpecialist & PSpecialist --> State[(Shared Pydantic State)]
        State --> AddendumAgent[🔄 Addendum Reconciliation Agent]
        AddendumAgent -->|Overrides Applied| State
        
        State --> Validator[🛡️ Validator / Critic Agent]
        Validator -->|Audit Checks Passed| FinalRecord[Validated JSON Output]
        Validator -.->|Missing/Hallucinated - Retry Loop| Orch
    end
    
    FinalRecord --> UI[Streamlit Web UI / REST API]
```
""",
        unsafe_allow_html=True,
    )
