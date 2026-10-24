"""
Document Ingestion & Parsing Module
Extracts clean text, structured tables, and metadata from HTML and PDF documents.
Supports doc_type classification, addendum detection, header/footer cleaning,
and robust error handling.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import pymupdf
import pdfplumber
from bs4 import BeautifulSoup

logger = logging.getLogger("rfp_ingestion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class Document:
    bid_id: str
    file_name: str
    doc_type: str  # bid_page, rfp, addendum, specs, affidavit, unknown
    page_number: int  # 1-indexed
    text: str
    addendum_number: Optional[int] = None
    document_date: Optional[str] = None
    tables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def citation_source(self) -> Dict[str, Any]:
        return {
            "file": self.file_name,
            "page": self.page_number,
            "doc_type": self.doc_type,
            "addendum_number": self.addendum_number,
        }


def classify_document(file_name: str, sample_text: str = "") -> tuple[str, Optional[int]]:
    """Classifies document into doc_type and extracts addendum_number if applicable."""
    lower_fn = file_name.lower()
    lower_text = sample_text.lower()[:1000]

    addendum_match = re.search(r'addendum\s*(?:no\.?|#)?\s*(\d+)', lower_fn) or re.search(
        r'addendum\s*(?:no\.?|#)?\s*(\d+)', lower_text
    )
    if "addendum" in lower_fn or "addendum" in lower_text:
        add_num = int(addendum_match.group(1)) if addendum_match else None
        return "addendum", add_num

    if "affidavit" in lower_fn or "affidavit" in lower_text:
        return "affidavit", None

    if "spec" in lower_fn or "specification" in lower_fn:
        return "specs", None

    if lower_fn.endswith(".html") or "bidnet direct" in lower_fn or "bid information" in lower_fn:
        return "bid_page", None

    if any(k in lower_fn for k in ["final", "rfp", "porfp", "request"]):
        return "rfp", None

    # Fallback to text inspection
    if "request for proposal" in lower_text or "porfp" in lower_text:
        return "rfp", None
    if "specification" in lower_text:
        return "specs", None

    return "rfp", None


def extract_document_date(text: str) -> Optional[str]:
    """Extracts prominent dates like YYYY-MM-DD, MM/DD/YYYY, or DD-Mon-YYYY."""
    patterns = [
        r'\b(\d{4}-\d{2}-\d{2})\b',
        r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
        r'\b(\d{1,2}-(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)-\d{4})\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def clean_text(text: str) -> str:
    """Removes repeated header/footer boilerplate, fixes hyphenation and whitespace."""
    if not text:
        return ""

    # Fix line-end hyphenation (e.g. "solici-\ntation" -> "solicitation")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Remove known recurrent headers & footers
    header_footer_patterns = [
        r'Dallas ISD rev \d+\.\d+\s+Page \d+ of \d+',
        r'Purchase Order Request for Proposals \(PORFP\)\s+Hardware Master Contract\s+\d+',
        r'Page \d+\s*\|\s*\d+',
        r'Page \d+\s*of\s*\d+',
    ]
    for pat in header_footer_patterns:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)

    # Normalize multiple newlines and spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_pdf(file_path: str, bid_id: str) -> List[Document]:
    """
    Parses a PDF file page by page using PyMuPDF and pdfplumber.
    Extracts text, 2D block layouts for specs, and table structures.
    """
    file_name = os.path.basename(file_path)
    documents = []

    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {file_path}: {e}")
        return documents

    # Determine doc_type from filename and first page
    first_page_text = doc[0].get_text() if len(doc) > 0 else ""
    doc_type, addendum_num = classify_document(file_name, first_page_text)
    doc_date = extract_document_date(first_page_text)

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]

        # Check for 2D column layout (e.g., Dell_Laptop_Specs.pdf)
        blocks = page.get_text("blocks")
        if "spec" in file_name.lower() and len(blocks) > 5:
            # Sort blocks primarily by y-coord (with rounding) then x-coord to pair SKUs with descriptions
            sorted_blocks = sorted(blocks, key=lambda b: (round(b[1] / 15), b[0]))
            raw_text = "\n".join(b[4].strip() for b in sorted_blocks if b[4].strip())
        else:
            raw_text = page.get_text()

        # Extract tables via PyMuPDF find_tables (fast C++ implementation)
        table_markdowns = []
        try:
            tab_finder = page.find_tables()
            for table in tab_finder.tables:
                extracted_rows = table.extract()
                if extracted_rows and len(extracted_rows) > 1:
                    cleaned_rows = []
                    for row in extracted_rows:
                        cleaned_row = [re.sub(r'\s+', ' ', (cell or "")).strip() for cell in row]
                        if any(cleaned_row):
                            cleaned_rows.append(cleaned_row)
                    if len(cleaned_rows) > 1:
                        # Render clean markdown table
                        header = "| " + " | ".join(cleaned_rows[0]) + " |"
                        sep = "| " + " | ".join(["---"] * len(cleaned_rows[0])) + " |"
                        body_rows = ["| " + " | ".join(r) + " |" for r in cleaned_rows[1:]]
                        md_table = "\n".join([header, sep] + body_rows)
                        table_markdowns.append(md_table)
        except Exception as e:
            logger.debug(f"PyMuPDF table extraction fallback on {file_name} p.{page_num}: {e}")

        # Combine text and table data
        combined_text = raw_text
        if table_markdowns:
            combined_text += "\n\n### Extracted Tables:\n" + "\n\n".join(table_markdowns)

        cleaned = clean_text(combined_text)

        if not cleaned:
            logger.warning(f"Empty or scanned page detected: {file_name} Page {page_num}")
            cleaned = f"[Empty or image-only page in {file_name} Page {page_num}]"

        doc_item = Document(
            bid_id=bid_id,
            file_name=file_name,
            doc_type=doc_type,
            page_number=page_num,
            text=cleaned,
            addendum_number=addendum_num,
            document_date=doc_date,
            tables=table_markdowns,
            metadata={
                "source_file": file_name,
                "page": page_num,
                "doc_type": doc_type,
                "addendum_number": addendum_num,
                "has_tables": len(table_markdowns) > 0,
            },
        )
        documents.append(doc_item)

    return documents


def parse_html(file_path: str, bid_id: str) -> List[Document]:
    """
    Parses an HTML bid page (e.g. BidNet Direct portal page),
    extracting key metadata, solicitation info, prebid dates, and requirements.
    """
    file_name = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
    except Exception as e:
        logger.error(f"Failed to read HTML file {file_path}: {e}")
        return []

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script, style, navigation headers
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    # Extract key labeled fields
    lines = []
    title = soup.title.string.strip() if soup.title else file_name
    lines.append(f"# Bid Portal Information: {title}\n")

    # Common portal key-value containers
    portal_data = {}
    for elem in soup.find_all(["dt", "th", "label", "span", "div"]):
        label = elem.get_text(strip=True)
        # Check if next sibling or adjacent element contains the value
        next_sibling = elem.find_next_sibling(["dd", "td", "span", "div"])
        if next_sibling:
            val = next_sibling.get_text(strip=True)
            if label and val and len(label) < 60 and len(val) < 500:
                clean_lbl = re.sub(r'[:\s]+$', '', label)
                if clean_lbl and clean_lbl not in portal_data:
                    portal_data[clean_lbl] = val

    if portal_data:
        lines.append("## Key Solicitation Details (Portal Metadata):")
        for k, v in portal_data.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("\n## Page Content:")

    # Add general visible body text
    body = soup.find("body") or soup
    body_text = clean_text(body.get_text(separator="\n"))
    lines.append(body_text)

    full_text = "\n".join(lines)
    doc_date = extract_document_date(full_text)

    doc_item = Document(
        bid_id=bid_id,
        file_name=file_name,
        doc_type="bid_page",
        page_number=1,
        text=full_text,
        addendum_number=None,
        document_date=doc_date,
        tables=[],
        metadata={
            "source_file": file_name,
            "page": 1,
            "doc_type": "bid_page",
            "is_html": True,
        },
    )
    return [doc_item]


def ingest_bid_folder(folder_path: str, bid_id: Optional[str] = None) -> List[Document]:
    """
    Ingests an entire bid folder containing HTML and PDF files.
    Returns a unified list of Document objects.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")

    assigned_bid_id = bid_id or os.path.basename(os.path.normpath(folder_path))
    all_documents = []

    files = sorted(os.listdir(folder_path))
    logger.info(f"Ingesting folder '{folder_path}' as bid '{assigned_bid_id}' ({len(files)} files found)")

    for f in files:
        full_path = os.path.join(folder_path, f)
        if os.path.isfile(full_path):
            if f.lower().endswith(".pdf"):
                docs = parse_pdf(full_path, assigned_bid_id)
                all_documents.extend(docs)
                logger.info(f"Parsed PDF: {f} ({len(docs)} pages)")
            elif f.lower().endswith(".html") or f.lower().endswith(".htm"):
                docs = parse_html(full_path, assigned_bid_id)
                all_documents.extend(docs)
                logger.info(f"Parsed HTML: {f}")
            else:
                logger.info(f"Skipping unsupported file format: {f}")

    logger.info(f"Completed ingestion for '{assigned_bid_id}': total {len(all_documents)} document pages.")
    return all_documents
