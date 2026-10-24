"""
Section and Table-Aware RFP Chunker
Splits ingested documents into semantically coherent chunks with overlap,
preserving table boundaries and attaching full source citations.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from ingestion.parser import Document


@dataclass
class Chunk:
    chunk_id: str
    text: str
    bid_id: str
    file_name: str
    doc_type: str
    page_number: int
    addendum_number: Optional[int] = None
    document_date: Optional[str] = None
    section_title: Optional[str] = None
    is_table: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "bid_id": self.bid_id,
            "file_name": self.file_name,
            "doc_type": self.doc_type,
            "page_number": self.page_number,
            "addendum_number": self.addendum_number,
            "document_date": self.document_date,
            "section_title": self.section_title,
            "is_table": self.is_table,
            "metadata": self.metadata,
        }

    @property
    def citation(self) -> Dict[str, Any]:
        return {
            "file": self.file_name,
            "page": self.page_number,
            "doc_type": self.doc_type,
            "addendum_number": self.addendum_number,
            "chunk_id": self.chunk_id,
        }


class RFPChunker:
    """
    RFP-optimized chunker that:
    1. Detects section titles & headers.
    2. Keeps extracted markdown tables intact to maintain tabular associations.
    3. Employs recursive boundary splitting (paragraphs -> sentences) with overlap.
    4. Prepends breadcrumbs [File | Page | Section] for rich context grounding.
    """

    def __init__(self, target_chunk_size: int = 900, chunk_overlap: int = 180):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def detect_section_header(self, text: str) -> Optional[str]:
        """Detects headers like 'Section 1 - General Information' or '1.1 Header Information'."""
        patterns = [
            r'^(?:Section\s+\d+|[0-9]+(?:\.[0-9]+)*)\s*[-:–]?\s*([A-Za-z0-9 /&()]+)',
            r'^(?:SCOPE AND SPECIFICATIONS|PURPOSE OF REQUEST|SCHEDULE OF EVENTS|BID SUBMISSION INSTRUCTIONS)',
            r'^(?:ADDENDUM\s+No\.\s*\d+)',
        ]
        for line in text.split("\n")[:5]:
            line_str = line.strip()
            for pat in patterns:
                match = re.match(pat, line_str, re.IGNORECASE)
                if match:
                    return line_str[:80]
        return None

    def chunk_document(self, doc: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        raw_text = doc.text.strip()
        if not raw_text:
            return chunks

        section_title = self.detect_section_header(raw_text) or doc.doc_type.upper()

        # Check if page has extracted tables
        parts = re.split(r'(### Extracted Tables:[\s\S]*)', raw_text)
        text_body = parts[0].strip()
        table_body = parts[1].strip() if len(parts) > 1 else ""

        # Process main text body
        if text_body:
            body_chunks = self._split_text(text_body)
            for idx, chunk_text in enumerate(body_chunks):
                chunk_id = f"{doc.bid_id}_{doc.file_name}_p{doc.page_number}_c{idx+1}"
                # Format chunk with contextual header
                enriched_text = f"[{doc.bid_id} | {doc.file_name} | Page {doc.page_number} | {section_title}]\n{chunk_text}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=enriched_text,
                        bid_id=doc.bid_id,
                        file_name=doc.file_name,
                        doc_type=doc.doc_type,
                        page_number=doc.page_number,
                        addendum_number=doc.addendum_number,
                        document_date=doc.document_date,
                        section_title=section_title,
                        is_table=False,
                        metadata={
                            "raw_text": chunk_text,
                            "source_file": doc.file_name,
                            "page": doc.page_number,
                            "doc_type": doc.doc_type,
                            "addendum_number": doc.addendum_number,
                            "has_tables": False,
                        },
                    )
                )

        # Process tables: keep table blocks intact where possible
        if table_body:
            table_chunks = self._split_tables(table_body)
            start_idx = len(chunks) + 1
            for idx, tbl_text in enumerate(table_chunks):
                chunk_id = f"{doc.bid_id}_{doc.file_name}_p{doc.page_number}_tbl{start_idx+idx}"
                enriched_text = f"[{doc.bid_id} | {doc.file_name} | Page {doc.page_number} | TABLE: {section_title}]\n{tbl_text}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=enriched_text,
                        bid_id=doc.bid_id,
                        file_name=doc.file_name,
                        doc_type=doc.doc_type,
                        page_number=doc.page_number,
                        addendum_number=doc.addendum_number,
                        document_date=doc.document_date,
                        section_title=section_title,
                        is_table=True,
                        metadata={
                            "raw_text": tbl_text,
                            "source_file": doc.file_name,
                            "page": doc.page_number,
                            "doc_type": doc.doc_type,
                            "addendum_number": doc.addendum_number,
                            "has_tables": True,
                        },
                    )
                )

        return chunks

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.target_chunk_size:
            return [text]

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            if len(current_chunk) + len(p) + 2 <= self.target_chunk_size:
                current_chunk = f"{current_chunk}\n\n{p}" if current_chunk else p
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Carry over overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = f"{overlap_text}\n\n{p}" if overlap_text else p
                else:
                    # Paragraph itself is too large; split by sentences or linebreaks
                    sub_chunks = self._split_large_paragraph(p)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Splits long paragraphs on sentence or newline boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+|\n', paragraph)
        chunks = []
        curr = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(curr) + len(s) + 1 <= self.target_chunk_size:
                curr = f"{curr} {s}" if curr else s
            else:
                if curr:
                    chunks.append(curr)
                curr = s
        if curr:
            chunks.append(curr)
        return chunks if chunks else [paragraph[:self.target_chunk_size]]

    def _split_tables(self, table_text: str) -> List[str]:
        """Splits large tables by groups of rows so tables don't exceed model limits."""
        lines = table_text.split("\n")
        if len(table_text) <= 1500:
            return [table_text]

        # Extract header lines if present
        header_lines = [l for l in lines[:3] if "|" in l]
        table_chunks = []
        curr_lines = list(header_lines)

        for line in lines[len(header_lines):]:
            if sum(len(l) for l in curr_lines) + len(line) + 1 <= 1400:
                curr_lines.append(line)
            else:
                if len(curr_lines) > len(header_lines):
                    table_chunks.append("\n".join(curr_lines))
                curr_lines = list(header_lines) + [line]

        if len(curr_lines) > len(header_lines):
            table_chunks.append("\n".join(curr_lines))

        return table_chunks if table_chunks else [table_text]

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
        return all_chunks
