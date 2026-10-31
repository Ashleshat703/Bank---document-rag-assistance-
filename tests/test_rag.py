"""
Unit tests for Bank Policy Document Assistant RAG pipeline.
Covers:
  - Text chunking
  - Empty documents and edge cases
  - TF-IDF retrieval
  - Abstention on out-of-domain / insufficient evidence queries
"""

from __future__ import annotations

import os
import sys

# Ensure parent directory is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from app import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    INSUFFICIENT_EVIDENCE_MSG,
    TfidfRetriever,
    chunk_document_pages,
)


# ---------------------------------------------------------------------------
# 1. Text Chunking Tests
# ---------------------------------------------------------------------------

def test_chunking_short_document():
    """Short text under chunk_size should produce a single chunk preserving metadata."""
    pages = [
        {
            "filename": "AML_Policy.pdf",
            "page_number": 1,
            "text": "The Bank adheres to the Bank Secrecy Act and anti-money laundering laws.",
        }
    ]
    chunks = chunk_document_pages(pages, chunk_size=500, chunk_overlap=100)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_id"] == 1
    assert chunk["filename"] == "AML_Policy.pdf"
    assert chunk["page_number"] == 1
    assert "Bank Secrecy Act" in chunk["text"]


def test_chunking_long_document_with_overlap():
    """Long text should be sliced into multiple overlapping chunks."""
    long_text = "Word " * 200  # 1000 characters
    pages = [{"filename": "Credit_Policy.pdf", "page_number": 2, "text": long_text}]
    chunk_size = 300
    chunk_overlap = 50

    chunks = chunk_document_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) > 1

    # Verify chunk IDs are sequential
    for i, c in enumerate(chunks, 1):
        assert c["chunk_id"] == i
        assert c["filename"] == "Credit_Policy.pdf"
        assert c["page_number"] == 2
        assert len(c["text"]) <= chunk_size


def test_chunking_invalid_parameters():
    """Chunking with non-positive chunk_size or excessive overlap should raise ValueError."""
    pages = [{"filename": "Test.pdf", "page_number": 1, "text": "Some text"}]

    with pytest.raises(ValueError):
        chunk_document_pages(pages, chunk_size=0, chunk_overlap=10)

    with pytest.raises(ValueError):
        chunk_document_pages(pages, chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        chunk_document_pages(pages, chunk_size=100, chunk_overlap=150)


# ---------------------------------------------------------------------------
# 2. Empty Document & Boundary Tests
# ---------------------------------------------------------------------------

def test_empty_pages_list():
    """Empty list of pages should return an empty chunk list."""
    chunks = chunk_document_pages([])
    assert chunks == []


def test_whitespace_and_blank_pages():
    """Pages containing only whitespace or blank text should be skipped."""
    pages = [
        {"filename": "Blank.pdf", "page_number": 1, "text": ""},
        {"filename": "Blank.pdf", "page_number": 2, "text": "   \n\t  "},
    ]
    chunks = chunk_document_pages(pages)
    assert chunks == []


def test_retriever_on_empty_corpus():
    """Retriever fitted on empty corpus should handle queries gracefully without crashing."""
    retriever = TfidfRetriever()
    retriever.fit([])

    passages, has_evidence = retriever.retrieve("What is the SAR limit?")
    assert passages == []
    assert has_evidence is False


def test_retriever_empty_query():
    """Empty or whitespace-only query string should return empty results and false evidence."""
    sample_chunks = [
        {"chunk_id": 1, "filename": "Test.pdf", "page_number": 1, "text": "Bank policy sample text"}
    ]
    retriever = TfidfRetriever()
    retriever.fit(sample_chunks)

    passages, has_evidence = retriever.retrieve("")
    assert passages == []
    assert has_evidence is False

    passages, has_evidence = retriever.retrieve("     \n\t")
    assert passages == []
    assert has_evidence is False


# ---------------------------------------------------------------------------
# 3. TF-IDF Retrieval Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_retriever():
    """Fixture providing a retriever indexed with diverse bank policy excerpts."""
    policy_chunks = [
        {
            "chunk_id": 1,
            "filename": "AML_Policy.pdf",
            "page_number": 2,
            "text": "A Suspicious Activity Report (SAR) must be filed for transactions of $5,000 or more.",
        },
        {
            "chunk_id": 2,
            "filename": "Wire_Policy.pdf",
            "page_number": 1,
            "text": "All outgoing wire transfers exceeding $100,000 require dual authorization.",
        },
        {
            "chunk_id": 3,
            "filename": "Credit_Policy.pdf",
            "page_number": 3,
            "text": "The maximum permissible back-end Debt-to-Income (DTI) ratio is capped at 43%.",
        },
        {
            "chunk_id": 4,
            "filename": "Expense_Policy.pdf",
            "page_number": 1,
            "text": "Employees receive a maximum daily domestic per diem meal allowance of $75.",
        },
    ]
    retriever = TfidfRetriever(similarity_threshold=0.15)
    retriever.fit(policy_chunks)
    return retriever


def test_retrieval_returns_relevant_passage(populated_retriever):
    """Querying for SAR threshold should return the AML policy chunk with score > 0."""
    passages, has_evidence = populated_retriever.retrieve(
        "What is the threshold for filing a Suspicious Activity Report SAR?",
        top_k=5,
    )
    assert has_evidence is True
    assert len(passages) >= 1
    top_hit = passages[0]
    assert top_hit["filename"] == "AML_Policy.pdf"
    assert top_hit["page_number"] == 2
    assert "SAR" in top_hit["text"]
    assert top_hit["similarity_score"] > 0.15


def test_retrieval_top_k_limit(populated_retriever):
    """Retrieval should strictly respect the top_k parameter."""
    passages, has_evidence = populated_retriever.retrieve("policy report allowance", top_k=2)
    assert len(passages) <= 2


# ---------------------------------------------------------------------------
# 4. Abstention Tests
# ---------------------------------------------------------------------------

def test_abstention_on_unrelated_query(populated_retriever):
    """Completely out-of-domain query should abstain (has_evidence=False, empty list)."""
    passages, has_evidence = populated_retriever.retrieve(
        "What is the baking recipe for chocolate chip walnut cookies?",
        top_k=5,
    )
    assert has_evidence is False
    assert passages == []


def test_abstention_on_out_of_vocabulary_query(populated_retriever):
    """Queries with terms completely absent from corpus should abstain cleanly."""
    passages, has_evidence = populated_retriever.retrieve(
        "astrophysics supernova spectrometry exoplanet",
        top_k=5,
    )
    assert has_evidence is False
    assert passages == []


def test_abstention_message_constant():
    """Verify standard insufficient evidence string."""
    assert INSUFFICIENT_EVIDENCE_MSG == (
        "There is insufficient evidence in the provided policy documents to answer this question."
    )

