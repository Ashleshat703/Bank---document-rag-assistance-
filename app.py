"""
Bank Policy Document Assistant using RAG
Streamlit application with pypdf, TF-IDF retrieval (scikit-learn),
and optional Google Gemini API key for answer synthesis.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Tuple

# Load environment variables if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional pypdf import
try:
    import pypdf
except ImportError:
    pypdf = None

# Optional scikit-learn import
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

# Optional Google Gemini import
try:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import google.generativeai as genai
except ImportError:
    genai = None

# Optional Streamlit import
try:
    import streamlit as st
except ImportError:
    st = None


INSUFFICIENT_EVIDENCE_MSG = (
    "There is insufficient evidence in the provided policy documents to answer this question."
)
DEFAULT_SIMILARITY_THRESHOLD = 0.15
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def extract_text_from_pdf(file_source: Any, filename: str) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF file stream, bytes, or file path.

    Returns:
        List of dicts: [{"filename": str, "page_number": int, "text": str}]
    """
    if pypdf is None:
        raise ImportError("pypdf is required to extract text from PDF files. Install it via pip install pypdf.")

    pages_data: List[Dict[str, Any]] = []

    if isinstance(file_source, (bytes, bytearray)):
        stream = io.BytesIO(file_source)
    elif isinstance(file_source, io.BytesIO):
        stream = file_source
    elif hasattr(file_source, "read"):
        content = file_source.read()
        stream = io.BytesIO(content)
    elif isinstance(file_source, str) and os.path.exists(file_source):
        stream = open(file_source, "rb")
    else:
        stream = io.BytesIO(b"")

    try:
        reader = pypdf.PdfReader(stream)
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            cleaned_text = page_text.strip()
            if cleaned_text:
                pages_data.append({
                    "filename": filename,
                    "page_number": page_idx + 1,
                    "text": cleaned_text,
                })
    finally:
        if isinstance(file_source, str) and not isinstance(stream, io.BytesIO):
            stream.close()

    return pages_data


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split extracted page text into overlapping windows while preserving page and file metadata.

    Returns:
        List of chunk dicts: [{"chunk_id": int, "filename": str, "page_number": int, "text": str}]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1

    for page in pages:
        text = (page.get("text") or "").strip()
        filename = page.get("filename", "unknown.pdf")
        page_number = page.get("page_number", 1)

        if not text:
            continue

        if len(text) <= chunk_size:
            chunks.append({
                "chunk_id": chunk_counter,
                "filename": filename,
                "page_number": page_number,
                "text": text,
            })
            chunk_counter += 1
            continue

        step = chunk_size - chunk_overlap
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_str = text[start:end].strip()
            if chunk_str:
                chunks.append({
                    "chunk_id": chunk_counter,
                    "filename": filename,
                    "page_number": page_number,
                    "text": chunk_str,
                })
                chunk_counter += 1
            if end == len(text):
                break
            start += step

    return chunks


class TfidfRetriever:
    """
    TF-IDF Vectorizer and Cosine Similarity retriever for document passages.
    """

    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self.vectorizer: Optional[Any] = None
        self.tfidf_matrix: Optional[Any] = None
        self.chunks: List[Dict[str, Any]] = []

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Fit the TF-IDF vectorizer over the corpus chunks.
        """
        self.chunks = chunks
        if not chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        if TfidfVectorizer is None:
            raise ImportError("scikit-learn is required for TF-IDF retrieval.")

        texts = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Retrieve top_k passages for the query.

        Returns:
            (retrieved_passages, has_sufficient_evidence)
            If top similarity is below the threshold or corpus is empty,
            returns ([], False).
        """
        if threshold is None:
            threshold = self.similarity_threshold

        query_clean = (query or "").strip()
        if not query_clean or not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return [], False

        query_vec = self.vectorizer.transform([query_clean])
        # If query contains only unknown/stop-words, vector norm will be 0
        if query_vec.nnz == 0:
            return [], False

        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        max_score = float(scores.max()) if len(scores) > 0 else 0.0

        if max_score < threshold:
            return [], False

        # Rank indices descending
        ranked_indices = scores.argsort()[::-1]

        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score < threshold:
                break
            chunk_copy = dict(self.chunks[idx])
            chunk_copy["similarity_score"] = round(score, 4)
            results.append(chunk_copy)
            if len(results) >= top_k:
                break

        if not results:
            return [], False

        return results, True


def fetch_supported_models(api_key: str) -> List[str]:
    """
    Fetch all models supporting generateContent for the provided Gemini API key.
    """
    if not api_key or genai is None:
        return []
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = m.name.replace("models/", "")
                models.append(name)
        return models
    except Exception:
        return []


def generate_answer_with_gemini(
    query: str,
    passages: List[Dict[str, Any]],
    api_key: str,
    model_name: str = "gemini-1.5-flash",
) -> str:
    """
    Generate a synthesized answer strictly grounded in retrieved passages using Google Gemini.
    Includes automatic fallback to valid supported models if the selected model raises a 404.
    """
    if genai is None:
        raise ImportError(
            "google-generativeai package is required for Gemini API synthesis. "
            "Install it via pip install google-generativeai."
        )

    if not passages:
        return INSUFFICIENT_EVIDENCE_MSG

    # Construct context block with explicit citation metadata
    context_blocks = []
    for idx, p in enumerate(passages, 1):
        context_blocks.append(
            f"Passage {idx} [Document: {p['filename']}, Page: {p['page_number']}]:\n{p['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    system_instruction = (
        "You are an assistant for bank policy compliance and internal regulations.\n"
        "Answer the user's question strictly using ONLY the provided policy passages.\n"
        "Rules:\n"
        "1. For every key fact, explicitly cite the source document and page number in brackets, e.g. [Document: filename, Page: X].\n"
        "2. Do not speculate, extrapolate, or assume facts not directly stated in the context.\n"
        f"3. If the provided passages do not contain sufficient evidence to answer the question, reply with: \"{INSUFFICIENT_EVIDENCE_MSG}\""
    )

    full_prompt = (
        f"{system_instruction}\n\n"
        f"Policy Passages:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    genai.configure(api_key=api_key)

    # Clean model identifier
    clean_model_name = model_name.replace("models/", "").strip()

    # Build prioritized candidate model list with fallbacks
    candidates = [clean_model_name]
    if "flash" in clean_model_name:
        candidates.extend(["gemini-1.5-flash-latest", "gemini-1.5-flash-001", "gemini-1.5-flash-002", "gemini-pro"])
    elif "pro" in clean_model_name:
        candidates.extend(["gemini-1.5-pro-latest", "gemini-1.5-pro-001", "gemini-pro"])
    else:
        candidates.extend(["gemini-1.5-flash", "gemini-pro", "gemini-1.5-pro"])

    # Deduplicate while preserving priority order
    seen = set()
    deduped_candidates = [m for m in candidates if not (m in seen or seen.add(m))]

    last_error: Optional[Exception] = None
    for cand in deduped_candidates:
        try:
            model = genai.GenerativeModel(model_name=cand)
            response = model.generate_content(full_prompt)
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception as err:
            last_error = err
            # Only fallback if error is 404 / model not found
            err_msg = str(err).lower()
            if "404" not in err_msg and "not found" not in err_msg:
                raise err

    # If all hardcoded candidates failed with 404, query API for actual models supported by key
    available_models = fetch_supported_models(api_key)
    if available_models:
        for avail in available_models:
            if avail in deduped_candidates:
                continue
            try:
                model = genai.GenerativeModel(model_name=avail)
                response = model.generate_content(full_prompt)
                if response and hasattr(response, "text") and response.text:
                    return response.text.strip()
            except Exception:
                continue

        raise RuntimeError(
            f"Model '{clean_model_name}' was not found. "
            f"Available models for your API key: {', '.join(available_models[:8])}"
        )

    if last_error:
        raise last_error

    return INSUFFICIENT_EVIDENCE_MSG


def run_streamlit_app() -> None:
    """Streamlit User Interface for the Bank Policy Assistant."""
    if st is None:
        raise ImportError("Streamlit is not installed. Please install streamlit to run the UI.")

    st.set_page_config(
        page_title="Bank Policy Assistant (RAG)",
        page_icon="🏦",
        layout="wide",
    )

    st.title("🏦 Bank Policy Document Assistant")
    st.markdown(
        "Upload bank policy PDFs, search through guidelines with TF-IDF retrieval, "
        "and inspect cited passages. If a Google Gemini API key is supplied, a synthesized "
        "answer citing document pages is generated."
    )

    # Sidebar settings
    st.sidebar.header("⚙️ Configuration")

    api_key_env = os.getenv("GEMINI_API_KEY", "")
    api_key = st.sidebar.text_input(
        "Google Gemini API Key (Optional)",
        value=api_key_env,
        type="password",
        help="Leave blank to view retrieved passages directly without LLM synthesis. Get a free key at https://aistudio.google.com/.",
    )

    default_models = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-pro",
        "gemini-2.0-flash-exp",
    ]

    discovered_models = fetch_supported_models(api_key) if api_key else []
    model_options = discovered_models if discovered_models else default_models

    model_name = st.sidebar.selectbox(
        "Gemini Model",
        options=model_options,
        index=0,
        help="Select a model available on your Gemini API key.",
    )

    threshold = st.sidebar.slider(
        "Retrieval Similarity Threshold",
        min_value=0.0,
        max_value=0.5,
        value=DEFAULT_SIMILARITY_THRESHOLD,
        step=0.02,
        help="Queries scoring below this cosine similarity will trigger an abstention response.",
    )

    top_k = st.sidebar.slider("Top Passages (Hit@K)", min_value=1, max_value=10, value=5)

    st.sidebar.divider()
    st.sidebar.subheader("📄 Upload Policy Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Bank Policy PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # Session state for parsed corpus and retriever
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
        st.session_state.chunks = []
        st.session_state.file_stats = {}

    # Ingest documents when files are uploaded or changed
    if uploaded_files:
        all_pages: List[Dict[str, Any]] = []
        file_stats: Dict[str, int] = {}

        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            pages = extract_text_from_pdf(file_bytes, uploaded_file.name)
            file_stats[uploaded_file.name] = len(pages)
            all_pages.extend(pages)

        chunks = chunk_document_pages(all_pages)
        retriever = TfidfRetriever(similarity_threshold=threshold)
        retriever.fit(chunks)

        st.session_state.retriever = retriever
        st.session_state.chunks = chunks
        st.session_state.file_stats = file_stats

        st.sidebar.success(f"Indexed {len(uploaded_files)} PDF(s) ({len(chunks)} chunks).")
        for fname, count in file_stats.items():
            st.sidebar.caption(f"• {fname} ({count} pages)")
    else:
        st.session_state.retriever = None
        st.session_state.chunks = []
        st.session_state.file_stats = {}

    # Query Input
    query = st.text_input(
        "Ask a policy question:",
        placeholder="e.g., What is the threshold for filing a Suspicious Activity Report (SAR)?",
    )

    if query:
        if not st.session_state.chunks or st.session_state.retriever is None:
            st.warning("Please upload at least one bank policy PDF in the sidebar first.")
            return

        retriever: TfidfRetriever = st.session_state.retriever
        passages, has_evidence = retriever.retrieve(query, top_k=top_k, threshold=threshold)

        if not has_evidence or not passages:
            st.warning(f"⚠️ {INSUFFICIENT_EVIDENCE_MSG}")
            st.info("Tip: Try rephrasing your question or checking your policy document contents.")
        else:
            st.success(f"Found {len(passages)} relevant passage(s).")

            # Answer synthesis branch
            if api_key:
                st.subheader("💡 Synthesized Answer (with Citations)")
                with st.spinner("Generating answer with Google Gemini..."):
                    try:
                        answer = generate_answer_with_gemini(
                            query,
                            passages,
                            api_key=api_key,
                            model_name=model_name,
                        )
                        st.markdown(answer)
                    except Exception as err:
                        st.error(f"Gemini Generation Error: {err}")
                        st.info("Falling back to direct passage display below.")
            else:
                st.info("ℹ️ No Gemini API key provided. Showing retrieved passages directly below:")

            # Retrieved passages branch
            st.subheader("📑 Retrieved Passages")
            for idx, p in enumerate(passages, 1):
                with st.expander(
                    f"#{idx} | {p['filename']} — Page {p['page_number']} (Score: {p['similarity_score']})",
                    expanded=(idx == 1),
                ):
                    st.write(p["text"])


if __name__ == "__main__":
    run_streamlit_app()
