# Bank Policy Document Assistant (RAG)

A minimal, production-grounded Retrieval-Augmented Generation (RAG) assistant designed for compliance officers, auditors, and bank staff to query internal bank policy documents.

Built with **Streamlit**, **pypdf**, and **scikit-learn** (TF-IDF retrieval), with optional LLM answer synthesis via **Google Gemini**.

---

## 🌟 Key Features

1. **PDF Document Ingestion**: Upload one or more bank policy documents (e.g., AML/KYC, credit authority, wire transfers, travel & expense policies).
2. **Metadata-Preserving Chunking**: Automatically segments page text into overlapping windows while retaining filename and 1-based page numbers.
3. **TF-IDF Passage Retrieval**: Uses `scikit-learn`'s `TfidfVectorizer` and cosine similarity to quickly index and rank relevant policy excerpts without requiring heavy vector databases.
4. **Strict Abstention for Out-of-Domain / Insufficient Evidence**:
   - If a query's highest similarity score falls below the configured threshold (default `0.15`), the system abstains and explicitly returns:
     > *"There is insufficient evidence in the provided policy documents to answer this question."*
5. **Flexible Operation Modes**:
   - **Without Gemini API Key**: Directly presents the top retrieved passages alongside document filenames, page numbers, and similarity scores.
   - **With Gemini API Key**: Generates a concise, compliance-oriented answer strictly grounded in the retrieved excerpts using Google Gemini (`gemini-1.5-flash`), requiring explicit bracketed source citations `[Document: <filename>, Page: <page>]`.

---

## 📂 Project Structure

```text
Bank Policies/
├── .env.example              # Environment variables template (GEMINI_API_KEY)
├── .gitignore                # Git ignore rules for Python, venv, and temp files
├── README.md                 # Project documentation and guide
├── requirements.txt          # Python project dependencies
├── pytest.ini                # Pytest configuration
├── app.py                    # Core Streamlit app and RAG pipeline
├── evaluate.py               # RAG evaluation script (Hit@5, abstention rates)
├── evaluation/
│   └── eval_questions.csv    # Benchmark dataset of answerable and unanswerable queries
└── tests/
    └── test_rag.py           # Unit tests (chunking, empty docs, retrieval, abstention)
```

---

## 🚀 Quickstart Guide (PowerShell on Windows)

### 1. Prerequisites
- Python 3.9+ installed and added to `PATH`.

### 2. Navigate and Activate the Virtual Environment

Open PowerShell in the project directory:

```powershell
cd "C:\Users\KIIT\OneDrive\Desktop\Bank Policies"
.\venv\Scripts\Activate.ps1
```

*(Note: If script execution is restricted in your PowerShell session, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### 3. Install Dependencies (if not already installed)

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)

If you wish to use Google Gemini for synthesized answers:

```powershell
Copy-Item .env.example .env
```

Open `.env` and enter your Gemini API key (get a free key at [Google AI Studio](https://aistudio.google.com/)):
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Alternatively, you can paste the key directly into the Streamlit sidebar at runtime).*

### 5. Launch the Streamlit App

```powershell
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Running Evaluation

The evaluation suite tests retrieval performance and abstention reliability across a curated dataset of bank policy questions:

```powershell
python evaluate.py
```

### Metrics Computed:
- **Retrieval Hit@5 (Answerable Questions)**: Percentage of answerable questions where the correct policy passage and page are retrieved within the top 5 results.
- **Abstention Rate on Unanswerable Questions (True Abstention / Specificity)**: Percentage of unanswerable questions where the system correctly returned insufficient evidence instead of hallucinating.
- **Abstention Rate on Answerable Questions (False Abstention)**: Percentage of answerable questions wrongly rejected due to low similarity.
- **Overall Abstention Accuracy**: Accuracy of the binary decision to answer vs. abstain.

---

## 🧪 Running Unit Tests

Unit tests verify the core RAG components (text chunking, empty document handling, TF-IDF retrieval, and threshold-based abstention):

```powershell
pytest -v
```

---

## 🔍 Metrics Requiring Manual Review

While retrieval Hit@5 and abstention rates are measured automatically, certain critical answer quality and citation metrics require human review:

1. **Faithfulness / Groundedness**:
   - Automated evaluation verifies whether the passage was retrieved; human review confirms that the LLM did not introduce external knowledge, make subtle numerical errors, or fabricate claims not present in the excerpt.
2. **Citation Precision & Attribution**:
   - Confirming that each statement maps to the exact page and paragraph cited, rather than citing an entire section or the wrong page.
3. **Completeness & Policy Nuance**:
   - Bank policies frequently include conditionality (e.g., *"Approval requires Director sign-off EXCEPT when transaction amount is under $5,000 and customer is low risk"*). Manual auditing verifies that exceptions, escalations, and edge conditions were not omitted.
4. **Tone & Regulatory Disclaimers**:
   - Ensuring the generated output maintains neutral compliance terminology without giving unqualified financial or legal guarantees.
