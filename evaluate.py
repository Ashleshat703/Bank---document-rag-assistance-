"""
Evaluation Script for Bank Policy Document Assistant RAG
Measures:
  1. Retrieval Hit@5 on Answerable Questions
  2. Abstention Rate on Unanswerable Questions (True Abstention / Specificity)
  3. False Abstention Rate on Answerable Questions
  4. Overall Abstention Decision Accuracy
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Any, Dict, List, Tuple

# Import RAG components from app
from app import (
    DEFAULT_SIMILARITY_THRESHOLD,
    TfidfRetriever,
    chunk_document_pages,
)

# Reference Bank Policy Corpus for deterministic, reproducible evaluation
SAMPLE_POLICY_PAGES = [
    # AML_Policy.pdf
    {
        "filename": "AML_Policy.pdf",
        "page_number": 1,
        "text": (
            "Section 1: General Anti-Money Laundering Framework and Governance. "
            "The Bank complies with the Bank Secrecy Act (BSA) and USA PATRIOT Act regulations. "
            "All operational departments must maintain active customer transaction surveillance."
        ),
    },
    {
        "filename": "AML_Policy.pdf",
        "page_number": 2,
        "text": (
            "Section 2: Suspicious Activity Reporting (SAR) Thresholds. "
            "A Suspicious Activity Report (SAR) must be filed with FinCEN for any suspicious transaction "
            "involving $5,000 or more when a suspect can be identified. If no suspect is identified, "
            "the mandatory filing threshold is $25,000. Filings must occur within 30 days of initial detection."
        ),
    },
    {
        "filename": "AML_Policy.pdf",
        "page_number": 3,
        "text": (
            "Section 3: Currency Transaction Reporting (CTR) and Record Retention. "
            "All cash transactions exceeding $10,000 conducted by or on behalf of one individual in a single "
            "business day require a CTR filing. All completed CTR filings and supporting audit logs must be "
            "retained for a minimum retention period of 5 years from the filing date."
        ),
    },
    {
        "filename": "AML_Policy.pdf",
        "page_number": 4,
        "text": (
            "Section 4: Politically Exposed Persons (PEPs) and High-Risk Due Diligence. "
            "Any prospective customer identified as a Politically Exposed Person (PEP), senior foreign political "
            "figure, or immediate family member requires Enhanced Due Diligence (EDD). Account onboarding requires "
            "written authorization from the Senior Compliance Officer or Managing Director prior to funding."
        ),
    },

    # Credit_Policy.pdf
    {
        "filename": "Credit_Policy.pdf",
        "page_number": 1,
        "text": (
            "Section 1: Retail Lending Authority Limits. "
            "Branch Managers possess single-signature loan approval authority up to $50,000 for unsecured credits "
            "and secured auto loans. Loans exceeding $50,000 up to $250,000 require dual signature approval from the "
            "Regional Credit Officer."
        ),
    },
    {
        "filename": "Credit_Policy.pdf",
        "page_number": 2,
        "text": (
            "Section 2: Commercial Credit Underwriting Standards. "
            "Commercial borrowers must demonstrate a Debt Service Coverage Ratio (DSCR) of at least 1.25x for the "
            "preceding two fiscal years based on audited financial statements."
        ),
    },
    {
        "filename": "Credit_Policy.pdf",
        "page_number": 3,
        "text": (
            "Section 3: Residential Mortgage Guidelines and Debt-to-Income (DTI). "
            "For conventional conforming residential mortgage loans, the maximum permissible back-end "
            "Debt-to-Income (DTI) ratio is capped at 43%. Exceptions up to 45% require Fannie Mae Desktop Underwriter "
            "automated approval and minimum FICO score of 720."
        ),
    },
    {
        "filename": "Credit_Policy.pdf",
        "page_number": 4,
        "text": (
            "Section 4: Commercial Loan Annual Credit Reviews. "
            "All active commercial borrowing relationships and lines of credit exceeding $250,000 total exposure "
            "must undergo a formal credit review annually (every 12 months) by the Credit Risk Committee."
        ),
    },

    # KYC_Policy.pdf
    {
        "filename": "KYC_Policy.pdf",
        "page_number": 1,
        "text": (
            "Section 1: Customer Identification Program (CIP). "
            "Individual account opening requires unexpired government-issued photo ID (valid passport, state driver's "
            "license, or national identity card) along with proof of address dated within 60 days."
        ),
    },
    {
        "filename": "KYC_Policy.pdf",
        "page_number": 2,
        "text": (
            "Section 2: Beneficial Ownership Rule for Legal Entities. "
            "Under FinCEN regulations, legal entity customers must certify beneficial ownership for every individual "
            "who owns 25% or greater equity interest in the entity, and at least one executive control person."
        ),
    },
    {
        "filename": "KYC_Policy.pdf",
        "page_number": 3,
        "text": (
            "Section 3: Address and Customer Profile Periodic Refresh. "
            "Standard retail accounts are reviewed every 36 months, whereas medium-risk entity profiles must be "
            "refreshed every 24 months."
        ),
    },
    {
        "filename": "KYC_Policy.pdf",
        "page_number": 4,
        "text": (
            "Section 4: Inactive and Dormant Accounts. "
            "A retail demand deposit or savings account with no customer-initiated financial activity for "
            "24 consecutive months will be officially designated as dormant and subject to abandoned property escheatment review."
        ),
    },

    # Wire_Transfer_Policy.pdf
    {
        "filename": "Wire_Transfer_Policy.pdf",
        "page_number": 1,
        "text": (
            "Section 1: Outgoing Wire Dual Authorization. "
            "All outgoing domestic or international wire transfers exceeding $100,000 require dual authorization "
            "from two authorized officers. High-value wires over $500,000 additionally require voice callback verification."
        ),
    },
    {
        "filename": "Wire_Transfer_Policy.pdf",
        "page_number": 2,
        "text": (
            "Section 2: Electronic Funds Transfer and Debit Card Withdrawal Limits. "
            "Standard consumer debit cards carry a default cash withdrawal limit of $1,000 per 24-hour cycle at ATMs. "
            "Point-of-sale debit purchases are limited to $5,000 per day unless temporary authorization is approved."
        ),
    },
    {
        "filename": "Wire_Transfer_Policy.pdf",
        "page_number": 3,
        "text": (
            "Section 3: Fedwire Operating Windows and Daily Cutoff Times. "
            "Same-day domestic Fedwire transfers must be submitted and approved by the daily cutoff time of 4:30 PM Eastern Time. "
            "Wires received after 4:30 PM will execute on the next business day."
        ),
    },

    # Expense_Policy.pdf
    {
        "filename": "Expense_Policy.pdf",
        "page_number": 1,
        "text": (
            "Section 1: Business Travel Meals and Per Diem. "
            "Employees on authorized overnight business travel are reimbursed up to a maximum daily domestic per diem "
            "meal allowance of $75. Itemized receipts are required for any single meal expense exceeding $25."
        ),
    },
    {
        "filename": "Expense_Policy.pdf",
        "page_number": 2,
        "text": (
            "Section 2: Lodging Standards and Caps. "
            "Domestic hotel accommodations are capped at $250 per night before standard taxes and resort fees. "
            "Lodging exceeding $250 per night requires prior written approval from the Department Vice President."
        ),
    },
    {
        "filename": "Expense_Policy.pdf",
        "page_number": 3,
        "text": (
            "Section 3: Expense Submission Deadlines. "
            "Employees must submit corporate expense reports with all supporting receipts within 30 calendar days "
            "following the completion of business travel or expense incurrence. Unsubmitted claims older than 60 days are forfeited."
        ),
    },
]


def load_eval_questions(csv_path: str) -> List[Dict[str, Any]]:
    """Load evaluation questions from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Evaluation questions file not found at: {csv_path}")

    questions = []
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "question_id": row["question_id"].strip(),
                "question": row["question"].strip(),
                "is_answerable": int(row["is_answerable"].strip()),
                "policy_domain": row["policy_domain"].strip(),
                "expected_filename": row["expected_filename"].strip(),
                "expected_page": int(row["expected_page"].strip()) if row["expected_page"].strip().isdigit() else 0,
                "ground_truth_keyword": row["ground_truth_keyword"].strip(),
            })
    return questions


def check_retrieval_hit(
    passages: List[Dict[str, Any]],
    expected_filename: str,
    expected_page: int,
    keywords_raw: str,
) -> bool:
    """
    Check if any of the top retrieved passages match the expected filename,
    expected page, and contain any of the ground-truth keywords.
    """
    if not passages:
        return False

    keywords = [k.strip().lower() for k in keywords_raw.split("|") if k.strip()]

    for p in passages:
        p_file = p.get("filename", "")
        p_page = p.get("page_number", 0)
        p_text = p.get("text", "").lower()

        # Check filename match
        file_match = (expected_filename.lower() in p_file.lower())
        # Check page match (allow exact page match)
        page_match = (expected_page == p_page)

        # Check keyword match
        keyword_match = any(kw in p_text for kw in keywords) if keywords else True

        if file_match and page_match and keyword_match:
            return True

    return False


def run_evaluation(csv_path: str) -> Dict[str, Any]:
    """Execute evaluation and compute Hit@5 and abstention metrics."""
    print("=" * 72)
    print("[BENCHMARK] BANK POLICY RAG EVALUATION BENCHMARK")
    print("=" * 72)

    # 1. Ingest corpus and index
    chunks = chunk_document_pages(SAMPLE_POLICY_PAGES)
    retriever = TfidfRetriever(similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD)
    retriever.fit(chunks)

    print(f"Corpus indexed: {len(SAMPLE_POLICY_PAGES)} pages -> {len(chunks)} text chunks.")
    print(f"Retrieval similarity threshold: {DEFAULT_SIMILARITY_THRESHOLD}\n")

    # 2. Load questions
    questions = load_eval_questions(csv_path)
    print(f"Loaded {len(questions)} evaluation questions from {csv_path}.\n")

    answerable_count = 0
    unanswerable_count = 0

    hits_at_5 = 0
    true_abstentions = 0      # Unanswerable correctly abstained (TN)
    false_positives = 0       # Unanswerable incorrectly retrieved (FP)
    false_abstentions = 0     # Answerable incorrectly abstained (FN)
    true_positives = 0        # Answerable retrieved (TP)

    print(f"{'ID':<5} | {'Type':<12} | {'Abstain?':<9} | {'Hit@5':<6} | Question")
    print("-" * 72)

    for q in questions:
        qid = q["question_id"]
        is_ans = q["is_answerable"] == 1
        q_type = "Answerable" if is_ans else "Unanswerable"

        passages, has_evidence = retriever.retrieve(q["question"], top_k=5)
        abstained = not has_evidence

        hit = False
        if is_ans:
            answerable_count += 1
            if abstained:
                false_abstentions += 1
                hit_str = "N/A"
            else:
                true_positives += 1
                hit = check_retrieval_hit(
                    passages,
                    q["expected_filename"],
                    q["expected_page"],
                    q["ground_truth_keyword"],
                )
                if hit:
                    hits_at_5 += 1
                hit_str = "YES" if hit else "NO"
        else:
            unanswerable_count += 1
            if abstained:
                true_abstentions += 1
                hit_str = "N/A"
            else:
                false_positives += 1
                hit_str = "FAIL"

        abstain_str = "YES" if abstained else "NO"
        print(f"{qid:<5} | {q_type:<12} | {abstain_str:<9} | {hit_str:<6} | {q['question'][:40]}...")

    # Calculate percentages
    hit_at_5_rate = (hits_at_5 / answerable_count * 100.0) if answerable_count > 0 else 0.0
    true_abstention_rate = (true_abstentions / unanswerable_count * 100.0) if unanswerable_count > 0 else 0.0
    false_abstention_rate = (false_abstentions / answerable_count * 100.0) if answerable_count > 0 else 0.0
    abstention_accuracy = (
        (true_positives + true_abstentions) / len(questions) * 100.0
    ) if questions else 0.0

    print("\n" + "=" * 72)
    print("[SUMMARY] EVALUATION RESULTS SUMMARY")
    print("=" * 72)
    print(f"Total Evaluated Questions:              {len(questions)}")
    print(f"  * Answerable Questions:                {answerable_count}")
    print(f"  * Unanswerable Questions:              {unanswerable_count}")
    print("-" * 72)
    print(f"Retrieval Hit@5 (Answerable):            {hits_at_5}/{answerable_count} ({hit_at_5_rate:.1f}%)")
    print(f"True Abstention Rate (Unanswerable):     {true_abstentions}/{unanswerable_count} ({true_abstention_rate:.1f}%)")
    print(f"False Abstention Rate (Answerable):      {false_abstentions}/{answerable_count} ({false_abstention_rate:.1f}%)")
    print(f"Overall Abstention Decision Accuracy:    {abstention_accuracy:.1f}%")
    print("=" * 72)

    # Human Review Metrics Guidance
    print("\n" + "=" * 72)
    print("[REVIEW] METRICS STILL REQUIRING MANUAL HUMAN REVIEW")
    print("=" * 72)
    print(
        "1. Faithfulness / Hallucination Rate:\n"
        "   Automated retrieval confirms relevant text was found; an auditor must confirm\n"
        "   that the synthesized LLM answer contains no ungrounded claims or numerical errors.\n\n"
        "2. Citation Precision & Attribution:\n"
        "   Verifying whether each claim directly maps to the specific page and section cited\n"
        "   rather than attributing facts to unrelated sections in the same document.\n\n"
        "3. Completeness of Policy Nuance & Exceptions:\n"
        "   Bank compliance rules involve conditional exceptions (e.g. 'requires VP sign-off\n"
        "   except when under $5,000'). Manual audit confirms edge conditions were not omitted.\n\n"
        "4. Regulatory Tone & Risk Posture:\n"
        "   Assessing that generated summaries use appropriate compliance terminology and do\n"
        "   not give absolute guarantees where policies dictate discretionary review."
    )
    print("=" * 72 + "\n")

    return {
        "hit_at_5_rate": hit_at_5_rate,
        "true_abstention_rate": true_abstention_rate,
        "false_abstention_rate": false_abstention_rate,
        "abstention_accuracy": abstention_accuracy,
    }


if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "evaluation", "eval_questions.csv")
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    run_evaluation(csv_file)
