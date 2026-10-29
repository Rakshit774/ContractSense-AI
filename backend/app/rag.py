
# ============================================================
# ContractSenseAI - app/rag.py
# RAG pipeline, risk analysis, AI explanations and summarization
# ============================================================

import os
import re
from pathlib import Path

import joblib
import pymupdf
import chromadb

from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ============================================================
# CONSTANTS
# ============================================================

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent

CHROMA_DIR = str(BACKEND_DIR / "chromadb_store")
CLASSIFIER_PATH = PROJECT_DIR / "models" / "baseline_lr_pipeline.pkl"

REVIEW_CONFIDENCE_THRESHOLD = 0.55


SYSTEM_PROMPT = """
You are ContractSenseAI, an assistant that explains contracts clearly.

Answer questions using only the supplied contract passages.
If the answer is not supported by those passages, say:
"This information is not available in the retrieved contract passages."

Use plain English, preserve exact dates and amounts, distinguish the
obligations of each party, and identify the relevant clause or section
when the supplied text provides that information.

Do not invent contractual facts or provide definitive legal conclusions.
"""


# ============================================================
# SINGLETON LOADERS
# ============================================================

_embedder = None
_chroma = None
_collection = None
_classifier = None
_groq = None


def get_embedder():
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)

    return _embedder


def get_collection():
    global _chroma, _collection

    if _collection is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)

        _chroma = chromadb.PersistentClient(path=CHROMA_DIR)

        _collection = _chroma.get_or_create_collection(
            name="contracts",
            metadata={"hnsw:space": "cosine"},
        )

    return _collection


def get_classifier():
    global _classifier

    if _classifier is None and CLASSIFIER_PATH.exists():
        _classifier = joblib.load(CLASSIFIER_PATH)

    return _classifier


def get_groq():
    global _groq

    if _groq is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Check your backend .env file."
            )

        _groq = Groq(api_key=api_key)

    return _groq


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF, with an optional OCR fallback."""

    with pymupdf.open(pdf_path) as doc:
        text = "\n\n".join(page.get_text() for page in doc)

        if len(text.strip()) < 100:
            try:
                import pytesseract
                from PIL import Image

                extracted_pages = []

                for page in doc:
                    pix = page.get_pixmap(dpi=200, alpha=False)

                    image = Image.frombytes(
                        "RGB",
                        [pix.width, pix.height],
                        pix.samples,
                    )

                    extracted_pages.append(
                        pytesseract.image_to_string(image)
                    )

                ocr_text = "\n\n".join(extracted_pages)

                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text

            except ImportError:
                pass

    text = re.sub(r"Page \d+\s*\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(text: str) -> list[str]:
    """Split extracted contract text into overlapping chunks."""

    if not text or not text.strip():
        return []

    return _splitter.split_text(text)


# ============================================================
# REVIEW FLAGS
# ============================================================

def _review_flags(text: str, confidence: float) -> list[str]:
    """
    Suggest manual review when confidence is low or text resembles
    a generic heading. These are heuristics, not legal conclusions.
    """

    reasons = []
    normalized = (text or "").strip()

    if confidence < REVIEW_CONFIDENCE_THRESHOLD:
        reasons.append(
            "The classifier confidence is below the review threshold."
        )

    generic_patterns = [
        r"^\s*(definitions?|general provisions?|miscellaneous)\s*$",
        r"^\s*(terms and conditions|agreement|appendix|exhibit)\s*$",
        r"^\s*(insurance|termination|payment|confidentiality)\s*:?\s*$",
    ]

    if any(
        re.match(pattern, normalized, flags=re.IGNORECASE)
        for pattern in generic_patterns
    ):
        reasons.append(
            "This text appears to be a short heading rather than "
            "a complete clause."
        )

    if not normalized:
        reasons.append("The clause text is empty.")

    return reasons


# ============================================================
# CONTRACT UPLOAD PIPELINE
# ============================================================

def process_contract(pdf_path: str, document_id: str) -> dict:
    """Extract, chunk, embed, classify and store a contract."""

    text = extract_text(pdf_path)

    if not text:
        raise ValueError(
            "No readable text was extracted from this PDF. "
            "If it is scanned, install and configure OCR dependencies."
        )

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No contract text chunks could be created.")

    embedder = get_embedder()
    collection = get_collection()
    classifier = get_classifier()

    embeddings = embedder.encode(
        chunks,
        show_progress_bar=False,
    )

    existing = collection.get(
        where={"document_id": document_id}
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    risk_counts = {
        "RED": 0,
        "YELLOW": 0,
        "GREEN": 0,
    }

    ids = []
    documents = []
    embedding_values = []
    metadatas = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        risk = "UNKNOWN"
        confidence = 0.0

        if classifier is not None:
            risk = str(classifier.predict([chunk])[0]).upper()

            if hasattr(classifier, "predict_proba"):
                confidence = float(
                    max(classifier.predict_proba([chunk])[0])
                )

        if risk in risk_counts:
            risk_counts[risk] += 1

        ids.append(f"{document_id}_{i:04d}")
        documents.append(chunk)
        embedding_values.append(embedding.tolist())

        metadatas.append({
            "document_id": document_id,
            "chunk_index": i,
            "risk_level": risk,
            "confidence": round(confidence, 4),
        })

    batch_size = 500

    for start in range(0, len(ids), batch_size):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embedding_values[start:end],
            metadatas=metadatas[start:end],
        )

    with pymupdf.open(pdf_path) as doc:
        page_count = len(doc)

    return {
        "document_id": document_id,
        "page_count": page_count,
        "chunk_count": len(chunks),
        "risk_counts": risk_counts,
        "full_text": text,
    }


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(
    query: str,
    document_id: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve semantically relevant chunks from one document."""

    if not query or not query.strip():
        return []

    collection = get_collection()
    embedder = get_embedder()

    matching = collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    matching_count = len(matching["ids"])

    if matching_count == 0:
        return []

    number_to_retrieve = max(1, min(top_k, matching_count))
    query_embedding = embedder.encode([query])[0].tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=number_to_retrieve,
        where={"document_id": document_id},
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    if not result["documents"] or not result["documents"][0]:
        return []

    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        chunks.append({
            "text": document,
            "risk_level": metadata.get("risk_level", "UNKNOWN"),
            "similarity": round(1 - distance, 4),
            "chunk_index": metadata.get("chunk_index", 0),
        })

    return chunks


# ============================================================
# CONTRACT QUESTION ANSWERING
# ============================================================

def ask_contract(
    question: str,
    document_id: str,
    top_k: int = 5,
) -> dict:
    """Answer a question using retrieved contract passages."""

    chunks = retrieve_chunks(question, document_id, top_k)

    if not chunks:
        return {
            "answer": "No relevant contract passages were found.",
            "chunks_used": 0,
            "top_similarity": 0,
            "retrieved_chunks": [],
        }

    risk_emoji = {
        "RED": "🔴",
        "YELLOW": "🟡",
        "GREEN": "🟢",
        "UNKNOWN": "⚪",
    }

    context = "\n\n".join(
        (
            f"[Clause {i + 1}] "
            f"{risk_emoji.get(chunk['risk_level'], '⚪')} "
            f"Risk: {chunk['risk_level']} | "
            f"Similarity: {chunk['similarity']}\n"
            f"{chunk['text']}"
        )
        for i, chunk in enumerate(chunks)
    )

    user_message = (
        f"CONTRACT PASSAGES:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )

    response = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content or ""

    return {
        "answer": answer.strip(),
        "chunks_used": len(chunks),
        "top_similarity": chunks[0]["similarity"],
        "retrieved_chunks": chunks,
    }


# ============================================================
# RISK SUMMARY WITH REVIEW FLAGS
# ============================================================

def get_risk_summary(document_id: str) -> dict:
    """Return chunks grouped by predicted risk, with review suggestions."""

    collection = get_collection()

    all_data = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )

    groups = {
        "RED": [],
        "YELLOW": [],
        "GREEN": [],
        "UNKNOWN": [],
    }

    for document, metadata in zip(
        all_data["documents"],
        all_data["metadatas"],
    ):
        confidence = float(metadata.get("confidence", 0))
        risk = str(metadata.get("risk_level", "UNKNOWN")).upper()

        if risk not in groups:
            risk = "UNKNOWN"

        reasons = _review_flags(document, confidence)

        groups[risk].append({
            "text": document,
            "confidence": confidence,
            "chunk_index": metadata.get("chunk_index", 0),
            "review_required": bool(reasons),
            "review_reasons": reasons,
        })

    return {
        "document_id": document_id,
        "red_clauses": groups["RED"],
        "yellow_clauses": groups["YELLOW"],
        "green_clauses": groups["GREEN"],
        "unknown_clauses": groups["UNKNOWN"],
        "counts": {
            "RED": len(groups["RED"]),
            "YELLOW": len(groups["YELLOW"]),
            "GREEN": len(groups["GREEN"]),
            "UNKNOWN": len(groups["UNKNOWN"]),
        },
    }


# ============================================================
# AI EXPLANATION FOR ONE CLAUSE
# ============================================================

def explain_risk_clause(document_id: str, chunk_index: int) -> dict:
    """
    Generate a grounded explanation for one stored clause.
    The classifier's risk label is treated as a prediction, not a verdict.
    """

    collection = get_collection()

    data = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )

    clause_text = None
    metadata_found = None

    for document, metadata in zip(
        data.get("documents") or [],
        data.get("metadatas") or [],
    ):
        if int(metadata.get("chunk_index", -1)) == chunk_index:
            clause_text = document
            metadata_found = metadata
            break

    if clause_text is None or metadata_found is None:
        raise ValueError("The requested clause was not found.")

    risk_level = str(
        metadata_found.get("risk_level", "UNKNOWN")
    ).upper()

    confidence = float(metadata_found.get("confidence", 0))
    review_reasons = _review_flags(clause_text, confidence)

    prompt = f"""
Explain the following contract passage to a non-lawyer.

Predicted classifier risk label: {risk_level}
Classifier confidence: {confidence:.4f}

Contract passage:
<clause>
{clause_text}
</clause>

Return these headings:

## What this clause means
Explain the passage in simple English.

## Why it may need attention
Explain specific concerns supported by the supplied wording.
If there is no clear concern in the passage, say so.

## What to verify
List concrete questions or conditions the reader should check.

Rules:
- Use only the supplied passage.
- Do not invent missing clauses, dates, parties, or obligations.
- Do not say the clause is illegal or unenforceable.
- The predicted risk label is a machine-learning prediction, not a legal verdict.
- If the passage lacks enough context, explicitly say that.
- This is general information, not legal advice.
"""

    response = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You explain contract passages cautiously and accurately. "
                    "Never invent contractual facts or provide definitive "
                    "legal conclusions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=700,
    )

    explanation = response.choices[0].message.content or ""

    if not explanation.strip():
        raise RuntimeError("The AI returned an empty clause explanation.")

    return {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "clause_text": clause_text,
        "risk_level": risk_level,
        "confidence": confidence,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "explanation": explanation.strip(),
        "disclaimer": (
            "AI-generated explanation for review assistance only; "
            "not legal advice or a guarantee of classification accuracy."
        ),
    }


# ============================================================
# CONTRACT SUMMARY
# ============================================================

def generate_summary(full_text: str) -> str:
    """Generate a concise, structured and accurate contract summary."""

    if not full_text or not full_text.strip():
        return (
            "Unable to generate a summary because no contract text "
            "was extracted."
        )

    max_input_characters = 60000
    contract_text = full_text[:max_input_characters]
    was_truncated = len(full_text) > max_input_characters

    truncation_note = ""

    if was_truncated:
        truncation_note = (
            "\nIMPORTANT: The source text was truncated. "
            "Do not imply that all contract sections were reviewed."
        )

    prompt = f"""
Prepare a clear and accurate contract summary for a non-lawyer.

Keep the summary concise, around 250-350 words.
Use valid Markdown and complete sentences.

PRIORITIES:
- Payment terms and invoice deadlines.
- Termination rights and exact notice periods.
- Insurance conditions and indemnification.
- Record retention and audit requirements.
- Important obligations and material risks.

STRICT ACCURACY RULES:
- Use only facts supported by the supplied contract text.
- Never invent names, dates, amounts, deadlines, or obligations.
- Preserve exact notice periods, payment deadlines and monetary amounts.
- Clearly distinguish the Commission's obligations from the Consultant's.
- If a detail is absent, say "Not specified in the extracted text."
- Preserve all important conditions and exceptions.
- Do not describe conditional insurance as universally mandatory.
- Professional liability insurance must be described as conditional
  on both parties initialling that provision if the source says so.
- Preserve any exceptions to record retention and audit periods.
- Do not invent details from missing exhibits or blank placeholders.
- Explain risks neutrally; do not declare a clause illegal.
- Avoid duplicate points and generic filler.
- Complete every section and every bullet before ending.
- If space is limited, prioritize material contractual facts.
- Do not end in the middle of a sentence or bullet.
{truncation_note}

Use these headings:

## Contract Overview
Identify the parties, agreement purpose and missing identifying details.

## Key Obligations
Use concise bullets, distinguishing the Commission and Consultant.

## Payment Terms
Explain payment timing, invoice deadlines, fees, reimbursements
and conditions for payment.

## Termination and Notice Periods
Explain who may terminate, notice periods and cure periods.

## Important Risks to Review
Provide up to three material risks. Complete each risk item.

## Other Important Conditions
Briefly cover insurance, indemnification, confidentiality,
intellectual property, record retention, audits and disputes
when supported by the supplied text.

Before answering, verify that:
1. All numbers match the supplied text.
2. Obligations are attributed to the correct party.
3. Conditional clauses remain conditional.
4. No unsupported facts are presented as certain.
5. All sections and bullets are complete.

CONTRACT TEXT:
<contract>
{contract_text}
</contract>
"""

    try:
        response = get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ContractSenseAI, a careful contract "
                        "summarization assistant. Accuracy, clarity and "
                        "complete sentences are essential. Never invent facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=3000,
        )

        if not response.choices:
            print("ERROR generating summary: Groq returned no choices.")
            return (
                "The contract was processed, but the summary could not "
                "be generated. Please check the backend logs and try again."
            )

        choice = response.choices[0]
        summary = (choice.message.content or "").strip()
        finish_reason = choice.finish_reason

        print(
            f"Contract summary generation finished. "
            f"finish_reason={finish_reason}"
        )

        if not summary:
            return (
                "The contract was processed, but the summary could not "
                "be generated. Please try again."
            )

        if finish_reason == "length":
            summary += (
                "\n\n> **Coverage note:** The model reached its output "
                "limit. Some summary details may be incomplete."
            )

        if was_truncated:
            summary += (
                "\n\n> **Coverage note:** The extracted contract text "
                "exceeded the processing limit. Some later sections may "
                "not have been included."
            )

        return summary

    except Exception as e:
        print(
            f"ERROR generating contract summary: "
            f"{type(e).__name__}: {e}"
        )
        return (
            "The contract was processed, but the summary could not "
            "be generated. Please check the backend logs and try again."
        )


# ============================================================
# COUNTER-PROPOSAL
# ============================================================

def generate_counter_proposal(clause_text: str) -> str:
    """Explain a potentially risky clause and suggest balanced wording."""

    if not clause_text or not clause_text.strip():
        return "Please provide the clause text to review."

    prompt = f"""
A contract clause has been flagged for review.

Original clause:
\"\"\"
{clause_text}
\"\"\"

Using only the supplied clause:

1. Explain the potential concern in 1-2 plain-English sentences.
2. Suggest balanced alternative wording in formal contract language.
3. Do not claim that the clause is illegal or unenforceable.
4. Do not invent facts about the parties or their intentions.
5. Clearly label the output as a suggested draft for professional review.

Use this format:

## Potential Concern
[Explain the concern.]

## Suggested Counter-Proposal
[Provide revised wording.]

This is drafting assistance, not legal advice.
"""

    response = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You help explain contract language and draft balanced "
                    "alternatives without inventing facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )

    result = response.choices[0].message.content or ""

    if not result.strip():
        return (
            "A counter-proposal could not be generated. Please try again."
        )

    return result.strip()
