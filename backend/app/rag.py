# ─────────────────────────────────────────────────────────────────────
#  ContractSenseAI · app/rag.py
#  All RAG pipeline logic — imported by the contract routes
#  This is the Week 5 notebook converted into a clean module
# ─────────────────────────────────────────────────────────────────────
import os
import re
import joblib
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# ── Constants ─────────────────────────────────────────────────────────
CHUNK_SIZE     = 600
CHUNK_OVERLAP  = 100
EMBED_MODEL    = "all-MiniLM-L6-v2"
GROQ_MODEL     = "openai/gpt-oss-20b"   # free, fast, 128K context
CHROMA_DIR     = "./chromadb_store"
CLASSIFIER_PATH = "../models/baseline_lr_pipeline.pkl"

SYSTEM_PROMPT = """You are ContractSenseAI, an expert legal document analyst.
Answer questions ONLY based on the contract clauses provided.
If the answer is not in the clauses, say 'This information is not in the contract.'
Always cite which part of the contract your answer comes from.
Use plain language — avoid legal jargon unless explaining a specific term."""

# ── Singleton loaders (load once, reuse across requests) ──────────────
_embedder   = None
_chroma     = None
_collection = None
_classifier = None
_groq       = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_collection():
    global _chroma, _collection
    if _collection is None:
        _chroma     = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma.get_or_create_collection(
            name     = "contracts",
            metadata = {"hnsw:space": "cosine"},
        )
    return _collection


def get_classifier():
    global _classifier
    if _classifier is None and os.path.exists(CLASSIFIER_PATH):
        _classifier = joblib.load(CLASSIFIER_PATH)
    return _classifier


def get_groq():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq


# ── Text extraction ───────────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    doc  = pymupdf.open(pdf_path)
    text = "".join(page.get_text() for page in doc)

    if len(text.strip()) < 100:          # scanned PDF fallback
        try:
            import pytesseract
            from PIL import Image
            text = ""
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img)
        except ImportError:
            pass

    text = re.sub(r'Page \d+\s*\n', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size    = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP,
    separators    = ["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(text: str) -> list[str]:
    return _splitter.split_text(text)


# ── Upload pipeline ───────────────────────────────────────────────────
def process_contract(pdf_path: str, document_id: str) -> dict:
    """
    Full upload pipeline:
    PDF → extract → chunk → embed → classify → store in ChromaDB
    Returns summary dict.
    """
    embedder   = get_embedder()
    collection = get_collection()
    classifier = get_classifier()

    text   = extract_text(pdf_path)
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks)

    # Remove existing entries for this doc (safe re-upload)
    existing = collection.get(where={"document_id": document_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    risk_counts = {"RED": 0, "YELLOW": 0, "GREEN": 0}
    ids, docs, embs, metas = [], [], [], []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        risk, conf = "UNKNOWN", 0.0
        if classifier:
            risk = classifier.predict([chunk])[0]
            conf = float(max(classifier.predict_proba([chunk])[0]))
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

        ids.append(f"{document_id}_{i:04d}")
        docs.append(chunk)
        embs.append(emb.tolist())
        metas.append({
            "document_id": document_id,
            "chunk_index": i,
            "risk_level" : risk,
            "confidence" : round(conf, 4),
        })

    collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    return {
        "document_id": document_id,
        "page_count" : pymupdf.open(pdf_path).page_count,
        "chunk_count": len(chunks),
        "risk_counts": risk_counts,
        "full_text"  : text,
    }


# ── Retrieval ─────────────────────────────────────────────────────────
def retrieve_chunks(query: str, document_id: str, top_k: int = 5) -> list[dict]:
    embedder   = get_embedder()
    collection = get_collection()

    q_emb = embedder.encode([query])[0].tolist()
    res   = collection.query(
        query_embeddings = [q_emb],
        n_results        = top_k,
        where            = {"document_id": document_id},
        include          = ["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        chunks.append({
            "text"       : doc,
            "risk_level" : meta.get("risk_level", "UNKNOWN"),
            "similarity" : round(1 - dist, 4),
            "chunk_index": meta.get("chunk_index", 0),
        })
    return chunks


# ── RAG Q&A ───────────────────────────────────────────────────────────
def ask_contract(question: str, document_id: str, top_k: int = 5) -> dict:
    chunks = retrieve_chunks(question, document_id, top_k)
    if not chunks:
        return {"answer": "No relevant clauses found.", "chunks": []}

    risk_emoji = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}
    context    = "\n\n".join(
        f"[Clause {i+1}] {risk_emoji.get(c['risk_level'],'')} "
        f"Risk: {c['risk_level']} | Relevance: {c['similarity']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    user_msg = f"CONTRACT CLAUSES:\n{context}\n\nQUESTION: {question}"
    response = get_groq().chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature = 0.1,
        max_tokens  = 800,
    )
    return {
        "answer"          : response.choices[0].message.content,
        "chunks_used"     : len(chunks),
        "top_similarity"  : chunks[0]["similarity"],
        "retrieved_chunks": chunks,
    }


# ── Risk summary ──────────────────────────────────────────────────────
def get_risk_summary(document_id: str) -> dict:
    collection = get_collection()
    all_data   = collection.get(
        where   = {"document_id": document_id},
        include = ["documents", "metadatas"],
    )
    red, yellow, green = [], [], []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        entry = {"text": doc, "confidence": meta.get("confidence", 0)}
        risk  = meta.get("risk_level", "UNKNOWN")
        if risk == "RED":
            red.append(entry)
        elif risk == "YELLOW":
            yellow.append(entry)
        else:
            green.append(entry)

    return {
        "document_id"   : document_id,
        "red_clauses"   : red,
        "yellow_clauses": yellow,
        "green_clauses" : green,
        "counts"        : {"RED": len(red), "YELLOW": len(yellow), "GREEN": len(green)},
    }


# ── Plain language summary ────────────────────────────────────────────
def generate_summary(full_text: str) -> str:
    prompt = f"""Summarize this contract in plain language (under 200 words).
Cover: what the agreement is about, who the parties are, payment terms,
termination conditions, and top 2 risks to watch out for.

CONTRACT:
{full_text[:3000]}"""

    response = get_groq().chat.completions.create(
        model    = GROQ_MODEL,
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.3,
        max_tokens  = 400,
    )
    return response.choices[0].message.content


# ── Counter-proposal ──────────────────────────────────────────────────
def generate_counter_proposal(clause_text: str) -> str:
    prompt = f"""A legal clause has been flagged as potentially unfair:

\"\"\"{clause_text}\"\"\"

1. Briefly explain the risk in plain language (1-2 sentences)
2. Write a balanced counter-proposal in formal legal language

Format:
RISK: <explanation>
COUNTER-PROPOSAL: <revised clause>"""

    response = get_groq().chat.completions.create(
        model    = GROQ_MODEL,
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.2,
        max_tokens  = 500,
    )
    return response.choices[0].message.content
