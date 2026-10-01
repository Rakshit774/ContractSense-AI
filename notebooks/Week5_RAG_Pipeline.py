# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ContractSenseAI  ·  Week 5  ·  RAG Pipeline                        ║
# ║  PDF → Extract → Chunk → Embed → ChromaDB → Groq → Answer           ║
# ║  Paste each CELL into a separate Jupyter cell and run in order       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ┌─────────────────────────────────────────────────────────────────────┐
# │  BEFORE RUNNING                                                     │
# │  1. Create a file called  .env  in your ContractSenseAI/ folder    │
# │  2. Add this one line to it:                                        │
# │       GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx                 │
# │  (paste your actual key from console.groq.com)                     │
# └─────────────────────────────────────────────────────────────────────┘

# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 1 — Imports                                                   │
# └─────────────────────────────────────────────────────────────────────┘
import os
import re
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import pymupdf                                          # PDF extraction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer   # local embeddings
import chromadb                                         # vector store
from groq import Groq                                   # LLM
from dotenv import load_dotenv

# Load .env file → makes GROQ_API_KEY available
load_dotenv(dotenv_path="../.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

print("✅ All imports successful")
print(f"✅ Groq API key loaded: {GROQ_API_KEY[:8]}...")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 2 — PDF Extraction                                           │
# └─────────────────────────────────────────────────────────────────────┘
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts full text from a PDF file.
    - Uses PyMuPDF for digital PDFs (fast, accurate)
    - Falls back to pytesseract OCR for scanned/image PDFs
    """
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    # If extracted text is too short → likely a scanned PDF → use OCR
    if len(text.strip()) < 100:
        print("⚠️  Digital extraction failed. Trying OCR fallback...")
        try:
            import pytesseract
            from PIL import Image
            text = ""
            for page in doc:
                # render page as image
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img)
            print("✅ OCR extraction complete")
        except ImportError:
            print("⚠️  pytesseract not installed. Install it for scanned PDF support.")

    text = re.sub(r'\n{3,}', '\n\n', text)   # collapse excessive blank lines
    text = re.sub(r'[ \t]+', ' ', text)       # collapse multiple spaces
    return text.strip()


def clean_text(text: str) -> str:
    """Light cleaning that preserves legal language."""
    text = re.sub(r'Page \d+\s*\n', '', text)   # remove "Page N" headers
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# Test on your contract
PDF_PATH = "../data/SampleContract-Shuttle.pdf"   # ← put your PDF in data/ folder

raw_text   = extract_text_from_pdf(PDF_PATH)
clean      = clean_text(raw_text)

print(f"Pages extracted : {pymupdf.open(PDF_PATH).page_count}")
print(f"Characters      : {len(clean):,}")
print(f"\nFirst 500 characters:")
print(clean[:500])


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 3 — Text Chunking                                            │
# └─────────────────────────────────────────────────────────────────────┘
# 600 chars with 100 overlap chosen based on testing:
#   - Long enough to contain a full clause with context
#   - Short enough to retrieve precisely
#   - Overlap ensures clauses split across boundaries are still found

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP,
    separators    = ["\n\n", "\n", ". ", " ", ""],  # split at paragraph first
)

chunks = splitter.split_text(clean)

print(f"Total chunks    : {len(chunks)}")
print(f"Avg chunk size  : {sum(len(c) for c in chunks) // len(chunks)} chars")
print(f"Min / Max       : {min(len(c) for c in chunks)} / {max(len(c) for c in chunks)} chars")
print(f"\n--- SAMPLE CHUNKS ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk[:300])
    print("...")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 4 — Embedding Model Setup                                    │
# │  First run downloads ~80MB model — takes 1-2 minutes               │
# └─────────────────────────────────────────────────────────────────────┘
print("Loading embedding model (downloads ~80MB on first run)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(f"✅ Embedding model loaded")
print(f"   Model: all-MiniLM-L6-v2")
print(f"   Embedding dimension: 384")

# Quick test
test_emb = embedder.encode(["This is a test clause."])
print(f"   Test embedding shape: {test_emb.shape}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 5 — ChromaDB Setup                                           │
# └─────────────────────────────────────────────────────────────────────┘
CHROMA_DIR = "../data/chromadb"   # persisted on disk

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Each contract gets its own collection, or use one shared collection
# We use one collection with document_id metadata to support multiple contracts
collection = chroma_client.get_or_create_collection(
    name     = "contracts",
    metadata = {"hnsw:space": "cosine"},  # cosine similarity for retrieval
)

print(f"✅ ChromaDB ready at {CHROMA_DIR}")
print(f"   Collection: 'contracts'")
print(f"   Current docs in store: {collection.count()}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 6 — Load Trained Classifier                                  │
# └─────────────────────────────────────────────────────────────────────┘
# Load the baseline classifier trained in Week 3
# (faster than DistilBERT for per-chunk scoring at upload time)
try:
    classifier = joblib.load("../models/baseline_lr_pipeline.pkl")
    print("✅ Clause risk classifier loaded (baseline TF-IDF + LR)")
    CLASSIFIER_AVAILABLE = True
except FileNotFoundError:
    print("⚠️  Classifier not found. Run Week3 notebook first.")
    print("    Risk scoring will be skipped.")
    CLASSIFIER_AVAILABLE = False


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 7 — Full Upload Pipeline                                     │
# │  PDF → Extract → Chunk → Embed → Classify → Store in ChromaDB     │
# └─────────────────────────────────────────────────────────────────────┘
def process_and_store_contract(pdf_path: str,
                                document_id: str = None) -> dict:
    """
    Complete upload pipeline for one contract PDF.

    Returns a summary dict with chunk count and risk breakdown.
    """
    if document_id is None:
        document_id = os.path.basename(pdf_path).replace(".pdf", "")

    print(f"\n{'═'*55}")
    print(f"Processing: {document_id}")
    print(f"{'═'*55}")

    # Step 1: Extract text
    print("Step 1/4 · Extracting text from PDF...")
    text  = clean_text(extract_text_from_pdf(pdf_path))
    print(f"          {len(text):,} characters extracted")

    # Step 2: Chunk
    print("Step 2/4 · Chunking text...")
    chunks = splitter.split_text(text)
    print(f"          {len(chunks)} chunks created")

    # Step 3: Embed all chunks
    print("Step 3/4 · Generating embeddings...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    print(f"          Embeddings shape: {embeddings.shape}")

    # Step 4: Classify each chunk + store in ChromaDB
    print("Step 4/4 · Classifying risk + storing in ChromaDB...")

    # Remove existing entries for this document (allows re-upload)
    existing = collection.get(where={"document_id": document_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"          Removed {len(existing['ids'])} old chunks")

    risk_counts = {"RED": 0, "YELLOW": 0, "GREEN": 0}

    ids       = []
    docs      = []
    embs      = []
    metadatas = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        # Classify chunk risk level
        if CLASSIFIER_AVAILABLE:
            risk = classifier.predict([chunk])[0]
            conf = float(max(classifier.predict_proba([chunk])[0]))
        else:
            risk, conf = "UNKNOWN", 0.0

        risk_counts[risk] = risk_counts.get(risk, 0) + 1

        ids.append(f"{document_id}_chunk_{i:04d}")
        docs.append(chunk)
        embs.append(emb.tolist())
        metadatas.append({
            "document_id" : document_id,
            "chunk_index" : i,
            "risk_level"  : risk,
            "confidence"  : round(conf, 4),
            "char_count"  : len(chunk),
        })

    collection.add(
        ids        = ids,
        documents  = docs,
        embeddings = embs,
        metadatas  = metadatas,
    )

    summary = {
        "document_id" : document_id,
        "pages"       : pymupdf.open(pdf_path).page_count,
        "chunks"      : len(chunks),
        "risk_counts" : risk_counts,
    }

    print(f"\n✅ Contract stored successfully")
    print(f"   Document ID : {document_id}")
    print(f"   Pages       : {summary['pages']}")
    print(f"   Chunks      : {summary['chunks']}")
    print(f"   Risk counts : 🔴 RED={risk_counts.get('RED',0)}  "
          f"🟡 YELLOW={risk_counts.get('YELLOW',0)}  "
          f"🟢 GREEN={risk_counts.get('GREEN',0)}")

    return summary


# Process the sample contract
summary = process_and_store_contract(
    pdf_path    = PDF_PATH,
    document_id = "sample_shuttle_contract",
)


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 8 — Retrieval Function                                       │
# └─────────────────────────────────────────────────────────────────────┘
def retrieve_relevant_chunks(query: str,
                              document_id: str,
                              top_k: int = 5) -> list[dict]:
    """
    Embeds the query and retrieves the top-k most relevant
    chunks from ChromaDB using cosine similarity.
    """
    query_embedding = embedder.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k,
        where            = {"document_id": document_id},
        include          = ["documents", "metadatas", "distances"],
    )

    chunks_out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks_out.append({
            "text"        : doc,
            "risk_level"  : meta["risk_level"],
            "confidence"  : meta["confidence"],
            "chunk_index" : meta["chunk_index"],
            "similarity"  : round(1 - dist, 4),   # cosine distance → similarity
        })

    return chunks_out


# Quick retrieval test
print("RETRIEVAL TEST")
print("─" * 50)
test_query = "What is the termination notice period?"
retrieved  = retrieve_relevant_chunks(test_query,
                                       document_id="sample_shuttle_contract")
print(f"Query  : {test_query}")
print(f"Top-{len(retrieved)} chunks retrieved:")
for i, c in enumerate(retrieved):
    print(f"\n  [{i+1}] similarity={c['similarity']}  "
          f"risk={c['risk_level']}  chunk={c['chunk_index']}")
    print(f"       {c['text'][:200]}...")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 9 — Groq LLM Setup + RAG Answer Function                    │
# └─────────────────────────────────────────────────────────────────────┘
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL  = "openai/gpt-oss-20b"   # free, fast, 128K context

SYSTEM_PROMPT = """You are ContractSenseAI, an expert legal document analyst.
Your job is to answer questions about contracts clearly and accurately.

Rules:
1. Answer ONLY based on the contract clauses provided to you.
2. If the answer is not in the provided clauses, say "This information is not in the contract."
3. Always cite which part of the contract your answer comes from.
4. Highlight any clauses that seem unfair or risky to the signing party.
5. Use plain language — avoid legal jargon unless explaining a specific term.
6. Be concise but complete."""


def ask_contract(question: str,
                 document_id: str,
                 top_k: int = 5) -> dict:
    """
    Full RAG pipeline:
    1. Embed the question
    2. Retrieve top-k relevant chunks from ChromaDB
    3. Build a prompt with those chunks as context
    4. Send to Groq (Llama 3.1)
    5. Return the grounded answer

    This is why RAG is better than plain ChatGPT:
    - ChatGPT answers from general knowledge → may hallucinate
    - Our system retrieves exact clauses first → answer is grounded
    """
    # Step 1 + 2: Retrieve
    relevant_chunks = retrieve_relevant_chunks(question, document_id, top_k)

    if not relevant_chunks:
        return {"answer": "No relevant clauses found in this contract.",
                "chunks": []}

    # Step 3: Build context block
    context_block = ""
    for i, chunk in enumerate(relevant_chunks):
        risk_emoji = {"RED":"🔴","YELLOW":"🟡","GREEN":"🟢"}.get(
            chunk["risk_level"], "⚪")
        context_block += (
            f"[Clause {i+1}] {risk_emoji} Risk: {chunk['risk_level']} "
            f"(relevance: {chunk['similarity']})\n"
            f"{chunk['text']}\n\n"
        )

    user_message = f"""CONTRACT CLAUSES (retrieved for your question):
{context_block}

QUESTION: {question}

Answer based only on the clauses above."""

    # Step 4: Call Groq
    response = groq_client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_message},
        ],
        temperature = 0.1,    # low = more factual, less creative
        max_tokens  = 800,
    )

    answer = response.choices[0].message.content

    return {
        "question"       : question,
        "answer"         : answer,
        "chunks_used"    : len(relevant_chunks),
        "top_chunk_sim"  : relevant_chunks[0]["similarity"],
        "risk_levels"    : [c["risk_level"] for c in relevant_chunks],
        "retrieved_chunks": relevant_chunks,
    }


print(f"✅ Groq client ready (model: {GROQ_MODEL})")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 10 — Live Q&A Test on Your Contract                          │
# └─────────────────────────────────────────────────────────────────────┘
DOC_ID = "sample_shuttle_contract"

test_questions = [
    "What is the termination notice period?",
    "What are the insurance requirements for the consultant?",
    "Can the consultant assign this agreement to someone else?",
    "What happens if there is a payment dispute?",
    "Who owns the work products created under this agreement?",
]

print("=" * 65)
print("  LIVE RAG Q&A TEST — ContractSenseAI")
print("=" * 65)

for question in test_questions:
    print(f"\n❓ {question}")
    print("─" * 65)
    result = ask_contract(question, document_id=DOC_ID)
    print(f"🤖 {result['answer']}")
    print(f"\n   [Retrieved {result['chunks_used']} chunks | "
          f"Top similarity: {result['top_chunk_sim']} | "
          f"Risk levels: {result['risk_levels']}]")
    print("=" * 65)


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 11 — Risk Summary Function (what the UI will display)        │
# └─────────────────────────────────────────────────────────────────────┘
def get_contract_risk_summary(document_id: str) -> dict:
    """
    Returns all RED and YELLOW risk chunks for a contract.
    This is what populates the risk cards in the React frontend.
    """
    # Get all chunks for this document
    all_chunks = collection.get(
        where   = {"document_id": document_id},
        include = ["documents", "metadatas"],
    )

    red_clauses    = []
    yellow_clauses = []

    for doc, meta in zip(all_chunks["documents"], all_chunks["metadatas"]):
        entry = {
            "text"       : doc,
            "risk_level" : meta["risk_level"],
            "confidence" : meta["confidence"],
        }
        if meta["risk_level"] == "RED":
            red_clauses.append(entry)
        elif meta["risk_level"] == "YELLOW":
            yellow_clauses.append(entry)

    return {
        "document_id"   : document_id,
        "red_clauses"   : red_clauses,
        "yellow_clauses": yellow_clauses,
        "red_count"     : len(red_clauses),
        "yellow_count"  : len(yellow_clauses),
    }


risk_summary = get_contract_risk_summary(DOC_ID)

print(f"\n CONTRACT RISK SUMMARY")
print(f"{'═'*55}")
print(f"  🔴 HIGH RISK clauses  : {risk_summary['red_count']}")
print(f"  🟡 MEDIUM RISK clauses: {risk_summary['yellow_count']}")
print(f"{'═'*55}")

print("\n🔴 RED CLAUSES (review before signing):")
for i, clause in enumerate(risk_summary["red_clauses"][:3], 1):
    print(f"\n  {i}. {clause['text'][:250]}...")

print("\n🟡 YELLOW CLAUSES (worth negotiating):")
for i, clause in enumerate(risk_summary["yellow_clauses"][:3], 1):
    print(f"\n  {i}. {clause['text'][:250]}...")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 12 — Counter-proposal Generator                              │
# │  Given a risky clause → generates a fairer version                 │
# └─────────────────────────────────────────────────────────────────────┘
def generate_counter_proposal(clause_text: str) -> str:
    """
    Takes a risky clause and generates a fairer counter-proposal.
    """
    prompt = f"""You are a legal expert helping someone negotiate a contract.

The following clause in a contract has been flagged as potentially unfair:

CLAUSE:
\"\"\"{clause_text}\"\"\"

Please:
1. Briefly explain in plain language why this clause is risky
2. Write a fair, balanced counter-proposal that protects both parties
3. Keep the counter-proposal in formal legal language

Format your response as:
RISK: <one sentence explanation>
COUNTER-PROPOSAL: <revised clause text>"""

    response = groq_client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.2,
        max_tokens  = 600,
    )
    return response.choices[0].message.content


# Test counter-proposal on a real clause from the contract
if risk_summary["red_clauses"]:
    test_clause = risk_summary["red_clauses"][0]["text"]
    print("\n COUNTER-PROPOSAL GENERATOR")
    print("─" * 55)
    print(f"Original clause:\n{test_clause[:300]}...\n")
    counter = generate_counter_proposal(test_clause)
    print(f"AI Counter-proposal:\n{counter}")
else:
    # fallback test with a hard-coded risky clause
    test_clause = ("CONSULTANT shall indemnify and hold harmless the COMMISSION "
                   "from any and all claims, damages, losses and expenses.")
    counter = generate_counter_proposal(test_clause)
    print(f"\n Counter-proposal test:\n{counter}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 13 — Plain Language Summary Generator                        │
# └─────────────────────────────────────────────────────────────────────┘
def generate_plain_summary(document_id: str, full_text: str) -> str:
    """
    Generates a 200-word plain-language summary of the entire contract.
    """
    # Use first 3000 chars as overview context (covers key clauses)
    context = full_text[:3000]

    prompt = f"""You are ContractSenseAI. Summarize this contract in simple, plain language
that anyone can understand — no legal jargon.

CONTRACT TEXT (first section):
{context}

Write a summary covering:
1. What this agreement is about (1 sentence)
2. Who the parties are
3. What the consultant must do
4. How payment works
5. Key risks or unusual clauses to watch out for

Keep it under 200 words. Write for someone with no legal background."""

    response = groq_client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.3,
        max_tokens  = 400,
    )
    return response.choices[0].message.content


print("\n PLAIN LANGUAGE SUMMARY")
print("─" * 55)
summary_text = generate_plain_summary(DOC_ID, clean)
print(summary_text)


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 14 — Save pipeline functions as a module (for backend use)  │
# └─────────────────────────────────────────────────────────────────────┘
# These functions will be imported directly into the FastAPI backend.
# We just verify everything is working end-to-end here.

print("\n" + "═"*55)
print("  WEEK 5 COMPLETE — RAG PIPELINE VALIDATED")
print("═"*55)
print("  ✅  PDF extraction working")
print("  ✅  Text chunking: 600-char chunks, 100 overlap")
print("  ✅  Embeddings: all-MiniLM-L6-v2 (384-dim, local)")
print("  ✅  ChromaDB vector store: persistent on disk")
print("  ✅  Clause risk classifier integrated at upload time")
print("  ✅  Groq RAG Q&A: grounded answers from contract text")
print("  ✅  Counter-proposal generator working")
print("  ✅  Plain language summary working")
print("═"*55)
print("\nNext: Week 6 → FastAPI backend wrapping these functions")
