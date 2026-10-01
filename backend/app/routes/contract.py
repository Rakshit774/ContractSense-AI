# ─────────────────────────────────────────────────────────────────────
#  ContractSenseAI · app/routes/contract.py
#
#  POST /contract/upload              — upload + process PDF
#  POST /contract/query               — ask a question (RAG)
#  GET  /contract/risk/{document_id}  — get clause risk summary
#  GET  /contract/summary/{document_id} — plain language summary
#  POST /contract/counter-proposal    — generate counter-proposal
#  GET  /contract/history             — list user's past documents
#  DELETE /contract/{document_id}     — delete a document
# ─────────────────────────────────────────────────────────────────────
import os
import uuid
import shutil
from fastapi import (APIRouter, Depends, HTTPException,
                     UploadFile, File, status)
from sqlalchemy.orm import Session

from ..database import get_db
from ..models   import User, Document
from ..schemas  import (DocumentResponse, QueryRequest, QueryResponse,
                        RiskSummaryResponse, CounterProposalRequest,
                        CounterProposalResponse)
from ..auth     import get_current_user
from .. import rag

router = APIRouter(prefix="/contract", tags=["Contracts"])

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── POST /contract/upload ─────────────────────────────────────────────
@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_contract(
    file        : UploadFile = File(...),
    db          : Session    = Depends(get_db),
    current_user: User       = Depends(get_current_user),
):
    """
    Upload a PDF contract.
    Pipeline: save → extract text → chunk → embed → classify → ChromaDB
    Returns document metadata including risk count breakdown.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "Only PDF files are accepted",
        )

    # Generate a unique document ID
    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    save_path   = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")

    # Save uploaded file temporarily
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Run the full RAG pipeline
        result = rag.process_contract(save_path, document_id)

        # Generate plain language summary
        summary_text = rag.generate_summary(result["full_text"])

        # Save document record to database
        doc = Document(
            document_id = document_id,
            filename    = file.filename,
            user_id     = current_user.id,
            page_count  = result["page_count"],
            chunk_count = result["chunk_count"],
            risk_counts = result["risk_counts"],
            summary     = summary_text,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

    except Exception as e:
        # Clean up on failure
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Processing failed: {str(e)}",
        )
    finally:
        # Always delete the temp file after processing
        if os.path.exists(save_path):
            os.remove(save_path)

    return doc


# ── POST /contract/query ──────────────────────────────────────────────
@router.post("/query", response_model=QueryResponse)
def query_contract(
    payload     : QueryRequest,
    db          : Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """
    Ask a question about a specific contract.
    Retrieves relevant chunks via ChromaDB, answers via Groq (RAG).
    """
    # Verify the document belongs to this user
    doc = db.query(Document).filter(
        Document.document_id == payload.document_id,
        Document.user_id     == current_user.id,
    ).first()

    if not doc:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "Document not found or access denied",
        )

    result = rag.ask_contract(
        question    = payload.question,
        document_id = payload.document_id,
        top_k       = payload.top_k,
    )

    return QueryResponse(
        question         = payload.question,
        answer           = result["answer"],
        chunks_used      = result["chunks_used"],
        top_similarity   = result["top_similarity"],
        retrieved_chunks = result["retrieved_chunks"],
    )


# ── GET /contract/risk/{document_id} ─────────────────────────────────
@router.get("/risk/{document_id}", response_model=RiskSummaryResponse)
def get_risk_summary(
    document_id : str,
    db          : Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """Return all RED and YELLOW clauses for a document."""

    doc = db.query(Document).filter(
        Document.document_id == document_id,
        Document.user_id     == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return rag.get_risk_summary(document_id)


# ── GET /contract/summary/{document_id} ──────────────────────────────
@router.get("/summary/{document_id}")
def get_summary(
    document_id : str,
    db          : Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """Return the plain language summary for a document."""

    doc = db.query(Document).filter(
        Document.document_id == document_id,
        Document.user_id     == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"document_id": document_id, "summary": doc.summary}


# ── POST /contract/counter-proposal ──────────────────────────────────
@router.post("/counter-proposal", response_model=CounterProposalResponse)
def counter_proposal(
    payload     : CounterProposalRequest,
    current_user: User = Depends(get_current_user),
):
    """Generate a fair counter-proposal for a risky clause."""

    result = rag.generate_counter_proposal(payload.clause_text)
    return CounterProposalResponse(
        original_clause  = payload.clause_text,
        counter_proposal = result,
    )


# ── GET /contract/history ─────────────────────────────────────────────
@router.get("/history", response_model=list[DocumentResponse])
def get_history(
    db          : Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """Return all documents uploaded by the current user."""

    docs = db.query(Document).filter(
        Document.user_id == current_user.id
    ).order_by(Document.uploaded_at.desc()).all()

    return docs


# ── DELETE /contract/{document_id} ───────────────────────────────────
@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id : str,
    db          : Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """Delete a document from the database and ChromaDB."""

    doc = db.query(Document).filter(
        Document.document_id == document_id,
        Document.user_id     == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from ChromaDB
    try:
        collection = rag.get_collection()
        existing   = collection.get(where={"document_id": document_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    db.delete(doc)
    db.commit()
