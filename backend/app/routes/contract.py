
import os
import uuid
import shutil

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    status,
)
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, Document
from ..schemas import (
    DocumentResponse,
    QueryRequest,
    QueryResponse,
    RiskSummaryResponse,
    CounterProposalRequest,
    CounterProposalResponse,
)
from ..auth import get_current_user
from .. import rag


router = APIRouter(prefix="/contract", tags=["Contracts"])

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_owned_document(
    document_id: str,
    db: Session,
    current_user: User,
):
    """Return a document only if it belongs to the logged-in user."""

    doc = (
        db.query(Document)
        .filter(
            Document.document_id == document_id,
            Document.user_id == current_user.id,
        )
        .first()
    )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or access denied",
        )

    return doc


# ============================================================
# POST /contract/upload
# ============================================================

@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_contract(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a PDF, process it, classify chunks and generate a summary."""

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted",
        )

    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    save_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")

    try:
        with open(save_path, "wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        result = rag.process_contract(save_path, document_id)
        summary_text = rag.generate_summary(result["full_text"])

        doc = Document(
            document_id=document_id,
            filename=file.filename,
            user_id=current_user.id,
            page_count=result["page_count"],
            chunk_count=result["chunk_count"],
            risk_counts=result["risk_counts"],
            summary=summary_text,
        )

        db.add(doc)
        db.commit()
        db.refresh(doc)

        return doc

    except HTTPException:
        db.rollback()
        raise

    except Exception as exc:
        db.rollback()

        try:
            collection = rag.get_collection()
            existing = collection.get(
                where={"document_id": document_id}
            )

            if existing.get("ids"):
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Contract processing failed: {str(exc)}",
        ) from exc

    finally:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass

        await file.close()


# ============================================================
# POST /contract/query
# ============================================================

@router.post("/query", response_model=QueryResponse)
def query_contract(
    payload: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Answer a question using relevant contract passages."""

    get_owned_document(
        payload.document_id,
        db,
        current_user,
    )

    try:
        result = rag.ask_contract(
            question=payload.question,
            document_id=payload.document_id,
            top_k=payload.top_k,
        )

        return QueryResponse(
            question=payload.question,
            answer=result["answer"],
            chunks_used=result["chunks_used"],
            top_similarity=result["top_similarity"],
            retrieved_chunks=result["retrieved_chunks"],
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not answer the contract question: {str(exc)}",
        ) from exc


# ============================================================
# GET /contract/risk/{document_id}
# ============================================================

@router.get(
    "/risk/{document_id}",
    response_model=RiskSummaryResponse,
)
def get_risk_summary(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the risk summary for the user's uploaded contract."""

    get_owned_document(
        document_id,
        db,
        current_user,
    )

    try:
        return rag.get_risk_summary(document_id)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve risk analysis: {str(exc)}",
        ) from exc


# ============================================================
# POST /contract/risk/{document_id}/explain/{chunk_index}
# ============================================================

@router.post("/risk/{document_id}/explain/{chunk_index}")
def explain_risk_clause(
    document_id: str,
    chunk_index: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Explain one classified clause using the AI."""

    if chunk_index < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_index must be zero or greater",
        )

    get_owned_document(
        document_id,
        db,
        current_user,
    )

    try:
        return rag.explain_risk_clause(
            document_id=document_id,
            chunk_index=chunk_index,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not explain the clause: {str(exc)}",
        ) from exc


# ============================================================
# GET /contract/summary/{document_id}
# ============================================================

@router.get("/summary/{document_id}")
def get_summary(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the saved plain-language contract summary."""

    doc = get_owned_document(
        document_id,
        db,
        current_user,
    )

    return {
        "document_id": document_id,
        "summary": doc.summary,
    }


# ============================================================
# POST /contract/counter-proposal
# ============================================================

@router.post(
    "/counter-proposal",
    response_model=CounterProposalResponse,
)
def counter_proposal(
    payload: CounterProposalRequest,
    current_user: User = Depends(get_current_user),
):
    """Generate a suggested alternative for a risky clause."""

    try:
        result = rag.generate_counter_proposal(payload.clause_text)

        return CounterProposalResponse(
            original_clause=payload.clause_text,
            counter_proposal=result,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not generate counter-proposal: {str(exc)}",
        ) from exc


# ============================================================
# GET /contract/history
# ============================================================

@router.get(
    "/history",
    response_model=list[DocumentResponse],
)
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the current user's uploaded contracts."""

    return (
        db.query(Document)
        .filter(Document.user_id == current_user.id)
        .order_by(Document.uploaded_at.desc())
        .all()
    )


# ============================================================
# DELETE /contract/{document_id}
# ============================================================

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete the user's document from the database and ChromaDB."""

    doc = get_owned_document(
        document_id,
        db,
        current_user,
    )

    try:
        collection = rag.get_collection()
        existing = collection.get(
            where={"document_id": document_id}
        )

        if existing.get("ids"):
            collection.delete(ids=existing["ids"])

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not remove contract chunks: {str(exc)}",
        ) from exc

    db.delete(doc)
    db.commit()

    return None
