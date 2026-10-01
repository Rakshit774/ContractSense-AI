# ─────────────────────────────────────────────────────────────────────
#  ContractSenseAI · app/schemas.py
#  Pydantic models — validate request bodies and shape responses
# ─────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional


# ── Auth ──────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email    : EmailStr
    password : str
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    email   : EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type  : str = "bearer"


class UserResponse(BaseModel):
    id        : int
    email     : str
    full_name : Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Contract ──────────────────────────────────────────────────────────
class DocumentResponse(BaseModel):
    document_id : str
    filename    : str
    uploaded_at : datetime
    page_count  : int
    chunk_count : int
    risk_counts : dict
    summary     : Optional[str]

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    document_id: str
    question   : str
    top_k      : int = 5


class QueryResponse(BaseModel):
    question        : str
    answer          : str
    chunks_used     : int
    top_similarity  : float
    retrieved_chunks: list[dict]


class RiskSummaryResponse(BaseModel):
    document_id   : str
    red_clauses   : list[dict]
    yellow_clauses: list[dict]
    green_clauses : list[dict]
    counts        : dict


class CounterProposalRequest(BaseModel):
    clause_text: str


class CounterProposalResponse(BaseModel):
    original_clause : str
    counter_proposal: str
