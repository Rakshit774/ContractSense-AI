# ─────────────────────────────────────────────────────────────────────
#  ContractSenseAI · app/models.py
#  SQLAlchemy ORM models — defines tables in contractsense.db
# ─────────────────────────────────────────────────────────────────────
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name     = Column(String, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="owner")


class Document(Base):
    __tablename__ = "documents"

    id          = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, unique=True, index=True)  # ChromaDB key
    filename    = Column(String, nullable=False)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    page_count  = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    risk_counts = Column(JSON, default={"RED": 0, "YELLOW": 0, "GREEN": 0})
    summary     = Column(String, nullable=True)  # plain language summary

    owner = relationship("User", back_populates="documents")
