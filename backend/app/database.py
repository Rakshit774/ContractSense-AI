# ─────────────────────────────────────────────────────────────────────
#  ContractSenseAI · app/database.py
#  SQLite database setup using SQLAlchemy ORM
#  Switch to PostgreSQL for production: change DATABASE_URL below
# ─────────────────────────────────────────────────────────────────────
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./contractsense.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session, closes it after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
