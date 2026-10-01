# ─────────────────────────────────────────────────────────────────────
#  ContractSenseAI · app/main.py
#  FastAPI application — entry point
#  Run with:  uvicorn app.main:app --reload --port 8000
# ─────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()   # load .env before anything else

from .database        import engine, Base
from .routes.auth     import router as auth_router
from .routes.contract import router as contract_router

# Create all database tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title       = "ContractSenseAI API",
    description = "AI-powered legal document analyzer — RAG + Clause Risk Classifier",
    version     = "1.0.0",
)

# ── CORS — allow React frontend to call the API ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173",  # Vite dev server
                         "http://localhost:3000",  # fallback
                         "https://your-vercel-app.vercel.app"],  # production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(contract_router)


# ── Health check ──────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status" : "running",
        "app"    : "ContractSenseAI API",
        "version": "1.0.0",
        "docs"   : "/docs",   # Swagger UI auto-generated
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
