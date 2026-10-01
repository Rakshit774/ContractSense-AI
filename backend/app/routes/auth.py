# ─────────────────────────────────────────────────────────────────────
#  ContractSenseAI · app/routes/auth.py
#  POST /auth/register   — create new user
#  POST /auth/login      — get JWT token
#  GET  /auth/me         — get current user info
# ─────────────────────────────────────────────────────────────────────
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..database import get_db
from ..models   import User
from ..schemas  import RegisterRequest, LoginRequest, TokenResponse, UserResponse
from ..auth     import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user and return a JWT token."""

    # Check email not already taken
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "Email already registered",
        )

    user = User(
        email         = payload.email,
        password_hash = hash_password(payload.password),
        full_name     = payload.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """Login with email + password, returns JWT token."""

    user = db.query(User).filter(User.email == payload.email).first()

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid email or password",
        )

    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Return the logged-in user's details."""
    return current_user
