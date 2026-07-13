import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.database.config import get_db
from src.database.models import User
from src.auth.security import hash_password, verify_password, create_access_token
from src.auth.schemas import UserRegisterRequest, UserResponse, TokenResponse
from src.auth.dependencies import get_current_user

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ---------------------------------------------------------------------------
# Register — creates user immediately (no email verification required)
# ---------------------------------------------------------------------------

@router.post("/register", response_model=UserResponse, status_code=201)
def register(request: UserRegisterRequest, db: Session = Depends(get_db)):
    """Create a new user account. No email verification required."""

    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        email=request.email,
        username=request.username,
        password_hash=hash_password(request.password),
        is_verified=True,  # Auto-verified — no OTP needed
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"New user registered: {request.email}")
    return user


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

@router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticate and return a JWT token."""

    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(data={"sub": user.email})
    return TokenResponse(access_token=token)


# ---------------------------------------------------------------------------
# Me
# ---------------------------------------------------------------------------

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Return the currently logged-in user's info. Requires a valid JWT."""
    return current_user
