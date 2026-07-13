import random
import string
import logging
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.database.config import get_db
from src.database.models import User
from src.auth.security import hash_password, verify_password, create_access_token
from src.auth.schemas import (
    UserRegisterRequest, UserLoginRequest,
    UserResponse, TokenResponse, OTPVerifyRequest,
)
from src.auth.dependencies import get_current_user
from src.auth.email import send_verification_email

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/auth", tags=["Authentication"])

OTP_EXPIRY_MINUTES = 10


def _generate_otp(length: int = 6) -> str:
    """Return a random numeric OTP of the given length."""
    return "".join(random.choices(string.digits, k=length))


# ---------------------------------------------------------------------------
# Register — creates user (unverified) and fires OTP email
# ---------------------------------------------------------------------------

@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    request: UserRegisterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a new, unverified user account and send an OTP to their email."""

    # Duplicate checks
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    otp = _generate_otp()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)

    user = User(
        email=request.email,
        username=request.username,
        password_hash=hash_password(request.password),
        is_verified=False,
        verification_otp=otp,
        otp_expires_at=expires_at,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Send OTP email in the background (won't block the response)
    background_tasks.add_task(send_verification_email, request.email, otp)
    logger.info(f"OTP for {request.email}: {otp}")  # helpful during dev / if email isn't configured

    return user


# ---------------------------------------------------------------------------
# Verify Email — validates OTP and marks user as verified
# ---------------------------------------------------------------------------

@router.post("/verify-email", status_code=200)
def verify_email(request: OTPVerifyRequest, db: Session = Depends(get_db)):
    """Verify a user's email with the OTP they received."""

    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="No account found with that email")

    if user.is_verified:
        return {"message": "Email already verified. You can log in."}

    if not user.verification_otp or user.verification_otp != request.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP. Please check the code and try again.")

    # Timezone-aware comparison
    now = datetime.now(timezone.utc)
    expires = user.otp_expires_at
    if expires and expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)

    if not expires or now > expires:
        raise HTTPException(
            status_code=400,
            detail="OTP has expired. Please register again or request a new code.",
        )

    # Mark verified and clear OTP
    user.is_verified = True
    user.verification_otp = None
    user.otp_expires_at = None
    db.commit()

    return {"message": "Email verified successfully! You can now log in."}


# ---------------------------------------------------------------------------
# Resend OTP — lets users get a fresh code if theirs expired
# ---------------------------------------------------------------------------

@router.post("/resend-otp", status_code=200)
async def resend_otp(
    request: UserLoginRequest,  # reuse email+password to avoid abuse
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Resend a verification OTP (requires email & password to prevent abuse)."""

    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if user.is_verified:
        return {"message": "Email already verified. You can log in."}

    otp = _generate_otp()
    user.verification_otp = otp
    user.otp_expires_at = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)
    db.commit()

    background_tasks.add_task(send_verification_email, request.email, otp)
    logger.info(f"Resent OTP for {request.email}: {otp}")

    return {"message": "A new verification code has been sent to your email."}


# ---------------------------------------------------------------------------
# Login — blocks unverified users
# ---------------------------------------------------------------------------

@router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticate and return a JWT token. Requires a verified email."""

    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="EMAIL_NOT_VERIFIED",
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
