from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from app.db import get_db
from app import models
from app.auth import verify_password, create_access_token, decode_token, hash_password
from app.schemas import SignupRequest, SignupResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> models.User:
    """Dependency to get current user from JWT token."""
    try:
        payload = decode_token(token)
        user_id = payload.get("uid")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.get(models.User, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


@router.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    email = form.username.strip().lower()
    user = db.execute(select(models.User).where(models.User.email == email)).scalar_one_or_none()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="Inactive user")

    token = create_access_token(sub=user.email, user_id=user.id)
    
    # Return token AND user data (including role)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role.value,  # .value to get string from enum
            "experiment_bucket": user.experiment_bucket,
        }
    }


@router.get("/me", response_model=UserResponse)
def get_me(current_user: models.User = Depends(get_current_user)):
    """Get current authenticated user."""
    return current_user


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()

    existing = db.query(models.User).filter(models.User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered.")

    try:
        user = models.User(
            email=email,
            password_hash=hash_password(payload.password),
            is_active=True,
            # role defaults to USER from model
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user  # UserResponse will serialize it

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Email already registered.")




