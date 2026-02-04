from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from app.db import get_db
from app import models
from app.auth import verify_password, create_access_token, decode_token, hash_password
from app.schemas import SignupRequest, SignupResponse

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@router.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # form.username will hold email in our case
    email = form.username.strip().lower()
    user = db.execute(select(models.User).where(models.User.email == email)).scalar_one_or_none()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="Inactive user")

    token = create_access_token(sub=user.email, user_id=user.id)
    return {"access_token": token, "token_type": "bearer"}


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()

    # Optional: quick existence check (nice error message)
    existing = db.query(models.User).filter(models.User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered.")

    try:
        user = models.User(
            email=email,
            password_hash=hash_password(payload.password),
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return SignupResponse(id=user.id, email=user.email)

    except ValueError as e:
        # bcrypt 72-byte limit or custom validation
        raise HTTPException(status_code=400, detail=str(e))

    except IntegrityError:
        # Handles race condition if two signups happen at once
        db.rollback()
        raise HTTPException(status_code=409, detail="Email already registered.")




