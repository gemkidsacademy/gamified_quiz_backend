# main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from datetime import datetime
import random
import os
from fastapi import Response
from sqlalchemy import select
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ---------------------------
# Database Setup
# ---------------------------

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# Models (DB tables)
# ---------------------------

class OTP(Base):
    __tablename__ = "otp_store"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True)
    otp_code = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Activity(Base):
    __tablename__ = "activities"
    activity_id = Column(Integer, primary_key=True, index=True)
    instructions = Column(String)
    questions = Column(JSON)  # list of dicts
    score_logic = Column(String)

class ActivityAttempt(Base):
    __tablename__ = "activity_attempts"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer)
    activity_id = Column(Integer)
    score = Column(Integer)
    time_taken = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True)
    phone_number = Column(String, unique=True)
    password = Column(String, nullable=False)  # store hashed password in production
    class_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# ---------------------------
# Pydantic Schemas
# ---------------------------


class OTPRequest(BaseModel):
    phone_number: str

class LoginRequest(BaseModel):
    phone_number: str
    otp: str

class OTPVerify(BaseModel):
    phone_number: str
    otp: str

class ActivitySubmit(BaseModel):
    student_id: int
    activity_id: int
    answers: List[str]

# ---------------------------
# FastAPI App
# ---------------------------

app = FastAPI(title="Gem Kids Gamified Quiz API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gamified-quiz-delta.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods
    allow_headers=["*"],  # allow all headers
)

# ---------------------------
# In-memory OTP storage
# ---------------------------
otp_dict = {}  # phone_number -> otp

# ---------------------------
# OTP Endpoints
# ---------------------------

# ---------------------------
# OTP Generation Endpoint
# ---------------------------

@app.post("/send-otp")
def send_otp(request: OTPRequest):
    if not request.phone_number:
        raise HTTPException(status_code=400, detail="Phone number is required")

    # Fixed OTP for testing
    otp_code = "123"
    otp_dict[request.phone_number] = otp_code

    # TODO: replace print with SMS/email sending in production
    print(f"[DEBUG] OTP for {request.phone_number}: {otp_code}")

    return {"message": "OTP sent successfully"}


# ---------------------------
# Login Endpoint
# ---------------------------
@app.post("/login")
def login(request: LoginRequest, response: Response, db: Session = Depends(get_db)):
    # Check if phone number exists in DB
    stmt = select(User).where(User.phone_number == request.phone_number)
    user = db.execute(stmt).scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Phone number not registered")

    # Check OTP from in-memory dict
    stored_otp = otp_dict.get(request.phone_number)
    if not stored_otp or request.otp != stored_otp:
        raise HTTPException(status_code=401, detail="Invalid OTP")

    # Set dummy session cookie
    response.set_cookie(
        key="session_token",
        value=f"session_{user.id}",
        httponly=True,
        max_age=3600  # 1 hour
    )

    # Optionally, remove OTP after successful login
    otp_dict.pop(request.phone_number, None)

    return {"message": "Login successful", "username": user.name, "id": user.id}
    
@app.post("/verify-otp")
def verify_otp(request: OTPVerify):
    stored_otp = otp_dict.get(request.phone_number)
    if not stored_otp or stored_otp != request.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    return {"message": "OTP verified successfully", "student_id": 1}

# ---------------------------
# Activity Endpoints
# ---------------------------

@app.get("/get-activity")
def get_activity(student_id: int, db: Session = Depends(get_db)):
    # TODO: fetch activity based on class/week from DB
    activity = db.query(Activity).first()  # placeholder: first activity
    if not activity:
        raise HTTPException(status_code=404, detail="No activity found")
    return {
        "activity_id": activity.activity_id,
        "instructions": activity.instructions,
        "questions": activity.questions,
        "score_logic": activity.score_logic
    }

@app.post("/submit-activity")
def submit_activity(submit: ActivitySubmit, db: Session = Depends(get_db)):
    activity = db.query(Activity).filter(Activity.activity_id == submit.activity_id).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")

    # Simple scoring logic
    score = 0
    for i, answer in enumerate(submit.answers):
        if i < len(activity.questions) and answer == activity.questions[i]["answer"]:
            score += 1

    # Record attempt
    attempt = ActivityAttempt(
        student_id=submit.student_id,
        activity_id=submit.activity_id,
        score=score,
        time_taken=random.randint(30, 180)
    )
    db.add(attempt)
    db.commit()

    return {"message": "Activity submitted", "score": score}

# ---------------------------
# Root Endpoint
# ---------------------------

@app.get("/")
def root():
    return {"message": "Welcome to Gem Kids Gamified Quiz API"}
