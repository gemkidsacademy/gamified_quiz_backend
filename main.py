# main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from datetime import datetime
import random
import os

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

# Create tables
Base.metadata.create_all(bind=engine)

# ---------------------------
# Pydantic Schemas
# ---------------------------

class OTPRequest(BaseModel):
    phone_number: str

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

# ---------------------------
# OTP Endpoints
# ---------------------------

@app.post("/login-otp")
def login_otp(request: OTPRequest, db: Session = Depends(get_db)):
    otp_code = str(random.randint(1000, 9999))
    otp_entry = db.query(OTP).filter(OTP.phone_number == request.phone_number).first()
    if otp_entry:
        otp_entry.otp_code = otp_code
        otp_entry.created_at = datetime.utcnow()
    else:
        otp_entry = OTP(phone_number=request.phone_number, otp_code=otp_code)
        db.add(otp_entry)
    db.commit()
    # TODO: send OTP via SMS/email
    print(f"DEBUG: OTP for {request.phone_number} is {otp_code}")
    return {"message": "OTP sent successfully"}

@app.post("/verify-otp")
def verify_otp(request: OTPVerify, db: Session = Depends(get_db)):
    otp_entry = db.query(OTP).filter(OTP.phone_number == request.phone_number).first()
    if not otp_entry or otp_entry.otp_code != request.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    # TODO: generate JWT or session token
    return {"message": "OTP verified successfully", "student_id": 1}  # placeholder

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
