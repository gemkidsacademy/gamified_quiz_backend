# main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from datetime import datetime
import random

app = FastAPI(title="Gem Kids Gamified Quiz API")

# ---------------------------
# Models
# ---------------------------

class OTPRequest(BaseModel):
    phone_number: str

class OTPVerify(BaseModel):
    phone_number: str
    otp: str

class ActivityRequest(BaseModel):
    student_id: int

class ActivitySubmit(BaseModel):
    student_id: int
    activity_id: int
    answers: List[str]

# ---------------------------
# Dummy in-memory storage (replace with DB later)
# ---------------------------

otp_store = {}  # phone_number -> otp
activities_store = {
    1: {
        "activity_id": 1,
        "instructions": "Solve the mini quiz",
        "questions": [
            {"q": "2 + 2 = ?", "options": ["3","4","5"], "answer": "4"},
            {"q": "Capital of Australia?", "options": ["Sydney","Melbourne","Canberra"], "answer": "Canberra"}
        ],
        "score_logic": "1 point per correct answer"
    }
}
activity_attempts = []

# ---------------------------
# OTP Endpoints
# ---------------------------

@app.post("/login-otp")
def login_otp(request: OTPRequest):
    otp = str(random.randint(1000, 9999))
    otp_store[request.phone_number] = otp
    # TODO: send OTP via SMS/Email
    print(f"DEBUG: OTP for {request.phone_number} is {otp}")
    return {"message": "OTP sent successfully"}

@app.post("/verify-otp")
def verify_otp(request: OTPVerify):
    stored_otp = otp_store.get(request.phone_number)
    if not stored_otp or stored_otp != request.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    # TODO: generate JWT or session token
    return {"message": "OTP verified successfully", "student_id": 1}  # Example

# ---------------------------
# Activity Endpoints
# ---------------------------

@app.get("/get-activity")
def get_activity(student_id: int):
    # TODO: fetch activity based on class/week from DB
    activity = activities_store.get(1)
    if not activity:
        raise HTTPException(status_code=404, detail="No activity found")
    return activity

@app.post("/submit-activity")
def submit_activity(submit: ActivitySubmit):
    # Simple scoring logic for demo
    activity = activities_store.get(submit.activity_id)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")

    score = 0
    for i, answer in enumerate(submit.answers):
        if i < len(activity["questions"]) and answer == activity["questions"][i]["answer"]:
            score += 1

    # Store attempt
    attempt = {
        "student_id": submit.student_id,
        "activity_id": submit.activity_id,
        "score": score,
        "time_taken": random.randint(30, 180),  # dummy time in seconds
        "timestamp": datetime.utcnow()
    }
    activity_attempts.append(attempt)

    return {"message": "Activity submitted", "score": score}

# ---------------------------
# Root Endpoint
# ---------------------------

@app.get("/")
def root():
    return {"message": "Welcome to Gem Kids Gamified Quiz API"}
