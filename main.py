# main.py
from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, JSON,
    DateTime, ForeignKey, select
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from openai import OpenAI
from celery import Celery
from celery.schedules import crontab

import os
import random
import json

from celery_app import celery_app
import scheduler

# ---------------------------
# Database Setup
# ---------------------------

DATABASE_URL = os.getenv("DATABASE_URL")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your_openai_api_key"))

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

class AnswerPayload(BaseModel):
    question_index: int  # 0-based index
    selected_option: str


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
    questions = Column(JSON)
    score_logic = Column(String)


class ActivityAttempt(Base):
    __tablename__ = "activity_attempts"
    attempt_id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"))
    quiz_id = Column(Integer, ForeignKey("student_quizzes.quiz_id"))
    score = Column(Integer)
    time_taken = Column(Integer)
    week_number = Column(Integer)
    term_number = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)


class StudentQuiz(Base):
    __tablename__ = "student_quizzes"
    quiz_id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("users.id"))
    quiz_json = Column(JSON)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True)
    phone_number = Column(String, unique=True)
    password = Column(String, nullable=False)
    class_name = Column(String)
    class_day = Column(String)
    status = Column(String, default="active")
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
    allow_origins=["https://gamified-quiz-delta.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# In-memory OTP storage
# ---------------------------

otp_dict = {}  # phone_number -> otp


# ---------------------------
# Quiz Endpoints
# ---------------------------
def generate_quizzes():
   

    print("\n==============================")
    print("[CELERY] Task Started: generate_quizzes()")
    print("==============================")

    db = SessionLocal()
    try:
        print("[DEBUG] Fetching active students...")
        students = db.query(User).filter(User.status == "active").all()
        print(f"[DEBUG] Active students found: {len(students)}")

        if not students:
            print("[DEBUG] No active students. Task ending.")
            return

        print("[DEBUG] Fetching activities...")
        activities = db.query(Activity).all()
        print(f"[DEBUG] Activities found: {len(activities)}")

        if not activities:
            print("[DEBUG] No activities found in DB. Task ending.")
            return

        print("[DEBUG] Starting quiz generation loop...")
        for student in students:
            print(f"\n[DEBUG] Generating quiz for student: {student.id} - {student.name}")

            activity = random.choice(activities)
            print(f"[DEBUG] Selected activity ID: {activity.activity_id}")

            try:
                prompt_template = activity.questions[0]["prompt"]
                topics = "Math, Science, English"
                prompt = prompt_template.replace("{topics}", topics)

                print("[DEBUG] Sending request to OpenAI...")
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.7
                )
                print("[DEBUG] OpenAI response received")

                quiz_content = response.choices[0].message.content

                # For testing without OpenAI, fallback to static JSON
                try:
                    parsed_json = json.loads(quiz_content)
                except Exception:
                    parsed_json = {
                        "questions": [
                            {"prompt": "Sample Q1", "correct_option": "A"},
                            {"prompt": "Sample Q2", "correct_option": "B"}
                        ]
                    }
                    print("[DEBUG] Using fallback static quiz JSON")

                student_quiz = StudentQuiz(
                    student_id=student.id,
                    activity_id=activity.activity_id,
                    quiz_json=parsed_json,
                    status="pending",
                    created_at=datetime.utcnow()
                )
                db.add(student_quiz)
                print("[DEBUG] Quiz saved for student")

            except Exception as ai_error:
                print(f"[ERROR] Failed to generate quiz for {student.name}: {ai_error}")
                continue

        db.commit()
        print("\n[CELERY] Successfully generated quizzes.")
        print("==============================\n")

    except Exception as e:
        print("[FATAL ERROR] Scheduler crashed:", e)
        db.rollback()

    finally:
        db.close()
        print("[DEBUG] Database session closed.")


@app.get("/get-pending-quiz/{user_id}")
def get_pending_quiz(user_id: int, db: Session = Depends(get_db)):
    quiz = db.query(StudentQuiz).filter_by(student_id=user_id, status="pending").first()
    if not quiz:
        return {"message": "No pending quiz found"}

    return {
        "quiz_id": quiz.quiz_id,
        "activity_id": quiz.activity_id,
        "quiz_json": quiz.quiz_json,
        "status": quiz.status,
        "created_at": quiz.created_at
    }





@app.get("/run-scheduler-now")
def run_scheduler_now():
    generate_quizzes()  # runs immediately
    return {"message": "Scheduler task executed"}
# ---------------------------
# Send OTP endpoint
# ---------------------------

@app.post("/send-otp")
def send_otp(request: OTPRequest, db: Session = Depends(get_db)):
    phone = request.phone_number.strip()

    if not phone:
        raise HTTPException(status_code=400, detail="Phone number is required")

    user = db.query(User).filter(User.phone_number == phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="Phone number not registered")

    otp_dict[phone] = "123"
    print(f"[DEBUG] OTP for {phone} is 123")

    return {"message": "OTP sent successfully"}


@app.post("/verify-otp")
def verify_otp(request: OTPVerify, db: Session = Depends(get_db)):
    phone = request.phone_number.strip()
    otp = request.otp.strip()

    if phone not in otp_dict or otp_dict[phone] != otp:
        raise HTTPException(status_code=401, detail="Invalid OTP")

    user = db.query(User).filter(User.phone_number == phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="Phone number not registered")

    otp_dict.pop(phone, None)

    return {
        "message": "OTP verified successfully",
        "student_id": user.id,
        "phone_number": user.phone_number,
        "name": user.name
    }


# ---------------------------
# Login Endpoint
# ---------------------------

@app.post("/login")
def login(request: LoginRequest, response: Response, db: Session = Depends(get_db)):
    stmt = select(User).where(User.phone_number == request.phone_number)
    user = db.execute(stmt).scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Phone number not registered")

    stored_otp = otp_dict.get(request.phone_number)
    if not stored_otp or request.otp != stored_otp:
        raise HTTPException(status_code=401, detail="Invalid OTP")

    response.set_cookie(
        key="session_token",
        value=f"session_{user.id}",
        httponly=True,
        max_age=3600
    )

    otp_dict.pop(request.phone_number, None)

    return {"message": "Login successful", "username": user.name, "id": user.id}


# ---------------------------
# Activity Endpoints
# ---------------------------

@app.get("/get-activity")
def get_activity(student_id: int, db: Session = Depends(get_db)):
    activity = db.query(Activity).first()
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

    score = 0
    for i, answer in enumerate(submit.answers):
        if i < len(activity.questions) and answer == activity.questions[i]["answer"]:
            score += 1

    attempt = ActivityAttempt(
        student_id=submit.student_id,
        activity_id=submit.activity_id,
        score=score,
        time_taken=random.randint(30, 180)
    )

    db.add(attempt)
    db.commit()

    return {"message": "Activity submitted", "score": score}


@app.post("/submit-quiz/{quiz_id}/answer")
def submit_quiz_answer(quiz_id: int, payload: AnswerPayload):
    db = SessionLocal()
    try:
        quiz = db.query(StudentQuiz).filter_by(quiz_id=quiz_id).first()
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")

        if not quiz.started_at:
            quiz.started_at = datetime.utcnow()

        if "student_answers" not in quiz.quiz_json:
            quiz.quiz_json["student_answers"] = {}

        q_key = f"q{payload.question_index + 1}"
        quiz.quiz_json["student_answers"][q_key] = payload.selected_option

        correct_count = 0
        for idx, question in enumerate(quiz.quiz_json["questions"]):
            q_key_check = f"q{idx+1}"
            if q_key_check in quiz.quiz_json["student_answers"]:
                if quiz.quiz_json["student_answers"][q_key_check] == question.get("correct_option"):
                    correct_count += 1

        quiz.quiz_json["score"] = correct_count

        if len(quiz.quiz_json["student_answers"]) == len(quiz.quiz_json["questions"]):
            quiz.status = "completed"
            quiz.completed_at = datetime.utcnow()

            time_taken_seconds = int((quiz.completed_at - quiz.started_at).total_seconds())

            attempt = ActivityAttempt(
                student_id=quiz.student_id,
                quiz_id=quiz.quiz_id,
                score=correct_count,
                time_taken=time_taken_seconds,
                week_number=quiz.quiz_json.get("week_number", 0),
                term_number=quiz.quiz_json.get("term_number", 0),
            )

            db.add(attempt)

        db.commit()

        return {
            "message": "Answer recorded successfully",
            "current_score": correct_count,
            "quiz_status": quiz.status
        }

    finally:
        db.close()


# ---------------------------
# Root Endpoint
# ---------------------------

@app.get("/")
def root():
    return {"message": "Welcome to Gem Kids Gamified Quiz API"}
