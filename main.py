# main.py
from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime
import os
import random
import json

from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, ForeignKey, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler

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
# OpenAI Client
# ---------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your_openai_api_key"))

# ---------------------------
# Models
# ---------------------------
class AnswerPayload(BaseModel):
    question_index: int
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

Base.metadata.create_all(bind=engine)

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
otp_dict = {}

# ---------------------------
# Quiz Generation
# ---------------------------
def generate_quizzes():
    print("\n==============================")
    print("[SCHEDULER] Task Started: generate_quizzes()")
    print("==============================")

    db = SessionLocal()
    try:
        students = db.query(User).filter(User.status == "active").all()
        if not students:
            print("[DEBUG] No active students. Task ending.")
            return

        activities = db.query(Activity).all()
        if not activities:
            print("[DEBUG] No activities found in DB. Task ending.")
            return

        for student in students:
            activity = random.choice(activities)
            try:
                prompt_template = activity.questions[0]["prompt"]
                topics = "Math, Science, English"
                prompt = prompt_template.replace("{topics}", topics)

                # OpenAI call
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.7
                )
                quiz_content = response.choices[0].message.content

                try:
                    parsed_json = json.loads(quiz_content)
                except Exception:
                    parsed_json = {
                        "questions": [
                            {"prompt": "Sample Q1", "correct_option": "A"},
                            {"prompt": "Sample Q2", "correct_option": "B"}
                        ]
                    }

                student_quiz = StudentQuiz(
                    student_id=student.id,
                    quiz_json=parsed_json,
                    status="pending",
                    created_at=datetime.utcnow()
                )
                db.add(student_quiz)
            except Exception as e:
                print(f"[ERROR] Failed to generate quiz for {student.name}: {e}")
                continue

        db.commit()
        print("[SCHEDULER] Successfully generated quizzes.")
    except Exception as e:
        print("[FATAL ERROR] Scheduler crashed:", e)
        db.rollback()
    finally:
        db.close()
        print("[DEBUG] Database session closed.")

# ---------------------------
# APScheduler Setup (weekly run)
# ---------------------------
scheduler = BackgroundScheduler()
scheduler.add_job(generate_quizzes, 'interval', weeks=1, next_run_time=datetime.utcnow())
scheduler.start()

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/run-scheduler-now")
def run_scheduler_now():
    generate_quizzes()
    return {"message": "Scheduler task executed"}

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

# (Other endpoints like OTP, login, activity submission, etc. remain unchanged)
