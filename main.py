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
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from celery.schedules import crontab
from openai import OpenAI
from celery import Celery
import json

   

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
# --------------------------

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
    questions = Column(JSON)  # list of dicts
    score_logic = Column(String)

class ActivityAttempt(Base):
    __tablename__ = "activity_attempts"
    attempt_id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"))
    quiz_id = Column(Integer, ForeignKey("student_quizzes.quiz_id"))
    score = Column(Integer)
    time_taken = Column(Integer)  # seconds
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
    password = Column(String, nullable=False)  # store hashed password in production
    class_name = Column(String)
    status = Column(String, default="active")  # new column to indicate active/inactive users
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

# ------------------ Celery Setup ------------------
celery_app = Celery(
    "gamified_scheduler",
    broker=os.getenv("CELERY_BROKER_URL")
)

celery_app.conf.update(
    timezone='Asia/Karachi',
    enable_utc=True
)

celery_app.conf.beat_schedule = {
    'generate-daily-quizzes': {
        'task': 'gamified_scheduler.generate_quizzes',
        'schedule': crontab(hour=19, minute=0),  # daily at 7 PM
    },
}


# ---------------------------
# In-memory OTP storage
# ---------------------------
otp_dict = {}  # phone_number -> otp


# ------------------ Scheduler Task ------------------
@celery_app.task
def generate_quizzes():
    db = SessionLocal()
    try:
        # 1. Fetch all active students
        students = db.query(User).filter(User.status == "active").all()
        if not students:
            print("No active students found.")
            return

        # 2. Fetch all activity prompts
        activities = db.query(Activity).all()
        if not activities:
            print("No activities found in the database.")
            return

        # 3. Generate quizzes for each student
        for student in students:
            activity = random.choice(activities)
            # Assume questions[0]["prompt"] contains the OpenAI template
            prompt_template = activity.questions[0]["prompt"]

            # You can replace {topics} dynamically if needed
            topics = "Math, Science, English"
            prompt = prompt_template.replace("{topics}", topics)

            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7
            )

            quiz_content = response['choices'][0]['message']['content']

            # Store the generated quiz
            student_quiz = StudentQuiz(
                student_id=student.id,
                activity_id=activity.activity_id,
                quiz_json=json.loads(quiz_content),
                status="pending",
                created_at=datetime.utcnow()
            )
            db.add(student_quiz)

        db.commit()
        print(f"Generated quizzes for {len(students)} students at {datetime.utcnow()}")

    except Exception as e:
        print("Error generating quizzes:", e)
        db.rollback()
    finally:
        db.close()


@app.get("/get-pending-quiz/{user_id}")
def get_pending_quiz(user_id: int):
    db = SessionLocal()
    try:
        quiz = db.query(StudentQuiz).filter_by(user_id=user_id, status="pending").first()
        if not quiz:
            raise HTTPException(status_code=404, detail="No pending quiz found for this user")
        return {
            "quiz_id": quiz.quiz_id,
            "activity_id": quiz.activity_id,
            "quiz_json": quiz.quiz_json,
            "status": quiz.status,
            "created_at": quiz.created_at
        }
    finally:
        db.close()

# ---------------------------
# Send OTP endpoint
# ---------------------------
@app.post("/send-otp")
def send_otp(request: OTPRequest, db: Session = Depends(get_db)):
    phone = request.phone_number.strip()
    if not phone:
        raise HTTPException(status_code=400, detail="Phone number is required")

    # Check if phone exists in DB
    user = db.query(User).filter(User.phone_number == phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="Phone number not registered")

    # For testing, OTP is always "123"
    otp_dict[phone] = "123"
    print(f"[DEBUG] OTP for {phone} is 123")  # debug only

    return {"message": "OTP sent successfully"}

@app.post("/verify-otp")
def verify_otp(request: OTPVerify, db: Session = Depends(get_db)):
    phone = request.phone_number.strip()
    otp = request.otp.strip()

    # Check OTP
    if phone not in otp_dict or otp_dict[phone] != otp:
        raise HTTPException(status_code=401, detail="Invalid OTP")

    # Lookup user in DB
    user = db.query(User).filter(User.phone_number == phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="Phone number not registered")

    # Remove OTP after successful login
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

@app.post("/submit-quiz/{quiz_id}/answer")
def submit_quiz_answer(quiz_id: int, payload: AnswerPayload):
    db = SessionLocal()
    try:
        quiz = db.query(StudentQuiz).filter_by(quiz_id=quiz_id).first()
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")

        # Record start time if this is the first answer
        if not quiz.started_at:
            quiz.started_at = datetime.utcnow()

        # Store student's answer inside quiz JSON
        if "student_answers" not in quiz.quiz_json:
            quiz.quiz_json["student_answers"] = {}

        q_key = f"q{payload.question_index + 1}"
        quiz.quiz_json["student_answers"][q_key] = payload.selected_option

        # Auto-calculate score so far
        correct_count = 0
        for idx, question in enumerate(quiz.quiz_json["questions"]):
            q_key_check = f"q{idx+1}"
            if q_key_check in quiz.quiz_json["student_answers"]:
                if quiz.quiz_json["student_answers"][q_key_check] == question.get("correct_option"):
                    correct_count += 1
        quiz.quiz_json["score"] = correct_count

        # Check if all questions answered
        if len(quiz.quiz_json["student_answers"]) == len(quiz.quiz_json["questions"]):
            quiz.status = "completed"
            quiz.completed_at = datetime.utcnow()
            time_taken_seconds = int((quiz.completed_at - quiz.started_at).total_seconds())

            # Insert into ActivityAttempt for leaderboard
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

if __name__ == "__main__":
    generate_quizzes()  # <-- This will run immediately
