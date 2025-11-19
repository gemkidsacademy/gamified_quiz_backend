# main.py
from fastapi import FastAPI, HTTPException, Depends, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime
import os
import random
import json
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, ForeignKey, select, func


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
class LoginRequest(BaseModel):
    phone_number: str
    password: str
    
class AnswerPayload(BaseModel):
    question_index: int
    selected_option: str
    student_id: int  # sent from frontend
    class_name: str  # sent from frontend


class OTPVerify(BaseModel):    
    phone: Optional[str] = None
    otp: str

class OTP(Base):
    __tablename__ = "otp_store"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True)
    otp_code = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class QuizResult(Base):
    __tablename__ = "quiz_results"
    
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, nullable=False)      # link to student
    class_name = Column(String, nullable=False)       # the class of the quiz
    total_score = Column(Integer, nullable=False)     # total correct answers
    total_questions = Column(Integer, nullable=False) # total questions in quiz
    submitted_at = Column(DateTime, default=datetime.utcnow)

class Activity(Base):
    __tablename__ = "activities"

    activity_id = Column(Integer, primary_key=True, index=True)
    instructions = Column(String)
    questions = Column(JSON)
    score_logic = Column(String)
    class_name = Column(String)   # new column for class name
    class_day = Column(String) 
    
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

class OTPRequest(BaseModel):    
    phone_number: str

class StudentQuiz(Base):
    __tablename__ = "student_quizzes"

    quiz_id = Column(Integer, primary_key=True)
    quiz_json = Column(JSON, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Class metadata
    class_name = Column(String, nullable=False)
    class_day = Column(String, nullable=True)
    
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
        # Get all distinct classes from activities
        activities = db.query(Activity).all()
        if not activities:
            print("[DEBUG] No activities found in DB. Task ending.")
            return

        for activity in activities:
            try:
                # Prepare prompt
                prompt_template = activity.questions[0]["prompt"] if activity.questions else "Create a simple quiz with {topics}."
                topics = "Math, Science, English"
                prompt = prompt_template.replace("{topics}", topics)

                # Try OpenAI API call
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",  # economical model
                        messages=[{"role": "system", "content": prompt}],
                        temperature=0.7
                    )
                    quiz_content = response.choices[0].message.content
                    try:
                        parsed_json = json.loads(quiz_content)
                    except Exception:
                        print(f"[WARNING] Failed to parse AI response for class {activity.class_name}. Using fallback quiz.")
                        parsed_json = None

                except Exception as e:
                    print(f"[ERROR] AI call failed for class {activity.class_name}: {e}")
                    parsed_json = None

                # Fallback quiz if AI fails
                if not parsed_json:
                    parsed_json = {
                        "quiz_title": f"Sample Quiz for {activity.class_name}",
                        "instructions": "This is a fallback quiz. Answer the questions carefully.",
                        "questions": [
                            {"category": "Math", "prompt": "What is 10 + 5?", "options": ["12","15","20","25"], "answer":"15"},
                            {"category": "Science", "prompt": "Which planet is closest to the Sun?", "options":["Earth","Mercury","Venus","Mars"], "answer":"Mercury"},
                            {"category": "English", "prompt": "Plural of 'mouse'?", "options":["Mouses","Mice","Mouseses","Mouse"], "answer":"Mice"}
                        ]
                    }

                # Save class-level quiz
                class_quiz = StudentQuiz(
                    quiz_json=parsed_json,
                    status="pending",
                    created_at=datetime.utcnow(),
                    class_name=activity.class_name,
                    class_day=activity.class_day
                )
                db.add(class_quiz)

            except Exception as e:
                print(f"[ERROR] Failed to generate quiz for class {activity.class_name}: {e}")
                continue

        db.commit()
        print("[SCHEDULER] Successfully generated quizzes for all classes.")

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
scheduler.add_job(generate_quizzes, 'interval', weeks=1)

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

@app.post("/send-otp")
def send_otp(request: OTPRequest, db: Session = Depends(get_db)):
    phone = request.phone_number.strip()
    if not phone:
        raise HTTPException(status_code=400, detail="Phone number required")
    user = db.query(User).filter(User.phone_number == phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="Phone number not registered")
    otp_dict[phone] = "123"
    print(f"[DEBUG] OTP for {phone}: 123")
    return {"message": "OTP sent successfully"}


@app.post("/verify-otp")
def verify_otp(request: OTPVerify, db: Session = Depends(get_db)):
    phone = request.phone.strip() if request.phone else None
    otp = request.otp.strip()

    if not phone:
        raise HTTPException(status_code=400, detail="Phone number is required")

    if phone not in otp_dict or otp_dict[phone] != otp:
        raise HTTPException(status_code=401, detail="Invalid OTP")

    user = db.query(User).filter(User.phone_number == phone).first()
    if not user:
        raise HTTPException(status_code=404, detail="Phone number not registered")

    # Remove OTP after successful verification
    otp_dict.pop(phone, None)

    # Include class_name in the response
    return {
        "message": "OTP verified",
        "student_id": user.id,
        "phone_number": user.phone_number,
        "name": user.name,
        "class_name": user.class_name  # <- added this
    }
    
@app.post("/login")
def login(request: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.phone_number == request.phone_number)).scalar_one_or_none()
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
    return {
        "message": "Login successful",
        "username": user.name,
        "id": user.id,
        "class_name": user.class_name  # <- include class_name here
    }



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

@app.get("/get-quiz")
def get_quiz(class_name: str = Query(..., description="Class name of the quiz"), db: Session = Depends(get_db)):
    """
    Fetch the latest quiz for a given class_name.
    This version trims whitespace and matches case-insensitively.
    """
    cleaned_class_name = class_name.strip()

    quiz = (
        db.query(StudentQuiz)
        .filter(func.lower(func.trim(StudentQuiz.class_name)) == cleaned_class_name.lower())
        .order_by(StudentQuiz.created_at.desc())
        .first()
    )

    if not quiz:
        raise HTTPException(status_code=404, detail=f"No quiz found for class '{class_name}'")

    return JSONResponse(content=quiz.quiz_json)

"""
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
        quiz_id=None,
        score=score,
        time_taken=random.randint(30, 180)
    )
    db.add(attempt)
    db.commit()
    return {"message": "Activity submitted", "score": score}
"""

@app.post("/submit-answer")
def submit_answer(class_name: str, question_index: int, selected_option: str, db: Session = Depends(get_db)):
    # Fetch the latest quiz for this class
    quiz = db.query(StudentQuiz).filter(StudentQuiz.class_name == class_name).order_by(StudentQuiz.created_at.desc()).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    # Initialize answers if not present
    if "student_answers" not in quiz.quiz_json:
        quiz.quiz_json["student_answers"] = {}

    quiz.quiz_json["student_answers"][f"q{question_index+1}"] = selected_option

    # Optional: Calculate current score
    score = 0
    for idx, q in enumerate(quiz.quiz_json["questions"]):
        answer = quiz.quiz_json["student_answers"].get(f"q{idx+1}")
        if answer == q.get("answer"):
            score += 1

    quiz.quiz_json["score"] = score
    db.commit()
    return {"current_score": score}
    
@app.post("/submit-quiz-answer")
def submit_quiz_answer(payload: AnswerPayload, db: Session = Depends(get_db)):
    """
    Receive a single answer from the student for the current question.
    Once all questions are answered, store final score in QuizResult table.
    """
    # Fetch the quiz for this class
    quiz = db.query(StudentQuiz).filter(
        StudentQuiz.class_name == payload.class_name
    ).first()
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    # Start quiz if not started
    if not quiz.started_at:
        quiz.started_at = datetime.utcnow()

    # Initialize student_answers dict
    if "student_answers" not in quiz.quiz_json:
        quiz.quiz_json["student_answers"] = {}

    # Record current answer
    q_key = f"q{payload.question_index + 1}"
    quiz.quiz_json["student_answers"][q_key] = payload.selected_option

    # Calculate current score
    correct_count = 0
    for idx, question in enumerate(quiz.quiz_json["questions"]):
        q_key_check = f"q{idx+1}"
        if q_key_check in quiz.quiz_json["student_answers"]:
            if quiz.quiz_json["student_answers"][q_key_check] == question.get("answer"):
                correct_count += 1

    quiz.quiz_json["score"] = correct_count

    # Check if quiz is completed
    if len(quiz.quiz_json["student_answers"]) == len(quiz.quiz_json["questions"]):
        quiz.status = "completed"
        quiz.completed_at = datetime.utcnow()
        time_taken_seconds = int((quiz.completed_at - quiz.started_at).total_seconds())

        # Save final result in QuizResult table
        result = QuizResult(
            student_id=payload.student_id,
            class_name=payload.class_name,
            total_score=correct_count,
            total_questions=len(quiz.quiz_json["questions"]),
            submitted_at=datetime.utcnow()
        )
        db.add(result)

    db.commit()
    return {
        "message": "Answer recorded",
        "current_score": correct_count,
        "quiz_status": quiz.status
    }




