# main.py
from fastapi import FastAPI, HTTPException, Depends, Response, Query, Path, File, UploadFile
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List
from sendgrid import SendGridAPIClient
from datetime import datetime, timedelta,date
import pandas as pd
import os
import random
import time
from sendgrid.helpers.mail import Mail
from typing import List, Dict, Any, Optional
import re 
import traceback
 

import uuid
from sqlalchemy.dialects.postgresql import UUID





 
import json

from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, ForeignKey, select, func, text, Boolean, Date, Text
from fastapi.responses import JSONResponse



from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.mutable import MutableDict



from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# ---------------------------
# Database Setup PGUSER,PGPASSWORD,PGHOST,PGPORT,PGDATABASE rewrite "" using PGPASSWORD=lgZmFsBTApVPJIyTegBttTLfdWnvccHj psql -h metro.proxy.rlwy.net -U postgres -p 31631 -d railway
# ---------------------------
#DATABASE_URL = os.getenv("DATABASE_URL")
# Confirm connection details
print("PGUSER:", os.environ.get("PGUSER"))
print("PGHOST:", os.environ.get("PGHOST"))
print("PGPORT:", os.environ.get("PGPORT"))
print("PGDATABASE:", os.environ.get("PGDATABASE"))

DATABASE_URL = os.environ.get("DATABASE_URL") or (
    f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
    f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
)
print("Connecting to DATABASE_URL:", DATABASE_URL)
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
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
otp_store = {}
# ---------------------------
# Models
# ---------------------------
#when generate exam is pressed we create a row here
class Exam(Base):
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True, index=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"), nullable=False)
    questions = Column(JSON, nullable=False)  # list of question objects
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    quiz = relationship("Quiz", back_populates="exams")
    student_exams = relationship("StudentExam", back_populates="exam")

class StudentExam(Base):
    __tablename__ = "student_exams"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)

    status = Column(String, default="pending")  # "pending", "completed"
    score = Column(Integer, default=0)

    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    exam = relationship("Exam", back_populates="student_exams")
    answers = relationship("StudentExamAnswer", back_populates="student_exam")

class StudentExamAnswer(Base):
    __tablename__ = "student_exam_answers"

    id = Column(Integer, primary_key=True, index=True)
    student_exam_id = Column(Integer, ForeignKey("student_exams.id"), nullable=False)

    question_id = Column(Integer, nullable=False)      # q_id inside questions JSON
    student_answer = Column(String, nullable=False)    # e.g. "A"
    correct_answer = Column(String, nullable=False)    # e.g. "C"
    is_correct = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    student_exam = relationship("StudentExam", back_populates="answers")

class Student(Base):
    __tablename__ = "students"
    
    id = Column(String, primary_key=True, index=True)   # Gem001, Gem002
    student_id = Column(String, unique=True, index=True, nullable=False)  # New field
    
    password = Column(String, nullable=False)           # store hashed password in production
    name = Column(String, nullable=False)

    parent_email = Column(String, unique=True, index=True, nullable=False)
    
    class_name = Column(String, nullable=False)         # e.g., Year 1, Year 2, Kindergarten
    class_day = Column(String, nullable=False)  

class TopicInput(BaseModel):
    name: str
    ai: int
    db: int
    total: int

class QuizCreate(BaseModel):
    class_name: str
    subject: str
    difficulty: str
    num_topics: int
    topics: List[TopicInput]

class Quiz(Base):
    __tablename__ = "quizzes"

    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String(50), nullable=False)      # kindergarten, selective, year1-6
    subject = Column(String(50), nullable=False)         # thinking_skills, mathematical_reasoning, reading, writing
    difficulty = Column(String(20), nullable=False)      # easy, medium, hard
    num_topics = Column(Integer, nullable=False)
    topics = Column(JSON, nullable=False)                # Stores list of topic objects
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)

    class_name = Column(String(50), nullable=False)
    subject = Column(String(50), nullable=False)
    difficulty = Column(String(20), nullable=False)

    question_type = Column(String(50), nullable=False)  
    # e.g. "domino_multi_image", "chart_mcq", "text_mcq", "two_diagram_mcq"

    question_text = Column(Text, nullable=True)

    # NEW — structured images
    # Example:
    # [
    #   {"role": "sequence", "url": "https://cdn.com/q1/sequence.png"},
    #   {"role": "option_A", "url": "..."}, 
    #   {"role": "option_B", "url": "..."}, 
    #   {"role": "option_C", "url": "..."}, 
    #   {"role": "option_D", "url": "..."}
    # ]
    images = Column(JSON, nullable=True)

    # NEW — structured options
    # Example:
    # { "A": "Option text", "B": "Option text", "C": "Option text", "D": "Option text" }
    options = Column(JSON, nullable=True)

    correct_answer = Column(String(5), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

class StudentLogin(BaseModel):
    student_id: str  # e.g., Gem001, Gem002
    password: str

class LeaderboardEntry(BaseModel):
    student_name: str
    class_name: str
    class_day: str
    total_score: int
    total_questions: int
    submitted_at: datetime   # <-- change from str to datetime
    week_number: int

class FranchiseLocation(Base):
    __tablename__ = "franchiselocation"
                     
    id = Column(Integer, primary_key=True, autoincrement=True)
    country = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)

class AdminTerm(Base):
    __tablename__ = "admin_term"

    id = Column(Integer, primary_key=True, index=True)
    year = Column(String, nullable=False)
    term_start_date = Column(Date, nullable=False)
    term_end_date = Column(Date, nullable=False)

class TermDataSchema(BaseModel):
    year: str
    termStartDate: str  # will parse as date
    termEndDate: str

    
class WeekRequest(BaseModel):
    date: str  #

class LoginRequest(BaseModel):
    phone_number: str
    password: str

class ActivityPayload(BaseModel):
    instructions: str
    questions: List[Dict[str, Any]]                 # expects a JSON array of question objects
    score_logic: Optional[str] = None              # can be used as activity type
    admin_prompt: Optional[str] = None             # optional prompt from front end
    class_name: str
    class_day: Optional[str] = None
    week_number: Optional[int] = None
    
class AnswerPayload(BaseModel):
    question_index: int
    selected_option: str
    student_id: int        # ID of the student submitting the answer
    student_name: str      # Name of the student (needed for leaderboard)
    class_name: str
    class_day: str  

class AdminDate(Base):
    __tablename__ = "admin_date"

    date = Column(String, primary_key=True) 

class TermStartDateSchema(BaseModel):
    date: date

class AdminDateSchema(BaseModel):
    date: str  # we can use str since your column is String


    

class TermStartDateResponse(BaseModel):
    date: str  # YYYY-MM-DD

class OTPVerify(BaseModel):
    email: EmailStr  # required email field
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
    quiz_id = Column(Integer, nullable=False)
    student_id = Column(Integer, nullable=False)
    student_name = Column(String, nullable=False)
    class_name = Column(String, nullable=False)
    class_day = Column(String, nullable=True)
    week_number = Column(Integer, nullable=True)   # <-- added
    total_score = Column(Integer, nullable=False)
    total_questions = Column(Integer, nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    
class Activity(Base):
    __tablename__ = "activities"

    activity_id = Column(Integer, primary_key=True, index=True)
    instructions = Column(String)
    admin_prompt = Column(String)       # <-- new column for admin prompt
    
    questions = Column(JSON)
    score_logic = Column(String)
    class_name = Column(String)         # new column for class name
    class_day = Column(String)
    week_number = Column(Integer)  
    
class ActivityAttempt(Base):
    __tablename__ = "activity_attempts"
    attempt_id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users_temp.id"))
    quiz_id = Column(Integer, ForeignKey("student_quizzes.quiz_id"))
    score = Column(Integer)
    time_taken = Column(Integer)
    week_number = Column(Integer)
    term_number = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

class OTPRequest(BaseModel):
    email: str
    
class StudentQuiz(Base):
    __tablename__ = "student_quizzes"

    quiz_id = Column(Integer, primary_key=True)
    
    # Use MutableDict so in-place updates to quiz_json are tracked
    quiz_json = Column(MutableDict.as_mutable(JSON), nullable=False, default=dict)
    
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Class metadata
    class_name = Column(String, nullable=False)
    class_day = Column(String, nullable=True)

"""    
class User(Base):
    __tablename__ = "users_temp"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True)
    phone_number = Column(String, unique=True)
    password = Column(String, nullable=False)
    class_name = Column(String)
    class_day = Column(String)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
"""
class SessionModel(Base):  # Handles authentication sessions
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_token = Column(UUID(as_uuid=True), unique=True, index=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))  # link to User
    is_authenticated = Column(Boolean, default=False)
    public_token = Column(UUID(as_uuid=True), unique=True, index=True, default=uuid.uuid4)

    # Establish relationship back to User
    user = relationship("User", back_populates="sessions")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone_number = Column(String, nullable=False)
    class_name = Column(String, nullable=False)
    class_day = Column(String, nullable=False)  # <-- added
    password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Establish relationship with sessions
    sessions = relationship("SessionModel", back_populates="user", cascade="all, delete-orphan")



try:
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully")
except Exception as e:
    print("Error creating tables:", e)


# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="Gem Kids Gamified Quiz API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gamified-quiz-delta.vercel.app",
        "https://leader-board-viewer-gamified-quiz.vercel.app",
        "https://leaderboard.gemkidsacademy.com.au",
        "https://gamifiedquiz.gemkidsacademy.com.au",
        "https://exam-module-pink.vercel.app"  # added origin
    ],
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
        activities = db.query(Activity).all()
        print(f"[DEBUG] Number of activities fetched: {len(activities)}")
        if not activities:
            print("[DEBUG] No activities found in DB. Task ending.")
            return

        for idx, activity in enumerate(activities, 1):
            print(f"\n[DEBUG] Processing activity {idx}/{len(activities)}: {activity}")
            print("[DEBUG] Activity object dict:", activity.__dict__)

            # --- Extract activity values ---
            raw_prompt = getattr(activity, "admin_prompt", "") or ""
            class_name = getattr(activity, "class_name", "") or ""
            topic_name = getattr(activity, "instructions", "") or ""
            activity_type=getattr(activity,"score_logic","") or ""
            print(f"[DEBUG] Extracted -> class_name: '{class_name}', topic_name: '{topic_name}'")
            print(f"[DEBUG] Admin prompt:\n{raw_prompt}")
            row = db.execute(select(FranchiseLocation)).scalar_one_or_none()
            if row:
               country = row.country
               state = row.state
               print(f"Country: {country}, State: {state}")
            else:
               print("No row found in FranchiseLocation table.")
               country = "UnknownCountry"
               state = "UnknownState"

            # --- Compose strict system prompt (single triple quotes) ---
            system_prompt = '''
            You are an expert quiz-generating AI and a creative educator. Create a gamified quiz for {country} {state} {class_name} class students on the topic: {topic_name}. The activity type is: "{activity_type}".
            
            Follow these rules STRICTLY:
            
            1. Return ONLY a single valid JSON object. NO explanations, NO extra text.
            2. Use standard double quotes " only.
            3. JSON MUST follow this exact structure ONLY:
            {{
              "quiz_title": "Sample Quiz",
              "instructions": "Answer all questions carefully",
              "questions": [
                {{"category": "{topic_name}", "prompt": "Question 1 here", "options": ["A","B","C","D"], "answer": "A"}},
                {{"category": "{topic_name}", "prompt": "Question 2 here", "options": ["A","B","C","D"], "answer": "B"}},
                {{"category": "{topic_name}", "prompt": "Question 3 here", "options": ["A","B","C","D"], "answer": "C"}},
                {{"category": "{topic_name}", "prompt": "Question 4 here", "options": ["A","B","C","D"], "answer": "D"}},
                {{"category": "{topic_name}", "prompt": "Question 5 here", "options": ["A","B","C","D"], "answer": "A"}}
              ]
            }}
            
            4. Each 'prompt' MUST contain only ONE clear question. Never include two questions in a single prompt.
            5. Each question MUST have exactly 4 options tied ONLY to that question.
            6. The 'answer' MUST match exactly one of the 4 options for that specific question.
            7. ENFORCED ACTIVITY TYPE LOGIC: Apply the rules below depending on the activity type.
               - If activity type is "Mini quiz":
                   *Create a short multiple-choice quiz with a clear learning goal.*
               - If activity type is "Single challenge question":
                   *Create one challenging question focused on problem-solving or reasoning.*
               - If activity type is "Story + question":
                   *Provide a brief story followed by a related multiple-choice question.*
               - If activity type is "Riddle":
                   *Create a fun riddle with 4 options, the correct answer must be among them.*
               - If activity type is "Image-based puzzle":
                   *Describe a visual puzzle clearly in the prompt. Provide 4 options describing possible solutions. Answer must be one of the options.*
               - If activity type is "Logic puzzle":
                   *Present a reasoning problem requiring logical deduction. Include 4 options, one of which is correct.*
               - If activity type is "Word scramble":
                   *Present 5 scrambled words. Each prompt shows a scrambled word, provide 4 options (the correct word + 3 plausible distractors). The 'answer' must match the correct unscrambled word exactly.*
               - If activity type is "Mini path-choice scenario":
                   *Present a short scenario with a decision to make. Student selects the correct action from 4 options.*
               - If activity type is "Word maze":
                   *Create a vocabulary-based puzzle inspired by a 'maze' theme. 
                    Each question presents a clue, 
                    and the 4 options are possible words the student might 'choose' while navigating the maze. 
                    The answer MUST be exactly one of the 4 options. 
                    NO directional paths, NO grids, NO letters like A/B/C/D, NO maze drawings.*
            
            8. For ANY other activity type passed through {activity_type}:
                   *Use the name to guide theme and creativity ONLY. DO NOT modify structure or rules.*
            
            9. Questions MUST be fun, clear, engaging, and age-appropriate for Year 5.
            10. NEVER blend two questions together. NEVER ask "What is X? Is it Y?" inside the same prompt.
            11. Admin prompt/context for this quiz: {raw_prompt}
            '''.format(
                country=country,
                state=state,
                class_name=class_name,
                topic_name=topic_name,
                activity_type=activity_type,
                raw_prompt=raw_prompt
            )



            parsed_json = None
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt}],
                    temperature=0.5
                )
                quiz_text = response.choices[0].message.content.strip()
                print(f"[DEBUG] Raw GPT response for class '{class_name}':\n{quiz_text}")

                # --- Direct JSON parsing (no regex) ---
                parsed_json = json.loads(quiz_text)
                print(f"[DEBUG] Successfully parsed JSON for class '{class_name}'")

            except json.JSONDecodeError as e_json:
                print(f"[ERROR] JSON parsing failed for class '{class_name}': {e_json}")
            except Exception as e_ai:
                print(f"[ERROR] AI call failed for class '{class_name}': {e_ai}")

            # ---- FALLBACK QUIZ ----
            if not parsed_json:
                print(f"[DEBUG] Using fallback quiz for class '{class_name}'")
                parsed_json = {
                    "quiz_title": f"Fallback Quiz for {class_name}",
                    "instructions": "This is a fallback quiz.",
                    "questions": [
                        {
                            "category": "Math",
                            "prompt": "What is 10 + 5?",
                            "options": ["12", "15", "20", "25"],
                            "answer": "15"
                        },
                        {
                            "category": "Science",
                            "prompt": "Which planet is closest to the Sun?",
                            "options": ["Earth", "Mercury", "Venus", "Mars"],
                            "answer": "Mercury"
                        },
                        {
                            "category": "English",
                            "prompt": "Plural of 'mouse'?","options": ["Mouses", "Mice", "Mouse", "Mices"],
                            "answer": "Mice"
                        }
                    ]
                }

            # ---- SAVE QUIZ TO DB ----
            try:
                class_quiz = StudentQuiz(
                    quiz_json=parsed_json,
                    status="pending",
                    created_at=datetime.utcnow(),
                    class_name=class_name,
                    class_day=getattr(activity, "class_day", "")
                )
                db.add(class_quiz)
                print(f"[DEBUG] Quiz added to DB for class '{class_name}'")
            except Exception as e_db:
                print(f"[ERROR] Failed to save quiz for class '{class_name}': {e_db}")

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
            
# ai_engine.py (for example)
def generate_exam_questions(quiz):
    """
    Generate exam questions using OpenAI based on quiz topics.
    """

    questions = []
    q_id = 1

    for topic in quiz.topics:
        topic_name = topic["name"]
        count = int(topic["total"])

        # Clean, safe multi-line prompt
        system_prompt = (
            "You are an expert exam generator. "
            f"Create {count} multiple-choice questions for the topic: {topic_name}.\n\n"
            "Return the output STRICTLY as a JSON list, like this:\n"
            "[\n"
            "  {\n"
            "    \"question\": \"...\",\n"
            "    \"options\": [\"A\", \"B\", \"C\", \"D\"],\n"
            "    \"correct\": \"A\"\n"
            "  }\n"
            "]\n\n"
            "Rules:\n"
            f"- Generate EXACTLY {count} questions.\n"
            "- Each question must have exactly 4 options.\n"
            "- The 'correct' field must match one of the 4 options.\n"
            "- Do NOT add any explanation, commentary, or text outside the JSON.\n"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.5
            )

            raw_output = response.choices[0].message.content.strip()
            generated = json.loads(raw_output)

        except Exception as e:
            raise Exception(f"AI generation failed for topic '{topic_name}': {str(e)}")

        for item in generated:
            questions.append({
                "q_id": q_id,
                "topic": topic_name,
                "question": item["question"],
                "options": item["options"],
                "correct": item["correct"]
            })
            q_id += 1

    return questions


@app.post("/api/student-exams/start")
def start_exam(student_id: int, exam_id: int, db: Session = Depends(get_db)):
    """
    A student starts an exam.
    We create a StudentExam row and return the exam questions to frontend.
    """
    exam = db.query(Exam).filter(Exam.id == exam_id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    # Create student_exam entry
    student_exam = StudentExam(
        student_id=student_id,
        exam_id=exam.id,
        status="started",
        started_at=datetime.utcnow()
    )

    db.add(student_exam)
    db.commit()
    db.refresh(student_exam)

    return {
        "message": "Exam started",
        "student_exam_id": student_exam.id,
        "exam_id": exam.id,
        "questions": exam.questions  # send questions to frontend
    }

@app.post("/api/student-exams/answer")
def submit_answer(
    student_exam_id: int,
    question_id: int,
    student_answer: str,
    db: Session = Depends(get_db)
):
    """
    Save an answer for a question and evaluate correctness.
    """
    # 1. Get the exam session
    student_exam = db.query(StudentExam).filter(
        StudentExam.id == student_exam_id
    ).first()

    if not student_exam:
        raise HTTPException(status_code=404, detail="Student exam session not found")

    # 2. Find the exam the student is taking
    exam = db.query(Exam).filter(Exam.id == student_exam.exam_id).first()

    # 3. Look for the question inside exam.questions JSON
    correct_answer = None
    for q in exam.questions:
        if q["q_id"] == question_id:
            correct_answer = q["correct"]
            break

    if correct_answer is None:
        raise HTTPException(status_code=400, detail="Invalid question ID")

    # 4. Save answer in DB
    answer_row = StudentExamAnswer(
        student_exam_id=student_exam_id,
        question_id=question_id,
        student_answer=student_answer,
        correct_answer=correct_answer,
        is_correct=(student_answer == correct_answer)
    )

    db.add(answer_row)
    db.commit()

    return {
        "message": "Answer saved",
        "correct": student_answer == correct_answer
    }

@app.post("/api/exams/generate/{quiz_id}")
def generate_exam(quiz_id: int, db: Session = Depends(get_db)):
    """
    1. Fetch quiz by quiz_id
    2. Generate questions using AI logic
    3. Save exam in 'exams' table
    4. Return exam JSON to frontend
    """
    # 1. Fetch quiz
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    # 2. Generate questions from quiz.topics
    try:
        questions = generate_exam_questions(quiz)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate exam: {str(e)}")

    if not questions:
        raise HTTPException(status_code=500, detail="No questions generated")

    # 3. Save exam in DB
    new_exam = Exam(
        quiz_id=quiz.id,
        questions=questions
    )
    db.add(new_exam)
    db.commit()
    db.refresh(new_exam)

    # 4. Return to frontend
    return {
        "message": "Exam generated successfully",
        "exam_id": new_exam.id,
        "quiz_id": quiz.id,
        "class_name": quiz.class_name,
        "subject": quiz.subject,
        "difficulty": quiz.difficulty,
        "questions": questions
    }


@app.post("/retrieve-week-number")
def retrieve_week_number(request: WeekRequest, db: Session = Depends(get_db)):    
    try:
        # Convert input date string to date object
        input_date = datetime.strptime(request.date, "%Y-%m-%d").date()
        # ------------------ Save or replace admin date ------------------
        existing_entry = db.query(AdminDate).first()
        if existing_entry:
            existing_entry.date = request.date
        else:
            existing_entry = AdminDate(date=request.date)
            db.add(existing_entry)

        db.commit()

        # ------------------ Calculate week number ------------------

        # Get the Monday of the week of the input date
        input_monday = input_date - timedelta(days=input_date.weekday())  # weekday(): Mon=0 ... Sun=6

        # Get the current date
        today = datetime.utcnow().date()
        current_monday = today - timedelta(days=today.weekday())

        # Calculate number of weeks between input week and current week
        delta_days = (current_monday - input_monday).days
        if delta_days < 0:
            raise HTTPException(status_code=400, detail="Selected date is in the future")

        week_number = (delta_days // 7) + 1  # first week is 1

        return {"week_number": week_number}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

@app.post("/login-exam-module")
def login_exam_module(login_data: StudentLogin, db: Session = Depends(get_db)):
    """
    Login a student using student_id and password
    """
    # Look up student by student_id
    student = db.query(Student).filter(Student.student_id == login_data.student_id).first()
    
    if not student:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Simple password check (plaintext; replace with hashed check in production)
    if login_data.password != student.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Return student info (without password)
    return {
        "student_id": student.student_id,
        "name": student.name,
        "parent_email": student.parent_email,
        "class_name": student.class_name,
        "class_day": student.class_day
    }


@app.get("/student-name")
def get_student_name(student_id: int = Query(...), db: Session = Depends(get_db)):
    """
    Return student name for a given student_id.
    """
    student = db.query(User).filter(User.id == student_id).first()

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    return {"student_id": student.id, "student_name": student.name}
    
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

#endpoint to save a certain exam for Exam module
@app.post("/api/quizzes")
def create_quiz(quiz: QuizCreate, db: Session = Depends(get_db)):
    """
    Create a new quiz with extensive debugging.
    """

    print("\n========== QUIZ CREATION START ==========")
    
    # Print the raw QuizCreate object
    print("🔍 Incoming QuizCreate object:", quiz)

    # Convert quiz to dict to inspect content
    try:
        quiz_dict = quiz.dict()
        print("📦 Parsed quiz payload:", quiz_dict)
    except Exception as e:
        print("❌ Failed to convert quiz to dict:", e)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Invalid quiz model: {str(e)}")

    # Debug each field
    print("➡️ class_name:", quiz.class_name, type(quiz.class_name))
    print("➡️ subject:", quiz.subject, type(quiz.subject))
    print("➡️ difficulty:", quiz.difficulty, type(quiz.difficulty))
    print("➡️ num_topics:", quiz.num_topics, type(quiz.num_topics))
    print("➡️ topics:", quiz.topics, type(quiz.topics))

    # Validate topics field
    if not isinstance(quiz.topics, list):
        print("❌ ERROR: topics is not a list")
        raise HTTPException(status_code=400, detail="topics must be a list")

    print("📝 Topics count:", len(quiz.topics))
    for i, t in enumerate(quiz.topics):
        print(f"   └─ Topic {i}: {t}")

    try:
        print("\n--- Creating SQLAlchemy Quiz object ---")
        new_quiz = Quiz(
            class_name = quiz.class_name,
            subject = quiz.subject,
            difficulty = quiz.difficulty,
            num_topics = quiz.num_topics,
            topics = [t.dict() for t in quiz.topics]   # Convert to JSON-safe dict
        )
        print("✅ SQLAlchemy object created:", new_quiz)

        print("\n--- Adding to DB session ---")
        db.add(new_quiz)

        print("\n--- Attempting commit ---")
        db.commit()
        print("✅ Commit successful!")

        db.refresh(new_quiz)
        print("🔄 Refreshed object:", new_quiz)

        print("========== QUIZ CREATION COMPLETE ==========\n")

        return {
            "message": "Quiz created successfully",
            "quiz_id": new_quiz.id
        }

    except Exception as e:
        print("\n❌ EXCEPTION DURING DB OPERATION ❌")
        print("Error message:", str(e))
        traceback.print_exc()

        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating quiz: {str(e)}")
     
def generate_otp():
    return random.randint(100000, 999999)

@app.post("/send-otp")
def send_otp_endpoint(request: OTPRequest, db: Session = Depends(get_db)):
    email = request.email.strip().lower()
    print(f"[DEBUG] Received OTP request for email: {email}")

    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")

    otp = generate_otp()
    otp_store[email] = {"otp": otp, "expiry": time.time() + 300}

    # Call your email sending function
    try:
        send_otp_email(email, otp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending email: {e}")

    return {"message": "OTP sent successfully"} 

@app.post("/retrieve-term-start-date", response_model=TermStartDateResponse)
def retrieve_term_start_date(db: Session = Depends(get_db)):
    try:
        stmt = select(AdminDate).limit(1)
        result = db.execute(stmt).scalars().first()

        if not result or not result.date:
            raise HTTPException(status_code=404, detail="Term start date not found")

        term_date = result.date
        # Convert to string if needed
        if isinstance(term_date, (date, datetime)):
            term_date_str = term_date.isoformat()
        else:
            term_date_str = str(term_date)  # already string

        return TermStartDateResponse(date=term_date_str)

    except Exception as e:
        print("Error retrieving term start date:", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/get_next_user_id_exam_module")
def get_next_user_id_exam_module(db: Session = Depends(get_db)):
    """
    Returns the next student ID as a simple number (1, 2, 3, ...)
    """
    try:
        # Get last student ordered by numeric ID
        last_student = db.query(Student).order_by(Student.id.desc()).first()

        if last_student:
            next_num = int(last_student.id) + 1
        else:
            next_num = 1

        return str(next_num)

    except Exception as e:
        print("[ERROR] Failed to generate next ID:", e)
        raise HTTPException(status_code=500, detail="Could not generate next user ID")

@app.post("/set-term-start-date")
def set_term_start_date(term_data: AdminDateSchema, db: Session = Depends(get_db)):
    try:
        # Clear any previous dates (since primary key only allows unique row)
        db.query(AdminDate).delete()
        
        # Add new date
        new_date = AdminDate(date=term_data.date)
        db.add(new_date)
        db.commit()
        db.refresh(new_date)

        return {"message": "Term start date updated successfully", "date": new_date.date}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set-term-data")
def set_term_data(term_data: TermDataSchema, db: Session = Depends(get_db)):
    try:
        # Convert strings to date objects
        term_start = datetime.fromisoformat(term_data.termStartDate).date()
        term_end = datetime.fromisoformat(term_data.termEndDate).date()

        # Optionally, delete old data or just insert new
        # db.query(AdminTerm).delete()  # if you want only 1 row

        new_term = AdminTerm(
            year=term_data.year,
            term_start_date=term_start,
            term_end_date=term_end
        )
        db.add(new_term)
        db.commit()
        db.refresh(new_term)

        return {
            "message": "Term data saved successfully",
            "term": {
                "year": new_term.year,
                "termStartDate": str(new_term.term_start_date),
                "termEndDate": str(new_term.term_end_date)
            }
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-term-dates")
def get_term_dates(db: Session = Depends(get_db)):
    try:
        terms = db.query(AdminTerm).all()
        return [
            {
                "id": t.id,
                "year": t.year,
                "termStartDate": t.term_start_date.isoformat(),
                "termEndDate": t.term_end_date.isoformat()
            }
            for t in terms
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


        
@app.post("/verify-otp")
def verify_otp(request: OTPVerify, db: Session = Depends(get_db)):
    email = request.email.strip().lower()
    otp = str(request.otp).strip()

    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    # --- Check if OTP exists for this email ---
    record = otp_store.get(email)
    if not record:
        raise HTTPException(status_code=400, detail="OTP not sent")

    # --- Check expiry ---
    if time.time() > record["expiry"]:
        otp_store.pop(email, None)
        raise HTTPException(status_code=400, detail="OTP expired")

    # --- Check OTP match ---
    if otp != str(record["otp"]):
        raise HTTPException(status_code=401, detail="Invalid OTP")

    # --- Fetch user by email ---
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")

    # --- Remove OTP after successful verification ---
    otp_store.pop(email, None)

    # --- Respond with user info ---
    return {
        "message": "OTP verified",
        "user": {
            "student_id": user.id,
            "email": user.email,
            "name": user.name,
            "class_name": user.class_name,
            "class_day": user.class_day
        }
    }
    
    
@app.post("/login")
def login(request: LoginRequest, response: Response, db: Session = Depends(get_db)):
    phone = request.phone_number.strip()
    print(f"[DEBUG] Received login request for phone: {phone}")

    if not phone:
        raise HTTPException(status_code=400, detail="Phone number is required")

    # --- Fetch user by phone ---
    user = db.query(User).filter(User.phone_number == phone).first()
    if not user:
        print(f"[WARNING] Phone number {phone} not registered")
        raise HTTPException(status_code=401, detail="Phone number not registered")

    # --- Retrieve OTP record ---
    record = otp_store.get(phone)
    if not record:
        print(f"[WARNING] No OTP record found for phone {phone}")
        raise HTTPException(status_code=400, detail="OTP not sent")

    if time.time() > record["expiry"]:
        print(f"[WARNING] OTP for phone {phone} has expired")
        otp_store.pop(phone, None)
        raise HTTPException(status_code=400, detail="OTP expired")

    if str(request.otp) != str(record["otp"]):
        print(f"[WARNING] Entered OTP ({request.otp}) does not match stored OTP ({record['otp']})")
        raise HTTPException(status_code=400, detail="Invalid OTP")

    print(f"[INFO] OTP for phone {phone} is valid")
    otp_store.pop(phone, None)  # clear OTP after verification

    # --- Clear previous user state if applicable ---
    if user.name in user_contexts:
        user_contexts[user.name] = []
        print(f"[DEBUG] Cleared previous context for user {user.name}")

    user_vectorstores_initialized[user.name] = False
    print(f"[DEBUG] vectorstores_initialized for user {user.name} set to False")

    # --- Remove existing session if using session management ---
    existing_session = db.query(SessionModel).filter(SessionModel.user_id == user.id).first()
    if existing_session:
        db.delete(existing_session)
        db.commit()
        print(f"[DEBUG] Cleared existing session for user {user.id}")

    # --- Set session cookie ---
    response.set_cookie(
        key="session_token",
        value=f"session_{user.id}",
        httponly=True,
        max_age=3600
    )

    # --- Prepare response ---
    user_info = {
        "id": user.id,
        "name": user.name,
        "phone_number": user.phone_number,
        "class_name": user.class_name,
        "class_day": user.class_day  # optional
    }

    print(f"[INFO] Login successful for user {user.name}")
    return {"message": "Login successful", "user": user_info}




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


@app.get("/api/leaderboard/accumulative/{class_name}")
def accumulative_leaderboard(
    class_name: str = Path(..., description="Class name"),
    day: str = Query(..., description="Class day, e.g., Monday"),
    db: Session = Depends(get_db)
):
    """
    Returns accumulative leaderboard for a given class and class day
    within 10 weeks from admin start date.
    """
    # --- Get start date from AdminDate ---
    admin_date_entry = db.query(AdminDate).first()
    if not admin_date_entry or not admin_date_entry.date:
        raise HTTPException(status_code=400, detail="Admin start date not set")
    
    # Convert to datetime if stored as string
    if isinstance(admin_date_entry.date, str):
        try:
            start_date = datetime.strptime(admin_date_entry.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=500, detail="Invalid date format in AdminDate")
    else:
        start_date = admin_date_entry.date

    # Calculate end date (10 weeks from start)
    end_date = start_date + timedelta(weeks=10)

    try:
        # Aggregate scores within date range and class day
        results = (
            db.query(
                QuizResult.student_id,
                QuizResult.student_name,
                func.sum(QuizResult.total_score).label("total_score")
            )
            .filter(QuizResult.class_name == class_name)
            .filter(QuizResult.class_day == day)
            .filter(QuizResult.submitted_at >= start_date)
            .filter(QuizResult.submitted_at <= end_date)
            .group_by(QuizResult.student_id, QuizResult.student_name)
            .order_by(func.sum(QuizResult.total_score).desc())
            .all()
        )

        leaderboard = [
            {
                "student_id": r.student_id,
                "student_name": r.student_name,
                "total_score": r.total_score
            }
            for r in results
        ]

        return {
            "class_name": class_name,
            "class_day": day,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "leaderboard": leaderboard
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/quiz-results")
def get_quiz_results(
    class_name: str = Query(..., description="Class name to filter results"),
    db: Session = Depends(get_db)
):
    """
    Return all quiz results for a specific class, sorted by score (descending).
    """
    cleaned_class_name = class_name.strip()

    results = (
        db.query(QuizResult)
        .filter(QuizResult.class_name == cleaned_class_name)
        .order_by(QuizResult.total_score.desc())
        .all()
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for class '{class_name}'"
        )

    response_data = [
        {
            "student_id": r.student_id,
            "student_name": r.student_name,
            "class_name": r.class_name,
            "total_score": r.total_score,
            "total_questions": r.total_questions,
            "submitted_at": r.submitted_at.isoformat()
        }
        for r in results
    ]

    return JSONResponse(content=response_data)


@app.post("/add-activities-from-csv")
async def add_activities_from_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload activities via CSV and create Activity entries identical to the /add-activity endpoint.
    """

    # --- Validate file type ---
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    # --- Read CSV ---
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    required_columns = ["instructions", "score_logic", "class_name", "class_day", "week_number"]
    for col in required_columns:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing required column: {col}")

    added_activities = []

    # --- Process each row ---
    for index, row in df.iterrows():

        instructions = str(row.get("instructions", "")).strip()
        score_logic = str(row.get("score_logic", "")).strip()
        class_name = str(row.get("class_name", "")).strip()
        class_day = str(row.get("class_day", "")).strip()
        week_number = row.get("week_number")

        # --- Validation (same as single-endpoint) ---
        if not instructions or not isinstance(instructions, str):
            continue
        if not score_logic or not isinstance(score_logic, str):
            continue
        if not class_name or not class_day:
            continue

        try:
            week_number = int(week_number)
        except:
            week_number = None

        # --- activity_type comes from score_logic ---
        activity_type = score_logic

        # --- SAME admin_prompt as /add-activity ---
        admin_prompt = (
            f"Create a gamified activity for students in {class_name}.\n"
            f"The activity format should be {activity_type}.\n"
            f"The topic taught is {instructions}."
        )

        # --- SAME default questions as /add-activity ---
        questions_json = [
            {
                "category": "General",
                "prompt": instructions,
                "options": ["A", "B", "C", "D"],
                "answer": "A"
            }
        ]

        # --- Create Activity entry ---
        activity = Activity(
            instructions=instructions,
            admin_prompt=admin_prompt,
            questions=questions_json,
            score_logic=score_logic,
            class_name=class_name,
            class_day=class_day,
            week_number=week_number
        )

        # --- Save ---
        try:
            db.add(activity)
            db.commit()
            db.refresh(activity)
            added_activities.append({
                "activity_id": activity.activity_id,
                "instructions": instructions,
                "week_number": week_number
            })
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Could not save activity at row {index}: {e}")
            continue

    if not added_activities:
        raise HTTPException(
            status_code=400,
            detail="CSV processed but no valid activities were added."
        )

    return {
        "message": f"{len(added_activities)} activities added successfully",
        "activities": added_activities
    }

 
@app.post("/add-activity")
def add_activity(payload: ActivityPayload, db: Session = Depends(get_db)):
    """
    Add a new activity. Front end provides instructions and score_logic (used as activity type).
    Backend constructs admin_prompt for GPT use.
    """

    # --- Validate payload ---
    if not payload.instructions or not isinstance(payload.instructions, str):
        raise HTTPException(status_code=400, detail="Instructions must be a non-empty string")
    if not payload.score_logic or not isinstance(payload.score_logic, str):
        raise HTTPException(status_code=400, detail="score_logic (used as activity type) must be a non-empty string")
    if not payload.class_name or not payload.class_day:
        raise HTTPException(status_code=400, detail="class_name and class_day are required")
    if payload.week_number is not None and not isinstance(payload.week_number, int):
        raise HTTPException(status_code=400, detail="week_number must be an integer if provided")

    # --- Use score_logic as activity_type in admin_prompt ---
    activity_type = payload.score_logic

    admin_prompt = (
        f"Create a gamified activity for students in {payload.class_name}.\n"
        f"The activity format should be {activity_type}.\n"  # dynamic injection from score_logic
        f"The topic taught is {payload.instructions}."
    )

    # --- Default or provided questions ---
    questions_json = payload.questions or [
        {
            "category": "General",
            "prompt": payload.instructions,
            "options": ["A", "B", "C", "D"],  # placeholder options
            "answer": "A"  # placeholder answer
        }
    ]

    # --- Create Activity object ---
    activity = Activity(
        instructions=payload.instructions,
        admin_prompt=admin_prompt,   # save admin prompt for AI
        questions=questions_json,
        score_logic=payload.score_logic,
        class_name=payload.class_name,
        class_day=payload.class_day,
        week_number=payload.week_number
    )

    # --- Save to DB ---
    try:
        db.add(activity)
        db.commit()
        db.refresh(activity)
        return {
            "message": "Activity added successfully",
            "activity_id": activity.activity_id,
            "admin_prompt": activity.admin_prompt  # Optional: return for debugging
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to add activity: {e}")

@app.get("/get-quiz")
def get_quiz(
    class_name: str = Query(..., description="Class name of the quiz"),
    student_id: int = Query(..., description="ID of the student requesting the quiz"),
    db: Session = Depends(get_db)
):
    """
    Fetch the latest quiz for a given class_name.
    Trims whitespace and matches case-insensitively.
    Returns the quiz JSON.
    """
    # ------------------ Calculate week_number ------------------
    admin_date_entry = db.query(AdminDate).first()
    if not admin_date_entry or not admin_date_entry.date:
        raise HTTPException(status_code=400, detail="Admin date not set")
    
    date_to_use = admin_date_entry.date
    week_number = calculate_week_number(date_to_use)

    # ------------------ Check if student already attempted ------------------
    existing_result = db.query(QuizResult).filter(
        QuizResult.student_id == student_id,
        QuizResult.class_name == class_name.strip(),
        QuizResult.week_number == week_number
    ).first()

    if existing_result:
        return JSONResponse(
            status_code=200,
            content={"message": "You have already attempted this week's quiz."}
        )

    cleaned_class_name = class_name.strip()

    quiz = (
        db.query(StudentQuiz)
        .filter(func.lower(func.trim(StudentQuiz.class_name)) == cleaned_class_name.lower())
        .order_by(StudentQuiz.created_at.desc())
        .first()
    )

    if not quiz:
        raise HTTPException(status_code=404, detail=f"No quiz found for class '{class_name}'")

    # Ensure quiz_json is always returned as a dict
    quiz_data = quiz.quiz_json or {}

    return JSONResponse(content=quiz_data)
    

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
def submit_answer(
    student_id: int,
    quiz_id: int,
    question_index: int,
    selected_option: str,
    db: Session = Depends(get_db)
):
    """
    Record a student's answer for a specific quiz.
    Does nothing if a result already exists for this student and quiz.
    """
    # Check if the student already has a QuizResult for this quiz
    existing_result = db.query(QuizResult).filter(
        QuizResult.student_id == student_id,
        QuizResult.quiz_id == quiz_id
    ).first()
    
    if existing_result:
        return {
            "message": "Result already exists for this student and quiz",
            "current_score": existing_result.total_score
        }

    # Fetch the quiz by quiz_id
    quiz = db.query(StudentQuiz).filter(StudentQuiz.quiz_id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    # Initialize student_answers dictionary if missing
    if "student_answers" not in quiz.quiz_json:
        quiz.quiz_json["student_answers"] = {}

    # Initialize student-specific answers
    if str(student_id) not in quiz.quiz_json["student_answers"]:
        quiz.quiz_json["student_answers"][str(student_id)] = {}

    # Record the answer
    quiz.quiz_json["student_answers"][str(student_id)][f"q{question_index+1}"] = selected_option

    # Recalculate student's score
    student_answers = quiz.quiz_json["student_answers"][str(student_id)]
    score = 0
    for idx, q in enumerate(quiz.quiz_json.get("questions", [])):
        if student_answers.get(f"q{idx+1}") == q.get("answer"):
            score += 1

    # Save the score in student_answers
    quiz.quiz_json["student_answers"][str(student_id)]["score"] = score

    db.commit()
    return {
        "message": "Answer recorded",
        "current_score": score
    }

    
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime

def send_otp_email(to_email: str, otp: str):
    message = Mail(
        from_email='noreply@gemkidsacademy.com.au',
        to_emails=to_email,
        subject='Your OTP Code',
        html_content=f'<p>Your OTP code is <strong>{otp}</strong>. It will expire in 5 minutes.</p>'
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(f"[INFO] Email sent to {to_email}, status code {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Failed to send email to {to_email}: {e}")
        raise


def calculate_week_number(date_str: str) -> int:
    try:
        input_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        input_monday = input_date - timedelta(days=input_date.weekday())

        today = datetime.utcnow().date()
        current_monday = today - timedelta(days=today.weekday())

        delta_days = (current_monday - input_monday).days
        if delta_days < 0:
            raise HTTPException(status_code=400, detail="Selected date is in the future")

        week_number = (delta_days // 7) + 1
        return week_number
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")


@app.post("/submit-quiz-answer")
def submit_quiz_answer(payload: AnswerPayload, db: Session = Depends(get_db)):
    """
    Submit a single answer for the current quiz question.
    Records answers in StudentQuiz, but creates only one QuizResult per student per quiz.
    """
    print("\n--- SUBMIT QUIZ ANSWER ---")
    print("Received payload:", payload)

    # Fetch the latest quiz for this class
    quiz = (
        db.query(StudentQuiz)
        .filter(StudentQuiz.class_name == payload.class_name)
        .order_by(StudentQuiz.created_at.desc())
        .first()
    )
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    # Start quiz if not started
    if not quiz.started_at:
        quiz.started_at = datetime.utcnow()

    # Load or initialize student_answers
    student_answers = quiz.quiz_json.get("student_answers") or {}

    # Record current answer
    q_key = f"q{payload.question_index + 1}"
    student_answers[q_key] = payload.selected_option
    quiz.quiz_json["student_answers"] = student_answers  # MutableDict tracks changes

    # Calculate current score
    correct_count = 0
    for idx, question in enumerate(quiz.quiz_json.get("questions", [])):
        q_key_check = f"q{idx+1}"
        student_answer = student_answers.get(q_key_check)
        if student_answer == question.get("answer"):
            correct_count += 1

    quiz.quiz_json["score"] = correct_count

    # Check if quiz is completed
    total_questions = len(quiz.quiz_json.get("questions", []))
    answered_questions = len(student_answers)
    quiz_completed = answered_questions >= total_questions

    if quiz_completed:
        quiz.status = "completed"
        quiz.completed_at = datetime.utcnow()

        # Check if a result already exists for this student for this quiz
        existing_result = db.query(QuizResult).filter(
            QuizResult.student_id == payload.student_id,
            QuizResult.quiz_id == quiz.quiz_id
        ).first()

        if existing_result:
            print("Result already exists, not creating a new row")
        else:
            admin_date_entry = db.query(AdminDate).first()
            date_to_use = admin_date_entry.date
            week_number = calculate_week_number(date_to_use)

            result = QuizResult(
                quiz_id=quiz.quiz_id,
                student_id=payload.student_id,
                student_name=payload.student_name,
                class_name=payload.class_name,
                class_day=getattr(payload, "class_day", None),
                week_number=week_number,   # <-- new
                total_score=correct_count,
                total_questions=total_questions,
                submitted_at=datetime.utcnow()
            )
            db.add(result)
            print("Result saved successfully")

    else:
        quiz.status = "in_progress"

    # Commit quiz updates (answers & score)
    db.commit()

    return {
        "message": "Answer recorded",
        "current_score": correct_count,
        "quiz_status": quiz.status
    }





