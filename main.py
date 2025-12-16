# main.py
from fastapi import FastAPI, HTTPException, Depends, Response, Query, Path, File, UploadFile, Body
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import docx
from io import BytesIO
from typing import List
from sendgrid import SendGridAPIClient
from datetime import datetime, timedelta,date, timezone
from sqlalchemy.dialects.postgresql import JSONB
import mammoth

 
import pandas as pd
import os
import random
import time
from sendgrid.helpers.mail import Mail
from typing import List, Dict, Any, Optional
import re 
import traceback
from openai import AsyncOpenAI
from google.oauth2.service_account import Credentials

import uuid
from sqlalchemy.dialects.postgresql import UUID
from google.cloud import storage





 
import json

from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, ForeignKey, select, func, text, Boolean, Date, Text, desc, Float, Index, case
from fastapi.responses import JSONResponse



from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.mutable import MutableDict



from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
GLOBAL_IMAGE_MAP = {}

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

QuestionSchema = {
    "type": "object",
    "properties": {
        "class_name": {"type": "string"},
        "subject": {"type": "string"},
        "topic": {"type": "string"},
        "difficulty": {"type": "string"},
        "question_text": {"type": "string"},
        "images": {
            "type": "array",
            "items": {"type": "string"}
        },
        "options": {
            "type": "object",
            "additionalProperties": {"type": "string"}
        },
        "correct_answer": {"type": "string"},
        "partial": {"type": "boolean"}
    },
    "required": ["question_text", "options", "correct_answer"],
    "additionalProperties": False
}

# -----------------------------
# Google Cloud Storage Setup
# -----------------------------
# Path to your service account JSON for GCS
service_account_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if not service_account_json:
    raise ValueError("Environment variable 'GOOGLE_APPLICATION_CREDENTIALS_JSON' is missing.")

# 2️⃣ Replace literal "\n" with actual newlines for the private key
service_account_info = json.loads(service_account_json)
service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

# 3️⃣ Initialize GCS client with credentials
gcs_client = storage.Client(
    credentials=Credentials.from_service_account_info(service_account_info),
    project=service_account_info["project_id"]
)

# 4️⃣ Access your bucket (GLOBAL)
BUCKET_NAME = "exammoduleimages"
gcs_bucket = gcs_client.bucket(BUCKET_NAME)

print(f"✅ Initialized GCS client for bucket: {gcs_bucket}")

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
client_save_questions = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
otp_store = {}
# ---------------------------
# Models
# ---------------------------
class StudentExamResultsThinkingSkills(Base):
    __tablename__ = "student_exam_results_thinking_skills"

    __table_args__ = (
        Index("idx_results_student", "student_id"),
    )

    id = Column(Integer, primary_key=True, index=True)

    student_id = Column(
        Text,
        ForeignKey("students.id"),
        nullable=False
    )

    exam_attempt_id = Column(
        Integer,
        ForeignKey("student_exams.id"),
        nullable=False,
        unique=True
    )

    total_questions = Column(Integer, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    wrong_answers = Column(Integer, nullable=False)
    accuracy_percent = Column(Float, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )
 
class ExamSubmissionRequest(BaseModel):
    session_id: int
    answers: Dict[int, str]

class StudentExamResponse(Base):
    __tablename__ = "student_exam_responses"

    id = Column(Integer, primary_key=True)

    student_id = Column(Integer, nullable=False)
    exam_id = Column(Integer, nullable=False)
    exam_attempt_id = Column(Integer, nullable=False)

    q_id = Column(Integer, nullable=False)
    topic = Column(Text, nullable=False)

    selected_option = Column(Text, nullable=True)
    correct_option = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )


class StudentExamAttemptThinkingSkills(Base):
    __tablename__ = "student_exam_attempts_thinkingskills"

    id = Column(Integer, primary_key=True)
    session_id = Column(Text, unique=True, nullable=False)
    student_id = Column(Text, nullable=False)
    exam_id = Column(Integer, nullable=False)

    started_at = Column(DateTime(timezone=True), nullable=False)
    submitted_at = Column(DateTime(timezone=True))
    is_completed = Column(Boolean, default=False)
    score = Column(Integer)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class StudentExamAnswerThinkingSkills(Base):
    __tablename__ = "student_exam_answers_thinkingskills"

    id = Column(Integer, primary_key=True)
    attempt_id = Column(
        Integer,
        ForeignKey(
            "student_exam_attempts_thinkingskills.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )
    question_id = Column(Integer, nullable=False)
    selected_option = Column(Text, nullable=False)
    is_correct = Column(Boolean)

    created_at = Column(DateTime(timezone=True), server_default=func.now())



class StudentExamReading(Base):
    __tablename__ = "student_exams_reading"

    id = Column(Integer, primary_key=True, index=True)

    # Keep consistent with writing exam (supports IDs like "Gem002")
    student_id = Column(Text, nullable=False)

    exam_id = Column(Integer, nullable=False)

    started_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    finished = Column(Boolean, default=False, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

class WritingGenerateSchema(BaseModel):
    class_name: str
    difficulty: str
 
class StudentExamWriting(Base):
    __tablename__ = "student_exam_writing"

    id = Column(Integer, primary_key=True, index=True)

    # Change from Integer → Text to support IDs like "Gem002"
    student_id = Column(Text, nullable=False)

    exam_id = Column(Integer, nullable=False)

    started_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    completed_at = Column(DateTime(timezone=True), nullable=True)

    duration_minutes = Column(Integer, default=40, nullable=False)

    answer_text = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

class WritingSubmitSchema(BaseModel):
    answer_text: str

class GeneratedExamWriting(Base):
    __tablename__ = "generated_exam_writing"

    id = Column(Integer, primary_key=True, index=True)

    class_name = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)

    question_text = Column(Text, nullable=False)

    duration_minutes = Column(Integer, default=40)

    is_current = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

class WritingQuestionBank(Base):
    __tablename__ = "writing_question_bank"

    id = Column(Integer, primary_key=True, index=True)

    class_name = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)

    question_text = Column(Text, nullable=False)

    source_file = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
 

class WritingQuizSchema(BaseModel):
    class_name: str
    subject: str
    topic: str
    difficulty: str
 
class GeneratedExamFoundational(Base):
    __tablename__ = "generated_exam_foundational"

    id = Column(Integer, primary_key=True, index=True)

    class_name = Column(String, nullable=False)
    subject = Column(String, nullable=False)

    total_questions = Column(Integer, nullable=False)

    exam_json = Column(JSON, nullable=False)

    is_current = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

class StudentsExamFoundational(Base):
    __tablename__ = "students_exam_foundational"

    id = Column(Integer, primary_key=True, index=True)

    student_id = Column(String, nullable=False)

    exam_id = Column(
        Integer,
        ForeignKey("generated_exam_foundational.id"),
        nullable=False
    )

    started_at = Column(DateTime(timezone=True), nullable=False)

    completed_at = Column(DateTime(timezone=True), nullable=True)

    current_section_index = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
 
class EmptyRequest(BaseModel):
    pass

class QuizSetupWriting(Base):
    __tablename__ = "quiz_setup_writing"

    id = Column(Integer, primary_key=True, index=True)

    class_name = Column(String, nullable=False)
    subject = Column(String, nullable=False)

    topic = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

class QuizSetupFoundational(Base):
    __tablename__ = "quiz_setup_foundational"

    id = Column(Integer, primary_key=True, index=True)

    class_name = Column(String, nullable=False)
    subject = Column(String, nullable=False)

    # -------- Section 1 --------
    section1_name = Column(String, nullable=False)
    section1_ai = Column(Integer, default=0)
    section1_db = Column(Integer, default=0)
    section1_total = Column(Integer, default=0)
    section1_time = Column(Integer, default=0)
    section1_intro = Column(Text, default="")

    # -------- Section 2 --------
    section2_name = Column(String, nullable=False)
    section2_ai = Column(Integer, default=0)
    section2_db = Column(Integer, default=0)
    section2_total = Column(Integer, default=0)
    section2_time = Column(Integer, default=0)
    section2_intro = Column(Text, default="")

    # -------- Section 3 (optional) --------
    section3_name = Column(String, nullable=True)
    section3_ai = Column(Integer, nullable=True)
    section3_db = Column(Integer, nullable=True)
    section3_total = Column(Integer, nullable=True)
    section3_time = Column(Integer, nullable=True)
    section3_intro = Column(Text, nullable=True) 

class SectionSchema(BaseModel):
    name: str
    ai: int
    db: int
    total: int
    time: int
    intro: str


class QuizSetupFoundationalSchema(BaseModel):
    class_name: str
    subject: str
    sections: List[SectionSchema]
 
class ReadingExamRequest(BaseModel):
    class_name: str
    difficulty: str

class AddStudentExamModuleRequest(BaseModel):
    id: str                 # Backend-suggested ID (e.g. "1" or UUID)
    student_id: str         # "Gem001"
    name: str
    parent_email: EmailStr
    class_name: str
    class_day: str

class GeneratedExamReading(Base):
    __tablename__ = "generated_exams_reading"

    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(Integer, ForeignKey("reading_exam_config.id"), nullable=True)
    class_name = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)
    total_questions = Column(Integer, nullable=False)
    exam_json = Column(JSONB, nullable=False)  # reading_material + questions[]
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class FinishExamRequestFoundational(BaseModel):
    student_id: str
    answers: Dict[str, str]
    reason: Optional[str] = "submitted"
 
class FinishExamRequest(BaseModel):
    student_id: str
    answers: Dict[str, str]
 
class StartExamRequest(BaseModel):
    student_id: str   # <-- MUST BE STRING
 
#when generate exam is pressed we create a row here
class StudentExam(Base):
    __tablename__ = "student_exams"

    id = Column(Integer, primary_key=True, index=True)

    # Student ID is a string (e.g., "Gem002")
    student_id = Column(
        String,
        ForeignKey("students.id"),
        nullable=False,
        index=True
    )

    exam_id = Column(
        Integer,
        ForeignKey("exams.id"),
        nullable=False,
        index=True
    )

    # Timezone-aware timestamps (UTC)
    started_at = Column(
        DateTime(timezone=True),
        nullable=False
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    duration_minutes = Column(Integer, default=40, nullable=False)

    # Relationships
    student = relationship(
        "Student",
        back_populates="student_exams"
    )

    exam = relationship(
        "Exam",
        back_populates="student_exams"
    )

    answers = relationship(
        "StudentExamAnswer",
        back_populates="student_exam",
        cascade="all, delete-orphan"
    )


class Exam_reading(Base):
    __tablename__ = "exams_reading"   # UPDATED TABLE NAME

    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String)
    subject = Column(String)
    difficulty = Column(String)
    topic = Column(String)
    total_questions = Column(Integer)
    reading_material = Column(JSON)  # dictionary of extracts/paragraphs



class Question_reading(Base):
    __tablename__ = "questions_reading"

    id = Column(Integer, primary_key=True, index=True)

    # FILTERABLE FIELDS
    class_name = Column(String, index=True)
    subject = Column(String, index=True)
    difficulty = Column(String, index=True)
    topic = Column(String, index=True)
    total_questions = Column(Integer)

    # Optional link to a full exam created by admin
    exam_id = Column(Integer, ForeignKey("exams_reading.id"), nullable=True)

    # COMPLETE EXAM CONTENT
    exam_bundle = Column(JSON)  # reading_material + questions[]


class QuestionReadingCreate(BaseModel):
    question_number: int
    question_text: str
    correct_answer: str

class ExamReadingCreate(BaseModel):
    class_name: str
    subject: str
    difficulty: str
    topic: str
    total_questions: int
    reading_material: Dict[str, str]
    answer_options: Dict[str, str] 
    questions: List[QuestionReadingCreate] 


class ReadingExamConfig(Base):
    __tablename__ = "reading_exam_config"

    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)
    num_topics = Column(Integer, nullable=False)
    topics = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ReadingTopicItem(BaseModel):
    name: str
    num_questions: int


class ReadingExamConfigCreate(BaseModel):
    class_name: str
    subject: str
    difficulty: str
    topics: List[ReadingTopicItem]

class Student(Base):
    __tablename__ = "students"

    id = Column(String, primary_key=True, index=True)  # <-- VARCHAR PK
    student_id = Column(String, unique=True, nullable=False)  # e.g. "Gem002"
    password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    parent_email = Column(String, nullable=False)
    class_name = Column(String, nullable=False)
    class_day = Column(String, nullable=True)

    student_exams = relationship("StudentExam", back_populates="student")
  

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
    class_name = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)
    num_topics = Column(Integer, nullable=False)
    topics = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    exams = relationship("Exam", back_populates="quiz")


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)

    class_name = Column(String(50), nullable=False)
    subject = Column(String(50), nullable=False)

    # NEW — topic extracted from Word/GPT parser
    topic = Column(String(100), nullable=True)

    difficulty = Column(String(20), nullable=False)

    question_type = Column(String(50), nullable=False)
    # e.g. "domino_multi_image", "chart_mcq", "text_mcq", "two_diagram_mcq"

    question_text = Column(Text, nullable=True)

    # Structured images (list of dicts)
    images = Column(JSON, nullable=True)

    # Structured MCQ options
    # e.g. {"A": "...", "B": "...", "C": "...", "D": "..."}
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

class Exam(Base):
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True, index=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"), nullable=False)

    class_name = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)

    questions = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    quiz = relationship("Quiz", back_populates="exams")

    student_exams = relationship(
        "StudentExam",
        back_populates="exam",
        cascade="all, delete-orphan"
    )



class StudentExamAnswer(Base):
    __tablename__ = "student_exam_answers"

    id = Column(Integer, primary_key=True, index=True)

    student_exam_id = Column(
        Integer,
        ForeignKey("student_exams.id"),
        nullable=False
    )

    question_id = Column(Integer, nullable=False)
    student_answer = Column(String, nullable=False)
    correct_answer = Column(String, nullable=False)
    is_correct = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    student_exam = relationship("StudentExam", back_populates="answers")





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
 
def parse_writing_with_openai(text: str) -> list[dict]:
    WRITING_PARSE_PROMPT = f"""
You are an exam content parser.

Extract writing questions from the text below.

STRICT RULES:
- Return ONLY valid JSON
- No markdown
- No explanations
- No comments
- No trailing commas
- Do not wrap in ``` blocks

If multiple questions exist, return a JSON array.
If one question exists, return a single JSON object.

Each question MUST follow this schema exactly:

{{
  "class_name": string,
  "subject": "Writing",
  "topic": string,
  "difficulty": string,
  "question_text": string
}}

Text to parse:
----------------
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You return strict JSON only."
            },
            {
                "role": "user",
                "content": WRITING_PARSE_PROMPT
            }
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"OpenAI returned invalid JSON.\nRaw response:\n{raw}"
        ) from e

    if isinstance(parsed, dict):
        return [parsed]

    if not isinstance(parsed, list):
        raise ValueError("Unexpected JSON structure from OpenAI")

    return parsed

 
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
                    created_at=datetime.now(timezone.utc),
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
def chunk_into_pages(paragraphs, per_page=30):
    pages = []
    for i in range(0, len(paragraphs), per_page):
        pages.append("\n".join(paragraphs[i:i+per_page]))
    return pages

def group_pages(pages, size=5):
    return [ "\n".join(pages[i:i+size]) for i in range(0, len(pages), size) ]
 
async def parse_with_gpt(block_text: str):
    completion = await client_save_questions.chat.completions.create(
        model="gpt-4o-mini",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "QuestionList",
                "schema": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": QuestionSchema
                        }
                    },
                    "required": ["questions"]
                }
            }
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document parser. Extract ALL quiz questions from the text. "
                    "Return ONLY valid JSON following the provided schema. "
                    "If a question seems cut off (e.g., ends mid-sentence), "
                    "set partial=true in that question object."
                )
            },
            {
                "role": "user",
                "content": block_text
            }
        ]
    )

    raw = completion.choices[0].message.content
    return json.loads(raw)


# ai_engine.py (for example)
def generate_exam_questions(quiz, db):
    print("\n===================== GENERATE EXAM START =====================\n")
    print("QUIZ RECEIVED FROM DATABASE:")
    print("---------------------------------------------------------------")
    print(f"Quiz ID: {quiz.id}")
    print(f"Class: {quiz.class_name}") 
    print(f"Subject: {quiz.subject}")  
    print(f"Difficulty: {quiz.difficulty}")
    print(f"Topics: {quiz.topics}")
    print("---------------------------------------------------------------\n")

    all_questions = []
    q_id = 1

    # Iterate over each quiz topic
    for topic in quiz.topics:
        print("\n===================== PROCESSING TOPIC =====================")
        print(f"Topic Raw Data: {topic}")

        topic_name = topic.get("name")
        ai_count = int(topic.get("ai", 0))
        db_count = int(topic.get("db", 0))

        print(f"Topic Name: {topic_name}")
        print(f"AI Count: {ai_count}")
        print(f"DB Count: {db_count}")
        print("============================================================\n")

        # ===============================================================
        # 1. FETCH QUESTIONS FROM DATABASE
        # ===============================================================
        print(f"[DB FETCH] Fetching {db_count} questions for topic: '{topic_name}'")

        try:
            db_questions = (
                db.query(Question)
                  .filter(Question.topic == topic_name)
                  .order_by(func.random())
                  .limit(db_count)
                  .all()
            )
        except Exception as e:
            print("[DB ERROR] While fetching DB questions:", str(e))
            raise

        print(f"[DB FETCH] Retrieved {len(db_questions)} questions.")
        for q in db_questions:
            print(f"  → DB Question ID: {q.id}, Text: {q.question_text[:60]}...")

        # Append DB questions to final list
        for q in db_questions:
            item = {
                "q_id": q_id,
                "topic": topic_name,
                "question": q.question_text,
                "options": q.options,
                "correct": q.correct_answer,
                "images": q.images or []
            }
            print(f"[APPEND DB] Adding question #{q_id}: {item}")
            all_questions.append(item)
            q_id += 1

        # ===============================================================
        # 2. GENERATE QUESTIONS USING OPENAI
        # ===============================================================
        if ai_count > 0:

            print(f"\n[AI GEN] Generating {ai_count} AI questions for topic '{topic_name}'")
            print("---------------------------------------------------------------")
        
            # -----------------------------
            # Fetch the single franchise location (first row)
            # -----------------------------
            location = db.query(FranchiseLocation).first()
        
            # Safety check (optional)
            if not location:
                raise Exception("No franchise location found in the database.")
        
            country = location.country
            state = location.state
        
            # -----------------------------
            # Build system prompt
            # -----------------------------
            system_prompt = (
                "You are an expert exam generator.\n"
                f"Create {ai_count} MCQs.\n\n"
                "Context for alignment:\n"
                f"- Country: {country}\n"
                f"- State/Region: {state}\n"
                f"- Class: {quiz.class_name}\n"
                f"- Subject: {quiz.subject}\n"
                f"- Topic: {topic_name}\n\n"
                "All questions MUST:\n"
                "- Match the curriculum style and difficulty of the given country/state.\n"
                "- Match the grade level of the class.\n"
                "- Match the specific subject and topic.\n\n"
                "Return STRICT JSON ONLY:\n"
                "[{\"question\":\"...\",\"options\":[\"A\",\"B\",\"C\",\"D\"],\"correct\":\"A\"}]\n"
                "NO explanations. NO commentary. NO extra text. ONLY valid JSON."
            )

            print("[AI GEN] PROMPT:")
            print(system_prompt)
            print("---------------------------------------------------------------")

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt}],
                    temperature=0.4
                )

                raw_output = response.choices[0].message.content.strip()

                print("[AI RAW OUTPUT]")
                print("---------------------------------------------------------------")
                print(raw_output)
                print("---------------------------------------------------------------\n")

                generated = json.loads(raw_output)

            except Exception as e:
                print("[AI ERROR] AI failed for topic:", topic_name)
                print("Error Details:", str(e))
                raise

            print(f"[AI GEN] Parsed {len(generated)} questions from OpenAI")

            for item in generated:
                question_obj = {
                    "q_id": q_id,
                    "topic": topic_name,
                    "question": item.get("question"),
                    "options": item.get("options"),
                    "correct": item.get("correct"),
                    "images": []
                }
                print(f"[APPEND AI] Adding AI question #{q_id}: {question_obj}")
                all_questions.append(question_obj)
                q_id += 1

    print("\n==================== FINAL QUESTION COUNT ====================")
    print(f"TOTAL QUESTIONS GENERATED: {len(all_questions)}")
    print("================================================================\n")

    return all_questions




def parse_question_text_v3(text: str):
    print("\n====== parse_question_text_v3 START ======")
    print("RAW BLOCK:")
    print(text)
    print("-----------------------------------------")

    data = {}

    def extract(label):
        pattern = rf"{label}\s*:\s*(.+)"
        match = re.search(pattern, text, re.IGNORECASE)
        value = match.group(1).strip() if match else None
        print(f"🔎 Extract {label}: {value}")
        return value

    def clean(val):
        if not val:
            return val
        return val.replace('"', "").replace("“", "").replace("”", "").strip()

    # Header fields
    data["class_name"] = clean(extract("CLASS"))
    data["subject"] = clean(extract("SUBJECT"))
    data["topic"] = clean(extract("TOPIC"))
    data["difficulty"] = clean(extract("DIFFICULTY"))
    data["question_type"] = extract("TYPE") or "multi_image_diagram_mcq"

    # Images
    images_block = extract("IMAGES")
    if images_block:
        imgs = [i.strip() for i in images_block.split(",")]
        data["images"] = imgs
    else:
        data["images"] = []

    print("🖼 IMAGES:", data["images"])

    # Question text
    q_text = re.search(
        r"QUESTION_TEXT:\s*(.+?)(OPTIONS:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if not q_text:
        raise Exception("QUESTION_TEXT missing.")

    data["question_text"] = q_text.group(1).strip()
    print("📝 QUESTION_TEXT:", data["question_text"])

    # Options
    options_match = re.search(
        r"OPTIONS:\s*(.+?)CORRECT_ANSWER:",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if not options_match:
        raise Exception("OPTIONS missing.")

    raw_options = options_match.group(1).strip()
    print("📄 OPTIONS BLOCK:\n", raw_options)

    options = {}
    for line in raw_options.split("\n"):
        line = line.strip()
        m = re.match(r"([A-Z])\.?\s*(.+)", line)
        if m:
            key, value = m.group(1), m.group(2)
            options[key] = value

    data["options"] = options
    print("✅ Parsed OPTIONS:", options)

    # Correct answer
    correct = extract("CORRECT_ANSWER")
    data["correct_answer"] = clean(correct)

    print("✔ CORRECT_ANSWER:", data["correct_answer"])
    print("====== parse_question_text_v3 END ======\n")

    return data

def extract_text_from_docx(file_bytes: bytes) -> str:
    # Wrap bytes in a file-like object
    file_obj = BytesIO(file_bytes)

    result = mammoth.extract_raw_text(file_obj)
    return result.value

def parse_exam_with_openai(raw_text: str):
    prompt = f"""
You will receive a Reading Comprehension exam document.

You MUST extract all exam data into this EXACT JSON wrapper:

{{
  "exams": [
    {{
      "class_name": "",
      "subject": "",
      "difficulty": "",
      "topic": "",
      "total_questions": 0,

      "reading_material": {{}},

      "answer_options": {{}},

      "questions": [
        {{
          "question_number": 1,
          "question_text": "",
          "correct_answer": "A"
        }}
      ]
    }}
  ]
}}

STRICT RULES:
- Every question in the Word document begins with a number followed by a period (e.g., "1. Question text").
- You MUST extract the number before the period as `question_number` (INT).
- The remaining text after the number + period must be `question_text`.
  Example:
    Input text: "4. Which extract discusses…"
    Output:
      "question_number": 4,
      "question_text": "Which extract discusses…"

- NEVER keep the number inside question_text.
- NEVER omit question_number.
- The output MUST ALWAYS include question_number for every question.

- Extract answer options EXACTLY as written. Include only letters present (A–G).
- Return ONLY valid JSON. No text, no backticks.
- JSON MUST start with "{{" and end with "}}".

INPUT TEXT:
{raw_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    # DEBUG LOG
    print("======== RAW GPT OUTPUT ========")
    print(content[:5000])
    print("================================")

    # Ensure JSON starts at first '{' and ends at last '}'
    start = content.find("{")
    end = content.rfind("}") + 1
    json_text = content[start:end]

    try:
        parsed = json.loads(json_text)

        # Ensure compatibility: return list of exams
        if "exams" in parsed:
            return parsed["exams"]
        else:
            raise ValueError("JSON missing 'exams' field")

    except Exception:
        raise ValueError("OpenAI returned invalid JSON:\n" + json_text)


def save_exam_to_db(db: Session, exam_data: ExamReadingCreate):

    # Still create exam metadata record (optional)
    exam = Exam_reading(
        class_name=exam_data.class_name,
        subject=exam_data.subject,
        difficulty=exam_data.difficulty,
        topic=exam_data.topic,
        total_questions=exam_data.total_questions
    )
    db.add(exam)
    db.flush()

    # Build full exam bundle
    exam_bundle = {
        "reading_material": exam_data.reading_material,
        "answer_options": exam_data.answer_options,  # <-- SAVE THEM
        "questions": [
            {
                "question_number": q.question_number,
                "question_text": q.question_text,
                "correct_answer": q.correct_answer
            }
            for q in exam_data.questions
        ]
    }

    # Save inside questions_reading with FULL METADATA
    qr = Question_reading(
        exam_id=exam.id,
        class_name=exam_data.class_name,
        subject=exam_data.subject,
        difficulty=exam_data.difficulty,
        topic=exam_data.topic,
        total_questions=exam_data.total_questions,
        exam_bundle=exam_bundle
    )

    db.add(qr)
    db.commit()

    return exam 

def upload_to_gcs(file_bytes: bytes, filename: str) -> str:
    """Upload a file to Google Cloud Storage and return the public URL."""

    print("\n================== GCS UPLOAD START ==================")
    print(f"[GCS] Incoming file: {filename}")
    print(f"[GCS] Bucket: {BUCKET_NAME}")

    if not BUCKET_NAME:
        raise Exception("BUCKET_NAME environment variable not set!")

    bucket = gcs_client.bucket(BUCKET_NAME)

    # Strip folder names so only the actual image name remains
    safe_filename = filename.split("/")[-1]
    print(f"[GCS] Clean filename: {safe_filename}")

    # Create a unique name
    unique_name = f"{uuid.uuid4()}_{safe_filename}"
    print(f"[GCS] Unique stored name: {unique_name}")

    blob = bucket.blob(unique_name)

    try:
        blob.upload_from_string(file_bytes, content_type="image/png")

        # NO make_public() because Uniform Bucket Access is ON
        public_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{unique_name}"

        print(f"[GCS] Upload successful → {public_url}")
        print("================== GCS UPLOAD END ==================\n")

        return public_url

    except Exception as e:
        print("[GCS ERROR] Upload failed:", str(e))
        print("================== GCS UPLOAD FAILED ==================\n")
        raise Exception(f"GCS upload failed: {str(e)}")

from sqlalchemy import func
@app.get("/api/student/exam-report/thinking-skills")
def get_thinking_skills_report(
    student_id: str = Query(..., description="External student id e.g. Gem002"),
    db: Session = Depends(get_db)
):
    # --------------------------------------------------
    # 1️⃣ Resolve student (external → internal)
    # --------------------------------------------------
    student = (
        db.query(Student)
        .filter(Student.student_id == student_id)
        .first()
    )

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # --------------------------------------------------
    # 2️⃣ Get latest completed exam attempt
    # --------------------------------------------------
    attempt = (
        db.query(StudentExam)
        .join(
            StudentExamResultsThinkingSkills,
            StudentExamResultsThinkingSkills.exam_attempt_id == StudentExam.id
        )
        .filter(
            StudentExam.student_id == student.id,
            StudentExam.completed_at.isnot(None)
        )
        .order_by(StudentExam.completed_at.desc())
        .first()
    )

    if not attempt:
        raise HTTPException(status_code=404, detail="No completed exam found")

    # --------------------------------------------------
    # 3️⃣ Load summary (donut / headline stats)
    # --------------------------------------------------
    summary = (
        db.query(StudentExamResultsThinkingSkills)
        .filter(
            StudentExamResultsThinkingSkills.exam_attempt_id == attempt.id
        )
        .first()
    )

    if not summary:
        raise HTTPException(status_code=404, detail="Exam result missing")

    # --------------------------------------------------
    # 4️⃣ Topic-wise breakdown (bar chart)
    # --------------------------------------------------
    topic_rows = (
        db.query(
            StudentExamResponse.topic,
            func.count(StudentExamResponse.id).label("total"),
            func.sum(
                case(
                    (StudentExamResponse.is_correct == True, 1),
                    else_=0
                )
            ).label("correct")
        )
        .filter(
            StudentExamResponse.exam_attempt_id == attempt.id
        )
        .group_by(StudentExamResponse.topic)
        .order_by(StudentExamResponse.topic)
        .all()
    )

    topic_breakdown = []
    for row in topic_rows:
        total = row.total
        correct = row.correct or 0

        topic_breakdown.append({
            "topic": row.topic,
            "total_questions": total,
            "correct_answers": correct,
            "wrong_answers": total - correct,
            "accuracy_percent": round((correct / total) * 100, 2) if total else 0
        })

    # --------------------------------------------------
    # 5️⃣ Final UI-ready response
    # --------------------------------------------------
    return {
        "summary": {
            "total_questions": summary.total_questions,
            "correct_answers": summary.correct_answers,
            "wrong_answers": summary.wrong_answers,
            "accuracy_percent": summary.accuracy_percent
        },
        "topic_breakdown": topic_breakdown
    }


@app.post("/api/student/save-exam-results-thinkingskills")
def save_exam_results_thinkingskills(
    payload: ExamSubmissionRequest,
    db: Session = Depends(get_db)
):
    # 1️⃣ Validate student
    student = (
        db.query(Student)
        .filter(Student.student_id == payload.student_id)
        .first()
    )

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # 2️⃣ Find ACTIVE exam attempt from student_exams
    attempt = (
        db.query(StudentExam)
        .filter(
            StudentExam.student_id == student.id,
            StudentExam.completed_at.is_(None)
        )
        .order_by(StudentExam.started_at.desc())
        .first()
    )

    if not attempt:
        # No active attempt → exam already completed or never started
        return {"status": "ignored"}

    # 3️⃣ Prevent double submission
    existing_submission = (
        db.query(StudentExamAttemptThinkingSkills)
        .filter(
            StudentExamAttemptThinkingSkills.student_exam_id == attempt.id
        )
        .first()
    )

    if existing_submission and existing_submission.is_completed:
        return {"status": "already_submitted"}

    # 4️⃣ Create attempt record if not exists
    if not existing_submission:
        existing_submission = StudentExamAttemptThinkingSkills(
            student_exam_id=attempt.id,
            submitted_at=None,
            is_completed=False
        )
        db.add(existing_submission)
        db.flush()  # get ID without commit

    # 5️⃣ Save answers (idempotent)
    for qid_str, selected_option in payload.answers.items():
        question_id = int(qid_str)

        exists = (
            db.query(StudentExamAnswerThinkingSkills)
            .filter(
                StudentExamAnswerThinkingSkills.attempt_id == existing_submission.id,
                StudentExamAnswerThinkingSkills.question_id == question_id
            )
            .first()
        )

        if exists:
            continue

        db.add(
            StudentExamAnswerThinkingSkills(
                attempt_id=existing_submission.id,
                question_id=question_id,
                selected_option=selected_option
            )
        )

    # 6️⃣ Mark submission complete
    existing_submission.submitted_at = datetime.now(timezone.utc)
    existing_submission.is_completed = True
    db.commit()

    return {"status": "saved"}

@app.post("/add_student_exam_module")
def add_student_exam_module(
    payload: AddStudentExamModuleRequest,
    db: Session = Depends(get_db)
):
    existing_student = (
        db.query(Student)
        .filter(Student.student_id == payload.student_id)
        .first()
    )

    if existing_student:
        raise HTTPException(
            status_code=400,
            detail="Student with this student_id already exists"
        )

    # ⚠️ Plain text password (as requested)
    plain_password = payload.student_id  # or any value you choose

    student = Student(
        id=payload.id,
        student_id=payload.student_id,
        name=payload.name,
        parent_email=payload.parent_email,
        class_name=payload.class_name,
        class_day=payload.class_day,
        password=plain_password,  # ← STORED AS IS
    )

    db.add(student)
    db.commit()
    db.refresh(student)

    return {
        "message": "Student added successfully",
        "student_id": student.student_id,
        "password": plain_password
    }

 
@app.post("/api/exams/submit-reading")
def submit_reading_exam(payload: dict, db: Session = Depends(get_db)):

    session_id = payload.get("session_id")
    answers = payload.get("answers", {})

    session = db.query(StudentExamReading).filter_by(id=session_id).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Mark as completed
    session.finished = True
    session.completed_at = datetime.utcnow()

    db.commit()

    return {"status": "submitted", "message": "Reading exam submitted successfully"}

@app.post("/api/exams/start-reading")
def start_reading_exam(student_id: str, db: Session = Depends(get_db)):
    """
    Start or resume a reading exam session for a student.
    Debug statements included.
    """

    print("\n\n==================== /start-reading REQUEST ====================")
    print("Incoming student_id:", student_id)

    # Load the latest generated reading exam
    exam = (
        db.query(GeneratedExamReading)
        .order_by(GeneratedExamReading.id.desc())
        .first()
    )

    print("Loaded exam ID:", exam.id if exam else None)

    if not exam:
        raise HTTPException(status_code=404, detail="No reading exam found")

    # Debug exam_json content
    print("Exam JSON keys:", list(exam.exam_json.keys()))
    print("duration_minutes FROM exam_json:", exam.exam_json.get("duration_minutes"))
    print("Full exam_json:", exam.exam_json)

    # Check if the student already has a session
    existing = (
        db.query(StudentExamReading)
        .filter_by(student_id=student_id, exam_id=exam.id)
        .first()
    )

    print("Existing session found?", bool(existing))

    # ----------------------------------------------------------
    # 🔒 Prevent second attempt if already finished
    # ----------------------------------------------------------
    if existing:

        print("Existing session ID:", existing.id)
        print("existing.started_at:", existing.started_at)
        print("existing.finished:", existing.finished)
        print("server_now:", datetime.utcnow())

        if existing.finished:
            print("❌ STUDENT ALREADY FINISHED. BLOCKING.")
            raise HTTPException(
                status_code=403,
                detail="You have already completed this exam. Only one attempt is allowed."
            )

        response = {
            "session_id": existing.id,
            "exam_json": exam.exam_json,
            "duration_minutes": exam.exam_json.get("duration_minutes", 40),
            "start_time": existing.started_at,
            "server_now": datetime.utcnow(),
        }

        print("Returning EXISTING session response:", response)
        print("==============================================================\n\n")

        return response

    # ----------------------------------------------------------
    # ✨ First-time attempt — Create a new session
    # ----------------------------------------------------------

    now = datetime.utcnow()
    print("Creating new session at:", now)

    new_session = StudentExamReading(
        student_id=student_id,
        exam_id=exam.id,
        started_at=now
    )

    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    response = {
        "session_id": new_session.id,
        "exam_json": exam.exam_json,
        "duration_minutes": exam.exam_json.get("duration_minutes", 40),
        "start_time": new_session.started_at,
        "server_now": datetime.utcnow(),
    }

    print("Returning NEW session response:", response)
    print("==============================================================\n\n")

    return response


@app.get("/api/foundational/classes")
def get_foundational_classes(db: Session = Depends(get_db)):
    rows = (
        db.query(QuizSetupFoundational.class_name)
        .distinct()
        .all()
    )

    return [{"class_name": r.class_name} for r in rows]
 
@app.post("/api/student/start-writing-exam")
def start_writing_exam(student_id: str, db: Session = Depends(get_db)):

    # 1️⃣ Check for existing active session
    existing_session = (
        db.query(StudentExamWriting)
        .filter(
            StudentExamWriting.student_id == student_id,
            StudentExamWriting.completed_at == None
        )
        .order_by(StudentExamWriting.started_at.desc())
        .first()
    )

    if existing_session:
        return {"status": "already_started"}

    # 2️⃣ Get current exam
    exam = (
        db.query(GeneratedExamWriting)
        .filter(GeneratedExamWriting.is_current == True)
        .order_by(GeneratedExamWriting.created_at.desc())
        .first()
    )

    if not exam:
        raise HTTPException(404, "No active writing exam")

    # 3️⃣ Create NEW session only if none exists
    new_session = StudentExamWriting(
        student_id=student_id,
        exam_id=exam.id,
        started_at=datetime.now(timezone.utc),
        duration_minutes=exam.duration_minutes
    )

    db.add(new_session)
    db.commit()

    return {"status": "started"}


@app.get("/api/get-quizzes-writing")
def get_quizzes_writing(db: Session = Depends(get_db)):
    rows = (
        db.query(
            QuizSetupWriting.class_name,
            QuizSetupWriting.difficulty
        )
        .distinct()
        .order_by(
            QuizSetupWriting.class_name,
            QuizSetupWriting.difficulty
        )
        .all()
    )

    return [
        {
            "class_name": row.class_name,
            "difficulty": row.difficulty
        }
        for row in rows
    ]

@app.post("/api/exams/writing/submit")
def submit_writing_exam(
    payload: WritingSubmitSchema,
    student_id: str,
    db: Session = Depends(get_db)
):
    exam_state = (
        db.query(StudentExamWriting)
        .filter(
            StudentExamWriting.student_id == student_id,
            StudentExamWriting.completed_at.is_(None)
        )
        .order_by(StudentExamWriting.started_at.desc())
        .first()
    )

    if not exam_state:
        raise HTTPException(
            status_code=404,
            detail="No active writing exam found"
        )

    exam_state.answer_text = payload.answer_text
    exam_state.completed_at = func.now()

    db.commit()

    return {
        "message": "Writing exam submitted successfully",
        "exam_id": exam_state.exam_id
    }



@app.get("/api/exams/writing/current")
def get_current_writing_exam(student_id: str, db: Session = Depends(get_db)):
    # ------------------------------------------------------------
    # 1️⃣ Get the student's ACTIVE writing session (NOT the exam)
    # ------------------------------------------------------------
    session = (
        db.query(StudentExamWriting)
        .filter(
            StudentExamWriting.student_id == student_id,
            StudentExamWriting.completed_at == None
        )
        .order_by(StudentExamWriting.started_at.desc())
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="No active writing exam session"
        )

    # ------------------------------------------------------------
    # 2️⃣ Fetch the exam linked to this session
    # ------------------------------------------------------------
    exam = (
        db.query(GeneratedExamWriting)
        .filter(GeneratedExamWriting.id == session.exam_id)
        .first()
    )

    if not exam:
        raise HTTPException(
            status_code=404,
            detail="Writing exam not found"
        )

    # ------------------------------------------------------------
    # 3️⃣ Calculate remaining time FROM session.started_at
    # ------------------------------------------------------------
    elapsed_seconds = (
        datetime.now(timezone.utc) - session.started_at
    ).total_seconds()

    total_seconds = session.duration_minutes * 60
    remaining_seconds = max(0, int(total_seconds - elapsed_seconds))

    return {
        "exam": {
            "exam_id": exam.id,
            "difficulty": exam.difficulty,
            "question_text": exam.question_text,
            "duration_minutes": session.duration_minutes
        },
        "remaining_seconds": remaining_seconds
    }

@app.post("/api/exams/generate-writing")
def generate_exam_writing(
    payload: WritingGenerateSchema,
    db: Session = Depends(get_db)
):
    # Normalize incoming values to lowercase
    class_name = payload.class_name.strip().lower()
    difficulty = payload.difficulty.strip().lower()

    # 1️⃣ Fetch ONE writing question with full case-insensitive matching
    question = (
        db.query(WritingQuestionBank)
        .filter(
            func.lower(WritingQuestionBank.class_name) == class_name,
            func.lower(WritingQuestionBank.difficulty) == difficulty
        )
        .order_by(func.random())
        .first()
    )

    if not question:
        raise HTTPException(
            status_code=404,
            detail=f"No writing question found for class '{class_name}' with difficulty '{difficulty}'."
        )

    # 2️⃣ Save generated exam (store normalized & pretty versions)
    exam = GeneratedExamWriting(
        class_name=class_name.capitalize(),       # year5 → Year5 (optional)
        subject="writing",
        difficulty=difficulty.capitalize(),       # easy → Easy
        question_text=question.question_text,
        duration_minutes=40
    )

    db.add(exam)
    db.commit()
    db.refresh(exam)

    # 3️⃣ Return response
    return {
        "exam_id": exam.id,
        "class_name": exam.class_name,
        "difficulty": exam.difficulty,
        "question_text": exam.question_text,
        "duration_minutes": exam.duration_minutes
    }

@app.post("/upload-word-writing")
async def upload_word_writing(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    print("\n=== UPLOAD WORD WRITING ===")

    if file.content_type != (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        raise HTTPException(400, "File must be .docx")

    raw = await file.read()
    extracted_text = extract_text_from_docx(raw)

    try:
        questions = parse_writing_with_openai(extracted_text)
    except Exception as e:
        print("❌ OpenAI parse error:", e)
        raise HTTPException(500, "Failed to parse writing questions")

    if not questions:
        raise HTTPException(400, "No writing questions detected")

    saved_ids = []

    for q in questions:
        obj = WritingQuestionBank(
            class_name=q["class_name"],
            subject=q["subject"],
            topic=q["topic"],
            difficulty=q["difficulty"],
            question_text=q["question_text"],
            source_file=file.filename
        )
        db.add(obj)
        db.flush()
        saved_ids.append(obj.id)

    db.commit()

    return {
        "message": "Writing questions uploaded successfully",
        "count": len(saved_ids),
        "question_ids": saved_ids
    }
 
@app.post("/api/quizzes-writing")
def save_writing_quiz(payload: WritingQuizSchema, db: Session = Depends(get_db)):
    quiz = QuizSetupWriting(
        class_name=payload.class_name,
        subject=payload.subject,
        topic=payload.topic,
        difficulty=payload.difficulty
    )

    db.add(quiz)
    db.commit()
    db.refresh(quiz)

    return {
        "message": "Writing quiz setup saved",
        "quiz_id": quiz.id
    }


@app.post("/api/student/start-exam/foundational-skills")
def start_or_resume_foundational_exam(
    student_id: str,
    db: Session = Depends(get_db)
):
    # ------------------------------------------------------------
    # 1️⃣ Get latest attempt
    # ------------------------------------------------------------
    attempt = (
        db.query(StudentsExamFoundational)
        .filter(StudentsExamFoundational.student_id == student_id)
        .order_by(StudentsExamFoundational.started_at.desc())
        .first()
    )

    # ------------------------------------------------------------
    # 2️⃣ Completed → load report
    # ------------------------------------------------------------
    if attempt and attempt.completed_at:
        return { "completed": True }

    # ------------------------------------------------------------
    # 3️⃣ Load current exam
    # ------------------------------------------------------------
    exam = (
        db.query(GeneratedExamFoundational)
        .filter(GeneratedExamFoundational.is_current == True)
        .order_by(desc(GeneratedExamFoundational.created_at))
        .first()
    )

    if not exam:
        raise HTTPException(status_code=404, detail="No active exam")

    sections = exam.exam_json.get("sections", [])

    if not sections:
        raise HTTPException(
            status_code=500,
            detail="Exam has no sections configured"
        )

    # ------------------------------------------------------------
    # 4️⃣ Create attempt if needed
    # ------------------------------------------------------------
    if not attempt:
        attempt = StudentsExamFoundational(
            student_id=student_id,
            exam_id=exam.id,
            started_at=datetime.now(timezone.utc),
            current_section_index=0
        )
        db.add(attempt)
        db.commit()
        db.refresh(attempt)

    # ------------------------------------------------------------
    # 5️⃣ Compute remaining time
    # ------------------------------------------------------------
    elapsed_seconds = int(
        (datetime.now(timezone.utc) - attempt.started_at).total_seconds()
    )

    remaining_time = max(
        exam.duration_minutes * 60 - elapsed_seconds,
        0
    )

    # ------------------------------------------------------------
    # 6️⃣ Return CURRENT section only
    # ------------------------------------------------------------
    current_section = sections[attempt.current_section_index]

    return {
        "completed": False,
        "current_section_index": attempt.current_section_index,
        "section": {
            "name": current_section.get("name"),
            "questions": current_section.get("questions", [])
        },
        "remaining_time": remaining_time
    }


@app.post("/api/exams/foundational/next-section")
def advance_foundational_section(
    student_id: str = Query(...),
    db: Session = Depends(get_db)
):
    # ------------------------------------------------------------
    # 1️⃣ Get active attempt (authoritative)
    # ------------------------------------------------------------
    attempt = (
        db.query(StudentsExamFoundational)
        .filter(
            StudentsExamFoundational.student_id == student_id,
            StudentsExamFoundational.completed_at.is_(None)
        )
        .order_by(StudentsExamFoundational.started_at.desc())
        .first()
    )

    if not attempt:
        raise HTTPException(
            status_code=404,
            detail="No active foundational exam attempt"
        )

    # ------------------------------------------------------------
    # 2️⃣ Load exam linked to this attempt
    # ------------------------------------------------------------
    exam = (
        db.query(GeneratedExamFoundational)
        .filter(GeneratedExamFoundational.id == attempt.exam_id)
        .first()
    )

    if not exam:
        raise HTTPException(
            status_code=404,
            detail="Exam not found for active attempt"
        )

    sections = exam.exam_json.get("sections", [])
    total_sections = len(sections)

    if total_sections == 0:
        raise HTTPException(
            status_code=500,
            detail="Exam has no sections configured"
        )

    # ------------------------------------------------------------
    # 3️⃣ If last section already reached → signal completion
    # ------------------------------------------------------------
    if attempt.current_section_index >= total_sections - 1:
        return {
            "completed": True,
            "message": "Last section already reached"
        }

    # ------------------------------------------------------------
    # 4️⃣ Advance section pointer
    # ------------------------------------------------------------
    attempt.current_section_index += 1
    db.commit()
    db.refresh(attempt)

    current_section = sections[attempt.current_section_index]

    return {
        "completed": False,
        "current_section_index": attempt.current_section_index,
        "section": {
            "name": current_section.get("name"),
            "questions": current_section.get("questions", [])
        }
    }


@app.post("/api/student/finish-exam/foundational-skills")
def finish_foundational_exam(
    payload: FinishExamRequestFoundational,
    db: Session = Depends(get_db)
):
    # ------------------------------------------------------------
    # 1️⃣ Get active attempt
    # ------------------------------------------------------------
    attempt = (
        db.query(StudentsExamFoundational)
        .filter(
            StudentsExamFoundational.student_id == payload.student_id,
            StudentsExamFoundational.completed_at.is_(None)
        )
        .order_by(StudentsExamFoundational.started_at.desc())
        .first()
    )

    # ------------------------------------------------------------
    # 2️⃣ Idempotency guard
    # ------------------------------------------------------------
    if not attempt:
        return {
            "success": True,
            "message": "Exam already completed or no active attempt"
        }

    # ------------------------------------------------------------
    # 3️⃣ Persist answers and complete exam
    # ------------------------------------------------------------
    attempt.answers_json = payload.answers
    attempt.completed_at = datetime.now(timezone.utc)
    attempt.completion_reason = payload.reason

    db.commit()
    db.refresh(attempt)

    return {
        "success": True,
        "message": "Exam completed successfully"
    }


# ------------------------------------------------------------
# Exam Report Endpoint
# ------------------------------------------------------------
@app.get("/api/student/exam-report/foundational-skills")
def get_foundational_exam_report(
    student_id: str = Query(...),
    db: Session = Depends(get_db)
):
    # ------------------------------------------------------------
    # 1️⃣ Fetch completed attempt
    # ------------------------------------------------------------
    attempt = (
        db.query(StudentsExamFoundational)
        .filter(
            StudentsExamFoundational.student_id == student_id,
            StudentsExamFoundational.completed_at.isnot(None)
        )
        .order_by(StudentsExamFoundational.completed_at.desc())
        .first()
    )

    if not attempt:
        raise HTTPException(
            status_code=404,
            detail="No completed foundational exam found"
        )

    # ------------------------------------------------------------
    # 2️⃣ Load exam definition
    # ------------------------------------------------------------
    exam = (
        db.query(GeneratedExamFoundational)
        .filter(GeneratedExamFoundational.id == attempt.exam_id)
        .first()
    )

    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    answers = attempt.answers_json or {}
    sections = exam.exam_json.get("sections", [])

    # ------------------------------------------------------------
    # 3️⃣ Compute results
    # ------------------------------------------------------------
    total_questions = 0
    correct_answers = 0
    topic_stats: Dict[str, Dict[str, int]] = {}

    for section in sections:
        topic = section.get("name", "Unknown")
        questions = section.get("questions", [])

        if topic not in topic_stats:
            topic_stats[topic] = {
                "total": 0,
                "correct": 0
            }

        for q in questions:
            qid = str(q.get("q_id"))
            correct_option = q.get("correct_answer")

            total_questions += 1
            topic_stats[topic]["total"] += 1

            if answers.get(qid) == correct_option:
                correct_answers += 1
                topic_stats[topic]["correct"] += 1

    wrong_answers = total_questions - correct_answers
    accuracy_percent = round(
        (correct_answers / total_questions) * 100
    ) if total_questions > 0 else 0

    # ------------------------------------------------------------
    # 4️⃣ Build topic breakdown
    # ------------------------------------------------------------
    topic_breakdown: List[Dict] = []

    for topic, stats in topic_stats.items():
        percent = round(
            (stats["correct"] / stats["total"]) * 100
        ) if stats["total"] > 0 else 0

        topic_breakdown.append({
            "topic": topic,
            "accuracy_percent": percent
        })

    # ------------------------------------------------------------
    # 5️⃣ Return report payload
    # ------------------------------------------------------------
    return {
        "summary": {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "wrong_answers": wrong_answers,
            "accuracy_percent": accuracy_percent
        },
        "topic_breakdown": topic_breakdown
    }




@app.post("/api/exams/generate-foundational")
def generate_exam_foundational(
    payload: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Generate a Foundational exam using the latest config for the given class.
    Backend now requires class_name from frontend.
    """

    print("\n==============================")
    print("📥 Generate Foundational Exam")
    print("==============================")

    # ------------------------------------------------------------
    # 0) Validate payload
    # ------------------------------------------------------------
    class_name = payload.get("class_name")
    if not class_name:
        raise HTTPException(400, "class_name is required")

    subject = "Foundational"

    # ------------------------------------------------------------
    # 1) Load latest foundational config FOR THIS CLASS ONLY
    # ------------------------------------------------------------
    cfg = (
        db.query(QuizSetupFoundational)
        .filter(func.lower(QuizSetupFoundational.class_name) == class_name.lower())
        .order_by(QuizSetupFoundational.id.desc())
        .first()
    )

    if not cfg:
        raise HTTPException(
            status_code=404,
            detail=f"No foundational config found for class '{class_name}'"
        )

    print(f"➡ Using config ID={cfg.id} for class={class_name}")

    # ------------------------------------------------------------
    # 2) Build sections list from config
    # ------------------------------------------------------------
    sections = [
        {
            "name": cfg.section1_name,
            "difficulty": cfg.section1_name,
            "total": cfg.section1_total,
            "time": cfg.section1_time,
            "intro": cfg.section1_intro,
        },
        {
            "name": cfg.section2_name,
            "difficulty": cfg.section2_name,
            "total": cfg.section2_total,
            "time": cfg.section2_time,
            "intro": cfg.section2_intro,
        }
    ]

    if cfg.section3_name:
        sections.append({
            "name": cfg.section3_name,
            "difficulty": cfg.section3_name,
            "total": cfg.section3_total,
            "time": cfg.section3_time,
            "intro": cfg.section3_intro,
        })

    final_questions = []
    warnings = []

    # ------------------------------------------------------------
    # 3) Fetch questions for each section
    # ------------------------------------------------------------
    for section in sections:
        difficulty = section["difficulty"]
        required = int(section["total"])

        print(f"\n➡ Section: {difficulty} — Need {required}")

        pool = (
            db.query(Question)
            .filter(
                func.lower(Question.class_name) == class_name.lower(),   # <-- Important fix
                func.lower(Question.subject) == subject.lower(),
                func.lower(Question.difficulty) == difficulty.lower(),
            )
            .all()
        )

        print(f"📦 Found {len(pool)} questions for {class_name} / {difficulty}")

        if not pool:
            warnings.append(f"No questions found for {difficulty}")
            continue

        random.shuffle(pool)
        chosen = pool[:required]

        if len(chosen) < required:
            warnings.append(
                f"Only {len(chosen)} questions available for {difficulty}"
            )

        for q in chosen:
            final_questions.append({
                "section": difficulty,
                "question_number": None,
                "question_text": q.question_text,
                "options": q.options,
                "correct_answer": q.correct_answer,
                "question_type": q.question_type,
                "images": q.images,
                "topic": q.topic,
            })

    # ------------------------------------------------------------
    # 4) Number questions
    # ------------------------------------------------------------
    if not final_questions:
        raise HTTPException(400, "No questions generated")

    for i, q in enumerate(final_questions, start=1):
        q["question_number"] = i

    # ------------------------------------------------------------
    # 5) Build exam JSON
    # ------------------------------------------------------------
    duration_minutes = sum(
        s["time"] for s in sections if s.get("time")
    )

    exam_json = {
        "class_name": class_name,
        "subject": subject,
        "total_questions": len(final_questions),
        "sections": sections,
        "questions": final_questions,
        "duration_minutes": duration_minutes,
    }

    # ------------------------------------------------------------
    # 6) Save exam
    # ------------------------------------------------------------
    saved = GeneratedExamFoundational(
        class_name=class_name,
        subject=subject,
        total_questions=len(final_questions),
        exam_json=exam_json,
        is_current=True
    )

    db.add(saved)
    db.commit()
    db.refresh(saved)

    print("✅ Saved Foundational Exam ID:", saved.id)

    return {
        "generated_exam_id": saved.id,
        "total_questions": len(final_questions),
        "exam_json": exam_json,
        "warnings": warnings
    }


@app.post("/api/quizzes-foundational")
def save_quiz_setup(payload: QuizSetupFoundationalSchema, db: Session = Depends(get_db)):

    if len(payload.sections) < 2:
        raise HTTPException(status_code=400, detail="At least 2 sections required")

    section1 = payload.sections[0]
    section2 = payload.sections[1]
    section3 = payload.sections[2] if len(payload.sections) == 3 else None

    total_questions = (
        section1.total +
        section2.total +
        (section3.total if section3 else 0)
    )

    if section3 is None and total_questions > 40:
        raise HTTPException(
            status_code=400,
            detail="Total questions cannot exceed 40 with 2 sections",
        )

    if section3 and total_questions > 50:
        raise HTTPException(
            status_code=400,
            detail="Total questions cannot exceed 50 with 3 sections",
        )

    quiz = QuizSetupFoundational(
        class_name=payload.class_name,
        subject=payload.subject,

        section1_name=section1.name,
        section1_ai=section1.ai,
        section1_db=section1.db,
        section1_total=section1.total,
        section1_time=section1.time,
        section1_intro=section1.intro,

        section2_name=section2.name,
        section2_ai=section2.ai,
        section2_db=section2.db,
        section2_total=section2.total,
        section2_time=section2.time,
        section2_intro=section2.intro,

        section3_name=section3.name if section3 else None,
        section3_ai=section3.ai if section3 else None,
        section3_db=section3.db if section3 else None,
        section3_total=section3.total if section3 else None,
        section3_time=section3.time if section3 else None,
        section3_intro=section3.intro if section3 else None,
    )

    db.add(quiz)
    db.commit()
    db.refresh(quiz)

    return {
        "message": "Quiz setup saved successfully",
        "quiz_id": quiz.id,
    }

@app.post("/api/exams/generate-reading")
def generate_exam_reading(payload: ReadingExamRequest, db: Session = Depends(get_db)):
    """
    Generate a reading exam using class_name + difficulty.
    Includes EXTENSIVE debug statements to trace failures.
    """

    print("\n\n===================================================")
    print("📥 [DEBUG] Incoming Generate Reading Exam Request")
    print("===================================================")
    print("➡ Raw Payload:", payload)
    print("➡ Payload as dict:", payload.dict())
    print("---------------------------------------------------")

    class_name = payload.class_name.strip()
    difficulty = payload.difficulty.strip()

    print(f"➡ Filters Received From Frontend: class_name='{class_name}', difficulty='{difficulty}'")
    print("---------------------------------------------------")

    # ------------------------------------------------------------
    # 1) LOAD CONFIG
    # ------------------------------------------------------------
    print("🔍 STEP 1: Searching for matching ReadingExamConfig...")
    cfg = (
        db.query(ReadingExamConfig)
        .filter(
            func.lower(ReadingExamConfig.class_name) == class_name.lower(),
            func.lower(ReadingExamConfig.difficulty) == difficulty.lower()
        )
        .first()
    )

    if not cfg:
        print("❌ [ERROR] NO CONFIG FOUND for these filters!")
        print(f"   class_name='{class_name}', difficulty='{difficulty}'")
        raise HTTPException(
            status_code=404,
            detail=f"No reading exam config found for class={class_name}, difficulty={difficulty}"
        )

    print("✅ Found Config!")
    print(f"   → Config ID: {cfg.id}")
    print(f"   → Config Subject: {cfg.subject}")
    print(f"   → Config Topics: {cfg.topics}")
    print("---------------------------------------------------")

    subject = cfg.subject
    topics = cfg.topics

    final_questions = []
    used_passages = {}
    merged_answer_options = {}
    warnings = []

    print("\n===================================================")
    print("🔁 STEP 2: TOPIC PROCESSING")
    print("===================================================")

    # ------------------------------------------------------------
    # 2) PROCESS EACH TOPIC
    # ------------------------------------------------------------
    for topic_spec in topics:
        topic_name = topic_spec.get("name", "").strip()
        required = int(topic_spec.get("num_questions", 0))

        print("\n---------------------------------------------------")
        print(f"📌 Working On Topic: '{topic_name}' (Need {required} questions)")
        print("---------------------------------------------------")

        # Debug: what EXACTLY are we filtering?
        print("🔍 Query Filters:")
        print(f"   class_name = '{class_name.lower()}'")
        print(f"   subject = '{subject.lower()}'")
        print(f"   difficulty = '{difficulty.lower()}'")
        print(f"   topic = '{topic_name.lower()}'")

        bundles = (
            db.query(Question_reading)
            .filter(
                func.lower(Question_reading.class_name) == class_name.lower(),
                func.lower(Question_reading.subject) == subject.lower(),
                func.lower(Question_reading.difficulty) == difficulty.lower(),
                func.lower(Question_reading.topic) == topic_name.lower(),
            )
            .all()
        )

        print(f"📦 Query Returned {len(bundles)} Bundles")

        # Special debug if nothing found
        if len(bundles) == 0:
            print("❌ [ERROR] NO QUESTION BUNDLES FOUND FOR THIS TOPIC!")
            print("   Possible causes:")
            print("   - Wrong subject spelling in DB?")
            print("   - Wrong topic name mismatch?")
            print("   - Difficulty mismatch?")
            print("   - Class mismatch?")
            warnings.append(f"No bundles found for topic '{topic_name}'")
            continue

        random.shuffle(bundles)
        collected = []

        # ------------------------------------------------------------
        # 3) Extract from bundles
        # ------------------------------------------------------------
        for b in bundles:
            print(f"➡ Inspecting Bundle ID={b.id}")

            bundle_json = b.exam_bundle or {}
            
            bundle_questions = bundle_json.get("questions", [])
            answer_options = bundle_json.get("answer_options", {})
            passages = bundle_json.get("reading_material", {})

            print(f"   • Questions inside bundle: {len(bundle_questions)}")
            print(f"   • Passages in bundle: {len(passages)}")
            print(f"   • Answer options found: {len(answer_options)}")

            # Save passages
            for label, text in passages.items():
                if label not in used_passages:
                    print(f"   ➕ Adding passage '{label}'")
                    used_passages[label] = text

            # Merge answer options
            for letter, text in answer_options.items():
                merged_answer_options[letter] = text

            # Randomize bundle questions
            random.shuffle(bundle_questions)

            # Collect required number
            for q in bundle_questions:
                collected.append({"bundle_id": b.id, "q": q})
                if len(collected) >= required:
                    break

            if len(collected) >= required:
                break

        print(f"📥 After collection: {len(collected)} questions gathered")

        if len(collected) < required:
            print(f"⚠ WARNING: Only {len(collected)} questions available for topic '{topic_name}'. Required: {required}")

        # Push into final list
        for item in collected[:required]:
            final_questions.append({
                "topic": topic_name,
                "question_number": None,
                "question_text": item["q"].get("question_text"),
                "correct_answer": item["q"].get("correct_answer"),
            })

    # ------------------------------------------------------------
    # 4) FINALIZE QUESTIONS
    # ------------------------------------------------------------
    print("\n===================================================")
    print("🔢 STEP 3: FINALIZING QUESTIONS")
    print("===================================================")

    if len(final_questions) == 0:
        print("❌ [FATAL ERROR] FINAL QUESTION LIST IS EMPTY — NOTHING GENERATED!")
        print("   → Most likely cause: filter mismatch OR misspelled subject/topic/difficulty.")
        raise HTTPException(status_code=400, detail="No questions generated.")

    for i, q in enumerate(final_questions, start=1):
        q["question_number"] = i

    print(f"✅ TOTAL QUESTIONS BUILT: {len(final_questions)}")
    print("---------------------------------------------------")

    # ------------------------------------------------------------
    # 5) BUILD FINAL EXAM JSON
    # ------------------------------------------------------------
    exam_json = {
        "class_name": class_name,
        "subject": subject,
        "difficulty": difficulty,
        "total_questions": len(final_questions),
        "reading_material": used_passages,
        "questions": final_questions,
        "answer_options": merged_answer_options,
        "duration_minutes": 40
    }

    # ------------------------------------------------------------
    # 6) SAVE EXAM
    # ------------------------------------------------------------
    print("\n===================================================")
    print("💾 STEP 4: SAVING GENERATED EXAM")
    print("===================================================")

    saved = GeneratedExamReading(
        config_id=cfg.id,
        class_name=class_name,
        subject=subject,
        difficulty=difficulty,
        total_questions=len(final_questions),
        exam_json=exam_json,
    )

    db.add(saved)
    db.commit()
    db.refresh(saved)

    print("🎉 EXAM SAVED SUCCESSFULLY!")
    print("➡ Generated Exam ID:", saved.id)
    print("===================================================\n\n")

    return {
        "generated_exam_id": saved.id,
        "total_questions": len(final_questions),
        "exam_json": exam_json,
        "warnings": warnings
    }




@app.get("/api/quizzes-reading")
def get_reading_quiz_dropdown(db: Session = Depends(get_db)):
    """
    Returns unique class_name + difficulty combinations 
    from reading_exam_config for populating the dropdown.
    """

    rows = (
        db.query(
            ReadingExamConfig.class_name,
            ReadingExamConfig.difficulty
        )
        .distinct()
        .all()
    )

    result = []
    seen = set()

    for row in rows:
        key = (row.class_name, row.difficulty)
        if key not in seen:
            seen.add(key)
            result.append({
                "class_name": row.class_name,
                "difficulty": row.difficulty,
                "label": f"{row.class_name} | {row.difficulty}"
            })

    return result


@app.post("/upload-word-reading")
async def upload_word(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    print("\n================= UPLOAD WORD READING REQUEST RECEIVED =================")

    # Validate file type
    print(f"[DEBUG] Uploaded file name: {file.filename}")
    print(f"[DEBUG] Uploaded file content-type: {file.content_type}")

    if file.content_type not in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]:
        print("[ERROR] Invalid file type. Must be .docx")
        raise HTTPException(status_code=400, detail="File must be a .docx Word document")

    # Read file bytes
    raw = await file.read()
    print(f"[DEBUG] Raw file size in bytes: {len(raw)}")
    print("[DEBUG] File read into memory successfully.")

    # ---------------------------
    # 1. Extract text
    # ---------------------------
    print("\n[STEP 1] Extracting text from DOCX...")
    extracted = extract_text_from_docx(raw)
    print(f"[DEBUG] Extracted text length: {len(extracted)} characters")
    print("[DEBUG] Extracted text preview:")
    print(extracted[:300])
    print("[DEBUG] Text extraction completed.")

    # ---------------------------
    # 2. Parse with OpenAI
    # ---------------------------
    print("\n[STEP 2] Parsing extracted text with OpenAI...")

    try:
        parsed = parse_exam_with_openai(extracted)
        print("[DEBUG] OpenAI parsing successful.")
    except Exception as e:
        print("[ERROR] Failed to parse JSON from OpenAI:")
        print(e)
        raise HTTPException(status_code=500, detail="OpenAI parsing error")

    print(f"[DEBUG] Parsed type: {type(parsed)}")
    print("[DEBUG] Parsed preview:")
    print(str(parsed)[:1200] + "...")

    # Support multiple exams in one file
    exams_to_store = parsed if isinstance(parsed, list) else [parsed]
    print(f"[DEBUG] Exams detected: {len(exams_to_store)}")

    # ---------------------------
    # 3. Save exams to DB
    # ---------------------------
    print("\n[STEP 3] Saving exams to database...")

    saved_ids = []

    for idx, exam_json in enumerate(exams_to_store, start=1):
        print(f"\n[DEBUG] Saving Exam #{idx}")
        print(f"[DEBUG] Exam keys found: {list(exam_json.keys())}")

        exam_obj = ExamReadingCreate(**exam_json)
        print("[DEBUG] ExamReadingCreate validation successful.")

        # Save using injected DB session
        saved_exam = save_exam_to_db(db, exam_obj)

        print(f"[DEBUG] Saved Exam #{idx} with ID: {saved_exam.id}")
        saved_ids.append(saved_exam.id)

    print("\n[STEP COMPLETE] All exams saved successfully.")
    print(f"[DEBUG] Returning exam IDs: {saved_ids}")
    print("================= REQUEST COMPLETE =================\n")

    return {
        "message": "Exam(s) uploaded and stored successfully.",
        "exam_ids": saved_ids
    }


     
@app.post("/api/admin/create-reading-config")
def create_reading_config(payload: ReadingExamConfigCreate, db: Session = Depends(get_db)):

    # Validate topics
    if len(payload.topics) == 0:
        raise HTTPException(status_code=400, detail="At least one topic must be provided.")

    num_topics = len(payload.topics)

    # Build JSON structure WITHOUT topic_id
    topics_json = [
        {
            "name": t.name,
            "num_questions": t.num_questions
        }
        for t in payload.topics
    ]

    # Create DB record
    new_config = ReadingExamConfig(
        class_name=payload.class_name,
        subject=payload.subject,
        difficulty=payload.difficulty,
        num_topics=num_topics,
        topics=topics_json
    )

    db.add(new_config)
    db.commit()
    db.refresh(new_config)

    return {
        "status": "success",
        "config_id": new_config.id,
        "message": "Reading exam configuration saved successfully.",
        "created_at": new_config.created_at
    }


@app.post("/api/student/start-exam")
def start_exam(
    req: StartExamRequest = Body(...),
    db: Session = Depends(get_db)
):
    print("\n🚀 START-EXAM REQUEST")
    print("➡ payload:", req.dict())

    # --------------------------------------------------
    # 1️⃣ Resolve student (external → internal)
    # --------------------------------------------------
    student = (
        db.query(Student)
        .filter(Student.student_id == req.student_id)
        .first()
    )

    if not student:
        print("❌ Student not found")
        raise HTTPException(status_code=404, detail="Student not found")

    print(f"✅ Student resolved: id={student.id}")

    # --------------------------------------------------
    # 2️⃣ Get latest attempt (if any)
    # --------------------------------------------------
    attempt = (
        db.query(StudentExam)
        .filter(StudentExam.student_id == student.id)
        .order_by(StudentExam.started_at.desc())
        .first()
    )

    if attempt:
        print(
            f"📘 Latest attempt found: "
            f"id={attempt.id}, completed_at={attempt.completed_at}"
        )
    else:
        print("📘 No previous attempts found")

    # --------------------------------------------------
    # 🟢 CASE A — COMPLETED attempt exists → SHOW REPORT
    # --------------------------------------------------
    if attempt and attempt.completed_at is not None:
        print("✅ Exam already completed → returning completed=true")

        return {
            "completed": True
        }

    # --------------------------------------------------
    # 3️⃣ Load quiz (class-based)
    # --------------------------------------------------
    quiz = (
        db.query(Quiz)
        .filter(func.lower(Quiz.class_name) == func.lower(student.class_name))
        .order_by(Quiz.id.desc())
        .first()
    )

    if not quiz:
        print("❌ Quiz not found for class")
        raise HTTPException(status_code=404, detail="Quiz not found")

    # --------------------------------------------------
    # 4️⃣ Load exam
    # --------------------------------------------------
    exam = (
        db.query(Exam)
        .filter(Exam.quiz_id == quiz.id)
        .order_by(Exam.id.desc())
        .first()
    )

    if not exam:
        print("❌ Exam not generated")
        raise HTTPException(status_code=404, detail="Exam not generated")

    # --------------------------------------------------
    # 🔧 Normalize questions for frontend
    # --------------------------------------------------
    def normalize_questions(raw_questions):
        normalized = []

        for q in raw_questions or []:
            fixed = dict(q)
            opts = fixed.get("options")

            if isinstance(opts, dict):
                fixed["options"] = [f"{k}) {v}" for k, v in opts.items()]
            elif isinstance(opts, list):
                fixed["options"] = opts
            else:
                fixed["options"] = []

            normalized.append(fixed)

        return normalized

    # --------------------------------------------------
    # 🟡 CASE B — ACTIVE attempt → RESUME
    # --------------------------------------------------
    if attempt and attempt.completed_at is None:
        print("⏳ Active attempt detected → resuming exam")

        started_at = attempt.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        elapsed = int((now - started_at).total_seconds())
        remaining = max(0, attempt.duration_minutes * 60 - elapsed)

        print(
            f"⏱ elapsed={elapsed}s, remaining={remaining}s"
        )

        # ⛔ Time expired → mark completed
        if remaining == 0:
            print("⛔ Time expired → marking completed")
            attempt.completed_at = now
            db.commit()

            return {
                "completed": True
            }

        return {
            "completed": False,
            "questions": normalize_questions(exam.questions),
            "remaining_time": remaining
        }

    # --------------------------------------------------
    # 🔵 CASE C — FIRST attempt → START NEW
    # --------------------------------------------------
    print("🆕 No attempt found → starting new exam")

    new_attempt = StudentExam(
        student_id=student.id,
        exam_id=exam.id,
        started_at=datetime.now(timezone.utc),
        duration_minutes=40
    )

    db.add(new_attempt)
    db.commit()

    print(f"✅ New attempt created: id={new_attempt.id}")

    return {
        "completed": False,
        "questions": normalize_questions(exam.questions),
        "remaining_time": new_attempt.duration_minutes * 60
    }

@app.get("/api/student/get-exam")
def get_exam(session_id: int, db: Session = Depends(get_db)):

    session = (
        db.query(StudentExam)
        .filter(StudentExam.id == session_id)
        .first()
    )

    # ❌ Invalid or deleted session → return safe payload
    if not session:
        return {
            "questions": [],
            "total_questions": 0,
            "remaining_time": 0,
            "completed": True
        }

    exam = (
        db.query(Exam)
        .filter(Exam.id == session.exam_id)
        .first()
    )

    if not exam:
        return {
            "questions": [],
            "total_questions": 0,
            "remaining_time": 0,
            "completed": True
        }

    # -------------------------------------
    # TIME CALCULATION (timezone-safe)
    # -------------------------------------
    now = datetime.now(timezone.utc)

    started_at = session.started_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)

    elapsed = (now - started_at).total_seconds()
    remaining = session.duration_minutes * 60 - elapsed

    if remaining <= 0:
        session.completed_at = now
        db.commit()
        remaining = 0

    # -------------------------------------
    # NORMALIZE QUESTIONS
    # -------------------------------------
    normalized_questions = []

    for q in exam.questions:
        fixed = dict(q)
        opts = fixed.get("options")

        if opts is None:
            fixed["options"] = []
        elif isinstance(opts, dict):
            fixed["options"] = [f"{k}) {v}" for k, v in opts.items()]
        elif not isinstance(opts, list):
            fixed["options"] = []

        normalized_questions.append(fixed)

    return {
        "questions": normalized_questions,
        "total_questions": len(normalized_questions),
        "remaining_time": int(max(remaining, 0)),
        "completed": session.completed_at is not None
    }
 
@app.post("/api/student/finish-exam")
def finish_exam(
    req: FinishExamRequest,
    db: Session = Depends(get_db)
):
    print("\n================ FINISH EXAM START ================")
    print("📥 Incoming payload:", req.dict())

    # --------------------------------------------------
    # 1️⃣ Validate student (external ID → internal ID)
    # --------------------------------------------------
    student = (
        db.query(Student)
        .filter(Student.student_id == req.student_id)
        .first()
    )

    print("👤 Student lookup result:", student)

    if not student:
        print("❌ Student NOT FOUND for student_id =", req.student_id)
        raise HTTPException(status_code=404, detail="Student not found")

    print("✅ Student found → internal id:", student.id)

    # --------------------------------------------------
    # 2️⃣ Get active exam attempt
    # --------------------------------------------------
    attempt = (
        db.query(StudentExam)
        .filter(
            StudentExam.student_id == student.id,
            StudentExam.completed_at.is_(None)
        )
        .order_by(StudentExam.started_at.desc())
        .first()
    )

    print("📝 Exam attempt lookup result:", attempt)

    if not attempt:
        print("⚠️ NO ACTIVE ATTEMPT FOUND for student.id =", student.id)
        return {"status": "completed"}

    print("✅ Active attempt found → attempt.id =", attempt.id)
    print("   started_at:", attempt.started_at)
    print("   completed_at (before):", attempt.completed_at)

    # --------------------------------------------------
    # 3️⃣ Idempotency guard
    # --------------------------------------------------
    existing_result = (
        db.query(StudentExamResultsThinkingSkills)
        .filter(
            StudentExamResultsThinkingSkills.exam_attempt_id == attempt.id
        )
        .first()
    )

    print("🧾 Existing result lookup:", existing_result)

    if existing_result:
        print("⚠️ Existing result already present for attempt.id =", attempt.id)

        if attempt.completed_at is None:
            print("🛠 Setting completed_at defensively")
            attempt.completed_at = datetime.now(timezone.utc)
            db.commit()
            print("✅ completed_at updated defensively")

        return {"status": "completed"}

    # --------------------------------------------------
    # 4️⃣ Load exam
    # --------------------------------------------------
    exam = db.query(Exam).filter(Exam.id == attempt.exam_id).first()

    print("📘 Exam lookup result:", exam)

    if not exam:
        print("❌ Exam NOT FOUND for exam_id =", attempt.exam_id)
        raise HTTPException(status_code=404, detail="Exam not found")

    questions = exam.questions or []
    total_questions = len(questions)

    print("📊 Total questions in exam:", total_questions)

    question_map = {q["q_id"]: q for q in questions}

    # --------------------------------------------------
    # 5️⃣ Save student responses
    # --------------------------------------------------
    correct = 0
    saved_responses = 0

    for q_id_str, selected in req.answers.items():
        print(f"➡️ Processing answer: q_id={q_id_str}, selected={selected}")

        try:
            q_id = int(q_id_str)
        except ValueError:
            print("❌ Invalid q_id (not int):", q_id_str)
            continue

        q = question_map.get(q_id)
        if not q:
            print("⚠️ Question not found in exam JSON for q_id =", q_id)
            continue

        is_correct = selected == q.get("correct")

        if is_correct:
            correct += 1

        db.add(
            StudentExamResponse(
                student_id=student.id,
                exam_id=exam.id,
                exam_attempt_id=attempt.id,
                q_id=q_id,
                topic=q.get("topic", "Unknown"),
                selected_option=selected,
                correct_option=q.get("correct"),
                is_correct=is_correct
            )
        )

        saved_responses += 1
        print(
            f"   💾 Saved response → q_id={q_id}, "
            f"is_correct={is_correct}"
        )

    print("📈 Responses saved:", saved_responses)
    print("✅ Correct answers count:", correct)

    wrong = total_questions - correct
    accuracy = round((correct / total_questions) * 100, 2) if total_questions else 0

    print("📊 Computed result:")
    print("   total_questions:", total_questions)
    print("   correct:", correct)
    print("   wrong:", wrong)
    print("   accuracy:", accuracy)

    # --------------------------------------------------
    # 6️⃣ Save summary result
    # --------------------------------------------------
    result_row = StudentExamResultsThinkingSkills(
        student_id=student.id,
        exam_attempt_id=attempt.id,
        total_questions=total_questions,
        correct_answers=correct,
        wrong_answers=wrong,
        accuracy_percent=accuracy
    )

    db.add(result_row)
    print("🧾 Summary result row added")

    # --------------------------------------------------
    # 7️⃣ Mark exam completed
    # --------------------------------------------------
    attempt.completed_at = datetime.now(timezone.utc)
    print("⏱ Setting completed_at to:", attempt.completed_at)

    db.commit()
    print("💾 DB COMMIT COMPLETED")

    print("================ FINISH EXAM END =================\n")

    return {
        "status": "completed",
        "total_questions": total_questions,
        "correct": correct,
        "wrong": wrong,
        "accuracy": accuracy
    }



@app.get("/api/student/get-quiz")
def get_quiz(student_id: str, subject: str, difficulty: str, db: Session = Depends(get_db)):
    """
    Returns full quiz questions for the student based on class, subject, difficulty.
    Includes detailed debug logs for diagnosing issues.
    """

    print("\n===========================")
    print("📥 API CALL: /api/student/get-quiz")
    print("===========================")
    print("🔸 Incoming parameters:")
    print(f"   student_id: {student_id}")
    print(f"   subject: {subject}")
    print(f"   difficulty: {difficulty}")
    print("---------------------------")

    # -------------------------------
    # VALIDATE STUDENT
    # -------------------------------
    student = db.query(Student).filter(Student.student_id == student_id).first()

    if not student:
        print("❌ ERROR: Student not found in database.")
        print(f"   student_id searched: {student_id}")
        raise HTTPException(status_code=404, detail="Student not found")

    print("✅ Student found:")
    print(f"   internal_id: {student.id}")
    print(f"   class_name: {student.class_name}")
    print("---------------------------")

    # -------------------------------
    # FIND MATCHING QUIZ
    # -------------------------------
    print("🔎 Searching for quiz definition matching:")
    print(f"   class_name: {student.class_name.lower()}")
    print(f"   subject: {subject}")
    print(f"   difficulty: {difficulty}")

    quiz = (
        db.query(Quiz)
        .filter(
            func.lower(Quiz.class_name) == student.class_name.lower(),
            Quiz.subject == subject,
            Quiz.difficulty == difficulty
        )
        .order_by(Quiz.id.desc())
        .first()
    )

    if not quiz:
        print("❌ ERROR: No quiz definition found matching given params.")
        raise HTTPException(status_code=404, detail="No quiz found")

    print("✅ Quiz definition found:")
    print(f"   quiz_id: {quiz.id}")
    print(f"   topics: {quiz.topics}")
    print("---------------------------")

    # -------------------------------
    # GET EXAM INSTANCE
    # -------------------------------
    print("🔎 Looking for generated exam linked to quiz...")

    exam = (
        db.query(Exam)
        .filter(Exam.quiz_id == quiz.id)
        .order_by(Exam.id.desc())
        .first()
    )

    if not exam:
        print("❌ ERROR: Exam instance not found — quiz not generated yet.")
        raise HTTPException(status_code=404, detail="Quiz not generated")

    print("✅ Exam found:")
    print(f"   exam_id: {exam.id}")
    print(f"   number_of_questions: {len(exam.questions) if exam.questions else 0}")
    print("---------------------------")

    # -------------------------------
    # RETURN QUESTIONS
    # -------------------------------
    print("📤 Preparing JSON response...")
    response = {
        "exam_id": exam.id,
        "quiz_id": quiz.id,
        "questions": exam.questions
    }

    print("📦 Returning response:")
    print(f"   exam_id: {response['exam_id']}")
    print(f"   quiz_id: {response['quiz_id']}")
    print(f"   total_questions: {len(response['questions'])}")
    print("===========================\n")

    return response



@app.post("/upload-image-folder")
async def upload_image_folder(images: List[UploadFile] = File(...)):
    print("\n====================== FOLDER UPLOAD START ======================\n")
    print(f"[INFO] Number of received files: {len(images)}")

    if not images:
        raise HTTPException(status_code=400, detail="No images received.")

    uploaded_urls = []

    for file in images:
        try:
            print("\n-----------------------------------------------------------")
            print(f"[FILE] Original filename: {file.filename}")
            print(f"[FILE] Content type: {file.content_type}")

            # Read file bytes
            file_bytes = await file.read()

            # Upload to GCS
            url = upload_to_gcs(file_bytes, file.filename)

            print(f"[UPLOAD SUCCESS] → {url}")

            # --- NEW: Extract clean filename ---
            clean_name = file.filename.split("/")[-1].split("\\")[-1]

            # --- NEW: Save to global map ---
            GLOBAL_IMAGE_MAP[clean_name] = url
            print(f"[MAP] {clean_name} → {url}")

            uploaded_urls.append({
                "original_name": clean_name,
                "url": url
            })

        except Exception as e:
            print(f"[ERROR] Failed to upload file '{file.filename}'")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

    print("\n====================== FOLDER UPLOAD END ========================\n")

    return {
        "status": "success",
        "message": f"{len(uploaded_urls)} images uploaded successfully.",
        "files": uploaded_urls
    }



@app.post("/upload-word")
async def upload_word(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    print("\n========== GPT-UPLOAD-WORD START ==========")
    print(f"📄 Incoming file: {file.filename} (type={file.content_type})")

    # -----------------------------
    # Validate file type
    # -----------------------------
    allowed = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain"
    }
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    # -----------------------------
    # Extract raw text
    # -----------------------------
    try:
        content = await file.read()
        doc = docx.Document(BytesIO(content))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    except Exception as e:
        print("❌ Error reading DOCX:", e)
        raise HTTPException(status_code=400, detail="Failed to read file.")

    # -----------------------------
    # Convert to 5-page chunks
    # -----------------------------
    pages = chunk_into_pages(paragraphs, per_page=30)
    page_groups = group_pages(pages, size=5)

    print(f"📄 Total pages: {len(pages)}")
    print(f"📦 Blocks to process (chunks of 5 pages): {len(page_groups)}")

    all_questions = []

    # -----------------------------
    # Process each chunk via GPT
    # -----------------------------
    for idx, block in enumerate(page_groups, start=1):
        print(f"\n--- GPT Parsing Block {idx}/{len(page_groups)} ---")

        result = await parse_with_gpt(block)
        questions = result.get("questions", [])

        print(f"🧩 GPT found {len(questions)} questions in block {idx}")
        all_questions.extend(questions)

    # -----------------------------
    # Save valid (non-partial) questions
    # -----------------------------
    saved_ids = []

    for q in all_questions:
        if q.get("partial"):
            print("⚠ Skipping partial question")
            continue
    
        # --- Image filename → URL mapping ---
        resolved_images = []
        for img in q.get("images") or []:
            if img in GLOBAL_IMAGE_MAP:
                resolved_images.append(GLOBAL_IMAGE_MAP[img])
                print(f"🔗 Mapped {img} → {GLOBAL_IMAGE_MAP[img]}")
            else:
                print(f"⚠ Image not found in map: {img}")
                resolved_images.append(img)  # fallback
                
        new_q = Question(
            class_name=q.get("class_name"),
            subject=q.get("subject"),
            topic=q.get("topic"),
            difficulty=q.get("difficulty"),
            question_type="multi_image_diagram_mcq",
            question_text=q.get("question_text"),
            images=resolved_images,
            options=q.get("options"),
            correct_answer=q.get("correct_answer")
        )
    
        db.add(new_q)
        db.commit()
        db.refresh(new_q)
    
        saved_ids.append(new_q.id)

    if not saved_ids:
        raise HTTPException(status_code=400, detail="No valid questions parsed.")

    print("\n🎉 GPT Upload complete. Questions saved:", saved_ids)
    print("========== GPT-UPLOAD-WORD END ==========\n")

    return {
        "status": "success",
        "saved_questions": saved_ids,
        "count": len(saved_ids)
    }




@app.get("/api/student/exam-status")
def exam_status(student_id: str, subject: str, db: Session = Depends(get_db)):
    """
    Returns:
    {
        "started": true/false,
        "completed": true/false,
        "attempts_used": 1,
        "attempts_allowed": 3,
        "total_questions": 40,
        "exam_id": 12
    }
    """

    # 1. Validate student exists
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    # 2. Find a quiz requirement for this student's class and subject
    quiz = (
        db.query(Quiz)
        .filter(
            func.lower(Quiz.class_name) == student.class_name.lower(),
            Quiz.subject == subject
        )
        .order_by(Quiz.id.desc())
        .first()
    )


    if not quiz:
        return {
            "started": False,
            "completed": False,
            "attempts_used": 0,
            "attempts_allowed": 3,
            "total_questions": 0,
            "exam_id": None
        }

    # Total number of questions in this quiz
    total_questions = sum(t["total"] for t in quiz.topics)

    # 3. Check if student has any exam attempts for this quiz
    attempts = (
        db.query(StudentExam)
        .filter(StudentExam.student_id == student.id, StudentExam.exam_id.isnot(None))
        .all()
    )

    attempts_used = len(attempts)
    attempts_allowed = 3  # your rule (frontend shows 1/3)

    # 4. Find the latest attempt for status
    latest_attempt = (
        db.query(StudentExam)
        .filter(StudentExam.student_id == student.id)
        .order_by(StudentExam.id.desc())
        .first()
    )

    # If no attempts at all
    if not latest_attempt:
        return {
            "started": False,
            "completed": False,
            "attempts_used": attempts_used,
            "attempts_allowed": attempts_allowed,
            "total_questions": total_questions,
            "quiz_id": quiz.id,
            "exam_id": None
        }

    # 5. Load the exam linked to latest attempt
    exam = db.query(Exam).filter(Exam.id == latest_attempt.exam_id).first()

    # Exam may not exist yet if not generated
    if not exam:
        return {
            "started": False,
            "completed": False,
            "attempts_used": attempts_used,
            "attempts_allowed": attempts_allowed,
            "total_questions": total_questions,
            "quiz_id": quiz.id,
            "exam_id": None
        }

    # Determine started & completed status
    started = latest_attempt.started_at is not None
    completed = latest_attempt.completed_at is not None

    return {
        "started": started,
        "completed": completed,
        "attempts_used": attempts_used,
        "attempts_allowed": attempts_allowed,
        "total_questions": len(exam.questions),
        "quiz_id": quiz.id,
        "exam_id": exam.id
    }


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
        started_at=datetime.now(timezone.utc)
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
    Generate an exam based on a quiz and its topics.

    Workflow:
    1. Fetch the quiz
    2. Generate exam questions (AI + DB)
    3. Store the exam record
    4. Return the generated exam
    """

    # 1. Fetch quiz
    quiz = (
        db.query(Quiz)
        .filter(Quiz.id == quiz_id)
        .first()
    )
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    # 2. Generate exam questions
    try:
        questions = generate_exam_questions(quiz, db)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate exam: {str(e)}"
        )

    if not questions:
        raise HTTPException(status_code=500, detail="No questions generated")

    # 3. Save exam (now includes class_name)
    new_exam = Exam(
        quiz_id=quiz.id,
        class_name=quiz.class_name,      # <-- saved here
        subject=quiz.subject,            # (optional but recommended)
        difficulty=quiz.difficulty,      # (optional but recommended)
        questions=questions
    )

    db.add(new_exam)
    db.commit()
    db.refresh(new_exam)

    # 4. Return payload
    return {
        "message": "Exam generated successfully",
        "exam_id": new_exam.id,
        "quiz_id": quiz.id,
        "class_name": quiz.class_name,
        "subject": quiz.subject,
        "difficulty": quiz.difficulty,
        "questions": questions,
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
        today = datetime.now(timezone.utc).date()
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


@app.get("/api/quizzes")
def get_quizzes(db: Session = Depends(get_db)):
    """
    Return a minimal list of quiz requirement entries for frontend selection.
    """
    try:
        quizzes = db.query(Quiz).order_by(Quiz.id.desc()).all()
        return [
            {
                "id": q.id,
                "class_name": q.class_name,
                "subject": q.subject,
                "difficulty": q.difficulty
            }
            for q in quizzes
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching quizzes: {str(e)}")


     
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

        today = datetime.now(timezone.utc).date()
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
        quiz.started_at = datetime.now(timezone.utc)

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
        quiz.completed_at = datetime.now(timezone.utc)

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
                submitted_at=datetime.now(timezone.utc)
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





