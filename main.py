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
from collections import defaultdict
from sqlalchemy.exc import IntegrityError

 
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

from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, ForeignKey, select, func, text, Boolean, Date, Text, desc, Float, Index, case, Numeric, distinct, cast
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

        # Full textual content only (concatenated text blocks)
        "question_text": {"type": "string"},

        # Ordered visual structure (text + images)
        "question_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["text", "image"]
                    },
                    "content": {
                        "type": "string"
                    },
                    "src": {
                        "type": "string"
                    }
                },
                "required": ["type"],
                "additionalProperties": False
            }
        },

        # Flat image list (legacy / convenience)
        "images": {
            "type": "array",
            "items": {"type": "string"}
        },

        # MCQ options
        "options": {
            "type": "object",
            "additionalProperties": {"type": "string"}
        },

        "correct_answer": {"type": "string"},
        "partial": {"type": "boolean"}
    },

    # Required fields for a valid question
    "required": [
        "question_text",
        "question_blocks",
        "options",
        "correct_answer"
    ],

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
class UploadedImage(Base):
    __tablename__ = "uploaded_images"

    id = Column(Integer, primary_key=True)
    original_name = Column(String, unique=True, nullable=False)
    gcs_url = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
 
class AdminOverallSelectiveReport(Base):
    __tablename__ = "admin_overall_selective_reports"

    id = Column(Integer, primary_key=True)
    student_id = Column(String, index=True)
    exam_date = Column(Date, index=True)

    overall_percent = Column(Float)
    readiness_band = Column(String)
    school_recommendation = Column(JSON)

    override_flag = Column(Boolean, default=False)
    override_message = Column(Text, nullable=True)

    components = Column(JSON)  # {reading: 78, maths: 65, ...}

    created_at = Column(DateTime, default=datetime.utcnow)

class AdminExamRawScore(Base):
    __tablename__ = "admin_exam_raw_scores"

    id = Column(Integer, primary_key=True, index=True)

    # internal student identifier (NO foreign key)
    student_id = Column(Integer, nullable=False)

    # exam attempt identifier (one row per attempt)
    exam_attempt_id = Column(Integer, nullable=False, unique=True)

    subject = Column(String, nullable=False)

    total_questions = Column(Integer, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    wrong_answers = Column(Integer, nullable=False)

    accuracy_percent = Column(Float, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AdminExamReport(Base):
    __tablename__ = "admin_exam_reports"

    id = Column(Integer, primary_key=True, index=True)

    # external student identifier (e.g. "Gem002")
    student_id = Column(String, nullable=False, index=True)

    # references exam attempt table (thinking / reading / etc)
    exam_attempt_id = Column(Integer, nullable=False, unique=True, index=True)

    exam_type = Column(String, nullable=False)
    # e.g. "selective"

    overall_score = Column(Float, nullable=True)

    readiness_band = Column(String, nullable=False)
    # e.g. "Strong Selective Potential"

    school_guidance_level = Column(String, nullable=False)
    # e.g. "Mid-tier Selective"

    summary_notes = Column(String, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )


class AdminExamSectionResult(Base):
    __tablename__ = "admin_exam_section_results"

    id = Column(Integer, primary_key=True, index=True)

    # references AdminExamReport.id
    admin_report_id = Column(Integer, nullable=False, index=True)

    section_name = Column(String, nullable=False)
    # "reading", "maths", "thinking", "writing"

    raw_score = Column(Float, nullable=True)
    # percent or marks

    performance_band = Column(String, nullable=False)
    # A / B / C / etc

    strengths_summary = Column(String, nullable=True)
    improvement_summary = Column(String, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
class AdminReadinessRuleApplied(Base):
    __tablename__ = "admin_readiness_rules_applied"

    id = Column(Integer, primary_key=True, index=True)

    # references AdminExamReport.id
    admin_report_id = Column(Integer, nullable=False, index=True)

    rule_code = Column(String, nullable=False)
    # e.g. "WRITING_MINIMUM"

    rule_description = Column(String, nullable=False)
    # human-readable explanation

    rule_result = Column(String, nullable=False)
    # "passed" or "failed"

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )


class StudentExamResultsMathematicalReasoning(Base):
    __tablename__ = "student_exam_results_mathematical_reasoning"

    id = Column(Integer, primary_key=True, index=True)

    student_id = Column(String, ForeignKey("students.id"), nullable=False)
    exam_attempt_id = Column(
        Integer,
        ForeignKey("student_exams.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    total_questions = Column(Integer, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    wrong_answers = Column(Integer, nullable=False)
    accuracy_percent = Column(Numeric(5, 2), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # relationships (optional but recommended)
    attempt = relationship("StudentExam")
    student = relationship("Student")
class StudentExamThinkingSkills(Base):
    __tablename__ = "student_exam_thinking_skills"

    id = Column(Integer, primary_key=True, index=True)

    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)

    started_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    duration_minutes = Column(Integer, nullable=False)

    # -----------------------------
    # Relationships (optional but recommended)
    # -----------------------------
    student = relationship("Student")
    exam = relationship("Exam")
    responses = relationship(
        "StudentExamResponseThinkingSkills",
        back_populates="attempt",
        cascade="all, delete-orphan"
    )

class StudentExamResponseThinkingSkills(Base):
    __tablename__ = "student_exam_response_thinking_skills"

    id = Column(Integer, primary_key=True, index=True)

    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    exam_attempt_id = Column(
        Integer,
        ForeignKey("student_exam_thinking_skills.id"),
        nullable=False
    )

    q_id = Column(Integer, nullable=False)      # stable per-attempt index
    topic = Column(String, nullable=True)

    selected_option = Column(String, nullable=True)
    correct_option = Column(String, nullable=True)

    is_correct = Column(Boolean, nullable=True) # NULL = not attempted

    # -----------------------------
    # Relationships
    # -----------------------------
    attempt = relationship(
        "StudentExamThinkingSkills",
        back_populates="responses"
    )
    student = relationship("Student")
    exam = relationship("Exam")


class TopicConfigMathematicalReasoning(BaseModel):
    name: str
    ai: int
    db: int
    total: int

class TopicConfigMathematicalReasoningCreate(BaseModel):
    class_name: str
    subject: str
    difficulty: str
    num_topics: int
    topics: List[TopicConfigMathematicalReasoning]

class QuizMathematicalReasoningCreate(BaseModel):
    class_name: str
    subject: str
    difficulty: str
    num_topics: int
    topics: List[TopicConfigMathematicalReasoning]
class QuizMathematicalReasoning(Base):
    __tablename__ = "quiz_mathematical_reasoning"

    id = Column(Integer, primary_key=True, index=True)

    class_name = Column(String, nullable=False)
    subject = Column(String, nullable=False)  # always mathematical_reasoning
    difficulty = Column(String, nullable=False)

    num_topics = Column(Integer, nullable=False)

    # Stores topic configs as JSON
    topics = Column(JSON, nullable=False)
 


class UpdateStudentRequest(BaseModel):
    student_id: str  # identifier (cannot be changed)

    name: Optional[str] = None
    parent_email: Optional[str] = None
    class_name: Optional[str] = None
    class_day: Optional[str] = None
    password: Optional[str] = None  # plain text (as per your system)
 
class StudentExamReportReading(Base):
    __tablename__ = "student_exam_report_reading"

    id = Column(Integer, primary_key=True, index=True)

    # 🔑 Identity
    student_id = Column(String, index=True, nullable=False)
    exam_id = Column(Integer, index=True, nullable=False)
    session_id = Column(Integer, index=True, nullable=False, unique=True)

    # 🔎 Question-level data
    topic = Column(String, index=True, nullable=False)
    question_id = Column(String, index=True, nullable=False)

    selected_answer = Column(String, nullable=True)
    correct_answer = Column(String, nullable=False)
    is_correct = Column(Boolean, nullable=False)

    # 🧾 Snapshot (optional but powerful)
    question_snapshot = Column(JSON, nullable=True)

    # 🕒 Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
 
class StudentExamResponseFoundational(Base):
     __tablename__ = "student_exam_response_foundational"
 
     id = Column(Integer, primary_key=True, index=True)
 
     student_id = Column(String, index=True, nullable=False)
     exam_id = Column(Integer, index=True, nullable=False)
     attempt_id = Column(Integer, index=True, nullable=False)
     topic = Column(String)

 
     section_name = Column(String, index=True, nullable=False)  # topic
     question_id = Column(String, index=True, nullable=False)
 
     selected_answer = Column(String, nullable=True)
     correct_answer = Column(String, nullable=False)
     is_correct = Column(Boolean, nullable=False)
 
     created_at = Column(DateTime(timezone=True), server_default=func.now())


class StartExamRequestFoundational(BaseModel):
    student_id: str
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
    report_json = Column(JSON, nullable=True)


class WritingGenerateSchema(BaseModel):
    class_name: str
    difficulty: str
 
class StudentExamWriting(Base):
    __tablename__ = "student_exam_writing"

    id = Column(Integer, primary_key=True, index=True)

    # External student identifier (e.g. "Gem002")
    student_id = Column(Text, nullable=False)

    # Generated writing exam ID
    exam_id = Column(Integer, nullable=False)

    # Exam lifecycle timestamps
    started_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    # Exam configuration
    duration_minutes = Column(
        Integer,
        default=40,
        nullable=False
    )

    # Student response
    answer_text = Column(
        Text,
        nullable=True
    )

    # -----------------------------------------
    # AI Writing Evaluation (Persisted)
    # -----------------------------------------
    ai_score = Column(
        Integer,
        nullable=True,
        comment="AI-evaluated writing score (0–20)"
    )

    ai_strengths = Column(
        Text,
        nullable=True,
        comment="AI-identified writing strengths"
    )

    ai_improvements = Column(
        Text,
        nullable=True,
        comment="AI-identified areas for improvement"
    )

    # Record creation timestamp
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
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

    # Total exam duration in minutes
    duration_minutes = Column(Integer, nullable=False)

    exam_json = Column(JSON, nullable=False)

    is_current = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

class StudentsExamFoundational(Base):
    __tablename__ = "students_exam_foundational"

    id = Column(Integer, primary_key=True, index=True)

    # External student identifier (e.g. "Gem002")
    student_id = Column(String, nullable=False, index=True)

    # Reference to generated_exam_foundational.id (NOT a foreign key)
    exam_id = Column(Integer, nullable=False, index=True)

    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    current_section_index = Column(Integer, nullable=False, default=0)

    answers_json = Column(JSONB, nullable=True)
    completion_reason = Column(String, nullable=True)

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

    class_name = Column(String(50), nullable=False)
    subject = Column(String(50), nullable=False)

    # -------- Section 1 --------
    section1_name = Column(String(20), nullable=False)
    section1_topic = Column(String(100), nullable=False)   # ✅ ADDED
    section1_ai = Column(Integer, default=0)
    section1_db = Column(Integer, default=0)
    section1_total = Column(Integer, default=0)
    section1_time = Column(Integer, default=0)
    section1_intro = Column(Text, default="")

    # -------- Section 2 --------
    section2_name = Column(String(20), nullable=False)
    section2_topic = Column(String(100), nullable=False)   # ✅ ADDED
    section2_ai = Column(Integer, default=0)
    section2_db = Column(Integer, default=0)
    section2_total = Column(Integer, default=0)
    section2_time = Column(Integer, default=0)
    section2_intro = Column(Text, default="")

    # -------- Section 3 (optional) --------
    section3_name = Column(String(20), nullable=True)
    section3_topic = Column(String(100), nullable=True)    # ✅ ADDED
    section3_ai = Column(Integer, nullable=True)
    section3_db = Column(Integer, nullable=True)
    section3_total = Column(Integer, nullable=True)
    section3_time = Column(Integer, nullable=True)
    section3_intro = Column(Text, nullable=True)

class SectionSchema(BaseModel):
    name: str                # Easy / Medium / Hard
    topic: str               # NEW: selected topic for this section
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

    # External student identifier (e.g. "Gem002")
    student_id = Column(String, nullable=False, index=True)

    # Reference to exams.id (NOT a foreign key)
    exam_id = Column(Integer, nullable=False, index=True)

    # Timezone-aware timestamps (UTC)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    duration_minutes = Column(Integer, default=40, nullable=False)


class Exam_reading(Base):
    __tablename__ = "exams_reading"   # UPDATED TABLE NAME

    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String)
    subject = Column(String)
    difficulty = Column(String)
    topic = Column(String)
    total_questions = Column(Integer)
    reading_material = Column(JSON)  # dictionary of extracts/paragraphs



class QuestionReading(Base):
    __tablename__ = "questions_reading"

    id = Column(Integer, primary_key=True, index=True)

    # --------------------------------------------------
    # FILTERABLE / SEARCH FIELDS
    # --------------------------------------------------
    class_name = Column(String, index=True, nullable=False)
    subject = Column(String, index=True, nullable=False)
    difficulty = Column(String, index=True, nullable=False)
    topic = Column(String, index=True, nullable=False)

    # Explicit for admin sanity & validation
    total_questions = Column(Integer, nullable=False)

    # --------------------------------------------------
    # OPTIONAL EXAM LINK (ADMIN-CREATED EXAMS)
    # --------------------------------------------------
    exam_id = Column(
        Integer,
        ForeignKey("exams_reading.id", ondelete="SET NULL"),
        nullable=True
    )

    # --------------------------------------------------
    # COMPLETE SELF-DESCRIBING QUIZ PAYLOAD
    # --------------------------------------------------
    exam_bundle = Column(JSON, nullable=False)


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

    id = Column(String, primary_key=True, index=True)   # internal PK
    student_id = Column(String, unique=True, nullable=False)  # e.g. "Gem002"

    password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    parent_email = Column(String, nullable=False)

    class_name = Column(String, nullable=False)
    class_day = Column(String, nullable=True)
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
    # ✅ NEW — ordered visual blocks (text + images)
    question_blocks = Column(JSON, nullable=True)

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

    # just a number, NO foreign key
    quiz_id = Column(Integer, nullable=True)

    class_name = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)

    questions = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())



class StudentExamAnswer(Base):
    __tablename__ = "student_exam_answers"

    id = Column(Integer, primary_key=True, index=True)

    # Reference to student_exams.id (NOT a foreign key)
    student_exam_id = Column(Integer, nullable=False, index=True)

    question_id = Column(Integer, nullable=False)
    student_answer = Column(String, nullable=False)
    correct_answer = Column(String, nullable=False)
    is_correct = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())





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
        "https://exam.gemkidsacademy.com.au",
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
def chunk_into_pages(paragraphs, per_page=18):
    pages = []
    for i in range(0, len(paragraphs), per_page):
        pages.append("\n\n".join(paragraphs[i:i+per_page]))
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
                "You are a deterministic exam-question parser that preserves data fidelity and visual order.\n\n"
            
                "CONTENT FIDELITY RULES:\n"
                "- The QUESTION_TEXT MUST include ALL setup, scenario, and descriptive text\n"
                "- Followed by the actual interrogative sentence\n"
                "- Preserve original wording and original order exactly\n"
                "- QUESTION_TEXT may span multiple sentences or multiple paragraphs\n"
                "- Do NOT summarize, shorten, paraphrase, or omit any content\n\n"
            
                "FORMATTING RULES:\n"
                "- Preserve paragraph boundaries exactly as provided in the input\n"
                "- Use double newline characters (\\n\\n) to separate paragraphs in QUESTION_TEXT\n"
                "- Do NOT collapse multiple paragraphs into a single line\n"
                "- Do NOT remove, trim, or normalize newline characters\n\n"
            
                "QUESTION_BLOCKS RULES:\n"
                "- In addition to QUESTION_TEXT, you MUST return QUESTION_BLOCKS\n"
                "- QUESTION_BLOCKS is an ordered list representing the visual flow of the question\n"
                "- Each block MUST be one of:\n"
                "  { \"type\": \"text\", \"content\": \"...\" }\n"
                "  { \"type\": \"image\", \"src\": \"...\" }\n"
                "- Preserve the exact order found in the source document\n"
                "- Do NOT merge text across image boundaries\n"
                "- All text appearing before an image MUST appear in a text block before that image\n"
                "- All text appearing after an image MUST appear in a text block after that image\n"
                "- QUESTION_TEXT MUST contain the full textual content only (text blocks concatenated with \\n\\n)\n\n"
            
                "STRUCTURE RULES:\n"
                "- Questions follow this structure:\n"
                "  Question X:\n"
                "  CLASS\n"
                "  SUBJECT\n"
                "  TOPIC\n"
                "  DIFFICULTY\n"
                "  QUESTION_TEXT\n"
                "  QUESTION_BLOCKS\n"
                "  (optional) IMAGES\n"
                "  OPTIONS\n"
                "  CORRECT_ANSWER\n\n"
            
                "TERMINATION RULES:\n"
                "- A question is COMPLETE if:\n"
                "  a) All required fields are present, OR\n"
                "  b) It is immediately followed by the marker END_OF_QUESTIONS\n"
                "- The marker END_OF_QUESTIONS indicates the end of the document\n"
                "- Ignore all content after END_OF_QUESTIONS\n\n"
            
                "PARTIAL RULES:\n"
                "- Mark partial=true ONLY if required fields are missing\n"
                "- Do NOT mark a question as partial due to document ending alone\n"
                "- Presence of IMAGES or QUESTION_BLOCKS does NOT imply incompleteness\n\n"
            
                "FAILURE HANDLING RULES:\n"
                "- If a question cannot be parsed into the required structure, OMIT it entirely\n"
                "- Do NOT emit placeholder text\n"
                "- Do NOT emit error messages\n"
                "- Do NOT emit strings inside the questions array\n"
                "- An empty questions array is valid\n\n"
            
                "OUTPUT RULES:\n"
                "- Return ONLY valid JSON following the provided schema\n"
                "- Do NOT include commentary, explanations, or markdown"
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
    print(f"Quiz ID       : {quiz.id}")
    print(f"Class         : {quiz.class_name}")
    print(f"Subject       : {quiz.subject}")
    print(f"Difficulty    : {quiz.difficulty}")
    print(f"Topics Config : {quiz.topics}")
    print("===============================================================\n")

    all_questions = []
    q_id = 1

    for topic in quiz.topics:
        print("\n===================== PROCESSING TOPIC =====================")

        topic_name = topic.get("name")
        ai_count = int(topic.get("ai", 0))
        db_count = int(topic.get("db", 0))

        print(f"Topic         : {topic_name}")
        print(f"Expected DB   : {db_count}")
        print(f"Expected AI   : {ai_count}")
        print("============================================================")

        # --------------------------------------------------
        # 1️⃣ DB PRE-FLIGHT CHECK
        # --------------------------------------------------
        available_db = (
            db.query(Question)
              .filter(func.lower(Question.topic) == topic_name.lower())
              .count()
        )

        print(f"[DB CHECK] Available questions: {available_db}")

        if available_db < db_count:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Not enough DB questions for topic '{topic_name}'. "
                    f"Required: {db_count}, Available: {available_db}"
                )
            )

        # --------------------------------------------------
        # 2️⃣ FETCH DB QUESTIONS
        # --------------------------------------------------
        db_questions = (
            db.query(Question)
              .filter(func.lower(Question.topic) == topic_name.lower())
              .order_by(func.random())
              .limit(db_count)
              .all()
        )

        print(f"[DB FETCH] Retrieved {len(db_questions)} questions")

        for q in db_questions:
            all_questions.append({
                "q_id": q_id,
                "topic": topic_name,
                "question": q.question_text,
                "options": q.options,
                "correct": q.correct_answer,
                "images": q.images or []
            })
            q_id += 1

        # --------------------------------------------------
        # 3️⃣ AI QUESTION GENERATION (STRICT)
        # --------------------------------------------------
        if ai_count > 0:
            print(f"\n[AI GEN] Requesting {ai_count} questions for '{topic_name}'")

            location = db.query(FranchiseLocation).first()
            if not location:
                raise HTTPException(500, "No franchise location found")

            system_prompt = (
                "You are an expert exam generator.\n"
                f"Create exactly {ai_count} MCQs.\n\n"
                f"Class: {quiz.class_name}\n"
                f"Subject: {quiz.subject}\n"
                f"Topic: {topic_name}\n"
                f"Country: {location.country}\n"
                f"State: {location.state}\n\n"
                "Return ONLY a valid JSON array.\n"
                "Do NOT include explanations, markdown, or extra text.\n"
                "Do NOT wrap in ```.\n"
                "JSON format:\n"
                "[{\"question\":\"...\",\"options\":[\"A\",\"B\",\"C\",\"D\"],\"correct\":\"A\"}]"

            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.4
            )

            raw_output = response.choices[0].message.content.strip()
            # --- DEBUG: inspect AI output ---
            print("\n[AI RAW OUTPUT]")
            print(raw_output)
            print("[END AI RAW OUTPUT]\n")
         
            try:
                generated = json.loads(raw_output)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=500,
                    detail="AI returned invalid JSON"
                )


            print(f"[AI CHECK] Generated {len(generated)} questions")

            if len(generated) != ai_count:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"AI generation failed for topic '{topic_name}'. "
                        f"Expected {ai_count}, got {len(generated)}."
                    )
                )

            for item in generated:
                all_questions.append({
                    "q_id": q_id,
                    "topic": topic_name,
                    "question": item["question"],
                    "options": item["options"],
                    "correct": item["correct"],
                    "images": []
                })
                q_id += 1

    print("\n==================== FINAL EXAM SUMMARY ====================")
    print(f"TOTAL QUESTIONS GENERATED: {len(all_questions)}")
    print("===========================================================\n")

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

def parse_exam_with_openai(extracted_text: str, question_type: str):

    """
    Extract exam data from a Word document.
    This function ONLY extracts data.
    It does NOT invent structure or fill missing information.
    """

    # -------------------------------
    # Task-specific rules
    # -------------------------------
    if question_type == "gapped_text":
        task_rules = """
GAPPED TEXT RULES:
- This is a GAPPED TEXT exam.
- Questions represent gaps, not full sentences.
- Each question MUST be extracted as:
  {
    "question_number": <gap_number>,
    "question_text": "Gap <gap_number> _____",
    "correct_answer": "<LETTER>"
  }
- DO NOT invent question wording.
- question_number MUST be the gap number (1–N), not inferred from position.
- DO NOT include passage text inside question_text.
- Extract correct answers ONLY from explicit CORRECT_ANSWER fields.
- Do NOT infer or guess missing answers.
"""
    else:
        task_rules = """
STANDARD READING RULES:
- Every question begins with a number followed by a period (e.g., "1. Question text").
- Extract the number as question_number (INT).
- The remaining text MUST be question_text.
- NEVER keep the number inside question_text.
"""

    # -------------------------------
    # Prompt
    # -------------------------------
    prompt = f"""
You will receive a Reading Comprehension exam document.

QUESTION TYPE: {question_type}

You MUST extract exam data into the EXACT JSON structure below.

{task_rules}

OUTPUT FORMAT (STRICT):
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

GLOBAL RULES:
- DO NOT add or remove fields.
- DO NOT rename keys.
- DO NOT include explanations or comments.
- Return ONLY valid JSON.
- JSON MUST start with "{{" and end with "}}".
- If something is unclear, leave fields empty — NEVER guess.

INPUT TEXT:
{extracted_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    # -------------------------------
    # Debug logging
    # -------------------------------
    print("======== RAW GPT OUTPUT ========")
    print(content[:5000])
    print("================================")

    # -------------------------------
    # Strict JSON extraction
    # -------------------------------
    start = content.find("{")
    end = content.rfind("}") + 1

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")

    json_text = content[start:end]

    try:
        parsed = json.loads(json_text)

        if "exams" not in parsed or not isinstance(parsed["exams"], list):
            raise ValueError("JSON missing 'exams' list")

        # IMPORTANT: Do not normalize here.
        # Normalization happens in the endpoint.
        return parsed["exams"]

    except Exception as e:
        raise ValueError(
            "OpenAI returned invalid JSON or schema mismatch:\n"
            + json_text
        ) from e


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

def sanitize_filename(filename: str) -> str:
    """
    Convert filename to a URL-safe, GCS-safe format.
    """
    filename = filename.split("/")[-1].split("\\")[-1]  # strip folders
    filename = filename.strip()
    filename = filename.replace(" ", "_")
    filename = re.sub(r"[^a-zA-Z0-9._-]", "", filename)
    return filename
 
def upload_to_gcs(file_bytes: bytes, filename: str) -> str:
    """Upload a file to Google Cloud Storage and return the public URL."""

    print("\n================== GCS UPLOAD START ==================")
    print(f"[GCS] Incoming file: {filename}")
    print(f"[GCS] Bucket: {BUCKET_NAME}")

    if not BUCKET_NAME:
        raise Exception("BUCKET_NAME environment variable not set!")

    bucket = gcs_client.bucket(BUCKET_NAME)

    # ✅ SANITIZE filename
    clean_filename = sanitize_filename(filename)
    print(f"[GCS] Sanitized filename: {clean_filename}")

    # ✅ Generate safe unique object name
    unique_name = f"{uuid.uuid4()}_{clean_filename}"
    print(f"[GCS] Unique stored name: {unique_name}")

    blob = bucket.blob(unique_name)

    try:
        blob.upload_from_string(
            file_bytes,
            content_type="image/png"
        )

        # Uniform Bucket Access → direct public URL
        public_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{unique_name}"

        print(f"[GCS] Upload successful → {public_url}")
        print("================== GCS UPLOAD END ==================\n")

        return public_url

    except Exception as e:
        print("[GCS ERROR] Upload failed:", str(e))
        print("================== GCS UPLOAD FAILED ==================\n")
        raise Exception(f"GCS upload failed: {str(e)}")

#api end points
@app.get("/api/writing/topics")
def get_writing_topics(
    difficulty: str = Query(...),
    db: Session = Depends(get_db),
):
    print("📥 Fetching Writing topics")
    print(f"   difficulty={difficulty}")

    DB_SUBJECT = "Writing"

    topics = (
        db.query(func.distinct(WritingQuestionBank.topic))
        .filter(
            func.lower(func.trim(WritingQuestionBank.subject)) == DB_SUBJECT.lower(),
            func.lower(func.trim(WritingQuestionBank.difficulty)) == difficulty.lower(),
        )
        .order_by(WritingQuestionBank.topic)
        .all()
    )

    topic_list = [{"name": t[0]} for t in topics]

    print(f"✅ Writing topics found: {len(topic_list)}")

    return topic_list

@app.get("/api/reading/topics")
def get_reading_topics(
    difficulty: str = Query(...),
    db: Session = Depends(get_db),
):
    print("📥 Fetching Reading topics")
    print(f"   difficulty={difficulty}")

    # DB VALUE (even if misspelled)
    DB_SUBJECT = "Reading Comprehension"

    topics = (
        db.query(func.distinct(QuestionReading.topic))
        .filter(
            func.lower(func.trim(QuestionReading.subject)) == DB_SUBJECT.lower(),
            func.lower(func.trim(QuestionReading.difficulty)) == difficulty.lower(),
        )
        .order_by(QuestionReading.topic)
        .all()
    )

    topic_list = [{"name": t[0]} for t in topics]

    print(f"✅ Reading topics found: {len(topic_list)}")

    return topic_list



@app.get("/api/topics")
def get_topics(
    class_name: str = Query(...),
    subject: str = Query(...),
    difficulty: str = Query(...),
    db: Session = Depends(get_db),
):
    print("📥 Fetching topics with filters:")
    print(f"   class_name={class_name}")
    print(f"   subject={subject}")
    print(f"   difficulty={difficulty}")

    normalized_subject = subject.lower().replace("_", " ")

    topics = (
        db.query(func.distinct(Question.topic))
        .filter(
            func.lower(func.trim(Question.class_name)) == class_name.lower(),
            func.lower(func.trim(Question.subject)) == normalized_subject,
            func.lower(func.trim(Question.difficulty)) == difficulty.lower(),
            Question.topic.isnot(None),
            Question.topic != "",
        )
        .order_by(Question.topic)
        .all()
    )


    topic_list = [{"name": t[0]} for t in topics]

    print(f"✅ Topics found: {len(topic_list)}")

    return topic_list


@app.post("/api/admin/bulk-users-exam-module")
async def bulk_users_exam_module(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    print("📥 Bulk user upload request received")
    print(f"📄 Uploaded filename: {file.filename}")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read CSV file")

    required_columns = {
        "student_id",
        "password",
        "name",
        "parent_email",
        "class_name",
        "class_day",
    }

    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing)}",
        )

    # 🔑 Get max numeric ID (ignore non-numeric / legacy UUIDs)
    last_id_int = (
        db.query(func.max(cast(Student.id, Integer)))
        .filter(Student.id.op("~")("^[0-9]+$"))
        .scalar()
    ) or 0

    next_id = last_id_int + 1

    print(f"🔢 Last ID in DB (numeric): {last_id_int}")
    print(f"🆕 Starting next_id: {next_id}")

    success = 0
    failed = 0
    errors = []

    for index, row in df.iterrows():
        print(
            f"➡️ Row {index + 2} | Assigning id={next_id}, "
            f"student_id={row['student_id']}"
        )

        try:
            student = Student(
                id=str(next_id),  # STRING id, sequential
                student_id=str(row["student_id"]).strip(),
                password=str(row["password"]).strip(),
                name=str(row["name"]).strip(),
                parent_email=str(row["parent_email"]).strip(),
                class_name=str(row["class_name"]).strip(),
                class_day=(
                    str(row["class_day"]).strip()
                    if not pd.isna(row["class_day"])
                    else None
                ),
            )

            db.add(student)
            db.commit()
            db.refresh(student)

            success += 1

        except IntegrityError:
            db.rollback()
            failed += 1
            errors.append(
                {
                    "row": index + 2,
                    "student_id": row["student_id"],
                    "error": "Integrity error (likely duplicate id or student_id)",
                }
            )

        except Exception as e:
            db.rollback()
            failed += 1
            errors.append(
                {
                    "row": index + 2,
                    "student_id": row["student_id"],
                    "error": str(e),
                }
            )

        finally:
            next_id += 1   # ✅ ALWAYS increment

    print("📊 Upload summary")
    print(f"   ✅ Success: {success}")
    print(f"   ❌ Failed: {failed}")

    print("❌ BULK UPLOAD ERRORS:")
    for err in errors:
        print(err)

    return {
        "success": success,
        "failed": failed,
        "errors": errors,
    }

@app.get("/api/topics-exam-setup", response_model=List[str])
def get_distinct_topics_exam_setup(
    class_name: str = Query(...),
    subject: str = Query(...),
    difficulty: str = Query(...),
    db: Session = Depends(get_db)
):
    print("\n================ TOPICS EXAM SETUP =================")
    print("class_name:", class_name)
    print("subject:", subject)
    print("difficulty:", difficulty)

    topics = (
        db.query(distinct(Question.topic))
        .filter(
            func.lower(Question.class_name) == class_name.lower(),
            func.lower(Question.subject) == subject.lower(),
            func.lower(Question.difficulty) == difficulty.lower(),
            Question.topic.isnot(None),
            Question.topic != ""
        )
        .order_by(Question.topic)
        .all()
    )

    return [t[0] for t in topics]


@app.get("/api/admin/students/{student_id}/selective-report-dates")
def get_selective_report_dates(
    student_id: str,
    db: Session = Depends(get_db)
):
    rows = (
        db.query(func.date(AdminExamReport.created_at))
        .filter(AdminExamReport.student_id == student_id)
        .distinct()
        .order_by(func.date(AdminExamReport.created_at).desc())
        .all()
    )

    return [r[0].isoformat() for r in rows]
@app.post("/api/admin/students/{student_id}/overall-selective-report")
def generate_overall_selective_report(
    student_id: str,
    exam_date: date,
    db: Session = Depends(get_db)
):
    """
    Generate (once) or fetch Overall Selective Readiness Report
    for a student on a given exam date.
    """

    # --------------------------------------------------
    # 1️⃣ Guard: already exists?
    # --------------------------------------------------
    existing = (
        db.query(AdminOverallSelectiveReport)
        .filter(
            AdminOverallSelectiveReport.student_id == student_id,
            AdminOverallSelectiveReport.exam_date == exam_date
        )
        .first()
    )

    if existing:
        return existing

    # --------------------------------------------------
    # 2️⃣ Fetch per-exam admin reports for that date
    # --------------------------------------------------
    reports = (
        db.query(AdminExamReport)
        .filter(
            AdminExamReport.student_id == student_id,
            func.date(AdminExamReport.created_at) == exam_date
        )
        .all()
    )

    if len(reports) < 4:
        raise HTTPException(
            status_code=400,
            detail="Overall readiness can be generated only after all four exams are completed."
        )

    # --------------------------------------------------
    # 3️⃣ Normalize exam types (🔥 FIX)
    # --------------------------------------------------
    def normalize_exam_type(raw: str) -> str:
        raw = raw.lower().strip()
        if raw == "thinking skills":
            return "thinking_skills"
        return raw

    reports_by_subject = {
        normalize_exam_type(r.exam_type): r for r in reports
    }

    required_subjects = {
        "reading",
        "mathematical_reasoning",
        "thinking_skills",
        "writing"
    }

    missing = required_subjects - reports_by_subject.keys()
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required exam reports: {', '.join(missing)}"
        )

    
    # --------------------------------------------------
    # 4️⃣ Compute normalized overall score (FIXED)
    # --------------------------------------------------
    MAX_SCORES = {
        "reading": 100,
        "mathematical_reasoning": 100,
        "thinking_skills": 100,
        "writing": 20,
    }
    
    # Raw scores (keep for overrides + storage)
    components = {
        subject: reports_by_subject[subject].overall_score
        for subject in MAX_SCORES
    }
    
    # Normalize all components to percentage
    normalized_components = {
        subject: round(
            (components[subject] / MAX_SCORES[subject]) * 100,
            2
        )
        for subject in components
    }
    
    # Equal-weight average of normalized scores
    overall_percent = round(
        sum(normalized_components.values()) / len(normalized_components),
        2
    )

    # --------------------------------------------------
    # 5️⃣ Map to readiness band
    # --------------------------------------------------
    if overall_percent >= 80:
        band = "Band 1 – Fully Selective Ready"
    elif overall_percent >= 70:
        band = "Band 2 – Strong Selective Potential"
    elif overall_percent >= 60:
        band = "Band 3 – Borderline Selective"
    else:
        band = "Band 4 – Not Selective Ready Yet"

    # --------------------------------------------------
    # 6️⃣ School recommendation matrix
    # --------------------------------------------------
    school_map = {
        "Band 1 – Fully Selective Ready": [
            "James Ruse",
            "Baulkham Hills",
            "North Sydney Boys/Girls",
            "Sydney Grammar (Academic profile)"
        ],
        "Band 2 – Strong Selective Potential": [
            "Hornsby Girls",
            "Chatswood",
            "Ryde",
            "Sydney Technical",
            "Parramatta",
            "Penrith Selective"
        ],
        "Band 3 – Borderline Selective": [
            "Local / partially selective schools",
            "Lower competition selective schools"
        ],
        "Band 4 – Not Selective Ready Yet": [
            "Focus on foundations",
            "Reassess selective pathway",
            "Consider enrichment before exam prep"
        ]
    }

    school_recommendation = school_map[band]

    # --------------------------------------------------
    # 7️⃣ Override rules (CRITICAL SAFETY)
    # --------------------------------------------------
    override_flag = False
    override_message = None

    if components["writing"] < 60:
        override_flag = True
        override_message = (
            "While the overall score is competitive, improvement in Writing "
            "is required for higher-tier selective schools."
        )

    for subject, score in components.items():
        if score < 55:
            override_flag = True
            override_message = (
                f"While the overall score is competitive, improvement in "
                f"{subject.replace('_', ' ').title()} is required for higher-tier selective schools."
            )
            break

    # --------------------------------------------------
    # 8️⃣ Store snapshot (IMMUTABLE)
    # --------------------------------------------------
    overall_report = AdminOverallSelectiveReport(
        student_id=student_id,
        exam_date=exam_date,
        overall_percent=overall_percent,
        readiness_band=band,
        school_recommendation=school_recommendation,
        override_flag=override_flag,
        override_message=override_message,
        components=components
    )

    db.add(overall_report)
    db.commit()
    db.refresh(overall_report)

    return overall_report


@app.get("/api/admin/students/{student_id}/selective-reports")
def get_student_selective_reports(
    student_id: str,
    exam_date: date,   # ✅ REQUIRED
    db: Session = Depends(get_db)
): 
    """
    Fetch all selective exam reports for a given student.
    Read-only aggregation over admin reporting tables.
    """

    # --------------------------------------------------
    # 1️⃣ Fetch admin exam reports for student
    # --------------------------------------------------
    reports = (
        db.query(AdminExamReport)
        .filter(
            AdminExamReport.student_id == student_id,
            func.date(AdminExamReport.created_at) == exam_date
        )
        .order_by(AdminExamReport.created_at.desc())
        .all()
    )

    if not reports:
        return []

    report_ids = [r.id for r in reports]

    # --------------------------------------------------
    # 2️⃣ Fetch section results
    # --------------------------------------------------
    sections = (
        db.query(AdminExamSectionResult)
        .filter(AdminExamSectionResult.admin_report_id.in_(report_ids))
        .all()
    )

    sections_by_report = defaultdict(list)
    for s in sections:
        sections_by_report[s.admin_report_id].append({
            "section_name": s.section_name,
            "raw_score": s.raw_score,
            "performance_band": s.performance_band,
            "strengths": s.strengths_summary,
            "improvements": s.improvement_summary
        })

    # --------------------------------------------------
    # 3️⃣ Fetch readiness rules (audit trail)
    # --------------------------------------------------
    rules = (
        db.query(AdminReadinessRuleApplied)
        .filter(AdminReadinessRuleApplied.admin_report_id.in_(report_ids))
        .all()
    )

    rules_by_report = defaultdict(list)
    for r in rules:
        rules_by_report[r.admin_report_id].append({
            "rule_code": r.rule_code,
            "result": r.rule_result,
            "description": r.rule_description
        })

    # --------------------------------------------------
    # 4️⃣ Shape final response
    # --------------------------------------------------
    response = []

    for report in reports:
        response.append({
            "id": report.id,                          # 🔑 REQUIRED
            "exam_type": report.exam_type,
            "exam_attempt_id": report.exam_attempt_id,
            "overall_score": report.overall_score,
            "readiness_band": report.readiness_band,
            "school_guidance_level": report.school_guidance_level,
            "summary_notes": report.summary_notes,
            "exam_date": report.created_at.date().isoformat(),  # 🔑 REQUIRED
            "disclaimer": (
                "This report is advisory only and does not guarantee placement."
            ),
    
            # ✅ sections are ALREADY dicts → just pass them through
            "sections": sections_by_report.get(report.id, []),
    
            # ✅ readiness rules are ALREADY dicts
            "readiness_rules": rules_by_report.get(report.id, [])
        })


    return response

@app.get("/api/admin/students")
def get_admin_students(db: Session = Depends(get_db)):
    students = (
        db.query(Student.student_id, Student.name)
        .order_by(Student.student_id)
        .all()
    )

    return [
        {
            "student_id": student_id,
            "name": name
        }
        for student_id, name in students
    ]


@app.get("/api/admin/students/{student_id}")
def get_student_details(student_id: str, db: Session = Depends(get_db)):
    student = (
        db.query(Student)
        .filter(Student.student_id == student_id)
        .first()
    )

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    return {
        "student_id": student.student_id,
        "name": student.name,
        "class_name": student.class_name,
        "class_day": student.class_day,
        "parent_email": student.parent_email,
    }
@app.get("/students/{student_id}/selective-reports")
def get_student_selective_reports(
    student_id: str,
    db: Session = Depends(get_db)
):
    """
    Return all selective readiness reports for a student.
    Reports are immutable snapshots.
    """

    reports = (
        db.query(AdminExamReport)
        .filter(AdminExamReport.student_id == student_id)
        .order_by(AdminExamReport.created_at.desc())
        .all()
    )

    response = []

    for report in reports:
        sections = (
            db.query(AdminExamSectionResult)
            .filter(
                AdminExamSectionResult.exam_attempt_id == report.exam_attempt_id
            )
            .order_by(AdminExamSectionResult.section_name)
            .all()
        )

        response.append({
            "id": report.id,
            "exam_date": report.created_at.date().isoformat(),
            "readiness_band": report.readiness_band,
            "school_guidance_level": report.school_guidance_level,
            "sections": [
                {
                    "section_name": s.section_name,
                    "performance_band": s.performance_band,
                    "strengths_summary": s.strengths_summary,
                    "improvement_summary": s.improvement_summary,
                }
                for s in sections_by_report.get(report.id, [])
            ],
            "disclaimer": (
                "This report is advisory only and does not guarantee placement."
            )
        })

    return response
 
@app.delete("/delete_student_exam_module/{id}")
def delete_student_exam_module(
    id: str,   # 🔴 CHANGE HERE (int → str)
    db: Session = Depends(get_db)
):
    print("\n================ DELETE STUDENT (EXAM MODULE) ================")
    print("➡ student id:", id)

    student = (
        db.query(Student)
        .filter(Student.id == id)
        .first()
    )

    if not student:
        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    db.delete(student)
    db.commit()

    return {
        "message": "Student deleted successfully",
        "deleted_id": id
    }

def generate_ai_questions_strict_Mathematical_Reasoning(
    *,
    quiz,
    topic_name,
    ai_count,
    db,
    max_attempts=6,
    batch_size=3
):
    collected = []
    attempts = 0

    location = db.query(FranchiseLocation).first()
    if not location:
        raise HTTPException(500, "No franchise location found")

    while len(collected) < ai_count and attempts < max_attempts:
        remaining = ai_count - len(collected)
        request_count = min(batch_size, remaining)
        attempts += 1

        print(
            f"[AI GEN] Attempt {attempts} | "
            f"Requesting {request_count} | "
            f"Collected so far: {len(collected)}"
        )

        system_prompt = (
            "You are an expert exam generator.\n"
            f"Create {request_count} MCQs.\n\n"
            f"Class: {quiz.class_name}\n"
            f"Subject: {quiz.subject}\n"
            f"Topic: {topic_name}\n"
            f"Country: {location.country}\n"
            f"State: {location.state}\n\n"
            "Rules:\n"
            "- Exactly one correct option\n"
            "- 4 options per question\n\n"
            "Return STRICT JSON ONLY:\n"
            "[{\"question\":\"...\",\"options\":[\"A\",\"B\",\"C\",\"D\"],\"correct\":\"A\"}]"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.4
        )

        raw_output = response.choices[0].message.content.strip()

        try:
            batch = json.loads(raw_output)
        except Exception as e:
            print("[AI ERROR] JSON parse failed:", str(e))
            continue

        valid = [
            q for q in batch
            if isinstance(q.get("question"), str)
            and isinstance(q.get("options"), list)
            and len(q["options"]) >= 4
            and q.get("correct") in ["A", "B", "C", "D"]
        ]

        print(f"[AI GEN] Valid questions received: {len(valid)}")
        collected.extend(valid)

    if len(collected) < ai_count:
        raise HTTPException(
            status_code=500,
            detail=(
                f"AI failed to generate required questions for topic "
                f"'{topic_name}'. Expected {ai_count}, got {len(collected)}."
            )
        )

    return collected[:ai_count]



@app.post("/api/quizzes/generate")
def generate_exam(
    payload: dict = Body(...),
    db: Session = Depends(get_db)
):
    difficulty = payload.get("difficulty")

    if not difficulty:
        raise HTTPException(
            status_code=400,
            detail="Difficulty is required"
        )

    # --------------------------------------------------
    # 1️⃣ Fetch latest quiz by subject + difficulty
    # --------------------------------------------------
    quiz = (
        db.query(QuizMathematicalReasoning)
        .filter(
            QuizMathematicalReasoning.subject == "mathematical_reasoning",
            QuizMathematicalReasoning.difficulty == difficulty
        )
        .order_by(QuizMathematicalReasoning.id.desc())
        .first()
    )

    if not quiz:
        raise HTTPException(
            status_code=404,
            detail="No Mathematical Reasoning quiz found"
        )

    # --------------------------------------------------
    # 2️⃣ PRE-FLIGHT DB VALIDATION (CRITICAL)
    # --------------------------------------------------
    for topic in quiz.topics:
        topic_name = topic["name"]
        db_required = int(topic.get("db", 0))

        available = (
            db.query(Question)
              .filter(func.lower(Question.topic) == topic_name.lower())
              .count()
        )

        if available < db_required:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot generate exam. "
                    f"Topic '{topic_name}' requires {db_required} DB questions, "
                    f"but only {available} exist. "
                    f"Please add more questions to the database."
                )
            )

    # --------------------------------------------------
    # 3️⃣ Generate questions (DB + AI)
    # --------------------------------------------------
    questions = []
    q_id = 1

    for topic in quiz.topics:
        topic_name = topic["name"]
        ai_count = int(topic.get("ai", 0))
        db_count = int(topic.get("db", 0))

        # ---- DB questions
        db_questions = (
            db.query(Question)
              .filter(func.lower(Question.topic) == topic_name.lower())
              .order_by(func.random())
              .limit(db_count)
              .all()
        )

        for q in db_questions:
            questions.append({
                "q_id": q_id,
                "topic": topic_name,
                "question": q.question_text,
                "options": q.options,
                "correct": q.correct_answer,
                "images": q.images or []
            })
            q_id += 1

        # ---- AI questions (STRICT + RETRY SAFE)
        if ai_count > 0:
            ai_questions = generate_ai_questions_strict_Mathematical_Reasoning(
                quiz=quiz,
                topic_name=topic_name,
                ai_count=ai_count,
                db=db
            )

            for item in ai_questions:
                questions.append({
                    "q_id": q_id,
                    "topic": topic_name,
                    "question": item["question"],
                    "options": item["options"],
                    "correct": item["correct"],
                    "images": []
                })
                q_id += 1

    # --------------------------------------------------
    # 4️⃣ FINAL TOTAL VALIDATION (NON-NEGOTIABLE)
    # --------------------------------------------------
    expected_total = sum(
        int(t.get("db", 0)) + int(t.get("ai", 0))
        for t in quiz.topics
    )

    actual_total = len(questions)

    if actual_total != expected_total:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Exam generation failed. "
                f"Expected {expected_total} questions, "
                f"but generated {actual_total}. "
                f"This indicates a generation inconsistency."
            )
        )

    # --------------------------------------------------
    # 5️⃣ Clear previous exams (SAFE NOW)
    # --------------------------------------------------
    exam_ids_subq = (
        db.query(Exam.id)
        .filter(Exam.subject == quiz.subject)
        .subquery()
    )

    db.query(StudentExam).filter(
        StudentExam.exam_id.in_(exam_ids_subq)
    ).delete(synchronize_session=False)

    db.query(Exam).filter(
        Exam.subject == quiz.subject
    ).delete(synchronize_session=False)

    db.commit()

    # --------------------------------------------------
    # 6️⃣ Save exam
    # --------------------------------------------------
    new_exam = Exam(
        quiz_id=quiz.id,
        class_name=quiz.class_name,
        subject=quiz.subject,
        difficulty=quiz.difficulty,
        questions=questions,
    )

    db.add(new_exam)
    db.commit()
    db.refresh(new_exam)

    # --------------------------------------------------
    # 7️⃣ Response
    # --------------------------------------------------
    return {
        "message": "Exam generated successfully",
        "exam_id": new_exam.id,
        "quiz_id": quiz.id,
        "class_name": quiz.class_name,
        "subject": quiz.subject,
        "difficulty": quiz.difficulty,
        "total_questions": len(questions),
        "questions": questions,
    }

@app.post("/api/exams/generate-thinking-skills")
def generate_thinking_skills_exam(
    payload: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Generate a Thinking Skills exam based on difficulty only.
    """

    difficulty = payload.get("difficulty")

    if not difficulty:
        raise HTTPException(
            status_code=400,
            detail="Difficulty is required to generate the exam"
        )
     # --------------------------------------------------
    # 0️⃣ Clear previous Thinking Skills exams
    # --------------------------------------------------
    # 1️⃣ Delete student exams for thinking skills
    # ✅ STEP 1: delete dependent StudentExam rows (SAFE)
    exam_ids_subq = db.query(Exam.id).filter(
        Exam.subject == "thinking_skills"
    ).subquery()
    
    db.query(StudentExam).filter(
        StudentExam.exam_id.in_(exam_ids_subq)
    ).delete(synchronize_session=False)
    
    # ✅ STEP 2: delete Exam rows
    db.query(Exam).filter(
        Exam.subject == "thinking_skills"
    ).delete(synchronize_session=False)
    
    db.commit()


    # --------------------------------------------------
    # 1️⃣ Fetch latest Thinking Skills quiz for difficulty
    # --------------------------------------------------
    quiz = (
        db.query(Quiz)
        .filter(Quiz.subject == "thinking_skills")
        .order_by(Quiz.id.desc())
        .first()
    ) 

    if not quiz:
        raise HTTPException(
            status_code=404,
            detail="No Thinking Skills quiz found for the given difficulty"
        )

    # --------------------------------------------------
    # 2️⃣ Generate exam questions
    # --------------------------------------------------
    try:
        questions = generate_exam_questions(quiz, db)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate Thinking Skills exam: {str(e)}"
        )

    if not questions:
        raise HTTPException(
            status_code=500,
            detail="No questions generated for Thinking Skills exam"
        )

    # --------------------------------------------------
    # 3️⃣ Save generated exam
    # --------------------------------------------------
    new_exam = Exam(
        quiz_id=quiz.id,
        class_name=quiz.class_name,
        subject=quiz.subject,
        difficulty=quiz.difficulty,
        questions=questions
    )

    db.add(new_exam)
    db.commit()
    db.refresh(new_exam)

    # --------------------------------------------------
    # 4️⃣ Return response
    # --------------------------------------------------
    return {
        "message": "Thinking Skills exam generated successfully",
        "exam_id": new_exam.id,
        "quiz_id": quiz.id,
        "class_name": quiz.class_name,
        "subject": quiz.subject,
        "difficulty": quiz.difficulty,
        "total_questions": len(questions),
        "questions": questions
    }

@app.post("/api/generate-exam-mathematical-reasoning")
def generate_exam_mathematical_reasoning(
    db: Session = Depends(get_db)
):
    """
    Generate Mathematical Reasoning exam based on latest quiz configuration
    """

    # --------------------------------------------------
    # 1️⃣ Fetch latest Mathematical Reasoning quiz
    # --------------------------------------------------
    quiz = (
        db.query(Quiz)
        .filter(Quiz.subject == "mathematical_reasoning")
        .order_by(Quiz.id.desc())
        .first()
    )

    if not quiz:
        raise HTTPException(
            status_code=404,
            detail="No Mathematical Reasoning quiz configuration found"
        )

    # --------------------------------------------------
    # 2️⃣ Clear previous Mathematical Reasoning exams
    # --------------------------------------------------
    exam_ids_subq = (
        db.query(Exam.id)
        .filter(Exam.subject == "mathematical_reasoning")
        .subquery()
    )

    # Delete dependent student exam attempts first
    db.query(StudentExam).filter(
        StudentExam.exam_id.in_(exam_ids_subq)
    ).delete(synchronize_session=False)

    # Delete previous exams
    db.query(Exam).filter(
        Exam.subject == "mathematical_reasoning"
    ).delete(synchronize_session=False)

    db.commit()

    # --------------------------------------------------
    # 3️⃣ Generate exam questions
    # --------------------------------------------------
    try:
        questions = (quiz, db)
    except Exception as e:
        print("❌ Question generation failed:", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate exam questions"
        )

    if not questions:
        raise HTTPException(
            status_code=500,
            detail="No questions generated"
        )

    # --------------------------------------------------
    # 4️⃣ Save generated exam
    # --------------------------------------------------
    new_exam = Exam(
        quiz_id=quiz.id,
        class_name=quiz.class_name,
        subject=quiz.subject,
        difficulty=quiz.difficulty,
        questions=questions
    )

    db.add(new_exam)
    db.commit()
    db.refresh(new_exam)

    # --------------------------------------------------
    # 5️⃣ Frontend-ready response
    # --------------------------------------------------
    return {
        "message": "Exam generated successfully",
        "exam_id": new_exam.id,
        "quiz_id": quiz.id,
        "class_name": quiz.class_name,
        "subject": quiz.subject,
        "difficulty": quiz.difficulty,
        "total_questions": len(questions),
        "questions": questions
    }


@app.get("/api/quizzes/thinking-skills/difficulty")
def get_thinking_skills_difficulty(db: Session = Depends(get_db)):
    """
    Return the difficulty level for the latest Thinking Skills quiz.
    """

    quiz = (
        db.query(Quiz)
        .filter(Quiz.subject == "thinking_skills")
        .order_by(Quiz.created_at.desc())
        .first()
    )

    if not quiz:
        raise HTTPException(
            status_code=404,
            detail="No Thinking Skills quiz found"
        )

    return quiz.difficulty

@app.post("/api/quizzes/mathematical-reasoning")
def create_quiz_mathematical_reasoning(
    quiz: QuizMathematicalReasoningCreate,
    db: Session = Depends(get_db)
):
    """
    Create a Mathematical Reasoning quiz (admin specification).
    Stored in quiz_mathematical_reasoning table.
    """

    print("\n========== MATHEMATICAL REASONING QUIZ CREATION START ==========")

    print("🔍 Incoming payload:", quiz)

    # Convert to dict for debugging
    try:
        quiz_dict = quiz.dict()
        print("📦 Parsed quiz payload:", quiz_dict)
    except Exception as e:
        print("❌ Invalid payload")
        raise HTTPException(status_code=400, detail=str(e))

    # Enforce subject correctness
    if quiz.subject != "mathematical_reasoning":
        raise HTTPException(
            status_code=400,
            detail="Invalid subject. Expected 'mathematical_reasoning'."
        )

    # Validate topics
    if not isinstance(quiz.topics, list):
        raise HTTPException(
            status_code=400,
            detail="topics must be a list"
        )

    print("📝 Topics count:", len(quiz.topics))
    for i, t in enumerate(quiz.topics):
        print(f"   └─ Topic {i}: {t}")

    try:
        print("\n--- Deleting existing Mathematical Reasoning quizzes ---")

        db.query(QuizMathematicalReasoning).delete()
        db.commit()

        print("🗑️ All previous quizzes deleted")
     
        print("\n--- Creating SQLAlchemy object ---")

        new_quiz = QuizMathematicalReasoning(
            class_name=quiz.class_name.strip(),
            subject="mathematical_reasoning",  # forced
            difficulty=quiz.difficulty.strip(),
            num_topics=quiz.num_topics,
            topics=[t.dict() for t in quiz.topics]
        )

        db.add(new_quiz)
        db.commit()
        db.refresh(new_quiz)

        print("✅ Quiz saved with ID:", new_quiz.id)
        print("========== QUIZ CREATION COMPLETE ==========\n")

        return {
            "message": "Mathematical Reasoning quiz created successfully",
            "quiz_id": new_quiz.id
        }

    except Exception as e:
        print("❌ DB error:", str(e))
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Error creating Mathematical Reasoning quiz"
        )

@app.post("/api/quizzes/mathematical-reasoning")
def create_topic_config_mathematical_reasoning(
    payload: TopicConfigMathematicalReasoningCreate,
    db: Session = Depends(get_db)
):
    """
    Create Mathematical Reasoning topic configuration.
    """

    print("\n========== MATHEMATICAL REASONING TOPIC CONFIG ==========")

    try:
        # Debug incoming payload
        payload_dict = payload.dict()
        print("📦 Incoming payload:", payload_dict)

        # Validate subject explicitly
        if payload.subject != "mathematical_reasoning":
            raise HTTPException(
                status_code=400,
                detail="Invalid subject. Expected 'mathematical_reasoning'."
            )

        # Create DB record
        quiz_config = QuizMathematicalReasoning(
            class_name=payload.class_name.strip(),
            subject=payload.subject.strip(),
            difficulty=payload.difficulty.strip(),
            num_topics=payload.num_topics,
            topics=[t.dict() for t in payload.topics]
        )

        db.add(quiz_config)
        db.commit()
        db.refresh(quiz_config)


        print(
            "✅ Saved topicConfig_mathematical_reasoning | ID:",
            quiz_config.id
        )
        print("=======================================================\n")

        return {
            "message": "Mathematical Reasoning topic config saved successfully",
            "topic_config_id": quiz_config.id
        }

    except HTTPException:
        raise

    except Exception as e:
        print("❌ ERROR saving Mathematical Reasoning topic config")
        print("Error:", str(e))
        traceback.print_exc()

        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Error saving Mathematical Reasoning topic configuration"
        )


@app.get("/get_all_students_exam_module")
def get_all_students_exam_module(
    db: Session = Depends(get_db)
):
    students = db.query(Student).order_by(Student.student_id.asc()).all()

    return [
        {
            "id": s.id,
            "student_id": s.student_id,
            "name": s.name,
            "parent_email": s.parent_email,
            "class_name": s.class_name,
            "class_day": s.class_day
            # ❌ password intentionally excluded
        }
        for s in students
    ]
 
@app.put("/edit_student_exam_module")
def edit_student_exam_module(
    payload: UpdateStudentRequest,
    db: Session = Depends(get_db)
):
    student = (
        db.query(Student)
        .filter(Student.student_id == payload.student_id)
        .first()
    )

    if not student:
        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    # --------------------------------------------------
    # Update fields ONLY if provided
    # --------------------------------------------------
    if payload.name is not None:
        student.name = payload.name

    if payload.parent_email is not None:
        student.parent_email = payload.parent_email

    if payload.class_name is not None:
        student.class_name = payload.class_name

    if payload.class_day is not None:
        student.class_day = payload.class_day

    if payload.password is not None:
        student.password = payload.password  # plain text (intentional)

    db.commit()
    db.refresh(student)

    return {
        "message": "Student updated successfully",
        "student_id": student.student_id
    }
 
@app.get("/api/exams/reading-report")
def get_reading_report(
    session_id: int = Query(..., description="Exam session ID"),
    db: Session = Depends(get_db)
):
    print("\n================ GET READING REPORT ================")
    print("🆔 session_id:", session_id)

    # --------------------------------------------------
    # 1️⃣ Load session
    # --------------------------------------------------
    session = (
        db.query(StudentExamReading)
        .filter(StudentExamReading.id == session_id)
        .first()
    )

    if not session:
        print("❌ Session not found")
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.finished:
        print("❌ Session not finished yet")
        raise HTTPException(status_code=400, detail="Exam not finished yet")

    if not session.report_json:
        print("❌ report_json missing")
        raise HTTPException(status_code=404, detail="Report not available")

    print("✅ Report found")

    # --------------------------------------------------
    # 2️⃣ Return authoritative report
    # --------------------------------------------------
    return session.report_json


@app.get("/api/student/exam-report/thinking-skills")
def get_thinking_skills_report(
    student_id: str = Query(..., description="External student id e.g. Gem002"),
    db: Session = Depends(get_db)
):
    # --------------------------------------------------
    # 1️⃣ Resolve student
    # --------------------------------------------------
    student = (
        db.query(Student)
        .filter(Student.student_id == student_id)
        .first()
    )

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # --------------------------------------------------
    # 2️⃣ Get latest completed THINKING SKILLS attempt
    # --------------------------------------------------
    attempt = (
        db.query(StudentExamThinkingSkills)
        .filter(
            StudentExamThinkingSkills.student_id == student.id,
            StudentExamThinkingSkills.completed_at.isnot(None)
        )
        .order_by(StudentExamThinkingSkills.completed_at.desc())
        .first()
    )

    if not attempt:
        raise HTTPException(
            status_code=404,
            detail="No completed Thinking Skills exam found"
        )

    # --------------------------------------------------
    # 3️⃣ Load all responses for this attempt
    # --------------------------------------------------
    responses = (
        db.query(StudentExamResponseThinkingSkills)
        .filter(
            StudentExamResponseThinkingSkills.exam_attempt_id == attempt.id
        )
        .all()
    )

    if not responses:
        raise HTTPException(
            status_code=404,
            detail="No responses found for Thinking Skills exam"
        )

    # --------------------------------------------------
    # 4️⃣ OVERALL SUMMARY (Report B)
    # --------------------------------------------------
    total_questions = len(responses)
    attempted = sum(1 for r in responses if r.is_correct is not None)
    correct = sum(1 for r in responses if r.is_correct is True)
    incorrect = sum(1 for r in responses if r.is_correct is False)
    not_attempted = total_questions - attempted

    accuracy_percent = round((correct / attempted) * 100, 2) if attempted else 0
    score_percent = round((correct / total_questions) * 100, 2) if total_questions else 0

    overall = {
        "total_questions": total_questions,
        "attempted": attempted,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "accuracy_percent": accuracy_percent,
        "score_percent": score_percent,
        "pass": None
    }

    # --------------------------------------------------
    # 5️⃣ TOPIC-WISE PERFORMANCE (Report A)
    # --------------------------------------------------
    topic_map = {}

    for r in responses:
        topic = r.topic or "Unknown"

        if topic not in topic_map:
            topic_map[topic] = {
                "topic": topic,
                "total": 0,
                "attempted": 0,
                "correct": 0,
                "incorrect": 0,
                "not_attempted": 0
            }

        topic_map[topic]["total"] += 1

        if r.is_correct is None:
            topic_map[topic]["not_attempted"] += 1
        else:
            topic_map[topic]["attempted"] += 1
            if r.is_correct:
                topic_map[topic]["correct"] += 1
            else:
                topic_map[topic]["incorrect"] += 1

    topic_wise_performance = list(topic_map.values())

    # --------------------------------------------------
    # 6️⃣ TOPIC ACCURACY & RESULT (Report C)
    # --------------------------------------------------
    topic_accuracy = []

    for t in topic_wise_performance:
        attempted_t = t["attempted"]
        accuracy_t = (
            round((t["correct"] / attempted_t) * 100, 2)
            if attempted_t else 0
        )
        score_t = round((t["correct"] / t["total"]) * 100, 2)

        topic_accuracy.append({
            "topic": t["topic"],
            "total_questions": t["total"],
            "attempted": attempted_t,
            "correct": t["correct"],
            "incorrect": t["incorrect"],
            "accuracy_percent": accuracy_t,
            "score_percent": score_t,
            "pass": None
        })

    # --------------------------------------------------
    # 7️⃣ IMPROVEMENT AREAS (Report D)
    # --------------------------------------------------
    improvement_areas = []

    for t in topic_accuracy:
        limited_data = t["total_questions"] < 5

        improvement_areas.append({
            "topic": t["topic"],
            "accuracy_percent": t["accuracy_percent"],
            "score_percent": t["score_percent"],
            "total_questions": t["total_questions"],
            "limited_data": limited_data
        })

    improvement_areas.sort(key=lambda x: x["accuracy_percent"])

    # --------------------------------------------------
    # 8️⃣ Final response
    # --------------------------------------------------
    return {
        "overall": overall,                                 # Report B
        "topic_wise_performance": topic_wise_performance,   # Report A
        "topic_accuracy": topic_accuracy,                   # Report C
        "improvement_areas": improvement_areas              # Report D
    }

@app.get("/api/student/exam-report/mathematical-reasoning")
def get_thinking_skills_report(
    student_id: str = Query(..., description="External student id e.g. Gem002"),
    db: Session = Depends(get_db)
):
    # --------------------------------------------------
    # 1️⃣ Resolve student
    # --------------------------------------------------
    student = (
        db.query(Student)
        .filter(Student.student_id == student_id)
        .first()
    )
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # --------------------------------------------------
    # 2️⃣ Get latest completed attempt
    # --------------------------------------------------
    attempt = (
        db.query(StudentExam)
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
    # 3️⃣ Load all responses for this attempt
    # --------------------------------------------------
    responses = (
        db.query(StudentExamResponse)
        .filter(StudentExamResponse.exam_attempt_id == attempt.id)
        .all()
    )

    if not responses:
        raise HTTPException(status_code=404, detail="No responses found")

    # --------------------------------------------------
    # 4️⃣ OVERALL SUMMARY (Report B)
    # --------------------------------------------------
    total_questions = len(responses)
    attempted = sum(1 for r in responses if r.is_correct is not None)
    correct = sum(1 for r in responses if r.is_correct is True)
    incorrect = sum(1 for r in responses if r.is_correct is False)
    not_attempted = total_questions - attempted

    accuracy_percent = round((correct / attempted) * 100, 2) if attempted else 0
    score_percent = round((correct / total_questions) * 100, 2) if total_questions else 0

    overall = {
        "total_questions": total_questions,
        "attempted": attempted,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "accuracy_percent": accuracy_percent,
        "score_percent": score_percent,
        "pass": None  # set rule later if required
    }

    # --------------------------------------------------
    # 5️⃣ TOPIC-WISE PERFORMANCE (Report A)
    # --------------------------------------------------
    topic_map = {}

    for r in responses:
        topic = r.topic
        if topic not in topic_map:
            topic_map[topic] = {
                "topic": topic,
                "total": 0,
                "attempted": 0,
                "correct": 0,
                "incorrect": 0,
                "not_attempted": 0
            }

        topic_map[topic]["total"] += 1

        if r.is_correct is None:
            topic_map[topic]["not_attempted"] += 1
        else:
            topic_map[topic]["attempted"] += 1
            if r.is_correct:
                topic_map[topic]["correct"] += 1
            else:
                topic_map[topic]["incorrect"] += 1

    topic_wise_performance = list(topic_map.values())

    # --------------------------------------------------
    # 6️⃣ TOPIC ACCURACY & RESULT (Report C)
    # --------------------------------------------------
    topic_accuracy = []

    for t in topic_wise_performance:
        attempted_t = t["attempted"]
        accuracy_t = (
            round((t["correct"] / attempted_t) * 100, 2)
            if attempted_t else 0
        )
        score_t = round((t["correct"] / t["total"]) * 100, 2)

        topic_accuracy.append({
            "topic": t["topic"],
            "total_questions": t["total"],
            "attempted": attempted_t,
            "correct": t["correct"],
            "incorrect": t["incorrect"],
            "accuracy_percent": accuracy_t,
            "score_percent": score_t,
            "pass": None
        })

    # --------------------------------------------------
    # 7️⃣ IMPROVEMENT AREAS (Report D)
    # --------------------------------------------------
    improvement_areas = []

    for t in topic_accuracy:
        limited_data = t["total_questions"] < 5

        improvement_areas.append({
            "topic": t["topic"],
            "accuracy_percent": t["accuracy_percent"],
            "score_percent": t["score_percent"],
            "total_questions": t["total_questions"],
            "limited_data": limited_data
        })

    improvement_areas.sort(key=lambda x: x["accuracy_percent"])

    # --------------------------------------------------
    # 8️⃣ Final response
    # --------------------------------------------------
    return {
        "overall": overall,                         # Report B
        "topic_wise_performance": topic_wise_performance,  # Report A
        "topic_accuracy": topic_accuracy,           # Report C
        "improvement_areas": improvement_areas      # Report D
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

def generate_admin_exam_report_reading(
    db: Session,
    student: Student,
    exam_attempt: StudentExamReading,
    overall_accuracy: float,
    topic_stats: dict
):
    """
    Generates an immutable admin report snapshot for a completed Reading exam.
    MUST NOT commit. Caller controls the transaction.
    """

    # -------------------------------
    # 1️⃣ Determine readiness & guidance
    # -------------------------------
    if overall_accuracy >= 80:
        readiness_band = "Strong Selective Potential"
        school_guidance = "Mid-tier Selective"
    elif overall_accuracy >= 60:
        readiness_band = "Borderline Selective"
        school_guidance = "Selective Preparation Required"
    else:
        readiness_band = "Not Yet Selective Ready"
        school_guidance = "General Stream Recommended"

    # -------------------------------
    # 2️⃣ Create admin report snapshot
    # -------------------------------
    admin_report = AdminExamReport(
        student_id=student.student_id,  # external ID (STRING)
        exam_attempt_id=exam_attempt.id,
        exam_type="reading",
        overall_score=overall_accuracy,
        readiness_band=readiness_band,
        school_guidance_level=school_guidance,
        summary_notes=f"Reading accuracy: {overall_accuracy}%"
    )

    db.add(admin_report)
    db.flush()  # obtain admin_report.id safely

    # -------------------------------
    # 3️⃣ Create section (topic) results
    # -------------------------------
    for topic, stats in topic_stats.items():
        attempted = stats.get("attempted", 0)
        correct = stats.get("correct", 0)

        accuracy = (
            round((correct / attempted) * 100, 2)
            if attempted > 0 else 0.0
        )

        if accuracy >= 80:
            band = "A"
        elif accuracy >= 60:
            band = "B"
        else:
            band = "C"

        section_result = AdminExamSectionResult(
            admin_report_id=admin_report.id,
            section_name=topic,
            raw_score=accuracy,
            performance_band=band,
            strengths_summary=None,
            improvement_summary=None
        )

        db.add(section_result)

    # -------------------------------
    # 4️⃣ Store applied rule (audit safety)
    # -------------------------------
    rule_applied = AdminReadinessRuleApplied(
        admin_report_id=admin_report.id,
        rule_code="READING_ACCURACY",
        rule_description="Reading accuracy used for readiness classification",
        rule_result="passed" if overall_accuracy >= 60 else "failed"
    )

    db.add(rule_applied)


 
@app.post("/api/exams/submit-reading")
def submit_reading_exam(payload: dict, db: Session = Depends(get_db)):

    print("\n================ SUBMIT READING EXAM ================")
    print("📥 Raw payload received:", payload)

    # --------------------------------------------------
    # 0️⃣ Validate payload
    # --------------------------------------------------
    session_id = payload.get("session_id")
    answers = payload.get("answers", {})

    print("🆔 session_id:", session_id)
    print("📝 answers keys:", list(answers.keys()) if isinstance(answers, dict) else answers)

    if not session_id:
        print("❌ ERROR: session_id missing")
        raise HTTPException(status_code=400, detail="session_id is required")

    if not isinstance(answers, dict):
        print("❌ ERROR: answers is not a dict")
        raise HTTPException(status_code=400, detail="answers must be a dictionary")

    # --------------------------------------------------
    # 1️⃣ Load session
    # --------------------------------------------------
    session = (
        db.query(StudentExamReading)
        .filter(StudentExamReading.id == session_id)
        .first()
    )

    if not session:
        print("❌ ERROR: Session not found for session_id:", session_id)
        raise HTTPException(status_code=404, detail="Session not found")

    print("✅ Session loaded:", {
        "session_id": session.id,
        "student_id": session.student_id,
        "exam_id": session.exam_id,
        "finished": session.finished
    })

    if session.finished:
        print("⚠️ Session already finished — idempotent exit")
        return {
            "status": "ok",
            "message": "Exam already submitted"
        }
    
    # --------------------------------------------------
    # Resolve student (external → internal)
    # --------------------------------------------------
    student = (
        db.query(Student)
        .filter(Student.student_id == session.student_id)
        .first()
    )
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # --------------------------------------------------
    # 2️⃣ Load exam
    # --------------------------------------------------
    exam = (
        db.query(GeneratedExamReading)
        .filter(GeneratedExamReading.id == session.exam_id)
        .first()
    )

    if not exam or not exam.exam_json:
        print("❌ ERROR: Exam or exam_json missing for exam_id:", session.exam_id)
        raise HTTPException(status_code=500, detail="Exam data not found")

    sections = exam.exam_json.get("sections", [])
    print("📚 Sections found:", len(sections))

    # --------------------------------------------------
    # 3️⃣ Initialize counters
    # --------------------------------------------------
    topic_stats = {}
    total_questions = attempted = correct = incorrect = not_attempted = 0

    # --------------------------------------------------
    # 4️⃣ Evaluate questions
    # --------------------------------------------------
    for section_idx, section in enumerate(sections):
        topic = section.get("topic", "Other")
        questions = section.get("questions", [])

        print(f"\n📂 Section {section_idx + 1}: {topic}")
        print("❓ Questions in section:", len(questions))

        if topic not in topic_stats:
            topic_stats[topic] = {
                "total": 0,
                "attempted": 0,
                "correct": 0,
                "incorrect": 0,
                "not_attempted": 0
            }

        for q_idx, q in enumerate(questions):
            question_id = q.get("question_id")
            correct_answer = q.get("correct_answer")
            selected_answer = answers.get(question_id)

            print(f"   🔹 Q{q_idx + 1} | ID:", question_id)
            print("      ➜ correct_answer:", correct_answer)
            print("      ➜ selected_answer:", selected_answer)

            is_attempted = selected_answer is not None
            is_correct = is_attempted and selected_answer == correct_answer

            total_questions += 1
            topic_stats[topic]["total"] += 1

            if is_attempted:
                attempted += 1
                topic_stats[topic]["attempted"] += 1
                if is_correct:
                    correct += 1
                    topic_stats[topic]["correct"] += 1
                    print("      ✅ Correct")
                else:
                    incorrect += 1
                    topic_stats[topic]["incorrect"] += 1
                    print("      ❌ Incorrect")
            else:
                not_attempted += 1
                topic_stats[topic]["not_attempted"] += 1
                print("      ⚠️ Not Attempted")

            # Persist per-question row
            db.add(StudentExamReportReading(
                student_id=session.student_id,
                exam_id=session.exam_id,
                session_id=session.id,
                topic=topic,
                question_id=question_id,
                selected_answer=selected_answer,
                correct_answer=correct_answer,
                is_correct=is_correct
            ))

    # --------------------------------------------------
    # 5️⃣ Build report
    # --------------------------------------------------
    print("\n📊 Building report summary")

    topics_report = []
    for topic, stats in topic_stats.items():
        accuracy = (
            round((stats["correct"] / stats["attempted"]) * 100, 2)
            if stats["attempted"] > 0 else 0.0
        )

        print(f"   📌 {topic} →", stats, "accuracy:", accuracy)

        topics_report.append({
            "topic": topic,
            **stats,
            "accuracy": accuracy
        })

    overall_accuracy = round((correct / attempted) * 100, 2) if attempted > 0 else 0.0
    # --------------------------------------------------
    # ADMIN RAW SCORE SNAPSHOT (Reading) — SAFE VERSION
    # --------------------------------------------------
    existing_raw_score = (
        db.query(AdminExamRawScore)
        .filter(AdminExamRawScore.exam_attempt_id == session.id)
        .first()
    )
    
    if existing_raw_score:
        # Update existing snapshot (idempotent)
        existing_raw_score.student_id = student.id
        existing_raw_score.subject = "reading"
        existing_raw_score.total_questions = total_questions
        existing_raw_score.correct_answers = correct
        existing_raw_score.wrong_answers = incorrect
        existing_raw_score.accuracy_percent = overall_accuracy
    else:
        # Insert new snapshot
        admin_raw_score = AdminExamRawScore(
            student_id=student.id,
            exam_attempt_id=session.id,
            subject="reading",
            total_questions=total_questions,
            correct_answers=correct,
            wrong_answers=incorrect,
            accuracy_percent=overall_accuracy
        )
        db.add(admin_raw_score)

    report_json = {
        "overall": {
            "total_questions": total_questions,
            "attempted": attempted,
            "correct": correct,
            "incorrect": incorrect,
            "not_attempted": not_attempted,
            "accuracy": overall_accuracy
        },
        "topics": topics_report,
        "improvement_order": [
            t["topic"] for t in sorted(topics_report, key=lambda x: x["accuracy"])
        ]
    }

    print("\n📈 Final overall report:", report_json["overall"])
    print("📉 Improvement order:", report_json["improvement_order"])

    # --------------------------------------------------
    # 6️⃣ Mark session finished
    # --------------------------------------------------
    session.finished = True
    session.completed_at = datetime.now(timezone.utc)
    session.report_json = report_json

    if not admin_report_exists(
        db=db,
        exam_attempt_id=session.id
    ):
        generate_admin_exam_report_reading(
            db=db,
            student=student,
            exam_attempt=session,
            overall_accuracy=overall_accuracy,
            topic_stats=topic_stats
        )


    db.commit()

    print("✅ Exam submission committed successfully")
    print("================ END SUBMIT READING EXAM ================\n")

    # --------------------------------------------------
    # 7️⃣ Return response
    # --------------------------------------------------
    return {
        "status": "submitted",
        "message": "Reading exam submitted successfully",
        "report": report_json
    }

from datetime import datetime, timezone
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

@app.post("/api/exams/start-reading")
def start_reading_exam(
    student_id: str,
    db: Session = Depends(get_db)
):
    """
    Start or resume a Reading Comprehension exam session.
    Assumes exam_json is already frontend-safe and self-contained.
    """

    print("\n==================== START READING EXAM ====================")
    print("Student ID:", student_id)

    # --------------------------------------------------
    # 1) Load latest generated reading exam
    # --------------------------------------------------
    exam = (
        db.query(GeneratedExamReading)
        .order_by(GeneratedExamReading.id.desc())
        .first()
    )

    if not exam:
        raise HTTPException(status_code=404, detail="No reading exam found")

    print("Loaded exam ID:", exam.id)

    exam_json = exam.exam_json or {}

    # --------------------------------------------------
    # 2) Check existing student session
    # --------------------------------------------------
    session = (
        db.query(StudentExamReading)
        .filter(
            StudentExamReading.student_id == student_id,
            StudentExamReading.exam_id == exam.id
        )
        .first()
    )

    # Always use timezone-aware UTC
    now = datetime.now(timezone.utc)

    # --------------------------------------------------
    # 3) Resume existing session
    # --------------------------------------------------
    if session:
        print("Existing session found:", session.id)

        start_time = session.started_at

        # Normalize old rows that may be timezone-naive
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        return {
            "session_id": session.id,
            "exam_json": exam_json,
            "duration_minutes": exam_json.get("duration_minutes", 40),
            "start_time": start_time,
            "server_now": now,
            "finished": bool(session.finished)
        }

    # --------------------------------------------------
    # 4) Create new session
    # --------------------------------------------------
    new_session = StudentExamReading(
        student_id=student_id,
        exam_id=exam.id,
        started_at=now
    )

    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    print("New session created:", new_session.id)

    return {
        "session_id": new_session.id,
        "exam_json": exam_json,
        "duration_minutes": exam_json.get("duration_minutes", 40),
        "start_time": new_session.started_at,
        "server_now": now,
        "finished": False
    }



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

    # --------------------------------------------------
    # 1️⃣ Get current writing exam
    # --------------------------------------------------
    exam = (
        db.query(GeneratedExamWriting)
        .filter(GeneratedExamWriting.is_current == True)
        .order_by(GeneratedExamWriting.created_at.desc())
        .first()
    )

    if not exam:
        raise HTTPException(404, "No active writing exam")

    # --------------------------------------------------
    # 2️⃣ Fetch latest attempt for THIS exam
    # --------------------------------------------------
    attempt = (
        db.query(StudentExamWriting)
        .filter(
            StudentExamWriting.student_id == student_id,
            StudentExamWriting.exam_id == exam.id
        )
        .order_by(StudentExamWriting.started_at.desc())
        .first()
    )

    # --------------------------------------------------
    # 🟥 CASE A — Exam already completed
    # --------------------------------------------------
    if attempt and attempt.completed_at:
        return {"completed": True}

    # --------------------------------------------------
    # 🟡 CASE B — Resume active attempt
    # --------------------------------------------------
    if attempt and attempt.completed_at is None:
        started_at = attempt.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        elapsed = int((now - started_at).total_seconds())
        remaining = max(0, attempt.duration_minutes * 60 - elapsed)

        if remaining == 0:
            attempt.completed_at = now
            db.commit()
            return {"completed": True}

        return {
            "completed": False,
            "remaining_time": remaining
        }

    # --------------------------------------------------
    # 🔵 CASE C — Start new writing attempt
    # --------------------------------------------------
    new_attempt = StudentExamWriting(
        student_id=student_id,
        exam_id=exam.id,
        started_at=datetime.now(timezone.utc),
        duration_minutes=exam.duration_minutes
    )

    db.add(new_attempt)
    db.commit()
    db.refresh(new_attempt)

    return {
        "completed": False,
        "remaining_time": new_attempt.duration_minutes * 60
    }



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
    student_id: str,  # kept for API compatibility
    db: Session = Depends(get_db)
):
    print("\n================ SUBMIT WRITING EXAM =================")
    print("➡️ Incoming student_id:", student_id)
    print("➡️ Payload received:", payload.dict())

    # --------------------------------------------------
    # 1️⃣ Load active writing attempt
    # --------------------------------------------------
    print("🔍 Looking for active writing attempt...")

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
        print("❌ No active writing exam found for student_id:", student_id)
        raise HTTPException(
            status_code=404,
            detail="No active writing exam found"
        )

    print("✅ Writing attempt found:", {
        "attempt_id": exam_state.id,
        "exam_id": exam_state.exam_id,
        "started_at": exam_state.started_at
    })

    # --------------------------------------------------
    # 2️⃣ Persist student answer
    # --------------------------------------------------
    print("💾 Saving student answer...")
    print("✏️ Essay length:", len(payload.answer_text or ""))

    exam_state.answer_text = payload.answer_text
    exam_state.completed_at = datetime.now(timezone.utc)

    # --------------------------------------------------
    # 3️⃣ Evaluate essay using OpenAI
    # --------------------------------------------------
    print("🤖 Preparing OpenAI prompt...")

    prompt = f"""
You are an automated exam marking system.

IMPORTANT RULES (MUST FOLLOW STRICTLY):
- Respond with ONLY a valid JSON object
- Do NOT include markdown
- Do NOT include ```json or ``` blocks
- Do NOT include explanations or extra text
- Output must be directly parsable by json.loads()

Required JSON schema:
{{
  "score": <integer between 0 and 20>,
  "strengths": "<short sentence>",
  "improvements": "<short sentence>"
}}

Evaluate the following student essay for selective school readiness writing quality.

Essay:
{payload.answer_text}
"""


    try:
        print("🚀 Calling OpenAI Responses API...")

        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0.4
        )

        print("🧠 Raw OpenAI response object:", response)

        ai_result = response.output_text
        print("🧠 OpenAI output_text:")
        print(ai_result)

    except Exception as e:
        print("❌ OpenAI evaluation failed!")
        print("❌ Exception type:", type(e))
        print("❌ Exception details:", str(e))
        raise HTTPException(
            status_code=500,
            detail="Writing evaluation failed"
        )

    # --------------------------------------------------
    # 4️⃣ Parse AI response
    # --------------------------------------------------
    print("🧪 Parsing AI response as JSON...")

    try:
        import json
        evaluation = json.loads(ai_result)

        writing_score = int(evaluation.get("score", 0))
        strengths = evaluation.get("strengths", "")
        improvements = evaluation.get("improvements", "")

        print("✅ Parsed AI evaluation:", {
            "score": writing_score,
            "strengths": strengths,
            "improvements": improvements
        })

    except Exception as e:
        print("❌ Failed to parse AI response as JSON")
        print("❌ Raw AI text was:")
        print(ai_result)
        print("❌ Exception:", str(e))
        raise HTTPException(
            status_code=500,
            detail="Invalid AI evaluation format"
        )

    # --------------------------------------------------
    # 5️⃣ Store AI evaluation on writing attempt
    # --------------------------------------------------
    print("💾 Storing AI evaluation on writing attempt...")

    exam_state.ai_score = writing_score
    exam_state.ai_strengths = strengths
    exam_state.ai_improvements = improvements

    # --------------------------------------------------
    # 6️⃣ Admin RAW SCORE snapshot
    # --------------------------------------------------
    print("📊 Creating AdminExamRawScore snapshot...")

    admin_raw_score = AdminExamRawScore(
        student_id=exam_state.student_id,
        exam_attempt_id=exam_state.id,
        subject="writing",
        total_questions=20,
        correct_answers=writing_score,
        wrong_answers=20 - writing_score,
        accuracy_percent=round((writing_score / 20) * 100, 2)
    )

    db.add(admin_raw_score)
    print("✅ Admin raw score prepared:", {
        "student_id": admin_raw_score.student_id,
        "attempt_id": admin_raw_score.exam_attempt_id,
        "accuracy": admin_raw_score.accuracy_percent
    })
    # --------------------------------------------------
    # 6️⃣➕ Writing Admin Report Snapshot (FINAL STEP)
    # --------------------------------------------------
    print("📄 Generating Writing admin report snapshot...")
    
    # --------------------------------------------------
    # A) Idempotency guard (CRITICAL)
    # --------------------------------------------------
    existing_report = (
        db.query(AdminExamReport)
        .filter(
            AdminExamReport.exam_attempt_id == exam_state.id
        )
        .first()
    )

    
    if existing_report:
        print("⚠️ Writing admin report already exists. Skipping snapshot generation.")
    else:
        # --------------------------------------------------
        # B) Determine performance band & readiness
        # --------------------------------------------------
        if writing_score >= 15:
            performance_band = "Strong"
            readiness_status = "Ready"
            guidance_text = "Student demonstrates strong writing skills suitable for selective readiness."
        elif writing_score >= 10:
            performance_band = "Developing"
            readiness_status = "Borderline"
            guidance_text = "Student shows developing writing skills and may benefit from targeted practice."
        else:
            performance_band = "Weak"
            readiness_status = "Not Ready"
            guidance_text = "Student requires significant improvement in writing fundamentals."
    
        print("🧠 Writing classification:", {
            "score": writing_score,
            "band": performance_band,
            "readiness": readiness_status
        })
    
        # --------------------------------------------------
        # C) Insert admin_exam_reports
        # --------------------------------------------------
        admin_report = AdminExamReport(
            exam_attempt_id=exam_state.id,
            student_id=exam_state.student_id,
            exam_type="writing",
            overall_score=writing_score,
            readiness_band=readiness_status,
            school_guidance_level=guidance_text,
            summary_notes=f"Writing score: {writing_score}/20"
        )


    
        db.add(admin_report)
        db.flush()  # 🔑 Needed to get admin_report.id
    
        print("✅ AdminExamReport created:", {
            "report_id": admin_report.id
        })
    
        # --------------------------------------------------
        # D) Insert admin_exam_section_results (single section)
        # --------------------------------------------------
        section_result = AdminExamSectionResult(
            admin_report_id=admin_report.id,
            section_name="Writing",
            raw_score=writing_score,
            performance_band=performance_band,
            strengths_summary=strengths,
            improvement_summary=improvements
        )

    
        db.add(section_result)
    
        print("✅ Writing section result stored")
    
        # --------------------------------------------------
        # E) Insert admin_readiness_rules_applied (audit trail)
        # --------------------------------------------------
        readiness_rule = AdminReadinessRuleApplied(
           admin_report_id=admin_report.id,
           rule_code="writing_score_threshold",
           rule_result=readiness_status,
           rule_description=f"Writing score {writing_score}/20 classified as {performance_band}"
        )

    
        db.add(readiness_rule)
    
        print("✅ Readiness rule audit stored")
    
    
        # --------------------------------------------------
        # 7️⃣ Commit atomically
        # --------------------------------------------------
        print("💾 Committing transaction to database...")
    
    try:
        db.commit()
        print("✅ Database commit successful")
    except Exception as e:
        print("❌ Database commit failed!")
        print("❌ Exception:", str(e))
        db.rollback()
        raise
    
    print("🎉 Writing exam submitted and evaluated successfully")
    print("====================================================\n")

    return {
        "message": "Writing exam submitted successfully",
        "exam_id": exam_state.exam_id,
        "writing_score": writing_score
    }



@app.get("/api/exams/writing/current")
def get_current_writing_exam(student_id: str, db: Session = Depends(get_db)):

    # 1️⃣ Get latest writing attempt (completed OR active)
    session = (
        db.query(StudentExamWriting)
        .filter(StudentExamWriting.student_id == student_id)
        .order_by(StudentExamWriting.started_at.desc())
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="No writing exam attempt found"
        )

    # 2️⃣ Completed → redirect
    if session.completed_at:
        return {"completed": True}

    # 3️⃣ Load exam
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

    # 4️⃣ Calculate remaining time
    started_at = session.started_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    elapsed = int((now - started_at).total_seconds())
    total = session.duration_minutes * 60
    remaining = max(0, total - elapsed)

    # 5️⃣ Timeout → auto-complete
    if remaining == 0:
        session.completed_at = now
        db.commit()
        return {"completed": True}

    # 6️⃣ Active attempt
    return {
        "completed": False,
        "remaining_seconds": remaining,
        "exam": {
            "exam_id": exam.id,
            "difficulty": exam.difficulty,
            "question_text": exam.question_text,
            "duration_minutes": session.duration_minutes
        }
    }


@app.post("/api/exams/generate-writing")
def generate_exam_writing(
    payload: WritingGenerateSchema,
    db: Session = Depends(get_db)
):
    # Normalize incoming values to lowercase
    class_name = payload.class_name.strip().lower()
    difficulty = payload.difficulty.strip().lower()
    db.query(GeneratedExamWriting).delete(synchronize_session=False)
    db.commit()

    # 1️⃣ Fetch ONE writing question with full case-insensitive matching
    question = (
        db.query(WritingQuestionBank)
        .filter(
            func.trim(func.lower(WritingQuestionBank.class_name)) == class_name,
            func.trim(func.lower(WritingQuestionBank.difficulty)) == difficulty
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
    print("\n--- Deleting previous Writing quiz setups ---")

    db.query(QuizSetupWriting).delete(synchronize_session=False)
    db.commit()

    print("🗑️ Previous writing quiz setups deleted")
    print("\n--- Creating new Writing quiz setup ---")
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


# ------------------------------------------------------------
# 🔧 Helper: Build sections with questions (FOUNDATIONAL)
# ------------------------------------------------------------
def build_sections_with_questions(exam_json):
    print("\n🧠 Normalizing sections from exam_json...")

    sections_meta = exam_json.get("sections", [])
    questions = exam_json.get("questions", [])

    print("   • sections_meta count:", len(sections_meta))
    print("   • questions count:", len(questions))

    section_map = {}

    # ✅ Initialize sections using DIFFICULTY (not name)
    for s in sections_meta:
        difficulty = s.get("difficulty")
        topic = s.get("topic")

        if not difficulty:
            print("⚠️ Skipping section with no difficulty:", s)
            continue

        section_map[difficulty] = {
            "difficulty": difficulty,
            "topic": topic,
            "time": s.get("time", 0),
            "intro": s.get("intro", ""),
            "questions": []
        }

    # ✅ Assign questions to sections
    for q in questions:
        section_key = q.get("section")

        if section_key in section_map:
            section_map[section_key]["questions"].append(q)
        else:
            print("⚠️ Question with unknown section:", q.get("section"))

    sections = list(section_map.values())
    print("📦 Sections normalized:", len(sections))

    for sec in sections:
        print(
            f"   • {sec['difficulty']} | topic={sec['topic']} | "
            f"questions={len(sec['questions'])}"
        )

    return sections



@app.post("/api/student/start-exam/foundational-skills")
def start_or_resume_foundational_exam(
    payload: StartExamRequestFoundational,
    db: Session = Depends(get_db)
):
    print("\n" + "=" * 70)
    print("🚀 START-EXAM (FOUNDATIONAL) REQUEST RECEIVED")
    print("=" * 70)

    # ------------------------------------------------------------
    # 0️⃣ Payload
    # ------------------------------------------------------------
    print("🧾 Raw payload:", payload)
    student_id = payload.student_id
    print("👤 student_id:", student_id)

    # ------------------------------------------------------------
    # 1️⃣ Fetch latest attempt
    # ------------------------------------------------------------
    print("\n🔍 Fetching latest exam attempt...")
    attempt = (
        db.query(StudentsExamFoundational)
        .filter(StudentsExamFoundational.student_id == student_id)
        .order_by(StudentsExamFoundational.started_at.desc())
        .first()
    )

    print("📌 Attempt found:", bool(attempt))
    if attempt:
        print("   • attempt.id:", attempt.id)
        print("   • completed_at:", attempt.completed_at)
        print("   • current_section_index:", attempt.current_section_index)

    if attempt and attempt.completed_at:
        print("✅ Attempt already completed — returning early")
        return {"completed": True}

    # ------------------------------------------------------------
    # 2️⃣ Load current exam
    # ------------------------------------------------------------
    print("\n📘 Fetching current GeneratedExamFoundational...")
    exam = (
        db.query(GeneratedExamFoundational)
        .filter(GeneratedExamFoundational.is_current == True)
        .order_by(desc(GeneratedExamFoundational.created_at))
        .first()
    )

    if not exam:
        print("❌ ERROR: No active exam found")
        raise HTTPException(404, "No active exam")

    print("✅ Exam loaded")
    print("   • exam.id:", exam.id)
    print("   • class_name:", exam.class_name)
    print("   • subject:", exam.subject)
    print("   • duration_minutes:", exam.duration_minutes)
    print("   • exam_json keys:", exam.exam_json.keys())

    # ------------------------------------------------------------
    # 3️⃣ Normalize sections + questions
    # ------------------------------------------------------------
    print("\n🧠 Normalizing sections from exam_json...")
    sections = build_sections_with_questions(exam.exam_json)

    print("📦 Sections normalized:", len(sections))
    for i, sec in enumerate(sections):
        print(f"   Section {i} keys:", sec.keys())

    if not sections:
        print("❌ ERROR: Sections list is empty after normalization")
        raise HTTPException(500, "No sections after normalization")

    # ------------------------------------------------------------
    # 4️⃣ Create attempt if needed
    # ------------------------------------------------------------
    if not attempt:
        print("\n🆕 Creating new exam attempt...")
        attempt = StudentsExamFoundational(
            student_id=student_id,
            exam_id=exam.id,
            started_at=datetime.now(timezone.utc),
            current_section_index=0
        )
        db.add(attempt)
        db.commit()
        db.refresh(attempt)

        print("✅ Attempt created")
        print("   • attempt.id:", attempt.id)

    # ------------------------------------------------------------
    # 5️⃣ Validate section index
    # ------------------------------------------------------------
    print("\n🧪 Validating section index...")
    print("   • current_section_index:", attempt.current_section_index)
    print("   • total sections:", len(sections))

    if attempt.current_section_index >= len(sections):
        print("❌ ERROR: current_section_index out of bounds")
        raise HTTPException(500, "Invalid section index")

    # ------------------------------------------------------------
    # 6️⃣ Timer calculation
    # ------------------------------------------------------------
    print("\n⏱ Calculating remaining time...")
    elapsed = int((datetime.now(timezone.utc) - attempt.started_at).total_seconds())
    print("   • elapsed seconds:", elapsed)

    if exam.duration_minutes is None:
        print("❌ ERROR: exam.duration_minutes is None")
        raise HTTPException(500, "Invalid exam duration")

    remaining_time = max(exam.duration_minutes * 60 - elapsed, 0)
    print("   • remaining_time (seconds):", remaining_time)

    # ------------------------------------------------------------
    # 7️⃣ Normalize questions for frontend
    # ------------------------------------------------------------
    print("\n🧩 Preparing current section for frontend...")
    current_section = sections[attempt.current_section_index]

    print("📌 Current section object:", current_section)
    print("📌 Current section keys:", current_section.keys())

    questions = current_section.get("questions")
    if questions is None:
        print("❌ ERROR: current_section['questions'] is None")
        raise HTTPException(500, "Section has no questions key")

    print("📊 Question count in section:", len(questions))

    normalized_questions = []
    for idx, q in enumerate(questions):
        print(f"   → Normalizing question {idx + 1}")
        fixed = dict(q)

        opts = fixed.get("options")

        if isinstance(opts, dict) and opts:
           fixed["options"] = [f"{k}) {v}" for k, v in opts.items()]
        else:
           # 🔒 Guarantee at least 1 safe option to avoid frontend crash
           fixed["options"] = ["A) Option unavailable"]


        normalized_questions.append(fixed)

    # ------------------------------------------------------------
    # 8️⃣ Final response
    # ------------------------------------------------------------
    print("\n✅ START-EXAM RESPONSE READY")
    print("   • section name:", current_section.get("difficulty"))
    print("   • questions returned:", len(normalized_questions))
    print("=" * 70 + "\n")

    return {
        "completed": False,
        "current_section_index": attempt.current_section_index,
        "section": {
            "name": current_section["difficulty"],
            "questions": normalized_questions
        },
        "remaining_time": remaining_time
    }



@app.post("/api/exams/foundational/next-section")
def advance_foundational_section(
    student_id: str = Query(...),
    db: Session = Depends(get_db)
):
    print("\n================ NEXT-SECTION (FOUNDATIONAL) ================")
    print("👤 student_id:", student_id)

    # 1️⃣ Get active attempt
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
        raise HTTPException(status_code=404, detail="No active attempt")

    print("🧪 attempt.id:", attempt.id)
    print("   ↳ current_section_index:", attempt.current_section_index)

    # 2️⃣ Load exam snapshot
    exam = db.query(GeneratedExamFoundational).get(attempt.exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    exam_json = exam.exam_json

    # 3️⃣ Build sections DIRECTLY from saved exam_json
    sections_meta = exam_json.get("sections", [])
    all_questions = exam_json.get("questions", [])

    if not sections_meta or not all_questions:
        raise HTTPException(
            status_code=500,
            detail="Exam data is incomplete"
        )

    # Group questions by section name
    sections = []
    for sec in sections_meta:
        sec_name = sec["difficulty"]  # ✅ FIX
    
        sec_questions = [
            q for q in all_questions
            if q.get("section") == sec_name
        ]
    
        sections.append({
            "name": sec_name,
            "questions": sec_questions
        })


    total_sections = len(sections)
    print("📂 Total sections:", total_sections)

    # 4️⃣ Last section check
    if attempt.current_section_index >= total_sections - 1:
        print("🏁 Last section reached")
        return {"completed": True}

    # 5️⃣ Advance section index
    attempt.current_section_index += 1
    db.commit()
    db.refresh(attempt)

    current_section = sections[attempt.current_section_index]

    print("➡️ Advanced to section:", current_section["name"])
    print("   ↳ questions:", len(current_section["questions"]))

    if not current_section["questions"]:
        print("⚠️ WARNING: Section has no questions")

    print("===========================================================\n")

    return {
        "completed": False,
        "current_section_index": attempt.current_section_index,
        "section": {
            "name": current_section["name"],
            "questions": current_section["questions"]
        }
    }


@app.post("/api/student/finish-exam/foundational-skills")
def finish_foundational_exam(
    payload: FinishExamRequestFoundational,
    db: Session = Depends(get_db)
):
    print("\n================ FINISH EXAM (FOUNDATIONAL) ================")
    print("📥 Payload received:", payload.dict())

    # ------------------------------------------------------------
    # 1️⃣ Fetch active attempt
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

    print("🔍 Active attempt:", attempt)

    if not attempt:
        print("⚠️ No active attempt found")
        return {
            "success": True,
            "message": "Exam already completed or no active attempt"
        }

    # ------------------------------------------------------------
    # 2️⃣ Load exam definition
    # ------------------------------------------------------------
    exam = db.query(GeneratedExamFoundational).get(attempt.exam_id)

    if not exam:
        print("❌ Exam not found for exam_id:", attempt.exam_id)
        return {"success": False, "message": "Exam not found"}

    questions = exam.exam_json.get("questions", [])
    print(f"📄 Questions in exam_json: {len(questions)}")

    # ------------------------------------------------------------
    # 3️⃣ Build question lookup (q_id → metadata)
    # ------------------------------------------------------------
    question_lookup = {}

    for q in questions:
        qid = q.get("q_id")
        if qid is None:
            continue

        question_lookup[qid] = {
            "section": q.get("section"),
            "topic": q.get("topic"),
            "correct_answer": q.get("correct")
        }

    print("🧠 Question lookup keys:", list(question_lookup.keys()))

    # ------------------------------------------------------------
    # 4️⃣ Persist responses (NO attempt-level idempotency guard)
    # ------------------------------------------------------------
    answers = payload.answers or {}
    print("📝 Answers received:", answers)

    inserted_count = 0

    for q_id_raw, selected_answer in answers.items():
        try:
            q_id = int(q_id_raw)
        except (TypeError, ValueError):
            print("❌ Invalid q_id:", q_id_raw)
            continue

        meta = question_lookup.get(q_id)
        if not meta:
            print("❌ No metadata for q_id:", q_id)
            continue

        correct_answer = meta["correct_answer"]
        is_correct = (
            selected_answer.upper() == correct_answer.upper()
            if selected_answer and correct_answer
            else False
        )

        response = StudentExamResponseFoundational(
            student_id=payload.student_id,
            exam_id=attempt.exam_id,
            attempt_id=attempt.id,
            section_name=meta["section"],
            topic=meta["topic"],                 # ✅ CRITICAL FIX
            question_id=q_id,
            selected_answer=selected_answer,
            correct_answer=correct_answer,
            is_correct=is_correct
        )

        db.add(response)
        inserted_count += 1

    print(f"📊 Responses added: {inserted_count}")

    # ------------------------------------------------------------
    # 5️⃣ Finalize attempt
    # ------------------------------------------------------------
    attempt.answers_json = answers
    attempt.completed_at = datetime.now(timezone.utc)
    attempt.completion_reason = payload.reason

    try:
        db.commit()
        print("✅ DB commit successful")
    except Exception as e:
        print("🔥 DB COMMIT FAILED:", str(e))
        db.rollback()
        raise

    print("🔄 Attempt finalized:", attempt.id)
    print("================ END FINISH EXAM =================\n")

    return {
        "success": True,
        "message": "Exam completed successfully"
    }



@app.get("/api/student/exam-report/foundational-skills")
def get_foundational_exam_report(
    student_id: str = Query(...),
    db: Session = Depends(get_db)
):
    print("\n================ EXAM-REPORT (FOUNDATIONAL) ================")
    print("👤 student_id:", student_id)

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

    exam = (
        db.query(GeneratedExamFoundational)
        .filter(GeneratedExamFoundational.id == attempt.exam_id)
        .first()
    )

    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    sections = build_sections_with_questions(exam.exam_json)

    topic_totals = {}
    total_questions = 0

    for sec in sections:
        for q in sec.get("questions", []):
            topic = q.get("topic", "Unknown")
            topic_totals.setdefault(topic, 0)
            topic_totals[topic] += 1
            total_questions += 1

    responses = (
        db.query(StudentExamResponseFoundational)
        .filter(StudentExamResponseFoundational.attempt_id == attempt.id)
        .all()
    )

    topic_stats = {}

    for topic, total in topic_totals.items():
        topic_stats[topic] = {
            "topic": topic,
            "total": total,
            "attempted": 0,
            "correct": 0,
            "incorrect": 0,
            "not_attempted": total
        }

    for r in responses:
        topic = r.topic
        stats = topic_stats.get(topic)

        if not stats:
            continue

        stats["attempted"] += 1
        stats["not_attempted"] -= 1

        if r.is_correct:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1

    topic_wise_performance = list(topic_stats.values())

    attempted = sum(t["attempted"] for t in topic_wise_performance)
    correct = sum(t["correct"] for t in topic_wise_performance)
    incorrect = sum(t["incorrect"] for t in topic_wise_performance)
    not_attempted = total_questions - attempted

    accuracy = round((correct / attempted) * 100, 1) if attempted else 0
    score = round((correct / total_questions) * 100, 1) if total_questions else 0

    improvement_areas = []

    for t in topic_wise_performance:
        acc = round((t["correct"] / t["total"]) * 100, 1) if t["total"] else 0
        improvement_areas.append({
            "topic": t["topic"],
            "accuracy": acc,
            "limited_data": t["total"] < 3
        })

    improvement_areas.sort(key=lambda x: x["accuracy"])

    return {
        "overall": {
            "total_questions": total_questions,
            "attempted": attempted,
            "correct": correct,
            "incorrect": incorrect,
            "not_attempted": not_attempted,
            "accuracy_percent": accuracy,
            "score_percent": score
        },
        "topic_wise_performance": topic_wise_performance,
        "improvement_areas": improvement_areas
    }
 
# def generate_ai_questions_foundational(
#     class_name: str,
#     subject: str,
#     difficulty: str,
#     topic: str,
#     count: int
# ):
#     """
#     Generate AI questions for a Foundational exam section.
#     Uses chunking + retries to reduce JSON corruption.
#     Compatible with current OpenAI Python SDK.
#     """
#
#     print("\n🤖 Calling AI to generate questions")
#     print(f"🤖 class={class_name}, subject={subject}, difficulty={difficulty}, count={count}")
#
#     if count <= 0:
#         return []
#
#     CHUNK_SIZE = 5
#     MAX_RETRIES = 2
#
#     remaining = count
#     all_questions = []
#
#     while remaining > 0:
#         batch_size = min(CHUNK_SIZE, remaining)
#
#         prompt = f"""
# You are a STRICT JSON generator for an automated exam system.
#
# TASK:
# Generate EXACTLY {batch_size} multiple-choice questions.
#
# ABSOLUTE RULES:
# - Output MUST be valid JSON
# - Output MUST be a JSON ARRAY
# - NO explanations
# - NO comments
# - NO markdown
# - NO text outside JSON
#
# REQUIRED FORMAT:
# [
#   {{
#     "question_text": "string",
#     "options": {{
#       "A": "string",
#       "B": "string",
#       "C": "string",
#       "D": "string"
#     }},
#     "correct_answer": "A|B|C|D",
#     "topic": "string"
#   }}
# ]
#
# CONSTRAINTS:
# - Class: {class_name}
# - Subject: {subject}
# - Difficulty: {difficulty}
#
# If you cannot comply perfectly, RETURN [] ONLY.
# """
#
#         success = False
#
#         for attempt in range(1, MAX_RETRIES + 1):
#             print(f"🤖 AI batch {batch_size} | attempt {attempt}")
#
#             try:
#                 response = client.responses.create(
#                     model="gpt-4o-mini",
#                     input=prompt,
#                     temperature=0.4
#                 )
#
#                 raw = response.output_text
#                 print("🤖 Raw AI response:", raw)
#
#                 parsed = json.loads(raw)
#
#                 if not isinstance(parsed, list):
#                     raise ValueError("AI output is not a list")
#
#                 if len(parsed) != batch_size:
#                     raise ValueError(
#                         f"Expected {batch_size} questions, got {len(parsed)}"
#                     )
#
#                 for q in parsed:
#                     all_questions.append({
#                         "section": difficulty,
#                         "question_number": None,
#                         "question_text": q["question_text"],
#                         "options": q["options"],
#                         "correct_answer": q["correct_answer"],
#                         "question_type": "mcq",
#                         "images": [],
#                         "topic": q.get("topic", "AI Generated"),
#                     })
#
#                 remaining -= batch_size
#                 success = True
#                 break
#
#             except Exception as e:
#                 print(f"❌ AI batch failed (attempt {attempt}): {e}")
#
#         if not success:
#             print("❌ AI failed after retries. Stopping generation.")
#             break
#
#     print(f"🤖 Successfully generated {len(all_questions)} AI questions")
#     return all_questions

def is_valid_ai_question(q: dict) -> bool:
    try:
        if not isinstance(q, dict):
            return False

        if not isinstance(q.get("question_text"), str):
            return False

        options = q.get("options")
        if not isinstance(options, dict):
            return False

        if set(options.keys()) != {"A", "B", "C", "D"}:
            return False

        if not all(isinstance(v, str) for v in options.values()):
            return False

        if q.get("correct_answer") not in {"A", "B", "C", "D"}:
            return False

        return True

    except Exception:
        return False


def generate_ai_questions_safely(
    class_name: str,
    subject: str,
    difficulty: str,
    topic: str,
    required_count: int,
):
    """
    Safely generate AI questions with strict validation.
    - Accepts partial success
    - Skips invalid questions
    - Bounded retries
    - NEVER raises on AI failure
    """

    print("\n🤖 GENERATE AI QUESTIONS SAFELY")
    print("--------------------------------------------------")
    print(f"Class      : {class_name}")
    print(f"Subject    : {subject}")
    print(f"Difficulty : {difficulty}")
    print(f"Topic      : {topic}")
    print(f"Required   : {required_count}")
    print("--------------------------------------------------")

    if required_count <= 0:
        return []

    CHUNK_SIZE = 5
    MAX_ATTEMPTS = 5

    valid_questions = []
    attempts = 0

    while len(valid_questions) < required_count and attempts < MAX_ATTEMPTS:
        attempts += 1
        remaining = required_count - len(valid_questions)
        batch_size = min(CHUNK_SIZE, remaining)

        print(f"\n🤖 Attempt {attempts}")
        print(f"➡ Requesting {batch_size} questions")

        prompt = f"""
You are a STRICT JSON generator.

RETURN ONLY VALID JSON.
RETURN ONLY A JSON ARRAY.
RETURN EXACTLY {batch_size} OBJECTS.

FORMAT (STRICT):
[
  {{
    "question_text": "string",
    "options": {{
      "A": "string",
      "B": "string",
      "C": "string",
      "D": "string"
    }},
    "correct_answer": "A|B|C|D"
  }}
]

CONSTRAINTS:
- Class: {class_name}
- Subject: {subject}
- Difficulty: {difficulty}
- Topic: {topic}

RULES:
- No extra keys
- No markdown
- No explanations
- If unsure, return []
"""

        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                temperature=0.3,
            )

            raw = response.output_text
            print("📥 Raw AI response:", raw)

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                print("❌ JSON parse failed:", e)
                continue

            if not isinstance(parsed, list):
                print("❌ AI output is not a list")
                continue

            print(f"📦 Parsed {len(parsed)} candidate questions")

            for idx, q in enumerate(parsed, start=1):
                if len(valid_questions) >= required_count:
                    break

                if is_valid_ai_question(q):
                    valid_questions.append({
                        "section": difficulty,
                        "question_text": q["question_text"],
                        "options": q["options"],
                        "correct_answer": q["correct_answer"],
                        "question_type": "text_mcq",
                        "images": [],
                        "topic": topic,        # 🔒 FORCE section topic
                        "source": "ai",
                    })
                    print(f"✅ Accepted question {len(valid_questions)}")
                else:
                    print(f"⚠️  Rejected invalid question #{idx}")

        except Exception as e:
            print("❌ AI call failed:", e)

    print("\n--------------------------------------------------")
    print(f"🤖 AI generation finished")
    print(f"✔ Valid questions collected: {len(valid_questions)}")
    print(f"⚠ Attempts used: {attempts}/{MAX_ATTEMPTS}")
    print("--------------------------------------------------\n")

    return valid_questions



@app.post("/api/exams/generate-foundational")
def generate_exam_foundational(
    payload: dict = Body(...),
    db: Session = Depends(get_db)
):
    print("\n" + "=" * 70)
    print("📥 GENERATE FOUNDATIONAL EXAM - REQUEST RECEIVED")
    print("=" * 70)

    # ------------------------------------------------------------
    # 0️⃣ Payload validation
    # ------------------------------------------------------------
    print("🧾 Raw payload:", payload)

    class_name = payload.get("class_name")
    if not class_name:
        print("❌ ERROR: class_name missing from payload")
        raise HTTPException(400, "class_name is required")

    print("✅ class_name:", class_name)

    # ------------------------------------------------------------
    # 1️⃣ Clear previous generated exams
    # ------------------------------------------------------------
    print("🧹 Clearing previous GeneratedExamFoundational records...")
    deleted = (
        db.query(GeneratedExamFoundational)
        .filter(func.lower(GeneratedExamFoundational.class_name) == class_name.lower())
        .delete(synchronize_session=False)
    )
    db.commit()
    print(f"🧹 Deleted {deleted} rows")

    # ------------------------------------------------------------
    # 2️⃣ Load latest quiz setup
    # ------------------------------------------------------------
    print("🔍 Fetching latest QuizSetupFoundational...")

    cfg = (
        db.query(QuizSetupFoundational)
        .filter(func.lower(QuizSetupFoundational.class_name) == class_name.lower())
        .order_by(QuizSetupFoundational.id.desc())
        .first()
    )

    if not cfg:
        print(f"❌ ERROR: No quiz setup found for class '{class_name}'")
        raise HTTPException(404, f"No quiz setup found for class '{class_name}'")

    print("✅ Quiz setup found")
    print("   ▸ Config ID:", cfg.id)
    print("   ▸ Subject:", cfg.subject)

    # ------------------------------------------------------------
    # 3️⃣ Build section metadata
    # ------------------------------------------------------------
    print("\n📦 Building section definitions from config...")

    sections = []

    def add_section(label, name, topic, time, intro):
        print(f"➡ Checking {label}...")
        if name:
            print(f"   ✔ Included")
            print(f"   • Difficulty:", name)
            print(f"   • Topic:", topic)
            print(f"   • Time:", time)
            sections.append({
                "difficulty": name,
                "topic": topic,
                "time": time or 0,
                "intro": intro or "",
                "label": label,
            })
        else:
            print(f"   ⏭ Skipped (no name)")

    add_section(
        "Section 1",
        cfg.section1_name,
        cfg.section1_topic,
        cfg.section1_time,
        cfg.section1_intro,
    )

    add_section(
        "Section 2",
        cfg.section2_name,
        cfg.section2_topic,
        cfg.section2_time,
        cfg.section2_intro,
    )

    add_section(
        "Section 3",
        cfg.section3_name,
        cfg.section3_topic,
        cfg.section3_time,
        cfg.section3_intro,
    )

    if not sections:
        print("❌ ERROR: No sections created from config")
        raise HTTPException(400, "No sections defined in quiz setup")

    print(f"✅ Total sections to generate: {len(sections)}")

    # ------------------------------------------------------------
    # 4️⃣ Generate questions per section
    # ------------------------------------------------------------
    final_questions = []
    warnings = []

    for section in sections:
        print("\n" + "-" * 60)
        print(f"📘 GENERATING SECTION: {section['label']}")
        print("-" * 60)

        difficulty = section["difficulty"]
        topic = section["topic"]

        print("➡ Difficulty:", difficulty)
        print("➡ Topic:", topic)

        # Resolve counts
        if difficulty == cfg.section1_name:
            db_required = cfg.section1_db or 0
            ai_required = cfg.section1_ai or 0
            total_required = cfg.section1_total or 0
        elif difficulty == cfg.section2_name:
            db_required = cfg.section2_db or 0
            ai_required = cfg.section2_ai or 0
            total_required = cfg.section2_total or 0
        elif difficulty == cfg.section3_name:
            db_required = cfg.section3_db or 0
            ai_required = cfg.section3_ai or 0
            total_required = cfg.section3_total or 0
        else:
            print("⚠️ WARNING: Difficulty does not match any config section")
            continue

        print("📊 Required counts:")
        print("   • DB:", db_required)
        print("   • AI:", ai_required)
        print("   • TOTAL:", total_required)

        section_questions = []

        # ---------------- DB QUESTIONS ----------------
        if db_required > 0:
            print("\n🗄 Fetching DB questions...")

            pool = (
                db.query(Question)
                .filter(
                    func.lower(Question.class_name) == class_name.lower(),
                    func.lower(Question.subject) == cfg.subject.lower(),
                    func.lower(Question.difficulty) == difficulty.lower(),
                    func.lower(Question.topic) == topic.lower(),
                )
                .all()
            )

            print(f"📦 DB pool size: {len(pool)}")

            random.shuffle(pool)
            chosen = pool[:db_required]

            print(f"🎯 Selected DB questions: {len(chosen)}")

            if len(chosen) < db_required:
                warning = (
                    f"{difficulty} ({topic}): "
                    f"only {len(chosen)} DB questions available"
                )
                print("⚠️ WARNING:", warning)
                warnings.append(warning)

            for q in chosen:
                section_questions.append({
                    "section": difficulty,
                    "topic": q.topic,
                    "question_text": q.question_text,
                    "options": q.options,
                    "correct_answer": q.correct_answer,
                    "question_type": q.question_type,
                    "images": q.images,
                    "source": "db",
                })

        # ---------------- AI QUESTIONS ----------------
        # ---------------- AI QUESTIONS (SAFE) ----------------
        if ai_required > 0:
            print("\n🤖 Generating AI questions (SAFE MODE)...")
            print("   • Difficulty:", difficulty)
            print("   • Topic:", topic)
            print("   • Required count:", ai_required)
        
            ai_questions = generate_ai_questions_safely(
                class_name=class_name,
                subject=cfg.subject,
                difficulty=difficulty,
                topic=topic,
                required_count=ai_required,
            )

        
            print(f"🤖 AI questions accepted: {len(ai_questions)}")
        
            for q in ai_questions:
                q["section"] = difficulty
                q["topic"] = topic
                q["source"] = "ai"
        
            section_questions.extend(ai_questions)
        
        # ---------------- HARD VALIDATION ----------------
        print("\n🧪 Validating section output...")
        
        if not section_questions:
            print(
                f"❌ ERROR: Section '{difficulty}' ({topic}) produced ZERO questions"
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Section '{difficulty}' ({topic}) produced zero questions. "
                    "Exam generation aborted."
                ),
            )
        
        print(f"✅ Section question count: {len(section_questions)}")
        
        if total_required and len(section_questions) != total_required:
            warning = (
                f"{difficulty} ({topic}): expected {total_required}, "
                f"got {len(section_questions)}"
            )
            print("⚠️ WARNING:", warning)
            warnings.append(warning)
        
        final_questions.extend(section_questions)
        # AFTER final_questions is built
 
    for idx, q in enumerate(final_questions, start=1):
            q["q_id"] = idx
    


    # ------------------------------------------------------------
    # 5️⃣ Final validation
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📦 FINAL EXAM SUMMARY")
    print("=" * 70)

    print("🧮 Total questions generated:", len(final_questions))

    if warnings:
        print("⚠️ WARNINGS:")
        for w in warnings:
            print("   -", w)
    else:
        print("✅ No warnings")

    if not final_questions:
        print("❌ ERROR: No questions generated at all")
        raise HTTPException(400, "Exam generation failed")

    print("🎉 EXAM GENERATION SUCCESSFUL")
    print("=" * 70 + "\n")
    
    # ------------------------------------------------------------
    # 6️⃣ Build exam JSON (for persistence)
    # ------------------------------------------------------------
    print("📦 Building exam_json payload...")
    normalized_questions = []

    for q in final_questions:
        normalized_questions.append({
            "q_id": q["q_id"],
            "section": q.get("section"),
            "difficulty": q.get("section"),  # same value in your system
            "topic": q.get("topic"),
            "question": q.get("question_text"),
            "options": q.get("options"),
            "correct": q.get("correct_answer"),
            "images": q.get("images") or []
        })
    
    exam_json = {
        "class_name": class_name,
        "subject": cfg.subject,
        "sections": sections,  # keep full section meta here
        "questions": normalized_questions,
        "total_questions": len(normalized_questions)
    }

    
    print("✅ exam_json built successfully")

    # ------------------------------------------------------------
    # 7️⃣ Persist generated exam
    # ------------------------------------------------------------
    print("💾 Saving generated exam to database...")
    
    saved_exam = GeneratedExamFoundational(
        class_name=class_name,
        subject=cfg.subject,
        total_questions=len(final_questions),
        duration_minutes=sum(s["time"] for s in sections),
        exam_json=exam_json,
        is_current=True,
    )
    
    db.add(saved_exam)
    db.commit()
    db.refresh(saved_exam)
    
    print("✅ Exam saved successfully")
    print("   ▸ Exam ID:", saved_exam.id)


    return {
        "generated_exam_id": saved_exam.id,
        "class_name": class_name,
        "subject": cfg.subject,
        "total_questions": len(final_questions),
        "warnings": warnings,
    }


"""
Before topic names were added to quiz setup foundational
@app.post("/api/exams/generate-foundational")
def generate_exam_foundational(
    payload: dict = Body(...),
    db: Session = Depends(get_db)
):
    '''
    Generate a Foundational exam using the LATEST quiz_setup_foundational
    (highest id) for the given class_name.
    Guarantees:
      - subject is taken from config
      - no empty sections
      - section totals respected
      - runtime-safe exam_json
    '''

    print("\n==============================")
    print("📥 Generate Foundational Exam")
    print("==============================")

    # ------------------------------------------------------------
    # 0️⃣ Validate payload
    # ------------------------------------------------------------
    class_name = payload.get("class_name")
    if not class_name:
        raise HTTPException(400, "class_name is required")

    db.query(GeneratedExamFoundational).delete(synchronize_session=False)
    db.commit()

    # ------------------------------------------------------------
    # 1️⃣ Load LATEST config for class (highest ID)
    # ------------------------------------------------------------
    cfg = (
        db.query(QuizSetupFoundational)
        .filter(func.lower(QuizSetupFoundational.class_name) == class_name.lower())
        .order_by(QuizSetupFoundational.id.desc())
        .first()
    )

    if not cfg:
        raise HTTPException(
            404,
            f"No quiz setup found for class '{class_name}'"
        )

    subject = cfg.subject
    print(f"➡ Using config ID={cfg.id}, subject={subject}")

    # ------------------------------------------------------------
    # 2️⃣ Build sections (from config only)
    # ------------------------------------------------------------
    sections = []

    def add_section(name, time, intro):
        if name:
            sections.append({
                "name": name,
                "difficulty": name,
                "time": time or 0,
                "intro": intro
            })

    add_section(cfg.section1_name, cfg.section1_time, cfg.section1_intro)
    add_section(cfg.section2_name, cfg.section2_time, cfg.section2_intro)
    add_section(cfg.section3_name, cfg.section3_time, cfg.section3_intro)

    if not sections:
        raise HTTPException(400, "No sections defined in config")

    final_questions = []
    warnings = []

    # ------------------------------------------------------------
    # 3️⃣ Generate questions PER SECTION
    # ------------------------------------------------------------
    for section in sections:
        difficulty = section["name"]

        if difficulty == cfg.section1_name:
            db_required = cfg.section1_db or 0
            ai_required = cfg.section1_ai or 0
            total_required = cfg.section1_total or 0
        elif difficulty == cfg.section2_name:
            db_required = cfg.section2_db or 0
            ai_required = cfg.section2_ai or 0
            total_required = cfg.section2_total or 0
        elif difficulty == cfg.section3_name:
            db_required = cfg.section3_db or 0
            ai_required = cfg.section3_ai or 0
            total_required = cfg.section3_total or 0
        else:
            continue

        print(f"\n➡ Section {difficulty}: DB={db_required}, AI={ai_required}, TOTAL={total_required}")

        section_questions = []

        if db_required > 0:
            pool = (
                db.query(Question)
                .filter(
                    func.lower(Question.class_name) == class_name.lower(),
                    func.lower(Question.subject) == subject.lower(),
                    func.lower(Question.difficulty) == difficulty.lower(),
                )
                .all()
            )

            random.shuffle(pool)
            chosen = pool[:db_required]

            if len(chosen) < db_required:
                warnings.append(
                    f"{difficulty}: only {len(chosen)} DB questions available"
                )

            for q in chosen:
                section_questions.append({
                    "section": difficulty,
                    "question_text": q.question_text,
                    "options": q.options,
                    "correct_answer": q.correct_answer,
                    "question_type": q.question_type,
                    "images": q.images,
                    "topic": q.topic,
                })

        if ai_required > 0:
            ai_questions = generate_ai_questions_foundational(
                class_name=class_name,
                subject=subject,
                difficulty=difficulty,
                count=ai_required
            )

            for q in ai_questions:
                q["section"] = difficulty

            section_questions.extend(ai_questions)

        if not section_questions:
            raise HTTPException(
                400,
                f"Section '{difficulty}' produced zero questions. Exam rejected."
            )

        if total_required and len(section_questions) != total_required:
            warnings.append(
                f"{difficulty}: expected {total_required}, got {len(section_questions)}"
            )

        final_questions.extend(section_questions)

    # ------------------------------------------------------------
    # 5️⃣ Number questions
    # ------------------------------------------------------------
    for i, q in enumerate(final_questions, start=1):
        q["q_id"] = i
        q["question_number"] = i

    # ------------------------------------------------------------
    # 6️⃣ Compute duration
    # ------------------------------------------------------------
    duration_minutes = sum(s["time"] for s in sections)
    if duration_minutes <= 0:
        raise HTTPException(400, "Invalid exam duration")

    # ------------------------------------------------------------
    # 7️⃣ Build exam JSON
    # ------------------------------------------------------------
    exam_json = {
        "class_name": class_name,
        "subject": subject,
        "total_questions": len(final_questions),
        "sections": sections,
        "questions": final_questions,
        "duration_minutes": duration_minutes,
    }

    # ------------------------------------------------------------
    # 8️⃣ Persist exam
    # ------------------------------------------------------------
    db.query(GeneratedExamFoundational).update({"is_current": False})

    saved = GeneratedExamFoundational(
        class_name=class_name,
        subject=subject,
        total_questions=len(final_questions),
        duration_minutes=duration_minutes,
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
        "duration_minutes": duration_minutes,
        "warnings": warnings
    }
"""




@app.post("/api/quizzes-foundational")
def save_quiz_setup(
    payload: QuizSetupFoundationalSchema,
    db: Session = Depends(get_db)
):
    print("\n================ SAVE QUIZ SETUP =================")
    print("📥 Raw payload received:")
    print(payload)

    print("\n➡️ Basic metadata:")
    print("class_name:", payload.class_name)
    print("subject:", payload.subject)
    print("number of sections:", len(payload.sections))

    if len(payload.sections) < 2:
        print("❌ ERROR: Less than 2 sections provided")
        raise HTTPException(
            status_code=400,
            detail="At least 2 sections required"
        )

    section1 = payload.sections[0]
    section2 = payload.sections[1]
    section3 = payload.sections[2] if len(payload.sections) == 3 else None

    print("\n➡️ Section breakdown:")
    print("SECTION 1:")
    print("  name:", section1.name)
    print("  topic:", section1.topic)
    print("  ai:", section1.ai)
    print("  db:", section1.db)
    print("  total:", section1.total)
    print("  time:", section1.time)

    print("\nSECTION 2:")
    print("  name:", section2.name)
    print("  topic:", section2.topic)
    print("  ai:", section2.ai)
    print("  db:", section2.db)
    print("  total:", section2.total)
    print("  time:", section2.time)

    if section3:
        print("\nSECTION 3:")
        print("  name:", section3.name)
        print("  topic:", section3.topic)
        print("  ai:", section3.ai)
        print("  db:", section3.db)
        print("  total:", section3.total)
        print("  time:", section3.time)
    else:
        print("\nSECTION 3: Not provided")

    total_questions = (
        section1.total +
        section2.total +
        (section3.total if section3 else 0)
    )

    print("\n➡️ Total questions calculated:", total_questions)

    if section3 is None and total_questions > 40:
        print("❌ ERROR: Total questions exceed 40 with 2 sections")
        raise HTTPException(
            status_code=400,
            detail="Total questions cannot exceed 40 with 2 sections"
        )

    if section3 and total_questions > 50:
        print("❌ ERROR: Total questions exceed 50 with 3 sections")
        raise HTTPException(
            status_code=400,
            detail="Total questions cannot exceed 50 with 3 sections"
        )

    print("\n🧹 Clearing existing quiz setup records...")
    db.query(QuizSetupFoundational).delete(synchronize_session=False)
    db.commit()
    print("✅ Existing records cleared")

    print("\n🛠 Creating QuizSetupFoundational object...")

    quiz = QuizSetupFoundational(
        class_name=payload.class_name,
        subject=payload.subject,

        section1_name=section1.name,
        section1_topic=section1.topic,
        section1_ai=section1.ai,
        section1_db=section1.db,
        section1_total=section1.total,
        section1_time=section1.time,
        section1_intro=section1.intro,

        section2_name=section2.name,
        section2_topic=section2.topic,
        section2_ai=section2.ai,
        section2_db=section2.db,
        section2_total=section2.total,
        section2_time=section2.time,
        section2_intro=section2.intro,

        section3_name=section3.name if section3 else None,
        section3_topic=section3.topic if section3 else None,
        section3_ai=section3.ai if section3 else None,
        section3_db=section3.db if section3 else None,
        section3_total=section3.total if section3 else None,
        section3_time=section3.time if section3 else None,
        section3_intro=section3.intro if section3 else None,
    )

    print("\n🧾 Final object values before DB insert:")
    print("section1_topic:", quiz.section1_topic)
    print("section2_topic:", quiz.section2_topic)
    print("section3_topic:", quiz.section3_topic)

    db.add(quiz)
    db.commit()
    db.refresh(quiz)

    print("\n✅ Quiz setup saved successfully")
    print("🆔 quiz_id:", quiz.id)
    print("==================================================\n")

    return {
        "message": "Quiz setup saved successfully",
        "quiz_id": quiz.id,
    }

@app.post("/api/exams/generate-reading")
def generate_exam_reading(
    payload: ReadingExamRequest,
    db: Session = Depends(get_db)
):
    """
    Generate a Reading Comprehension exam.
    Produces a STRICT, frontend-safe exam_json with SECTIONED structure.
    """

    print("\n================ GENERATE READING EXAM ================")
    print("Incoming payload:", payload.dict())

    class_name = payload.class_name.strip()
    difficulty = payload.difficulty.strip()
    db.query(GeneratedExamReading).delete(synchronize_session=False)
    db.commit()
    # --------------------------------------------------
    # 1️⃣ LOAD CONFIG
    # --------------------------------------------------
    cfg = (
        db.query(ReadingExamConfig)
        .filter(
            func.lower(ReadingExamConfig.class_name) == class_name.lower(),
            func.lower(ReadingExamConfig.difficulty) == difficulty.lower(),
        )
        .first()
    )

    if not cfg:
        raise HTTPException(404, "No reading exam config found")

    subject = cfg.subject
    topics = cfg.topics
    warnings = []

    sections = []

    # --------------------------------------------------
    # 2️⃣ PROCESS EACH TOPIC AS A SECTION
    # --------------------------------------------------
    for topic_spec in topics:
        topic_name = topic_spec["name"].strip()
        required = int(topic_spec["num_questions"])
        topic_lower = topic_name.lower()

        print(f"\n▶ Topic: {topic_name} | Required: {required}")

        # --------------------------------------------------
        # Load question bundles
        # --------------------------------------------------
        bundles = (
            db.query(QuestionReading)
            .filter(
                func.lower(QuestionReading.class_name) == class_name.lower(),
                func.lower(QuestionReading.subject) == subject.lower(),
                func.lower(QuestionReading.difficulty) == difficulty.lower(),
                func.lower(QuestionReading.topic) == topic_lower,
            )
            .all()
        )


        if not bundles:
            warnings.append(f"No bundles found for topic '{topic_name}'")
            continue

        random.shuffle(bundles)

        collected_questions = []
        reading_material = None
        answer_options = None
        question_type = None

        # --------------------------------------------------
        # Collect questions safely
        # --------------------------------------------------
        for bundle in bundles:
            bundle_json = bundle.exam_bundle or {}

            if not question_type:
                question_type = bundle_json.get("question_type")

            if not reading_material:
                reading_material = bundle_json.get("reading_material")

            if not answer_options:
                answer_options = bundle_json.get("answer_options")

            for q in bundle_json.get("questions", []):
                collected_questions.append(q)
                if len(collected_questions) >= required:
                    break

            if len(collected_questions) >= required:
                break

        if len(collected_questions) < required:
            warnings.append(
                f"Only {len(collected_questions)} questions found for '{topic_name}'"
            )

        if not collected_questions:
            continue

        # --------------------------------------------------
        # Normalize question numbering (per section)
        # --------------------------------------------------
        for idx, q in enumerate(collected_questions, start=1):
            q["question_number"] = idx

        # --------------------------------------------------
        # Build SECTION (frontend-safe)
        # --------------------------------------------------
        section = {
            "question_type": question_type,
            "topic": topic_name,
            "reading_material": reading_material,
            "questions": collected_questions,
        }

        # Only attach answer_options if they exist
        if answer_options:
            section["answer_options"] = answer_options

        sections.append(section)

    # --------------------------------------------------
    # 3️⃣ FINALIZE EXAM
    # --------------------------------------------------
    if not sections:
        raise HTTPException(400, "No exam sections generated")

    total_questions = sum(len(s["questions"]) for s in sections)

    exam_json = {
        "class_name": class_name,
        "subject": subject,
        "difficulty": difficulty,
        "duration_minutes": 40,
        "total_questions": total_questions,
        "sections": sections,
    }

    saved = GeneratedExamReading(
        config_id=cfg.id,
        class_name=class_name,
        subject=subject,
        difficulty=difficulty,
        total_questions=total_questions,
        exam_json=exam_json,
    )

    db.add(saved)
    db.commit()
    db.refresh(saved)

    print("✅ Exam generated. ID:", saved.id)

    return {
        "generated_exam_id": saved.id,
        "total_questions": total_questions,
        "warnings": warnings,
        "exam_json": exam_json,
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

 
CANONICAL_READING_TOPICS = {
    "comparative analysis": "Comparative analysis",
    "gapped text": "Gapped Text",
    "main idea & summary": "Main Idea & Summary",
    "main idea and summary": "Main Idea & Summary",
}

def normalize_topic(raw_topic: str) -> str:
    if not raw_topic:
        raise HTTPException(status_code=400, detail="Missing topic")

    key = raw_topic.strip().lower()

    if key not in CANONICAL_READING_TOPICS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid topic '{raw_topic}'. Must match admin-defined topics."
        )

    return CANONICAL_READING_TOPICS[key]

 
@app.post("/upload-word-reading")
async def upload_word_reading(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    print("\n================ UPLOAD WORD READING =================")

    # --------------------------------------------------
    # 1️⃣ Validate file
    # --------------------------------------------------
    if file.content_type != (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        raise HTTPException(status_code=400, detail="File must be a .docx")

    raw = await file.read()

    extracted = extract_text_from_docx(raw)
    if not extracted or len(extracted.strip()) < 50:
        raise HTTPException(status_code=400, detail="Invalid or empty document")

    # --------------------------------------------------
    # 2️⃣ Detect question type
    # --------------------------------------------------
    lower_text = extracted.lower()
    if "gapped text" in lower_text or "gap 1" in lower_text:
        question_type = "gapped_text"
    else:
        question_type = "standard_reading"

    print("Detected question type:", question_type)

    # --------------------------------------------------
    # 3️⃣ Parse with OpenAI
    # --------------------------------------------------
    try:
        parsed = parse_exam_with_openai(
            extracted_text=extracted,
            question_type=question_type
        )
    except Exception as e:
        print("❌ OpenAI parsing error:", e)
        raise HTTPException(status_code=500, detail="Parsing failed")

    if not parsed:
        raise HTTPException(status_code=400, detail="No exam data parsed")

    exams = parsed if isinstance(parsed, list) else [parsed]
    saved_ids = []

    # --------------------------------------------------
    # 4️⃣ Normalize + Save bundles
    # --------------------------------------------------
    for exam_json in exams:
        raw_topic = exam_json.get("topic")
        if not raw_topic:
            raise HTTPException(
                status_code=400,
                detail="Parsed exam missing topic"
            )

        topic = normalize_topic(raw_topic)

        raw_questions = exam_json.get("questions", [])
        if not raw_questions:
            raise HTTPException(
                status_code=400,
                detail=f"No questions found for topic '{topic}'"
            )

        questions = []
        for q in raw_questions:
            if not q.get("question_text") or not q.get("correct_answer"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid question structure detected"
                )

            questions.append({
                "question_text": q["question_text"],
                "correct_answer": q["correct_answer"],
            })

        bundle = {
            "topic": topic,
            "reading_material": exam_json.get("reading_material", {}),
            "questions": questions,
            "answer_options": exam_json.get("answer_options", {}),
        }

        # 🔒 SYSTEM FIELDS ARE CONTROLLED HERE (NOT BY OPENAI)
        obj = Question_reading(
            class_name="selective",
            subject="Reading Comprehension",
            difficulty="medium",
            topic=topic,
            exam_bundle=bundle,
        )

        db.add(obj)
        db.commit()
        db.refresh(obj)

        saved_ids.append(obj.id)
        print(f"Saved bundle ID: {obj.id} | Topic: {topic}")

    print("✅ Upload complete")

    return {
        "message": "Word reading exams uploaded successfully",
        "bundle_ids": saved_ids,
    }

@app.post("/upload-word-reading-gapped-multi-ai")
async def upload_word_reading_gapped_multi_ai(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    print("\n" + "=" * 70)
    print("📄 START: MULTI GAPPED TEXT WORD UPLOAD (AI - STRICT MODE)")
    print("=" * 70)

    # --------------------------------------------------
    # 1️⃣ File validation
    # --------------------------------------------------
    print("📥 STEP 1: File validation")
    print("   → Filename:", file.filename)
    print("   → Content-Type:", file.content_type)

    if file.content_type != (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        print("❌ INVALID FILE TYPE")
        raise HTTPException(status_code=400, detail="File must be .docx")

    raw = await file.read()
    print("✅ File read into memory | bytes:", len(raw))

    # --------------------------------------------------
    # 2️⃣ Text extraction
    # --------------------------------------------------
    print("\n📄 STEP 2: Extracting text from DOCX")

    try:
        full_text = extract_text_from_docx(raw)
    except Exception as e:
        print("❌ DOCX TEXT EXTRACTION FAILED")
        print("   → Exception:", str(e))
        raise HTTPException(status_code=500, detail="Failed to extract document text")

    if not full_text:
        print("❌ Extracted text is EMPTY")
        raise HTTPException(status_code=400, detail="Empty document")

    print("✅ Text extracted successfully")
    print("   → Character count:", len(full_text))

    if len(full_text.strip()) < 300:
        print("❌ Document too short after extraction")
        raise HTTPException(status_code=400, detail="Invalid document content")

    # --------------------------------------------------
    # 3️⃣ Exam block detection
    # --------------------------------------------------
    print("\n🧩 STEP 3: Locating EXAM blocks")

    start_token = "=== EXAM START ==="
    end_token = "=== EXAM END ==="

    parts = full_text.split(start_token)
    print("   → Total START tokens found:", len(parts) - 1)

    blocks = []

    for idx, part in enumerate(parts[1:], start=1):
        print(f"   → Processing potential block {idx}")

        if end_token not in part:
            print("     ⚠️ END token missing, skipping block")
            continue

        block = part.split(end_token)[0].strip()
        print("     → Block length:", len(block))

        if len(block) < 500:
            print("     ⚠️ Block too short, skipped")
            continue

        blocks.append(block)
        print("     ✅ Block accepted")

    if not blocks:
        print("❌ NO VALID EXAM BLOCKS FOUND")
        raise HTTPException(
            status_code=400,
            detail="No exam blocks found. Missing EXAM START / EXAM END markers."
        )

    print(f"✅ Total valid exam blocks detected: {len(blocks)}")

    saved_ids = []

    # --------------------------------------------------
    # 4️⃣ AI extraction prompt
    # --------------------------------------------------
    print("\n🤖 STEP 4: Preparing AI extraction")

    system_prompt = """
You are an exam content extraction engine.

You MUST extract ONE COMPLETE GAPPED TEXT reading exam.

The document contains a METADATA section with the following keys:
- class_name
- subject
- topic
- difficulty

You MUST extract these fields exactly as written.
If any of these metadata fields are missing or empty, RETURN {}.

CRITICAL RULES (FAIL HARD):
- DO NOT generate, infer, or rewrite content
- Extract ONLY what exists in the document
- reading_material.content MUST contain the FULL passage
- answer_options MUST contain ALL options A through G
- questions MUST be EXACTLY 6
- correct_answer MUST be one of: A|B|C|D|E|F|G
- If ANY required field is missing or empty, RETURN {}

OUTPUT:
- VALID JSON ONLY
- No markdown
- No explanations
"""

    # --------------------------------------------------
    # 5️⃣ Process each exam block
    # --------------------------------------------------
    for block_idx, block_text in enumerate(blocks, start=1):
        print("\n" + "-" * 70)
        print(f"🔍 STEP 5: Processing block {block_idx}/{len(blocks)}")
        print("   → Block text length:", len(block_text))

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"BEGIN_DOCUMENT\n{block_text}\nEND_DOCUMENT"
                    }
                ]
            )
        except Exception as e:
            print("❌ OpenAI API CALL FAILED")
            print("   → Exception:", str(e))
            continue

        raw_output = response.choices[0].message.content.strip()
        print("🧠 AI raw response length:", len(raw_output))

        if not raw_output.startswith("{"):
            print("❌ AI OUTPUT IS NOT JSON")
            print("   → Preview:", raw_output[:200])
            continue

        try:
            parsed = json.loads(raw_output)
            print("✅ JSON parsed successfully")
        except Exception as e:
            print("❌ JSON PARSING FAILED")
            print("   → Exception:", str(e))
            continue

        if not parsed:
            print("⚠️ AI returned EMPTY JSON")
            continue

        # --------------------------------------------------
        # 6️⃣ Hard validation
        # --------------------------------------------------
        print("\n🧪 STEP 6: Validating extracted content")

        rm = parsed.get("reading_material", {})
        if not rm.get("content"):
            print("❌ Missing reading material content")
            continue

        if len(rm["content"].strip()) < 300:
            print("❌ Reading material too short")
            continue

        opts = parsed.get("answer_options", {})
        required_opts = ["A", "B", "C", "D", "E", "F", "G"]

        missing_opts = [k for k in required_opts if not opts.get(k)]
        if missing_opts:
            print("❌ Missing answer options:", missing_opts)
            continue

        questions = parsed.get("questions", [])
        print("   → Questions extracted:", len(questions))

        if len(questions) != 6:
            print("❌ Invalid question count")
            continue

        print("✅ Validation passed")

        # --------------------------------------------------
        # 7️⃣ Enrich questions
        # --------------------------------------------------
        enriched_questions = []
        for i, q in enumerate(questions, start=1):
            enriched_questions.append({
                "question_id": f"GT_Q{i}",
                "gap_number": i,
                "question_text": f"Gap {i}",
                "correct_answer": q["correct_answer"]
            })
         
        # --------------------------------------------------
        # 6.5️⃣ Metadata validation (NEW)
        # --------------------------------------------------
        required_meta = ["class_name", "subject", "topic", "difficulty"]
        missing_meta = [k for k in required_meta if not parsed.get(k)]
        
        if missing_meta:
            print("❌ Missing required metadata:", missing_meta)
            continue
        # --------------------------------------------------
        # 8️⃣ Save to DB
        # --------------------------------------------------
        # --------------------------------------------------
        # 7️⃣ Final bundle (QUIZ-READY)
        # --------------------------------------------------
        bundle = {
            "question_type": "gapped_text",
            "topic": parsed["topic"],
            "reading_material": rm,
            "answer_options": opts,
            "questions": enriched_questions
        }
        
        # --------------------------------------------------
        # 8️⃣ Save ONE row per exam
        # --------------------------------------------------
        try:
            obj = QuestionReading(
                class_name=parsed["class_name"].lower(),
                subject=parsed["subject"],
                difficulty=parsed["difficulty"].lower(),
                topic=parsed["topic"],
                total_questions=len(enriched_questions),
                exam_bundle=bundle
            )
        
            db.add(obj)
            db.commit()
            db.refresh(obj)
        
            saved_ids.append(obj.id)
            print(f"✅ Block {block_idx} saved successfully | ID={obj.id}")
        
        except Exception as e:
            db.rollback()
            print("❌ DATABASE SAVE FAILED")
            print("   → Exception:", str(e))
            continue

    # --------------------------------------------------
    # 9️⃣ Final response
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("🎉 UPLOAD COMPLETE")
    print("   → Total saved exams:", len(saved_ids))
    print("   → IDs:", saved_ids)
    print("=" * 70)

    return {
        "message": "Gapped Text documents processed",
        "saved_count": len(saved_ids),
        "bundle_ids": saved_ids
    }


@app.post("/upload-word-reading-main-idea-ai")
async def upload_word_reading_main_idea_ai(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    print("\n" + "=" * 70)
    print("📄 START: MAIN IDEA & SUMMARY WORD UPLOAD (AI - DEBUG MODE)")
    print("=" * 70)

    # --------------------------------------------------
    # 1️⃣ File validation
    # --------------------------------------------------
    print("📥 STEP 1: File validation")
    print("   → Filename:", file.filename)
    print("   → Content-Type:", file.content_type)

    if file.content_type != (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        print("❌ INVALID FILE TYPE")
        raise HTTPException(status_code=400, detail="File must be .docx")

    raw = await file.read()
    print("✅ File read successfully | bytes:", len(raw))

    # --------------------------------------------------
    # 2️⃣ Text extraction
    # --------------------------------------------------
    print("\n📄 STEP 2: Extracting text from DOCX")

    try:
        full_text = extract_text_from_docx(raw)
    except Exception as e:
        print("❌ DOCX TEXT EXTRACTION FAILED")
        print("   → Exception:", str(e))
        raise HTTPException(status_code=500, detail="Failed to extract document text")

    if not full_text:
        print("❌ Extracted text is EMPTY")
        raise HTTPException(status_code=400, detail="Empty document")

    print("✅ Text extracted")
    print("   → Character count:", len(full_text))

    if len(full_text.strip()) < 300:
        print("❌ Document too short")
        raise HTTPException(status_code=400, detail="Invalid document")

    # --------------------------------------------------
    # 3️⃣ Exam block detection
    # --------------------------------------------------
    print("\n🧩 STEP 3: Detecting exam blocks")

    start_token = "=== EXAM START ==="
    end_token = "=== EXAM END ==="

    parts = full_text.split(start_token)
    print("   → START tokens found:", len(parts) - 1)

    blocks = []

    for idx, part in enumerate(parts[1:], start=1):
        print(f"   → Inspecting block candidate {idx}")

        if end_token not in part:
            print("     ⚠️ END token missing — skipped")
            continue

        block = part.split(end_token)[0].strip()
        print("     → Block length:", len(block))

        if len(block) < 500:
            print("     ⚠️ Block too short — skipped")
            continue

        blocks.append(block)
        print("     ✅ Block accepted")

    if not blocks:
        print("❌ NO VALID EXAM BLOCKS FOUND")
        raise HTTPException(
            status_code=400,
            detail="No exam blocks found. Missing EXAM START / EXAM END markers."
        )

    print(f"✅ Total valid exam blocks: {len(blocks)}")
    saved_ids = []

    # --------------------------------------------------
    # 4️⃣ AI extraction prompt
    # --------------------------------------------------
    print("\n🤖 STEP 4: Preparing AI extraction prompt")

    system_prompt = """
You are an exam content extraction engine.

You MUST extract ONE COMPLETE MAIN IDEA & SUMMARY reading exam.

CRITICAL RULES (FAIL HARD):
- DO NOT generate, infer, or rewrite content
- Extract ONLY what exists in the document
- reading_material MUST contain ALL paragraphs
- answer_options MUST be shared and contain A–G
- questions MUST match Total_Questions exactly
- correct_answer MUST be one of A–G
- If ANY required field is missing or empty, RETURN {}

OUTPUT:
- VALID JSON ONLY
- No markdown
- No explanations
"""

    # --------------------------------------------------
    # 5️⃣ Process each exam block
    # --------------------------------------------------
    for block_idx, block_text in enumerate(blocks, start=1):
        print("\n" + "-" * 70)
        print(f"🔍 STEP 5: Processing block {block_idx}/{len(blocks)}")
        print("   → Block length:", len(block_text))

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"BEGIN_DOCUMENT\n{block_text}\nEND_DOCUMENT"
                    }
                ]
            )
        except Exception as e:
            print("❌ OpenAI API CALL FAILED")
            print("   → Exception:", str(e))
            continue

        raw_output = response.choices[0].message.content.strip()
        print("🧠 AI response length:", len(raw_output))

        if not raw_output.startswith("{"):
            print("❌ NON-JSON AI OUTPUT")
            print("   → Preview:", raw_output[:200])
            continue

        try:
            parsed = json.loads(raw_output)
            print("✅ JSON parsed successfully")
        except Exception as e:
            print("❌ JSON PARSE FAILED")
            print("   → Exception:", str(e))
            continue

        if not parsed:
            print("⚠️ EMPTY JSON returned by AI")
            continue

        # --------------------------------------------------
        # 6️⃣ Hard validation
        # --------------------------------------------------
        print("\n🧪 STEP 6: Validating extracted content")

        rm = parsed.get("reading_material", {})
        paragraphs = rm.get("paragraphs", {})

        if not paragraphs:
            print("❌ Missing paragraphs")
            continue

        print("   → Paragraphs found:", len(paragraphs))

        opts = parsed.get("answer_options", {})
        required_opts = ["A", "B", "C", "D", "E", "F", "G"]

        missing_opts = [k for k in required_opts if not opts.get(k)]
        if missing_opts:
            print("❌ Missing answer options:", missing_opts)
            continue

        questions = parsed.get("questions", [])
        print("   → Questions found:", len(questions))

        if not questions:
            print("❌ No questions extracted")
            continue

        invalid_q = False
        for i, q in enumerate(questions, start=1):
            if (
                "paragraph" not in q
                or "correct_answer" not in q
                or q["correct_answer"] not in required_opts
            ):
                print(f"❌ Invalid question format at index {i}")
                invalid_q = True
                break

        if invalid_q:
            continue

        print("✅ Validation passed")

        # --------------------------------------------------
        # 7️⃣ Enrich questions
        # --------------------------------------------------
        print("\n🧩 STEP 7: Enriching questions")

        enriched_questions = []
        for i, q in enumerate(questions, start=1):
            enriched_questions.append({
                "question_id": f"MI_Q{i}",
                "paragraph": q["paragraph"],
                "question_text": f"Paragraph {q['paragraph']}",
                "correct_answer": q["correct_answer"]
            })

        # --------------------------------------------------
        # 8️⃣ Bundle (RENDER-SAFE)
        # --------------------------------------------------
        bundle = {
            "question_type": "main_idea",
            "topic": parsed["topic"],
            "reading_material": rm,
            "answer_options": opts,
            "questions": enriched_questions
        }

        # --------------------------------------------------
        # 9️⃣ Save to DB
        # --------------------------------------------------
        print("\n💾 STEP 8: Saving exam to database")

        try:
            obj = QuestionReading(
                class_name=parsed["class_name"].lower(),
                subject=parsed["subject"],
                difficulty=parsed["difficulty"].lower(),
                topic=parsed["topic"],
                total_questions=len(enriched_questions),
                exam_bundle=bundle
            )

            db.add(obj)
            db.commit()
            db.refresh(obj)

            saved_ids.append(obj.id)
            print(f"✅ Block {block_idx} saved | ID={obj.id}")

        except Exception as e:
            db.rollback()
            print("❌ DATABASE SAVE FAILED")
            print("   → Exception:", str(e))
            continue

    # --------------------------------------------------
    # 🔟 Final response
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("🎉 UPLOAD COMPLETE")
    print("   → Total saved exams:", len(saved_ids))
    print("   → IDs:", saved_ids)
    print("=" * 70)

    return {
        "message": "Main Idea & Summary documents processed",
        "saved_count": len(saved_ids),
        "bundle_ids": saved_ids
    }


@app.post("/upload-word-reading-comparative-ai")
async def upload_word_reading_comparative_ai(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    print("\n" + "=" * 70)
    print("📄 START: COMPARATIVE ANALYSIS WORD UPLOAD (AI - DEBUG MODE)")
    print("=" * 70)

    # --------------------------------------------------
    # 1️⃣ File validation
    # --------------------------------------------------
    print("📥 STEP 1: File validation")
    print("   → Filename:", file.filename)
    print("   → Content-Type:", file.content_type)

    if file.content_type != (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        print("❌ INVALID FILE TYPE")
        raise HTTPException(status_code=400, detail="File must be .docx")

    raw = await file.read()
    print("✅ File read successfully | bytes:", len(raw))

    # --------------------------------------------------
    # 2️⃣ Text extraction
    # --------------------------------------------------
    print("\n📄 STEP 2: Extracting text from DOCX")

    try:
        full_text = extract_text_from_docx(raw)
    except Exception as e:
        print("❌ DOCX TEXT EXTRACTION FAILED")
        print("   → Exception:", str(e))
        raise HTTPException(status_code=500, detail="Failed to extract document text")

    if not full_text:
        print("❌ Extracted text is EMPTY")
        raise HTTPException(status_code=400, detail="Empty document")

    print("✅ Text extracted")
    print("   → Character count:", len(full_text))

    if len(full_text.strip()) < 300:
        print("❌ Document too short")
        raise HTTPException(status_code=400, detail="Invalid document")

    # --------------------------------------------------
    # 3️⃣ Exam block detection
    # --------------------------------------------------
    print("\n🧩 STEP 3: Detecting exam blocks")

    start_token = "=== EXAM START ==="
    end_token = "=== EXAM END ==="

    parts = full_text.split(start_token)
    print("   → START tokens found:", len(parts) - 1)

    blocks = []

    for idx, part in enumerate(parts[1:], start=1):
        print(f"   → Inspecting block candidate {idx}")

        if end_token not in part:
            print("     ⚠️ END token missing — skipped")
            continue

        block = part.split(end_token)[0].strip()
        print("     → Block length:", len(block))

        if len(block) < 200:
            print("     ⚠️ Block too short — skipped")
            continue

        blocks.append(block)
        print("     ✅ Block accepted")

    if not blocks:
        print("❌ NO VALID EXAM BLOCKS FOUND")
        raise HTTPException(
            status_code=400,
            detail="No exam blocks found. Missing EXAM START / EXAM END markers."
        )

    print(f"✅ Total valid exam blocks: {len(blocks)}")

    saved_ids = []

    # --------------------------------------------------
    # 4️⃣ AI extraction prompt
    # --------------------------------------------------
    print("\n🤖 STEP 4: Preparing AI extraction prompt")

    system_prompt = """
You are an exam content extraction engine.

You MUST extract ONE COMPLETE COMPARATIVE ANALYSIS reading exam.

The document contains a METADATA section with the following fields:
- class_name
- subject
- topic
- difficulty

You MUST extract these fields EXACTLY as written and include them
as top-level keys in the JSON output.

CRITICAL SCHEMA REQUIREMENTS (FAIL HARD):

The JSON output MUST contain:
{
  "class_name": string,
  "subject": string,
  "topic": string,
  "difficulty": string,
  "reading_material": {
    "extracts": {
      "A": string,
      "B": string,
      "C": string,
      "D": string
    }
  },
  "questions": [
    {
      "question_text": string,
      "correct_answer": "A|B|C|D"
    }
  ]
}

If ANY required field is missing or empty, RETURN {}.

CRITICAL RULES:
- DO NOT generate, infer, or rewrite content
- Extract ONLY what exists in the document
- Preserve wording exactly
- questions MUST match Total_Questions exactly
- correct_answer MUST be one of: A, B, C, D

OUTPUT RULES:
- VALID JSON ONLY
- No markdown
- No explanations
"""


    # --------------------------------------------------
    # 5️⃣ Process each exam block
    # --------------------------------------------------
    for block_idx, block_text in enumerate(blocks, start=1):
        print("\n" + "-" * 70)
        print(f"🔍 STEP 5: Processing block {block_idx}/{len(blocks)}")
        print("   → Block length:", len(block_text))

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"BEGIN_DOCUMENT\n{block_text}\nEND_DOCUMENT"
                    }
                ]
            )
        except Exception as e:
            print("❌ OpenAI API CALL FAILED")
            print("   → Exception:", str(e))
            continue

        raw_output = response.choices[0].message.content.strip()
        print("🧠 AI response length:", len(raw_output))

        if not raw_output.startswith("{"):
            print("❌ NON-JSON AI OUTPUT")
            print("   → Preview:", raw_output[:200])
            continue

        try:
            parsed = json.loads(raw_output)
            print("✅ JSON parsed successfully")
        except Exception as e:
            print("❌ JSON PARSE FAILED")
            print("   → Exception:", str(e))
            continue

        if not parsed:
            print("⚠️ EMPTY JSON returned by AI")
            continue

        # --------------------------------------------------
        # 6️⃣ Hard validation
        # --------------------------------------------------
        print("\n🧪 STEP 6: Validating extracted content")

        rm = parsed.get("reading_material", {})
        extracts = rm.get("extracts", {})

        missing_extracts = [
            k for k in ["A", "B", "C", "D"]
            if not extracts.get(k) or not extracts[k].strip()
        ]

        if missing_extracts:
            print("❌ Missing extracts:", missing_extracts)
            continue

        questions = parsed.get("questions", [])
        print("   → Questions found:", len(questions))

        if not questions:
            print("❌ No questions extracted")
            continue

        invalid_q = False
        for i, q in enumerate(questions, start=1):
            if (
                "question_text" not in q
                or "correct_answer" not in q
                or q["correct_answer"] not in ["A", "B", "C", "D"]
            ):
                print(f"❌ Invalid question format at index {i}")
                invalid_q = True
                break

        if invalid_q:
            continue

        print("✅ Validation passed")

        # --------------------------------------------------
        # 7️⃣ Enrich bundle (RENDER-SAFE)
        # --------------------------------------------------
        print("\n🧩 STEP 7: Enriching bundle")

        answer_options = {
            "A": "Extract A",
            "B": "Extract B",
            "C": "Extract C",
            "D": "Extract D"
        }

        for i, q in enumerate(questions, start=1):
            q["question_id"] = f"CA_Q{i}"

        bundle = {
            "question_type": "comparative_analysis",
            "topic": parsed["topic"],
            "reading_material": rm,
            "answer_options": answer_options,
            "questions": questions
        }

        # --------------------------------------------------
        # 8️⃣ Save to DB
        # --------------------------------------------------
        print("\n💾 STEP 8: Saving exam to database")

        try:
            obj = QuestionReading(
                class_name=parsed["class_name"].lower(),
                subject=parsed["subject"],
                difficulty=parsed["difficulty"].lower(),
                topic=parsed["topic"],
                total_questions=len(questions),
                exam_bundle=bundle
            )

            db.add(obj)
            db.commit()
            db.refresh(obj)

            saved_ids.append(obj.id)
            print(f"✅ Block {block_idx} saved | ID={obj.id}")

        except Exception as e:
            db.rollback()
            print("❌ DATABASE SAVE FAILED")
            print("   → Exception:", str(e))
            continue

    # --------------------------------------------------
    # 9️⃣ Final response
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("🎉 UPLOAD COMPLETE")
    print("   → Total saved exams:", len(saved_ids))
    print("   → IDs:", saved_ids)
    print("=" * 70)

    return {
        "message": "Comparative Analysis document processed",
        "saved_count": len(saved_ids),
        "bundle_ids": saved_ids
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
    print("\n--- Deleting dependent Reading generated exams ---")

    db.query(GeneratedExamReading).delete(synchronize_session=False)
    print("\n--- Deleting previous Reading exam configs ---")

    db.query(ReadingExamConfig).delete(synchronize_session=False)
    db.commit()

    print("🗑️ Previous reading configs deleted")

    print("\n--- Creating new Reading exam config ---")

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
        .filter(
            func.lower(Student.student_id) ==
            func.lower(req.student_id.strip())
        )
        .first()
    )
    
    if not student:
        print(f"❌ Student not found (raw={repr(req.student_id)})")
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
        .filter(
            func.lower(Exam.class_name) == func.lower(student.class_name),
            Exam.subject == "mathematical_reasoning"
        )
        .order_by(Exam.created_at.desc())
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

    # --------------------------------------------------
    # ✅ PRE-CREATE RESPONSE ROWS (for reporting)
    # --------------------------------------------------
    for idx, q in enumerate(exam.questions or []):
        db.add(StudentExamResponse(
            student_id=student.id,
            exam_id=exam.id,
            exam_attempt_id=new_attempt.id,
    
            q_id=idx + 1,                     # stable per-attempt question index
            topic=q.get("topic"),
    
            selected_option=None,
            correct_option=q.get("correct_answer"),
            is_correct=None                  # NULL = not attempted
        ))
    
    db.commit()

    return {
        "completed": False,
        "questions": normalize_questions(exam.questions),
        "remaining_time": new_attempt.duration_minutes * 60
    }

@app.post("/api/student/start-exam-thinkingskills")
def start_exam(
    req: StartExamRequest = Body(...),
    db: Session = Depends(get_db)
):
    print("\n🚀 START-EXAM REQUEST")
    print("➡ payload:", req.dict())

    # --------------------------------------------------
    # 1️⃣ Resolve student
    # --------------------------------------------------
    student = (
        db.query(Student)
        .filter(
            func.lower(Student.student_id) ==
            func.lower(req.student_id.strip())
        )
        .first()
    )

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    print(f"✅ Student resolved: id={student.id}")

    # --------------------------------------------------
    # 2️⃣ Get latest THINKING SKILLS attempt
    # --------------------------------------------------
    attempt = (
        db.query(StudentExamThinkingSkills)
        .filter(StudentExamThinkingSkills.student_id == student.id)
        .order_by(StudentExamThinkingSkills.started_at.desc())
        .first()
    )

    if attempt and attempt.completed_at:
        print("✅ Exam already completed")
        return {"completed": True}

    # --------------------------------------------------
    # 3️⃣ Load latest THINKING SKILLS exam
    # --------------------------------------------------
    exam = (
        db.query(Exam)
        .filter(
            func.lower(Exam.class_name) ==
            func.lower(student.class_name),
            Exam.subject == "thinking_skills"
        )
        .order_by(Exam.created_at.desc())
        .first()
    )

    if not exam:
        raise HTTPException(status_code=404, detail="Thinking Skills exam not found")

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
    # 🟡 CASE B — Resume active attempt
    # --------------------------------------------------
    if attempt and attempt.completed_at is None:
        print("⏳ Resuming active attempt")

        started_at = attempt.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        elapsed = int((now - started_at).total_seconds())
        remaining = max(0, attempt.duration_minutes * 60 - elapsed)

        if remaining == 0:
            attempt.completed_at = now
            db.commit()
            return {"completed": True}

        return {
            "completed": False,
            "questions": normalize_questions(exam.questions),
            "remaining_time": remaining
        }

    # --------------------------------------------------
    # 🔵 CASE C — Start new THINKING SKILLS attempt
    # --------------------------------------------------
    print("🆕 Starting new Thinking Skills attempt")

    new_attempt = StudentExamThinkingSkills(
        student_id=student.id,
        exam_id=exam.id,
        started_at=datetime.now(timezone.utc),
        duration_minutes=40
    )

    db.add(new_attempt)
    db.commit()
    db.refresh(new_attempt)

    # --------------------------------------------------
    # ✅ PRE-CREATE RESPONSE ROWS (THINKING SKILLS)
    # --------------------------------------------------
    for idx, q in enumerate(exam.questions or []):
        db.add(
            StudentExamResponseThinkingSkills(
                student_id=student.id,
                exam_id=exam.id,
                exam_attempt_id=new_attempt.id,

                q_id=idx + 1,
                topic=q.get("topic"),

                selected_option=None,
                correct_option=q.get("correct"),
                is_correct=None
            )
        )

    db.commit()

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
def admin_report_exists(db: Session, exam_attempt_id: int) -> bool:
    return (
        db.query(AdminExamReport.id)
        .filter(AdminExamReport.exam_attempt_id == exam_attempt_id)
        .first()
        is not None
    )
 
def generate_admin_exam_report(
    db: Session,
    student: Student,
    exam_attempt: StudentExamThinkingSkills,
    subject: str,
    accuracy: float,
    correct: int,
    wrong: int,
    total_questions: int
):
    """
    Generates an immutable admin report snapshot for a completed exam attempt.
    This function MUST NOT commit. Caller controls the transaction.
    """

    # -------------------------------
    # 1️⃣ Determine performance band
    # -------------------------------
    if accuracy >= 80:
        performance_band = "A"
        readiness_band = "Strong Selective Potential"
        school_guidance = "Mid-tier Selective"
    elif accuracy >= 60:
        performance_band = "B"
        readiness_band = "Borderline Selective"
        school_guidance = "Selective Preparation Required"
    else:
        performance_band = "C"
        readiness_band = "Not Yet Selective Ready"
        school_guidance = "General Stream Recommended"

    # -------------------------------
    # 2️⃣ Create main admin report
    # -------------------------------
    admin_report = AdminExamReport(
        student_id=student.student_id,
        exam_attempt_id=exam_attempt.id,
        exam_type="thinking_skills",
        overall_score=accuracy,
        readiness_band=readiness_band,
        school_guidance_level=school_guidance,
        summary_notes=f"Thinking Skills accuracy: {accuracy}%"
    )

    db.add(admin_report)
    db.flush()  # get admin_report.id without commit

    # -------------------------------
    # 3️⃣ Create section result
    # -------------------------------
    section_result = AdminExamSectionResult(
        admin_report_id=admin_report.id,
        section_name="thinking",
        raw_score=accuracy,
        performance_band=performance_band,
        strengths_summary="Logical reasoning applied correctly",
        improvement_summary="Improve speed and multi-step reasoning"
    )

    db.add(section_result)

    # -------------------------------
    # 4️⃣ Store applied rule (audit safety)
    # -------------------------------
    rule_applied = AdminReadinessRuleApplied(
        admin_report_id=admin_report.id,
        rule_code="THINKING_SKILLS_ACCURACY",
        rule_description="Thinking Skills accuracy used for readiness classification",
        rule_result="passed" if accuracy >= 60 else "failed"
    )

    db.add(rule_applied)


@app.post("/api/student/finish-exam/thinking-skills")
def finish_thinking_skills_exam(
    req: FinishExamRequest,
    db: Session = Depends(get_db)
):
    print("\n================ FINISH THINKING SKILLS EXAM START ================")
    print("📥 Incoming payload:", req.dict())

    # --------------------------------------------------
    # 1️⃣ Validate student (external ID → internal ID)
    # --------------------------------------------------
    student = (
        db.query(Student)
        .filter(Student.student_id == req.student_id)
        .first()
    )

    if not student:
        print("❌ Student NOT FOUND for student_id =", req.student_id)
        raise HTTPException(status_code=404, detail="Student not found")

    print("✅ Student found → internal id:", student.id)

    # --------------------------------------------------
    # 2️⃣ Get active THINKING SKILLS attempt
    # --------------------------------------------------
    attempt = (
        db.query(StudentExamThinkingSkills)
        .filter(
            StudentExamThinkingSkills.student_id == student.id,
            StudentExamThinkingSkills.completed_at.is_(None)
        )
        .order_by(StudentExamThinkingSkills.started_at.desc())
        .first()
    )

    if not attempt:
        print("⚠️ No active Thinking Skills attempt found")
        return {"status": "completed"}

    print("✅ Active attempt found → attempt.id =", attempt.id)

    # --------------------------------------------------
    # 3️⃣ Idempotency guard (result already saved)
    # --------------------------------------------------
    existing_result = (
        db.query(StudentExamResultsThinkingSkills)
        .filter(
            StudentExamResultsThinkingSkills.exam_attempt_id == attempt.id
        )
        .first()
    )

    if existing_result:
        print("⚠️ Result already exists → idempotent return")

        if attempt.completed_at is None:
            attempt.completed_at = datetime.now(timezone.utc)
            db.commit()

        return {"status": "completed"}

    # --------------------------------------------------
    # 4️⃣ Load exam (Thinking Skills only)
    # --------------------------------------------------
    exam = (
        db.query(Exam)
        .filter(
            Exam.id == attempt.exam_id,
            Exam.subject == "thinking_skills"
        )
        .first()
    )

    if not exam:
        print("❌ Exam NOT FOUND for exam_id =", attempt.exam_id)
        raise HTTPException(status_code=404, detail="Exam not found")

    questions = exam.questions or []
    total_questions = len(questions)

    question_map = {q["q_id"]: q for q in questions}

    print("📊 Total questions in exam:", total_questions)

    # --------------------------------------------------
    # 5️⃣ Update student responses (DO NOT INSERT)
    # --------------------------------------------------
    correct = 0
    saved_responses = 0

    for q_id_str, selected in req.answers.items():
        print(f"➡️ Processing answer: q_id={q_id_str}, selected={selected}")

        try:
            q_id = int(q_id_str)
        except ValueError:
            print("❌ Invalid q_id:", q_id_str)
            continue

        q = question_map.get(q_id)
        if not q:
            print("⚠️ Question not found in exam JSON for q_id =", q_id)
            continue

        is_correct = selected == q.get("correct")
        if is_correct:
            correct += 1

        response = (
            db.query(StudentExamResponseThinkingSkills)
            .filter(
                StudentExamResponseThinkingSkills.exam_attempt_id == attempt.id,
                StudentExamResponseThinkingSkills.q_id == q_id
            )
            .first()
        )

        if not response:
            print("❌ Response row missing for q_id =", q_id)
            continue

        response.selected_option = selected
        response.correct_option = q.get("correct")
        response.is_correct = is_correct

        saved_responses += 1

    print("📈 Responses updated:", saved_responses)
    print("✅ Correct answers:", correct)

    wrong = saved_responses - correct
    accuracy = round((correct / saved_responses) * 100, 2) if saved_responses else 0

    # 🔒 HARD SAFETY CHECK — attempt must still exist
    attempt_exists = (
        db.query(StudentExamThinkingSkills.id)
        .filter(StudentExamThinkingSkills.id == attempt.id)
        .first()
    )
    
    if not attempt_exists:
        raise HTTPException(
            status_code=409,
            detail="Thinking Skills exam attempt no longer exists. Please restart the exam."
        )

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

    # --------------------------------------------------
    # 7️⃣ Mark attempt completed
    # --------------------------------------------------
    attempt.completed_at = datetime.now(timezone.utc)
    if not admin_report_exists(
        db=db,
        exam_attempt_id=attempt.id
    ):

        generate_admin_exam_report(
            db=db,
            student=student,
            exam_attempt=attempt,
            subject="thinking_skills",
            accuracy=accuracy,
            correct=correct,
            wrong=wrong,
            total_questions=total_questions
        )

    db.commit()
    # 8️⃣ Generate Admin Report Snapshot (NEW)
    

    print("================ FINISH THINKING SKILLS EXAM END =================\n")

    return {
        "status": "completed",
        "total_questions": total_questions,
        "attempted": saved_responses,
        "correct": correct,
        "wrong": wrong,
        "accuracy": accuracy
    }

def aggregate_math_sections(db: Session, exam_attempt_id: int):
    """
    Aggregates Mathematical Reasoning responses into topic-based sections.

    Returns:
    {
        "Arithmetic": { "accuracy": 75.0 },
        "Word Problems": { "accuracy": 60.0 }
    }
    """

    responses = (
        db.query(StudentExamResponse)
        .filter(
            StudentExamResponse.exam_attempt_id == exam_attempt_id
        )
        .all()
    )

    section_totals = {}

    for r in responses:
        section = r.topic  # ✅ USE topic as section

        if not section:
            section = "General"

        if section not in section_totals:
            section_totals[section] = {"correct": 0, "total": 0}

        section_totals[section]["total"] += 1
        if r.is_correct:
            section_totals[section]["correct"] += 1

    section_results = {}

    for section, stats in section_totals.items():
        accuracy = (
            round((stats["correct"] / stats["total"]) * 100, 1)
            if stats["total"] else 0
        )

        section_results[section] = {
            "accuracy": accuracy
        }

    return section_results


def generate_admin_exam_report_math(
    db: Session,
    student: Student,
    exam_attempt,
    accuracy: float,
    correct: int,
    wrong: int,
    total_questions: int
):
    """
    Generates admin report + section results for Mathematical Reasoning.
    MUST NOT commit.
    """

    # -------------------------------
    # 1️⃣ Readiness & guidance rules
    # -------------------------------
    if accuracy >= 80:
        readiness_band = "Strong Selective Potential"
        school_guidance = "Mid-tier Selective"
    elif accuracy >= 60:
        readiness_band = "Borderline Selective"
        school_guidance = "Selective Preparation Required"
    else:
        readiness_band = "Not Yet Selective Ready"
        school_guidance = "General Stream Recommended"

    # -------------------------------
    # 2️⃣ Create admin report snapshot
    # -------------------------------
    admin_report = AdminExamReport(
        student_id=student.student_id,  # external ID (IMPORTANT)
        exam_attempt_id=exam_attempt.id,
        exam_type="mathematical_reasoning",
        overall_score=accuracy,
        readiness_band=readiness_band,
        school_guidance_level=school_guidance,
        summary_notes=f"Mathematical Reasoning accuracy: {accuracy}%"
    )

    db.add(admin_report)
    db.flush()  # get admin_report.id

    # -------------------------------
    # 3️⃣ Aggregate maths sections
    # -------------------------------
    section_stats = aggregate_math_sections(
        db=db,
        exam_attempt_id=exam_attempt.id
    )

    # -------------------------------
    # 4️⃣ Insert section rows
    # -------------------------------
    for section_name, stats in section_stats.items():
        raw_score = stats["accuracy"]

        if raw_score >= 80:
            band = "A"
        elif raw_score >= 60:
            band = "B"
        else:
            band = "C"

        db.add(
            AdminExamSectionResult(
                admin_report_id=admin_report.id,
                section_name=section_name,
                raw_score=raw_score,
                performance_band=band,
                strengths_summary=None,
                improvement_summary=None
            )
        )

    # -------------------------------
    # 5️⃣ Store applied rule (audit)
    # -------------------------------
    db.add(
        AdminReadinessRuleApplied(
            admin_report_id=admin_report.id,
            rule_code="MATHS_ACCURACY",
            rule_description="Mathematical Reasoning accuracy used for readiness classification",
            rule_result="passed" if accuracy >= 60 else "failed"
        )
    )


@app.post("/api/student/finish-exam")
def finish_exam(
    req: FinishExamRequest,
    db: Session = Depends(get_db)
):
    print("\n================ FINISH MATHEMATICAL REASONING EXAM START ================")
    print("📥 Incoming payload:", req.dict())

    # --------------------------------------------------
    # 1️⃣ Resolve student (external → internal)
    # --------------------------------------------------
    student = (
        db.query(Student)
        .filter(Student.student_id == req.student_id)
        .first()
    )

    if not student:
        print("❌ Student NOT FOUND:", req.student_id)
        raise HTTPException(status_code=404, detail="Student not found")

    print("✅ Student resolved → id:", student.id)

    # --------------------------------------------------
    # 2️⃣ Get active Mathematical Reasoning attempt
    # --------------------------------------------------
    attempt = (
        db.query(StudentExam)
        .join(Exam, StudentExam.exam_id == Exam.id)
        .filter(
            StudentExam.student_id == student.id,
            StudentExam.completed_at.is_(None),
            Exam.subject == "mathematical_reasoning"
        )
        .order_by(StudentExam.started_at.desc())
        .first()
    )


    if not attempt:
        print("⚠️ No active attempt found")
        return {"status": "completed"}

    print("✅ Active attempt found → id:", attempt.id)

    # --------------------------------------------------
    # 3️⃣ Load exam (STRICT subject guard)
    # --------------------------------------------------
    exam = (
        db.query(Exam)
        .filter(
            Exam.id == attempt.exam_id,
            Exam.subject == "mathematical_reasoning"
        )
        .first()
    )

    if not exam:
        raise HTTPException(
            status_code=400,
            detail="Finish endpoint called for non-mathematical reasoning exam"
        )

    questions = exam.questions or []
    total_questions = len(questions)

    print("📘 Exam loaded → questions:", total_questions)

    question_map = {q["q_id"]: q for q in questions}

    # --------------------------------------------------
    # Clear previous reports for this student + subject
    # --------------------------------------------------
    db.query(StudentExamResultsMathematicalReasoning).filter(
        StudentExamResultsMathematicalReasoning.student_id == student.id
    ).delete(synchronize_session=False)
    
    db.commit()

    
    # --------------------------------------------------
    # 5️⃣ Update student responses (NO inserts)
    # --------------------------------------------------
    correct = 0
    saved_responses = 0

    for q_id_str, selected in req.answers.items():
        try:
            q_id = int(q_id_str)
        except ValueError:
            continue
    
        q = question_map.get(q_id)
        if not q:
            continue
    
        correct_answer = q.get("correct")
    
        # 🔍 DEBUG LOGS (THIS IS THE KEY)
        print("🧪 EVALUATING QUESTION")
        print("Question ID:", q_id)
        print("Student Answer:", selected)
        print("Correct Answer (DB):", correct_answer)
    
        is_correct = selected == correct_answer
    
        print("➡️ MATCH RESULT:", is_correct)
        print("------------------------------------")
    
        if is_correct:
            correct += 1
        response = (
            db.query(StudentExamResponse)
            .filter(
                StudentExamResponse.exam_attempt_id == attempt.id,
                StudentExamResponse.q_id == q_id
            )
            .first()
        )

        if not response:
            continue

        response.selected_option = selected
        response.correct_option = q.get("correct")
        response.is_correct = is_correct

        saved_responses += 1

    wrong = saved_responses - correct
    accuracy = round((correct / saved_responses) * 100, 2) if saved_responses else 0

    print("📊 Result computed → correct:", correct, "wrong:", wrong)
    existing_raw = (
        db.query(AdminExamRawScore)
        .filter(
            AdminExamRawScore.exam_attempt_id == attempt.id,
            AdminExamRawScore.subject == "mathematical_reasoning"
        )
        .first()
    )
    
    if not existing_raw:
        raw_score = AdminExamRawScore(
            student_id=student.id,
            exam_attempt_id=attempt.id,
            subject="mathematical_reasoning",
            total_questions=total_questions,
            correct_answers=correct,
            wrong_answers=wrong,
            accuracy_percent=accuracy
        )
        db.add(raw_score)


    # --------------------------------------------------
    # 6️⃣ Save summary (NEW TABLE)
    # --------------------------------------------------
    result_row = StudentExamResultsMathematicalReasoning(
        student_id=student.id,
        exam_attempt_id=attempt.id,
        total_questions=total_questions,
        correct_answers=correct,
        wrong_answers=wrong,
        accuracy_percent=accuracy
    )

    db.add(result_row)

    # --------------------------------------------------
    # 7️⃣ Mark attempt completed
    # --------------------------------------------------
    attempt.completed_at = datetime.now(timezone.utc)
    # --------------------------------------------------
# 8️⃣ Generate Admin Report Snapshot (ADMIN)
# --------------------------------------------------
    if not admin_report_exists(
        db=db,
        exam_attempt_id=attempt.id
    ):
        generate_admin_exam_report_math(
            db=db,
            student=student,
            exam_attempt=attempt,
            accuracy=accuracy,
            correct=correct,
            wrong=wrong,
            total_questions=total_questions
        )
    
    # --------------------------------------------------
    # 9️⃣ Final commit (single transaction)
    # --------------------------------------------------
    db.commit()

    print("================ FINISH MATHEMATICAL REASONING EXAM END =================\n")
    

    return {
        "status": "completed",
        "total_questions": total_questions,
        "attempted": saved_responses,
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
async def upload_image_folder(
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    print("\n====================== FOLDER UPLOAD START ======================")
    print(f"[INFO] Number of received files: {len(images)}")

    if not images:
        raise HTTPException(status_code=400, detail="No images received.")

    uploaded_urls = []

    for file in images:
        try:
            print("\n-----------------------------------------------------------")
            print(f"[FILE] Incoming filename: {file.filename}")
            print(f"[FILE] Content type: {file.content_type}")

            # ✅ Normalize + sanitize filename (CRITICAL)
            original_name = sanitize_filename(file.filename)
            print(f"[FILE] Sanitized original_name: {original_name}")

            # 🔍 Check DB FIRST (idempotent)
            existing = (
                db.query(UploadedImage)
                .filter(UploadedImage.original_name == original_name)
                .first()
            )

            if existing:
                print("[SKIP] Image already exists in DB")
                print(f"   • original_name : {existing.original_name}")
                print(f"   • gcs_url       : {existing.gcs_url}")

                uploaded_urls.append({
                    "original_name": existing.original_name,
                    "url": existing.gcs_url,
                    "status": "already_exists"
                })
                continue

            # ⬆️ Only upload if NOT already present
            file_bytes = await file.read()
            gcs_url = upload_to_gcs(file_bytes, original_name)

            print("[UPLOAD SUCCESS]")
            print(f"   • original_name : {original_name}")
            print(f"   • gcs_url       : {gcs_url}")

            # 💾 Persist mapping
            record = UploadedImage(
                original_name=original_name,
                gcs_url=gcs_url
            )
            db.add(record)
            db.commit()

            uploaded_urls.append({
                "original_name": original_name,
                "url": gcs_url,
                "status": "uploaded"
            })

        except Exception as e:
            print(f"[ERROR] Failed to process file '{file.filename}'")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    print("\n====================== FOLDER UPLOAD END ========================")

    return {
        "status": "success",
        "message": f"{len(uploaded_urls)} images processed.",
        "files": uploaded_urls
    }
 
def chunk_by_question(paragraphs):
    blocks = []
    current = []

    for p in paragraphs:
        if p.strip().startswith("METADATA:") and current:
            blocks.append("\n\n".join(current))
            current = []
        current.append(p)

    if current:
        blocks.append("\n\n".join(current))

    return blocks


@app.post("/upload-word")
async def upload_word(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    request_id = str(uuid.uuid4())[:8]

    print(f"\n========== GPT-UPLOAD-WORD START [{request_id}] ==========")
    print(f"[{request_id}] 📄 Incoming file: {file.filename}")
    print(f"[{request_id}] 📄 Content-Type: {file.content_type}")

    # -----------------------------
    # Validate file type
    # -----------------------------
    allowed = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain"
    }

    if file.content_type not in allowed:
        print(f"[{request_id}] ❌ Invalid file type")
        raise HTTPException(status_code=400, detail="Invalid file type.")

    # -----------------------------
    # Extract raw text
    # -----------------------------
    try:
        content = await file.read()
        doc = docx.Document(BytesIO(content))
        paragraphs = []
        for p in doc.paragraphs:
            text = p.text.rstrip()
            if text:
                paragraphs.append(text)

        print(f"[{request_id}] 📄 Extracted paragraphs: {len(paragraphs)}")
    except Exception as e:
        print(f"[{request_id}] ❌ DOCX read error: {e}")
        raise HTTPException(status_code=400, detail="Failed to read file.")

    # -----------------------------
    # Convert to chunks
    # -----------------------------
    blocks = chunk_by_question(paragraphs)


    

    print(f"[{request_id}] 📦 Total blocks: {len(blocks)}")

    all_questions = []

    # -----------------------------
    # GPT parsing
    # -----------------------------
    for block_idx, block in enumerate(blocks, start=1):
        print(
            f"\n[{request_id}] ▶️ GPT BLOCK {block_idx}/{len(blocks)} "
            f"| chars={len(block)}"
        )
    
        result = await parse_with_gpt(block)
        questions = result.get("questions", [])
    
        print(f"[{request_id}] 🧩 GPT returned {len(questions)} questions")
    
        for qi, q in enumerate(questions, start=1):
            print(
                f"[{request_id}] 🔎 B{block_idx}-Q{qi} | "
                f"partial={q.get('partial')} | "
                f"class={q.get('class_name')} | "
                f"topic={q.get('topic')} | "
                f"images={q.get('images')}"
            )
    
        all_questions.extend(questions)

    print(f"\n[{request_id}] 📊 Total questions detected: {len(all_questions)}")

    # -----------------------------
    # Save valid questions
    # -----------------------------
    saved_ids = []
    skipped_partial = 0

    for idx, q in enumerate(all_questions, start=1):
        qid = f"Q{idx}"

        if q.get("partial") is True:
            skipped_partial += 1
            print(
                f"[{request_id}] ⚠ SKIP {qid} | "
                f"reason=incomplete (partial={q.get('partial')})"
            )
            continue
        

        # --- Normalize question_text formatting (defensive) ---
        qt = q.get("question_text")
        if qt:
            q["question_text"] = qt.replace("\n\n\n", "\n\n").strip()
        resolved_images = []

        for img in q.get("images") or []:
            img_raw = img
            img_norm = img.strip().lower()

            print(
                f"[{request_id}] 🖼️ RESOLVE {qid} | "
                f"img_raw='{img_raw}' | img_norm='{img_norm}'"
            )

            record = (
                db.query(UploadedImage)
                .filter(func.lower(func.trim(UploadedImage.original_name)) == img_norm)
                .first()
            )

            if not record:
                print(
                    f"[{request_id}] ❌ IMAGE NOT FOUND {qid} | "
                    f"img='{img_raw}'"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Image '{img_raw}' not uploaded yet"
                )

            print(
                f"[{request_id}] ✅ IMAGE OK {qid} | "
                f"gcs_url='{record.gcs_url}'"
            )

            resolved_images.append(record.gcs_url)

        new_q = Question(
            class_name=q.get("class_name"),
            subject=q.get("subject"),
            topic=q.get("topic"),
            difficulty=q.get("difficulty"),
            question_type="multi_image_diagram_mcq",
        
            question_text=q.get("question_text"),
            question_blocks=q.get("question_blocks"),  # ✅ ADD THIS
        
            images=resolved_images,
            options=q.get("options"),
            correct_answer=q.get("correct_answer")
        )

        db.add(new_q)
        db.commit()
        db.refresh(new_q)

        saved_ids.append(new_q.id)
        print(f"[{request_id}] 💾 SAVED {qid} | db_id={new_q.id}")

    # -----------------------------
    # Final summary
    # -----------------------------
    print(
        f"\n[{request_id}] 🏁 UPLOAD COMPLETE | "
        f"saved={len(saved_ids)} | "
        f"partial_skipped={skipped_partial} | "
        f"total_detected={len(all_questions)}"
    )
    print(f"========== GPT-UPLOAD-WORD END [{request_id}] ==========\n")

    return {
        "status": "success",
        "saved_questions": saved_ids,
        "count": len(saved_ids),
        "skipped_partial": skipped_partial
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
        print("\n--- Deleting previous data (Exam → Quiz) ---")

        # 1️⃣ Delete dependent exams (StudentExam deleted automatically via cascade)
        db.query(Exam).delete(synchronize_session=False)

        # 2️⃣ Delete quizzes
        db.query(Quiz).delete(synchronize_session=False)

        db.commit()
        print("🗑️ All previous exams and quizzes deleted")

        print("\n--- Creating SQLAlchemy Quiz object ---")

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
    Returns the next numeric student id (as string),
    safely handling string-based IDs.
    """

    try:
        last_id_int = (
            db.query(func.max(cast(Student.id, Integer)))
            .filter(Student.id.op("~")("^[0-9]+$"))  # only numeric strings
            .scalar()
        ) or 0

        next_id = last_id_int + 1
        return str(next_id)

    except Exception as e:
        print("[ERROR] Failed to get next student id:", e)
        raise HTTPException(
            status_code=500,
            detail="Could not generate next user ID",
        )
     
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





