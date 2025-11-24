# main.py
from fastapi import FastAPI, HTTPException, Depends, Response, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List
from sendgrid import SendGridAPIClient
from datetime import datetime, timedelta
import os
import random
import time
from sendgrid.helpers.mail import Mail
from typing import List, Dict, Any, Optional
import re 
import uuid
from sqlalchemy.dialects.postgresql import UUID





import json

from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, ForeignKey, select, func, text, Boolean
from fastapi.responses import JSONResponse



from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.mutable import MutableDict



from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler

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

class LeaderboardEntry(BaseModel):
    student_name: str
    class_name: str
    class_day: str
    total_score: int
    total_questions: int
    submitted_at: datetime   # <-- change from str to datetime
    week_number: int
    
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
        "https://leader-board-viewer-gamified-quiz.vercel.app"
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

            # --- Compose strict system prompt (single triple quotes) ---
            system_prompt = '''
            You are an expert quiz-generating AI and a creative educator. Create a gamified quiz for Australia NSW {class_name} class students on the topic: {topic_name}. The activity type is: "{activity_type}".
            
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
               - If activity type is "Mini Path – choice answer":
                   *Create a short decision step where the student chooses the correct path or action.*
               - If activity type is "Pattern hunt puzzle – choice answer":
                   *Include a hidden clue, rule, or pattern the student must identify before choosing the correct option.*
               - If activity type is "Character mission challenge – Choice answer":
                   *Present a mission or adventure with a character where the student selects the correct action to succeed.*
            
            8. For ANY other activity type passed through {activity_type}:
                   *Use the name to guide theme and creativity ONLY. DO NOT modify structure or rules.*
            
            9. Questions MUST be fun, clear, engaging, and age-appropriate for Year 5.
            
            10. NEVER blend two questions together. NEVER ask "What is X? Is it Y?" inside the same prompt.
            
            11. Admin prompt/context for this quiz: {raw_prompt}
            '''.format(
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
    """
    Retrieve the term start date from admin_date table.
    """
    try:
        # Fetch the first row in the admin_date table
        stmt = select(AdminDate).limit(1)
        result = db.execute(stmt).scalars().first()

        if not result or not result.date:
            raise HTTPException(status_code=404, detail="Term start date not found")

        # Return date as ISO string
        return TermStartDateResponse(date=result.date.isoformat())

    except Exception as e:
        print("Error retrieving term start date:", e)
        raise HTTPException(status_code=500, detail="Internal server error")
        

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





