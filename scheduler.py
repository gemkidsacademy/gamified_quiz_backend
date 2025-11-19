import json
import random
from datetime import datetime

from celery_app import celery_app

from openai_client import client   # your OpenAI wrapper
from main import SessionLocal, User, Activity, StudentQuiz


# ------------------ Scheduler Task ------------------
@celery_app.task
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

                parsed_json = json.loads(quiz_content)

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
