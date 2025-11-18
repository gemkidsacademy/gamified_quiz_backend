import os
from celery import Celery
from celery.schedules import crontab

# Celery Initialization
celery_app = Celery(
    "gamified_scheduler",
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_RESULT_BACKEND")
)

celery_app.conf.update(
    timezone="Asia/Karachi",
    enable_utc=True
)

# Celery Beat Schedule
celery_app.conf.beat_schedule = {
    'generate-daily-quizzes': {
        'task': 'scheduler.generate_quizzes',
        'schedule': crontab(hour=19, minute=0),  # run daily at 7 PM
    },
}
