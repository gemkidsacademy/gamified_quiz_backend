# Procfile for Railway deployment

# Run FastAPI app
web: python -m uvicorn main:app --host 0.0.0.0 --port $PORT

# Run Celery worker
worker: celery -A main.celery_app worker --loglevel=info

# Run Celery beat for scheduled tasks
beat: celery -A main.celery_app beat --loglevel=info
