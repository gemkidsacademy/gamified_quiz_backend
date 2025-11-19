# Run FastAPI app
web: python -m uvicorn main:app --host 0.0.0.0 --port $PORT

# Run Celery worker
worker: celery -A celery_app.celery_app worker --loglevel=info

# Run Celery beat
beat: celery -A celery_app.celery_app beat --loglevel=info
