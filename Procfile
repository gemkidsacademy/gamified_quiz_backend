web: python -m uvicorn main:app --host 0.0.0.0 --port 8080
worker: celery -A celery_app.celery_app worker --loglevel=info
beat: celery -A celery_app.celery_app beat --loglevel=info
