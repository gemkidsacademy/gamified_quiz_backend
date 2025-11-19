web: python start.py
worker: celery -A celery_app.celery_app worker --loglevel=info
beat: celery -A celery_app.celery_app beat --loglevel=info
