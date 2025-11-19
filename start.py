# start.py
import os
import uvicorn

port = int(os.environ.get("PORT", 8000))  # read $PORT, default 8000 if missing

uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
