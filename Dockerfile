# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8080

# Start both services: web + worker
# Use a process manager to run multiple commands
RUN pip install --no-cache-dir honcho

# Copy Procfile
COPY Procfile /app/Procfile

# Start processes
CMD ["honcho", "start", "-f", "Procfile"]
