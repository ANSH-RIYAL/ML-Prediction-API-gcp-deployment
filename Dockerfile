FROM python:3.11-slim

WORKDIR /app

# Install system deps for scipy/numpy compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (better Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + model files
COPY . .

# Cloud Run uses PORT env variable
ENV PORT=8080
EXPOSE 8080

# Run with gunicorn for production (Flask app)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 120 app:app
