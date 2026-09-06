FROM python:3.11-slim

# Install system dependencies required by pydub/audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose container port
EXPOSE 8080

# Run application using main.py
CMD ["python", "main.py"]
