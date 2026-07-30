FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre‑download the Demucs model (htdemucs) using the CLI.
# This ensures the model is cached in the container during the build,
# so the first user does NOT wait for a download.
RUN demucs --model htdemucs --out /tmp /dev/null

# Copy the rest of the application
COPY . .

# Render expects the app to listen on port 10000
ENV PORT=10000
EXPOSE 10000

# Start the Flask server
CMD ["python", "app.py"]
