FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download the model (htdemucs) on the server during build
RUN python -c "import demucs.separate; demucs.separate.main(['--model', 'htdemucs', '--out', '/tmp', '/dev/null'])"
COPY . .
ENV PORT=10000
EXPOSE 10000
CMD ["python", "app.py"]