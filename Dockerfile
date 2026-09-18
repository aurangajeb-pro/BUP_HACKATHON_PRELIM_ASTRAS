FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    OLLAMA_URL=http://ollama:11434 \
    OLLAMA_MODEL=gemma4:31b-cloud

EXPOSE 8000

# Sync endpoints run in FastAPI's thread pool. Inference has its own timeout.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
