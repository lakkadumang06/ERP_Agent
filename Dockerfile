FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data/sample_docs ./data/sample_docs
COPY scripts ./scripts

# Unprivileged user; /app/data holds the SQLite file (mounted as a volume).
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV DATABASE_URL=sqlite:////app/data/erp.db
EXPOSE 8000

# Seed only when the database file does not exist yet (seeding drops tables).
CMD ["sh", "-c", "[ -f /app/data/erp.db ] || python -m app.seed; exec uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
