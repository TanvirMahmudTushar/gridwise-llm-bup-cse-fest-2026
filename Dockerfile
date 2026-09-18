FROM python:3.12-slim

# CBC solver binary bundled by PuLP's Python wheel is used by default on
# linux/amd64 and linux/arm64; no system package install is required.

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8000
EXPOSE 8000

# No secrets are baked into the image -- GROQ_API_KEY and friends must be
# supplied at `docker run` time via -e / --env-file.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
