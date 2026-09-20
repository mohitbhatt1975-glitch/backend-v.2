# Works unchanged on Render, Hugging Face Spaces and Fly.io.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Dependencies first, so edits to the app don't invalidate the wheel cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the trained surrogate. training_data.csv is excluded by
# .dockerignore: it is only needed to retrain, never to serve.
COPY physics.py solar.py materials_library.py main.py ./
COPY thermal_model_v4.pkl ./
COPY static/ ./static/

EXPOSE 8000

# Hosts inject $PORT; default to 8000 when they don't. Single worker keeps the
# image inside a 512 MB free tier -- the process measures ~185 MB resident.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
