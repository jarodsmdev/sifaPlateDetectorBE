# ---------- STAGE 0: base OS ----------
FROM python:3.10-slim AS base

RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*


# ---------- STAGE 1: system deps ----------
FROM base AS system-deps

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


# ---------- STAGE 2: builder ----------
FROM system-deps AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- STAGE 3: runtime ----------
FROM system-deps AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY app ./app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]