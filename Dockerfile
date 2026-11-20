# ---------- STAGE 0: base OS ----------
FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d AS base

RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*


# ---------- STAGE 1: system deps ----------
FROM base AS system-deps

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# ---------- STAGE 2: builder ----------
FROM system-deps AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --require-hashes --prefix=/install -r requirements.txt


# ---------- STAGE 3: runtime ----------
FROM system-deps AS runtime

WORKDIR /app

# Artefacto versionado de modelos (ver scripts/fetch_models.sh).
# Al cambiar de modelo: publicar un release nuevo y actualizar URL y SHA256.
ARG PLATE_MODELS_URL="https://github.com/jarodsmdev/sifaPlateDetectorBE/releases/download/models-v1/plate-models-v1.tar.gz"
ARG PLATE_MODELS_SHA256="7814edec45d8cb2b7a38e92639f82ff336bafc4f6f9c9494cd5fc621982dbca6"
LABEL org.sifa.models.sha256="${PLATE_MODELS_SHA256}"

COPY --from=builder /install /usr/local
COPY scripts/fetch_models.sh /usr/local/bin/fetch_models.sh

RUN chmod +x /usr/local/bin/fetch_models.sh \
    && /usr/local/bin/fetch_models.sh

COPY app ./app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
