#!/usr/bin/env bash
# Descarga, verifica e instala los modelos de IA desde un artefacto versionado.
# Pensado para ejecutarse durante el build de la imagen, pero también sirve en
# local con PADDLE_CACHE_DIR/YOLO_CACHE_DIR.
set -euo pipefail

URL="${PLATE_MODELS_URL:?falta PLATE_MODELS_URL}"
SHA="${PLATE_MODELS_SHA256:?falta PLATE_MODELS_SHA256}"
PADDLE_DIR="${PADDLE_CACHE_DIR:-/root/.paddleocr}"
YOLO_DIR="${YOLO_CACHE_DIR:-/root/.cache/open-image-models}"

if [ -f "$PADDLE_DIR/.models-version" ] &&
   [ "$(cat "$PADDLE_DIR/.models-version")" = "$SHA" ] &&
   [ -f "$YOLO_DIR/.models-version" ] &&
   [ "$(cat "$YOLO_DIR/.models-version")" = "$SHA" ]; then
  echo "Modelos ya presentes y verificados ($SHA). Nada que hacer."
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Descargando modelos desde $URL"
curl --fail --location --retry 3 --retry-delay 2 \
     --connect-timeout 10 --max-time 300 \
     --output "$tmp/models.tar.gz" "$URL"

echo "$SHA  $tmp/models.tar.gz" | sha256sum --check --status || {
  echo "ERROR: el SHA256 del artefacto no coincide. Se aborta." >&2
  exit 1
}

tar -xzf "$tmp/models.tar.gz" -C "$tmp"

rm -rf "$PADDLE_DIR" "$YOLO_DIR"
mkdir -p "$PADDLE_DIR" "$YOLO_DIR"
cp -a "$tmp/paddle_cache/." "$PADDLE_DIR/"
cp -a "$tmp/yolo_cache/." "$YOLO_DIR/"

printf '%s' "$SHA" > "$PADDLE_DIR/.models-version"
printf '%s' "$SHA" > "$YOLO_DIR/.models-version"

echo "Modelos instalados y verificados ($SHA)."
