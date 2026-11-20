# API de detección y lectura de patentes

Servicio en FastAPI para detectar patentes vehiculares en imágenes y extraer su texto mediante OCR.

## Qué hace este proyecto

- Detecta patentes usando un modelo YOLO de `open-image-models`.
- Procesa la imagen en memoria (no guarda capturas en disco).
- Ejecuta OCR con PaddleOCR sobre el recorte de cada patente.
- Separa la confianza de detección (YOLO) de la confianza de lectura (OCR).
- Valida el texto contra los formatos chilenos de autos y motos.

## Estructura principal

- `app/main.py`: API FastAPI y endpoint `/plate/api/v1/detect`.
- `app/model.py`: carga y ejecución del detector de patentes.
- `app/ocr.py`: ejecución de PaddleOCR y extracción de candidatos.
- `app/plate_format.py`: normalización, formatos chilenos y selección del mejor candidato.
- `app/services.py`: pipeline en memoria (redimensionar → YOLO → OCR).
- `app/schemas.py`: contrato de respuesta.
- `Dockerfile`: imagen de ejecución con dependencias del sistema.
- `Dockerfile.test`: imagen mínima para pruebas unitarias (sin Paddle/YOLO).
- `docker-compose.yml`: despliegue local en contenedor.

## Requisitos

### Opción recomendada (Docker)

- Docker
- Docker Compose

### Opción local (sin Docker)

- Python 3.10 exacto (el lockfile y sus hashes se generaron con 3.10)
- Dependencias de Python instaladas desde el lockfile (`requirements.txt`)

Para ejecutar solo las pruebas unitarias **no necesitas Python en el host**:
usa la imagen de `Dockerfile.test` (ver más abajo).

## Levantar con Docker (recomendado)

```bash
docker compose up --build
```

La imagen incluye los modelos de IA. El primer build (o uno sin caché de capas)
descarga el artefacto versionado desde GitHub y verifica su SHA256; requiere
conexión a internet, igual que la instalación de dependencias.

El servicio quedará disponible en:

```text
http://localhost:8000/plate/api/v1/detect
```

Para detener:

```bash
docker compose down
```

## Levantar en local (sin Docker)

Requiere `python3.10` instalado. En Ubuntu no existe el comando `python` por
defecto; usa `python3.10` explícitamente.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Si el `.venv` fue creado con otra versión de Python (por ejemplo 3.12 o 3.14),
bórralo y vuelve a crearlo con `python3.10`; los paquetes y hashes del lock no
son compatibles con otras versiones.

### Modelos de IA en local

Los modelos se instalan en `~/.paddleocr` y `~/.cache/open-image-models`. Para
obtenerlos sin construir la imagen:

```bash
export PLATE_MODELS_URL="https://github.com/jarodsmdev/sifaPlateDetectorBE/releases/download/models-v1/plate-models-v1.tar.gz"
export PLATE_MODELS_SHA256="7814edec45d8cb2b7a38e92639f82ff336bafc4f6f9c9494cd5fc621982dbca6"
./scripts/fetch_models.sh
```

El script descarga el artefacto indicado por `PLATE_MODELS_URL`, verifica
`PLATE_MODELS_SHA256` y lo deja en esas rutas. Es idempotente: si la versión ya
está instalada, no descarga de nuevo. En Docker, este paso ocurre durante el
build. También se pueden usar `PADDLE_CACHE_DIR` y `YOLO_CACHE_DIR` para
instalarlos dentro del proyecto.

Para actualizar los modelos: publicar un release nuevo en GitHub (nunca
sobrescribir el anterior), actualizar `PLATE_MODELS_URL` y
`PLATE_MODELS_SHA256` en `Dockerfile`/`.envExample`, y reconstruir la imagen.
El detalle completo está en
[`docu/IMPLEMENTACION_POOL_MODELOS_RELEASE.md`](docu/IMPLEMENTACION_POOL_MODELOS_RELEASE.md).

## Lockfile y dependencias

`requirements.in` contiene las dependencias directas con versión exacta.
`requirements.txt` es el lockfile generado (todas las dependencias, con hashes)
y **no se edita a mano**.

`requirements-dev.in` y `requirements-dev.txt` son el lockfile de las pruebas
unitarias: incluyen el runtime mínimo para importar la app y `pytest`, `httpx` y
`requests`. No incluyen PaddleOCR, PaddlePaddle ni open-image-models porque
`test/conftest.py` los reemplaza por mocks.

Regenerar ambos locks con Python 3.10 / Linux x86_64 (vía Docker, sin instalar
nada en el host):

```bash
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD":/w -w /w python:3.10-slim bash -c '
    python -m venv /tmp/venv &&
    /tmp/venv/bin/pip install pip-tools==7.7.0 &&
    /tmp/venv/bin/pip-compile --generate-hashes \
      --output-file requirements.txt requirements.in &&
    /tmp/venv/bin/pip-compile --generate-hashes \
      --constraint requirements.txt \
      --output-file requirements-dev.txt requirements-dev.in
  '
```

Las instalaciones usan `--require-hashes`: si un paquete no coincide con el hash
esperado, la instalación falla.

## Uso del endpoint

### `POST /plate/api/v1/detect`

Recibe una imagen (`jpg` o `png`) como `multipart/form-data`.

- Tamaño máximo por archivo: 5 MB.
- Tipos permitidos: `image/jpeg`, `image/png`.
- Requiere un JWT con rol `USER_APP`.

Ejemplo con `curl`:

```bash
curl -X POST "http://localhost:8000/plate/api/v1/detect" \
  -H "accept: application/json" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/ruta/a/tu/imagen.jpg"
```

Respuesta esperada (ejemplo):

```json
{
  "result": [
    {
      "plate": "BBCC11",
      "success": true,
      "plate_valid": true,
      "status": "OK",
      "detection_confidence": 0.9654,
      "ocr_confidence": 0.9123,
      "bbox": {
        "x1": 100,
        "y1": 220,
        "x2": 280,
        "y2": 300
      }
    }
  ],
  "timestamp": "2026-03-27T12:34:56.000000Z"
}
```

### Campos de cada detección

| Campo | Descripción |
|---|---|
| `plate` | Texto leído por OCR (mejor esfuerzo). |
| `success` | Igual a `plate_valid`; se mantiene por compatibilidad. |
| `plate_valid` | `true` si `plate` calza con un formato chileno conocido. |
| `status` | `OK`, `Formato de patente no reconocido` o error de OCR. |
| `detection_confidence` | Confianza de YOLO al localizar el recuadro. |
| `ocr_confidence` | Confianza de PaddleOCR para el texto elegido. |
| `bbox` | Coordenadas del recuadro en la imagen procesada. |

## Códigos de error comunes

- `400`: formato no soportado o contenido que no es una imagen válida.
- `401`: token inválido o expirado.
- `408`: el procesamiento superó el límite de 5 segundos.
- `413`: archivo demasiado grande (máximo 5 MB).
- `422`: falta el campo `file` o la petición no es multipart.
- `503`: el servicio está saturado (pool y cola llenos). Reintentar según `Retry-After`.

## Notas importantes

- La captura se decodifica y procesa en memoria; no se escribe en `/data`.
- Los modelos vienen dentro de la imagen Docker; no se montan volúmenes de caché.
- El endpoint procesa como máximo `PLATE_MAX_CONCURRENCY` imágenes a la vez y
  encola hasta `PLATE_MAX_QUEUE`. Al saturarse responde `503`.
- Al vencer `PLATE_TIMEOUT_S`, el trabajo se cancela en el próximo punto de
  control (entre etapas), no a mitad de una inferencia.
- La precisión del OCR depende de calidad, enfoque, iluminación y ángulo de la imagen.
- El contenedor expone el puerto interno `8000` y lo publica en `8000` del host.

## Datos de prueba y evaluación

Las imágenes de validación se guardan en `data/` y no se versionan. La convención
de nombres está documentada en [`data/README.md`](data/README.md):

```text
<dia|noche>_[etiqueta...]_<PATENTE>.jpg
```

## Pruebas y Evaluación de Métricas

### Preparación del entorno de pruebas (Docker)

La imagen de pruebas instala solo `requirements-dev.txt`: runtime mínimo para
importar la app y las herramientas de test. No instala PaddleOCR ni YOLO, y no
necesita los modelos ni un Python local. Los hashes del lock son para Linux
x86_64.

```bash
docker build -f Dockerfile.test -t sifa-plate-tests .
```

### Pruebas unitarias (`pytest`)

Las pruebas usan *mocks* para no cargar los modelos pesados. El montaje del
código permite editar tests sin reconstruir la imagen:

```bash
docker run --rm -it --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD":/app -w /app sifa-plate-tests
```

Son 39 pruebas y no requieren conexión a internet ni el artefacto de modelos.
Para ver el reporte de cobertura:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD":/app -w /app sifa-plate-tests \
  python -m pytest --cov=app --cov-report=term-missing --cov-report=html
```

### Evaluación de precisión de la IA

Procesa todas las imágenes de `data/` con los modelos reales y calcula métricas
de detección, acierto OCR, validez de formato, confianzas y tiempos. Requiere el
runtime completo y los modelos, no la imagen `sifa-plate-tests`. Con la imagen de
la app ya construida:

```bash
docker compose run --rm -v "$PWD":/app plate-api \
  python test/evaluate_validation.py
# alias histórico:
docker compose run --rm -v "$PWD":/app plate-api \
  python test/test_precision.py
```

Para exigir que todos los nombres cumplan la convención:

```bash
docker compose run --rm -v "$PWD":/app plate-api \
  python test/evaluate_validation.py --strict-tags
```

### Pruebas de carga

Requiere el servidor local levantado y, en Linux, usa la red del host:

```bash
docker run --rm --network host -v "$PWD":/app -w /app \
  sifa-plate-tests python test/test_carga.py
```
