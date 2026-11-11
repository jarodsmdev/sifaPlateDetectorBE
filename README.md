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
- `docker-compose.yml`: despliegue local en contenedor.

## Requisitos

### Opción recomendada (Docker)

- Docker
- Docker Compose

### Opción local (sin Docker)

- Python 3.10+
- Dependencias de Python (`requirements.txt`)

## Levantar con Docker (recomendado)

```bash
docker compose up --build
```

El servicio quedará disponible en:

```text
http://localhost:8000/plate/api/v1/detect
```

Para detener:

```bash
docker compose down
```

## Levantar en local (sin Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

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

## Notas importantes

- La captura se decodifica y procesa en memoria; no se escribe en `/data`.
- En el primer uso, los modelos pueden tardar más por carga/descarga inicial.
- La precisión del OCR depende de calidad, enfoque, iluminación y ángulo de la imagen.
- El contenedor expone el puerto interno `8000` y lo publica en `8000` del host.

## Datos de prueba y evaluación

Las imágenes de validación se guardan en `data/` y no se versionan. La convención
de nombres está documentada en [`data/README.md`](data/README.md):

```text
<dia|noche>_[etiqueta...]_<PATENTE>.jpg
```

## Pruebas y Evaluación de Métricas

### Preparación del entorno local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov httpx
```

### Pruebas unitarias (`pytest`)

Las pruebas usan *mocks* para no descargar los modelos pesados:

```bash
pytest
```

Para ver el reporte de cobertura:

```bash
pytest --cov=app --cov-report=html
```

### Evaluación de precisión de la IA

Procesa todas las imágenes de `data/` con los modelos reales y calcula métricas
de detección, acierto OCR, validez de formato, confianzas y tiempos:

```bash
python test/evaluate_validation.py
# alias histórico:
python test/test_precision.py
```

Para exigir que todos los nombres cumplan la convención:

```bash
python test/evaluate_validation.py --strict-tags
```

### Pruebas de carga

```bash
# Requiere el servidor local levantado
python test/test_carga.py
```
