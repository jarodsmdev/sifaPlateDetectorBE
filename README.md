# API de detección y lectura de patentes

Servicio en FastAPI para detectar patentes vehiculares en imágenes y extraer su texto mediante OCR.

## Qué hace este proyecto

- Detecta patentes usando un modelo YOLO de `open-image-models`.
- Recorta la zona detectada y aplica preprocesamiento de imagen.
- Ejecuta OCR con Tesseract para obtener el texto final de la patente.
- Expone un endpoint HTTP para integrarlo con otros sistemas.

## Estructura principal

- `app/main.py`: API FastAPI y endpoint `/detect`.
- `app/model.py`: carga y ejecución del detector de patentes.
- `app/ocr.py`: lógica de OCR y selección del mejor resultado.
- `Dockerfile`: imagen de ejecución con dependencias del sistema.
- `docker-compose.yml`: despliegue local en contenedor.

## Requisitos

### Opción recomendada (Docker)

- Docker
- Docker Compose

### Opción local (sin Docker)

- Python 3.10
- Dependencias de Python (`requirements.txt`)

## Levantar con Docker (recomendado)

1. Desde la carpeta del proyecto, construir y levantar:

```bash
docker compose up --build
```

1. El servicio quedará disponible en:

```text
http://localhost:8001/api/v1/plate/detect
```

Para detener:

```bash
docker compose down
```

## Levantar en local (sin Docker)

1. Crear y activar entorno virtual:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

Verificar instalacion:

1. Ejecutar API en local:

```bash
uvicorn app.main:app --reload --port 8001
```

## Uso del endpoint

### `POST /api/v1/plate/detect`

Recibe una imagen (`jpg` o `png`) como `multipart/form-data`.

- Tamaño máximo por archivo: 5 MB.
- Tipos permitidos: `image/jpeg`, `image/png`.

Ejemplo con `curl`:

```bash
curl -X POST "http://localhost:8001/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/ruta/a/tu/imagen.jpg"
```

Respuesta esperada (ejemplo):

```json
{
  "result": [
    {
      "plate": "ABCD12",
      "success": true,
      "status": "OK",
      "confidence": 0.93,
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

## Códigos de error comunes

- `400`: formato no soportado.
- `413`: archivo demasiado grande (máximo 5 MB).

## Notas importantes

- En el primer uso, el modelo de detección puede tardar más por carga/descarga inicial.
- La precisión del OCR depende de calidad, enfoque, iluminación y ángulo de la imagen.
- El contenedor expone el puerto interno `8000` y lo publica en `8001` del host.

## Datos de prueba

Puedes guardar imágenes de prueba en la carpeta `data/` (montada en el contenedor) para probar rápidamente con `curl`.
