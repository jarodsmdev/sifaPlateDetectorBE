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

- Python 3.10+
- `tesseract-ocr` instalado en el sistema
- Dependencias de Python (`requirements.txt`)

## Levantar con Docker (recomendado)

1. Desde la carpeta del proyecto, construir y levantar:

```bash
docker compose up --build
```

1. El servicio quedará disponible en:

```text
http://localhost:8001/detect
```

Para detener:

```bash
docker compose down
```

## Levantar en local (sin Docker)

1. Crear y activar entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

1. Instalar Tesseract en el sistema operativo (esto NO lo instala `pip`):

Fedora:

```bash
sudo dnf install -y tesseract
```

Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

Verificar instalacion:

```bash
tesseract --version
```

1. Ejecutar API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Si `tesseract` no esta en tu `PATH`, puedes definirlo manualmente:

```bash
export TESSERACT_CMD=/usr/bin/tesseract
```

## Uso del endpoint

### `POST /detect`

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

Puedes guardar imágenes de prueba en la carpeta `data/` para evaluarlas o probar rápidamente con `curl`.

---

## Pruebas y Evaluación de Métricas

El proyecto cuenta con un entorno de pruebas unitarias (`pytest`) y un módulo de evaluación de métricas de precisión de la IA para presentar estadísticas en tiempo real.

### Preparación del entorno local (Desde Cero)
Antes de ejecutar cualquier prueba o script en tu máquina local, debes preparar tu entorno de desarrollo siguiendo estos pasos:

```bash
# 1. Crear el entorno virtual (si no lo has creado antes)
python -m venv .venv

# 2. Activar el entorno virtual
# En Windows (PowerShell):
.venv\Scripts\Activate.ps1
# En Linux/macOS o Git Bash:
# source .venv/bin/activate

# 3. Instalar las dependencias principales de la aplicación (Modelos de IA, YOLO, PaddleOCR, OpenCV, etc.)
pip install -r requirements.txt

# 4. Instalar las dependencias adicionales para las pruebas y la evaluación de métricas
pip install pytest pytest-cov httpx pillow
```


### 1. Pruebas Unitarias y Cobertura (`pytest`)
Ejecuta las pruebas unitarias automáticas (las cuales utilizan *mocks* para no requerir la descarga de los modelos pesados de IA y correr en milisegundos):
```bash
pytest
```
* **Ver reporte de cobertura**: Puedes abrir el reporte interactivo HTML generado en `htmlcov/index.html`. En Windows PowerShell puedes abrirlo ejecutando:
  ```powershell
  Start-Process "htmlcov/index.html"
  ```

### 2. Módulo de Evaluación de Precisión de la IA (Métricas)
Este script procesa todas las imágenes en la carpeta `data/` utilizando los modelos de IA locales reales (YOLOv9 y PaddleOCR). Mide la exactitud de lectura y calcula estadísticas detalladas sobre los tiempos de respuesta del servidor (Media, Mediana y Moda).

> [!IMPORTANT]
> Para calcular la precisión del OCR, debes nombrar las imágenes dentro de `data/` con su patente real (ej: `ABCD12.jpg` o `ABCD12_auto.png`). Las imágenes con nombres genéricos se omitirán del cálculo de precisión.

Para iniciar la evaluación:
```bash
python test/test_precision.py
```

### 3. Pruebas de Carga y Simulación de Estrés
Para simular peticiones simultáneas y paralelas de múltiples fiscalizadores interactuando con la API local:
```bash
# Recuerda levantar primero el servidor local (ej: uvicorn app.main:app)
python test/test_carga.py
```
