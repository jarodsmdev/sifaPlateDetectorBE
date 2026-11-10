"""
Punto de entrada principal de la API FastAPI para SIFA.
Coordina la recepción de imágenes, la validación, la detección del modelo YOLO
y la extracción de texto mediante PaddleOCR.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.services import process_image_pipeline

# Configuración de logging: Usamos el logger de uvicorn para que nuestros
# mensajes se impriman junto con los logs de acceso por defecto del servidor.
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# Importaciones de configuración y esquemas extraídos
from app.config import (
    API_TITLE, API_VERSION, API_DESCRIPTION, API_SERVERS,
    LOCAL_DEV_ORIGIN_REGEX, MAX_FILE_SIZE
)
from app.schemas import PlateDetectionResponseSchema

# Importar dependencia de autenticación y autorización por rol
from app.auth import require_role
from app.channel_guard import ChannelKeyMiddleware, channel_key_from_env, warn_if_disabled


# Inicialización de la aplicación utilizando la configuración externa
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    servers=API_SERVERS
)

# Guardia del canal interno: exige X-Internal-Key en cada petición que no venga
# de loopback. Se registra ANTES que CORS porque en Starlette lo último que se
# añade queda por fuera: así CORS sigue siendo el exterior y se encarga del
# preflight, y este guardia solo ve las peticiones que sí llegan al backend.
_channel_key = channel_key_from_env()
warn_if_disabled(_channel_key)
app.add_middleware(ChannelKeyMiddleware, channel_key=_channel_key)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=LOCAL_DEV_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
    "/plate/api/v1/detect",
    response_model=PlateDetectionResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Procesar captura multimedia y extraer caracteres alfanuméricos",
    tags=["Procesamiento de Visión Artificial"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Token inválido o expirado."},
        status.HTTP_400_BAD_REQUEST: {"description": "Formato o contenido de imagen inválido. Solo JPG/PNG."},
        status.HTTP_408_REQUEST_TIMEOUT: {"description": "El procesamiento excedió el tiempo máximo permitido (5 segundos)."},
        status.HTTP_413_CONTENT_TOO_LARGE: {"description": "El archivo excede el límite estructural de 5MB."},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Error de validación: el campo 'file' no fue enviado o la petición no es multipart/form-data."}
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "Fotografía capturada por el fiscalizador (JPG/PNG)"
                            }
                        },
                        "required": ["file"]
                    }
                }
            },
            "required": True
        }
    }
)
async def detect(
    file: UploadFile = File(...),
    user: dict = Depends(require_role("USER_APP"))
    ):
    """
    Endpoint principal para detectar y leer patentes.

    1. Recibe la imagen y valida su formato y tamaño.
    2. Decodifica la imagen en memoria, sin escribirla en disco.
    3. Delega el procesamiento pesado a un hilo secundario con un límite de 5 segundos.
    4. Devuelve los resultados estructurados.
    """

    logger.info("---------------------------------")

    # Validación de formato
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning(f"Rechazado: Formato de archivo no soportado ({file.content_type}).")
        raise HTTPException(status_code=400, detail="Formato no soportado")

    # Validar tamaño leyendo como máximo un byte extra
    contents = await file.read(MAX_FILE_SIZE + 1)
    file_size_kb = len(contents) / 1024
    if len(contents) > MAX_FILE_SIZE:
        logger.warning(f"Rechazado: Archivo demasiado grande ({file_size_kb:.2f} KB).")
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máximo 5MB)")

    # Decodificar en memoria: evita persistir la captura y valida el contenido real
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Rechazado: el contenido no es una imagen válida.")
        raise HTTPException(status_code=400, detail="Imagen inválida o corrupta")

    logger.info(f"Petición aceptada: {file.content_type} ({file_size_kb:.2f} KB).")

    start_time = time.time()
    try:
        # Enviar el trabajo pesado a otro hilo, quedando libre para recibir al segundo fiscalizador al instante.
        # Esperamos un máximo de 5.0 segundos por el procesamiento completo.
        output = await asyncio.wait_for(
            asyncio.to_thread(process_image_pipeline, img),
            timeout=5.0,
        )

        processing_time = time.time() - start_time
        logger.info(f"Procesamiento exitoso en {processing_time:.2f}s. Detecciones: {len(output)}.")
    except asyncio.TimeoutError:
        logger.error("Timeout al procesar la imagen. Tiempo excedió los 5 segundos.")
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="El procesamiento de la imagen superó el tiempo límite de 5 segundos.")
    except ValueError:
        logger.warning("La imagen no tiene contenido procesable.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Imagen inválida o corrupta")
    except Exception:
        logger.exception("Error interno procesando la imagen.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno procesando imagen")

    # Agregar marca de tiempo en formato ISO 8601 (estándar UTC)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return {"result": output, "timestamp": timestamp}
