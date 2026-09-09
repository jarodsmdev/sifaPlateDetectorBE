"""
Punto de entrada principal de la API FastAPI para SIFA.
Coordina la recepción de imágenes, la validación, la detección del modelo YOLO 
y la extracción de texto mediante PaddleOCR.
"""

import asyncio
import cv2
import time
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from datetime import datetime, timezone
from app.services import process_image_pipeline

# Configuración de logging: Usamos el logger de uvicorn para que nuestros
# mensajes se impriman junto con los logs de acceso por defecto del servidor.
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# Importaciones de configuración y esquemas extraídos
from app.config import (
    API_TITLE, API_VERSION, API_DESCRIPTION, API_SERVERS, 
    LOCAL_DEV_ORIGIN_REGEX, UPLOAD_DIR, MAX_FILE_SIZE
)
from app.schemas import PlateDetectionResponseSchema

# Importar dependencia de autenticación y autorización por rol
from app.auth import require_role


# Inicialización de la aplicación utilizando la configuración externa
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    servers=API_SERVERS
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=LOCAL_DEV_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post(
    "/plate/api/v1/detect",
    response_model=PlateDetectionResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Procesar captura multimedia y extraer caracteres alfanuméricos",
    tags=["Procesamiento de Visión Artificial"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Token inválido o expirado."},
        status.HTTP_400_BAD_REQUEST: {"description": "Formato de archivo inválido. Solo JPG/PNG."},
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
    2. Guarda la imagen temporalmente en el disco.
    3. Delega el procesamiento pesado a un hilo secundario con un límite de 5 segundos.
    4. Devuelve los resultados estructurados y elimina el archivo temporal.
    """

    logger.info("---------------------------------")

    # Generar un nombre único con fecha y hora para ordenar fácilmente el historial de fotos
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = uuid.uuid4().hex[:6]
    filename = f"patente_{timestamp_str}_{random_id}.jpg"
    
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Validación de formato
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning(f"Rechazado: Formato de archivo no soportado ({file.content_type}).")
        raise HTTPException(status_code=400, detail="Formato no soportado")

    # Validar tamaño del archivo
    contents = await file.read()
    file_size_kb = len(contents) / 1024
    if len(contents) > MAX_FILE_SIZE:
        logger.warning(f"Rechazado: Archivo demasiado grande ({file_size_kb:.2f} KB).")
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máximo 5MB)")

    logger.info(f"Petición aceptada: {filename} ({file_size_kb:.2f} KB).")

    # Guardar archivo en disco para que los modelos de IA puedan procesarlo
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    start_time = time.time()
    try:
        # Enviar el trabajo pesado a otro hilo, quedando libre para recibir al segundo fiscalizador al instante.
        # Esperamos un máximo de 5.0 segundos por el procesamiento completo.
        output = await asyncio.wait_for(asyncio.to_thread(process_image_pipeline, file_path), timeout=5.0)
        
        # 'output' es una lista de diccionarios, tomamos el primero
        plate_str = output[0].get("plate") if output else "Ninguna"
        
        processing_time = time.time() - start_time
        logger.info(f"Procesamiento exitoso para {filename} en {processing_time:.2f}s. Patente: {plate_str}.")
    except asyncio.TimeoutError:
        logger.error(f"Timeout al procesar {filename}. Tiempo excedió los 5 segundos.")
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="El procesamiento de la imagen superó el tiempo límite de 5 segundos.")
    except Exception as e:
        logger.error(f"Error interno procesando {filename}: {str(e)}", exc_info=True)
        # Capturamos otros errores (por ejemplo ValueError de la imagen)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error interno procesando imagen: {str(e)}")
    finally:
        # Limpieza: Asegurar eliminación de la foto del servidor para no llenar el disco,
        # incluso si ocurre un timeout o error.
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass # Evitar que falle silenciosamente si el archivo estaba bloqueado

    # Agregar marca de tiempo en formato ISO 8601 (estándar UTC)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # UTC en ISO 8601

    return {"result": output, "timestamp": timestamp}