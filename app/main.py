"""
Punto de entrada principal de la API FastAPI para SIFA.
Coordina la recepción de imágenes, la validación, la detección del modelo YOLO 
y la extracción de texto mediante PaddleOCR.
"""

import asyncio
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from datetime import datetime
from app.model import detect_plate
from app.ocr import read_plate, encode_plate_crop_base64

# Importaciones de configuración y esquemas extraídos
from app.config import (
    API_TITLE, API_VERSION, API_DESCRIPTION, API_SERVERS, 
    LOCAL_DEV_ORIGIN_REGEX, UPLOAD_DIR, MAX_FILE_SIZE
)
from app.schemas import PlateDetectionResponseSchema


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
        status.HTTP_400_BAD_REQUEST: {"description": "Formato de archivo inválido. Solo JPG/PNG."},
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"description": "El archivo excede el límite estructural de 5MB."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Fallo crítico interno en el motor OCR o IA."}
    }
)
async def detect(file: UploadFile = File(..., description="Fotografía capturada por el fiscalizador (JPG/PNG)"))):
    """
    Endpoint principal para detectar y leer patentes.
    
    1. Recibe la imagen y valida su formato y tamaño.
    2. Guarda la imagen temporalmente en el disco.
    3. Busca las coordenadas de la patente con YOLO.
    4. Recorta y lee el texto de la patente con PaddleOCR.
    5. Devuelve los resultados estructurados y elimina el archivo temporal.
    """

    # Generar un nombre único con fecha y hora para ordenar fácilmente el historial de fotos
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = uuid.uuid4().hex[:6]
    filename = f"patente_{timestamp_str}_{random_id}.jpg"
    
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Validación de formato
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Formato no soportado")

    # Validar tamaño del archivo
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máximo 5MB)")

    # Guardar archivo en disco para que los modelos de IA puedan procesarlo
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    img = cv2.imread(file_path)

    # Obtener dimensiones originales
    alto, ancho = img.shape[:2]

    # Si la imagen es más grande que FullHD, la achicamos manteniendo la proporción
    if ancho > 1920 or alto > 1080:
        escala = min(1920/ancho, 1080/alto)
        nuevo_ancho = int(ancho * escala)
        nuevo_alto = int(alto * escala)
        img_reducida = cv2.resize(img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
        cv2.imwrite(file_path, img_reducida) # Sobrescribir con la versión ligera

    # Procesar detecciones (obtener coordenadas con YOLO) 
    # FastAPI atiende la petición, pero manda el trabajo 
    # matemático pesado a otro hilo, quedando libre para recibir al segundo fiscalizador al instante.
    detections = await asyncio.to_thread(detect_plate, file_path)
    output = []

    # Iterar sobre cada patente encontrada en la foto (pueden ser varias)
    for det in detections:
        bbox = {
            "x1": int(det.bounding_box.x1),
            "y1": int(det.bounding_box.y1),
            "x2": int(det.bounding_box.x2),
            "y2": int(det.bounding_box.y2),
        }

        # Convertir el recorte a Base64 y leer el texto con PaddleOCR
        image_base64 = encode_plate_crop_base64(file_path, bbox)
        plate_result = read_plate(file_path, bbox)

        # Estructurar la respuesta
        # se quitó el campo "image" de la respuesta ya que toma mucho tiempo para enviar por la red
        output.append({
            "plate": plate_result["plate"],
            "success": plate_result["success"],
            "status": plate_result["status"],
            "confidence": float(det.confidence),
            "bbox": bbox,
        })

    # Limpieza: Eliminar la foto del servidor para no llenar el disco
    # os.remove(file_path)

    # Agregar marca de tiempo en formato ISO 8601 (estándar UTC)
    timestamp = datetime.utcnow().isoformat() + "Z"  # UTC en ISO 8601

    return {"result": output, "timestamp": timestamp}