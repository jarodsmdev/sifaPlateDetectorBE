"""
Punto de entrada principal de la API FastAPI para SIFA.
Coordina la recepción de imágenes, la validación, la detección del modelo YOLO 
y la extracción de texto mediante PaddleOCR.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from datetime import datetime
from app.model import detect_plate
from app.ocr import read_plate, encode_plate_crop_base64

app = FastAPI()

# Configuración de CORS para permitir peticiones desde el emulador Android o localhost
# LOCAL_DEV_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

# app.add_middleware(
#     CORSMiddleware,
#     allow_origin_regex=LOCAL_DEV_ORIGIN_REGEX,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Configuración del almacenamiento temporal
# Crear directorio temporal para uploads
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Validación de seguridad: límite de 5MB por foto para no saturar la RAM
MAX_FILE_SIZE = 5 * 1024 * 1024

@app.post("/api/v1/plate/detect")
async def detect(file: UploadFile = File(...)):
    """
    Endpoint principal para detectar y leer patentes.
    
    1. Recibe la imagen y valida su formato y tamaño.
    2. Guarda la imagen temporalmente en el disco.
    3. Busca las coordenadas de la patente con YOLO.
    4. Recorta y lee el texto de la patente con PaddleOCR.
    5. Devuelve los resultados estructurados y elimina el archivo temporal.
    """
    try:
        # Generar un nombre único para evitar colisiones si hay peticiones simultáneas
        filename = f"{uuid.uuid4()}.jpg"
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

        # Procesar detecciones(obtener coordenadas con YOLO)
        detections = detect_plate(file_path)

        if not detections:
            raise HTTPException(status_code=404, detail="No se detectaron patentes")

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

            if not plate_result["success"]:
                raise HTTPException(status_code=422, detail="Error leyendo patente")

            # Estructurar la respuesta
            output.append({
                "plate": plate_result["plate"],
                "success": plate_result["success"],
                "status": plate_result["status"],
                "confidence": float(det.confidence),
                "bbox": bbox,
                "image": image_base64
            })

        # Agregar timestamp
        timestamp = datetime.utcnow().isoformat() + "Z"  # UTC en ISO 8601

        return {"result": output, "timestamp": timestamp}

    except HTTPException as e:
        raise e # Respeta errores definidos

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail="Error interno procesando imagen") from e

    finally:
        # Limpiar el archivo temporal para no saturar el disco
        if os.path.exists(file_path):
            os.remove(file_path)
