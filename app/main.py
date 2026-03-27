from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import uuid
import os
from datetime import datetime
from app.model import detect_plate
from app.ocr import read_plate

app = FastAPI()

UPLOAD_DIR = "/temp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Validación de tamaño (máximo 5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Validación de formato
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Formato no soportado")

    # Validar tamaño del archivo
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máximo 5MB)")

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    # Procesar detecciones
    detections = detect_plate(file_path)
    output = []

    for det in detections:
        bbox = {
            "x1": int(det.bounding_box.x1),
            "y1": int(det.bounding_box.y1),
            "x2": int(det.bounding_box.x2),
            "y2": int(det.bounding_box.y2),
        }

        plate_result = read_plate(file_path, bbox)
        output.append({
            "plate": plate_result["plate"],
            "success": plate_result["success"],
            "status": plate_result["status"],
            "confidence": float(det.confidence),
            "bbox": bbox
        })

    os.remove(file_path)

    # Agregar timestamp
    timestamp = datetime.utcnow().isoformat() + "Z"  # UTC en ISO 8601

    return {"result": output, "timestamp": timestamp}