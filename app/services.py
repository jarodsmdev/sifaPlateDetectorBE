"""
Módulo de Servicios de Negocio.
Orquesta el flujo de procesamiento de imágenes, conectando la detección (YOLO)
y el reconocimiento óptico (PaddleOCR).
"""
import logging

import cv2
import numpy as np

from app.model import detect_plate
from app.ocr import read_plate

logger = logging.getLogger("sifa.services")


def process_image_pipeline(img: np.ndarray):
    """
    Pipeline completo en memoria:
    1. Redimensiona la imagen si supera FullHD.
    2. Detecta patentes usando YOLO.
    3. Extrae texto y valida el formato de cada patente usando PaddleOCR.
    """
    if img is None or getattr(img, "size", 0) == 0:
        logger.error("Fallo al procesar una imagen vacía.")
        raise ValueError("No se pudo cargar la imagen")

    alto, ancho = img.shape[:2]
    logger.debug(f"Dimensiones originales de la imagen: {ancho}x{alto}")

    # Si la imagen es más grande que FullHD, la achicamos manteniendo la proporción
    if ancho > 1920 or alto > 1080:
        escala = min(1920 / ancho, 1080 / alto)
        nuevo_ancho = int(ancho * escala)
        nuevo_alto = int(alto * escala)
        img = cv2.resize(img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

    # Procesar detecciones (obtener coordenadas con YOLO)
    detections = detect_plate(img)
    output_data = []
    logger.info(f"YOLO encontró {len(detections)} posible(s) patente(s).")

    # Iterar sobre cada patente encontrada en la foto (pueden ser varias)
    for det in detections:
        bbox = {
            "x1": int(det.bounding_box.x1),
            "y1": int(det.bounding_box.y1),
            "x2": int(det.bounding_box.x2),
            "y2": int(det.bounding_box.y2),
        }

        plate_result = read_plate(img, bbox)

        output_data.append({
            "plate": plate_result["plate"],
            "success": plate_result["plate_valid"],
            "plate_valid": plate_result["plate_valid"],
            "status": plate_result["status"],
            "detection_confidence": float(det.confidence),
            "ocr_confidence": float(plate_result["ocr_confidence"]),
            "bbox": bbox,
        })
    return output_data
