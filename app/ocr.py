"""
Módulo de Reconocimiento Óptico de Caracteres (OCR).
Utiliza PaddleOCR para extraer texto de recortes de imagen y delega la
normalización, corrección y validación de patentes a ``app.plate_format``.
"""

import logging

from paddleocr import PaddleOCR

from app.plate_format import parse_ocr_candidates

# INICIALIZACIÓN GLOBAL DEL MODELO
# Silenciamos los logs de depuración nativos de Paddle y cargamos el modelo.
# Se inicializa fuera de las funciones para que FastAPI lo mantenga en RAM.
logging.getLogger("ppocr").setLevel(logging.ERROR)
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

# Configuración de márgenes
PLATE_CROP_PADDING = -5


def _get_plate_crop(img, bbox, pad=PLATE_CROP_PADDING):
    """
    Recorta la imagen original utilizando las coordenadas del bounding box,
    aplicando un margen (padding) para centrar mejor el texto.
    """
    h_img, w_img = img.shape[:2]

    # Evitar coordenadas negativas o recortes que excedan las dimensiones de la imagen
    x1 = max(0, bbox["x1"] - pad)
    y1 = max(0, bbox["y1"] - pad)
    x2 = min(w_img, bbox["x2"] + pad)
    y2 = min(h_img, bbox["y2"] + pad)
    return img[y1:y2, x1:x2]


def _extract_candidates(result):
    """Convierte la salida de PaddleOCR en pares ``(texto, confianza)``."""
    if not result or not result[0]:
        return []

    raw = result[0]
    if isinstance(raw, tuple) and len(raw) >= 2 and isinstance(raw[0], str):
        raw = [raw]

    candidates = []
    for line in raw:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        if not isinstance(line[0], str):
            continue
        try:
            score = float(line[1])
        except (TypeError, ValueError):
            score = 0.0
        candidates.append((line[0], score))

    return candidates


def read_plate(img, bbox):
    """
    Ejecuta el OCR sobre el área específica de la patente.

    Args:
        img (numpy.ndarray): Imagen en memoria (OpenCV).
        bbox (dict): Coordenadas de la patente.

    Returns:
        dict: Patente, validez de formato, estado y confianza del OCR.
    """
    if img is None:
        return {
            "plate": "",
            "plate_valid": False,
            "success": False,
            "status": "Imagen no cargada",
            "ocr_confidence": 0.0,
        }

    crop = _get_plate_crop(img, bbox)
    if crop.size == 0:
        return {
            "plate": "",
            "plate_valid": False,
            "success": False,
            "status": "Recorte de patente inválido",
            "ocr_confidence": 0.0,
        }

    # det=False: YOLO ya localizó el texto; cls=False: el recorte viene derecho.
    result = ocr.ocr(crop, det=False, cls=False)
    candidates = _extract_candidates(result)

    if not candidates:
        return {
            "plate": "",
            "plate_valid": False,
            "success": False,
            "status": "OCR ha fallado",
            "ocr_confidence": 0.0,
        }

    parsed = parse_ocr_candidates(candidates)
    parsed["success"] = parsed["plate_valid"]
    return parsed
