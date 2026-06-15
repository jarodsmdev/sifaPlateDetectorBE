"""
Módulo de Servicios de Negocio.
Orquesta el flujo de procesamiento de imágenes, conectando la detección (YOLO)
y el reconocimiento óptico (PaddleOCR).
"""
import cv2
import logging
from app.model import detect_plate
from app.ocr import read_plate, encode_plate_crop_base64

logger = logging.getLogger("sifa.services")

def process_image_pipeline(path: str):
    """
    Pipeline completo:
    1. Carga y opcionalmente redimensiona la imagen.
    2. Detecta patentes usando YOLO.
    3. Extrae texto de cada patente usando PaddleOCR.
    """
    logger.info(f"Iniciando procesamiento de imagen en la ruta: {path}")
    img = cv2.imread(path)
    if img is None:
         logger.error(f"Fallo al cargar la imagen con OpenCV en la ruta: {path}")
         raise ValueError("No se pudo cargar la imagen")

    # Obtener dimensiones originales
    alto, ancho = img.shape[:2]
    logger.debug(f"Dimensiones originales de la imagen: {ancho}x{alto}")

    # Si la imagen es más grande que FullHD, la achicamos manteniendo la proporción
    if ancho > 1920 or alto > 1080:
        escala = min(1920/ancho, 1080/alto)
        nuevo_ancho = int(ancho * escala)
        nuevo_alto = int(alto * escala)
        img = cv2.resize(img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, img) # Sobrescribir con la versión ligera para YOLO

    # Procesar detecciones (obtener coordenadas con YOLO) 
    detections = detect_plate(path)
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

        # Convertir el recorte a Base64 y leer el texto con PaddleOCR
        # Se pasa la imagen cargada en memoria para evitar lecturas de disco repetidas
        image_base64 = encode_plate_crop_base64(img, bbox)
        plate_result = read_plate(img, bbox)

        # Estructurar la respuesta
        output_data.append({
            "plate": plate_result["plate"],
            "success": plate_result["success"],
            "status": plate_result["status"],
            "confidence": float(det.confidence),
            "bbox": bbox,
        })
    return output_data