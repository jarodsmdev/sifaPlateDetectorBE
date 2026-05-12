"""
Módulo de Reconocimiento Óptico de Caracteres (OCR).
Utiliza PaddleOCR para extraer texto de recortes de imagen y aplica
heurísticas específicas para limpiar y corregir patentes chilenas.
"""

import cv2
import re
import numpy as np
import base64
import logging
from paddleocr import PaddleOCR

# INICIALIZACIÓN GLOBAL DEL MODELO
# Silenciamos los logs de depuración nativos de Paddle y cargamos el modelo.
# Se inicializa fuera de las funciones para que FastAPI lo mantenga en RAM.
logging.getLogger("ppocr").setLevel(logging.ERROR)
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

# Configuración de márgenes y tamaños
PLATE_CROP_PADDING = -5
MIN_PLATE_LEN = 5
MAX_PLATE_LEN = 6

# Expresión regular para validar matrices chilenas:
# Opción 1: 2 Letras + 4 Números (Ej: AB1234)
# Opción 2: 4 Letras + 2 Números excluyendo vocales y la Ñ (Ej: BCDF12)
CHILE_PLATE_PATTERN = re.compile(r'^([A-Z]{2}[0-9]{4}|[B-DF-HJ-NP-TV-Z]{4}[0-9]{2})$')

def correct_common_mistakes(text):
    """
    Aplica correcciones lógicas al texto detectado basándose en los 
    formatos de patentes chilenas para evitar confusiones comunes de OCR 
    (Ej: confundir el número '0' con la letra 'O').
    """

    if len(text) != 6:
        return text
    
    # Diccionarios de confusiones visuales comunes
    letter_to_number = {'O': '0', 'I': '1', 'A': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}
    number_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '7': 'T'}
    
    corrected = list(text)
    
    # Regla Formato Nuevo: 4 letras, 2 números (Ej: TZPW11)
    if corrected[0].isalpha() and corrected[1].isalpha() and corrected[2].isalpha():
        # Los dos últimos caracteres DEBEN ser forzados a números
        for i in [4, 5]:
            if corrected[i].isalpha() and corrected[i] in letter_to_number:
                corrected[i] = letter_to_number[corrected[i]]
        
        # Los primeros cuatro caracteres DEBEN ser forzados a letras
        for i in [0, 1, 2, 3]:
            if corrected[i].isdigit() and corrected[i] in number_to_letter:
                corrected[i] = number_to_letter[corrected[i]]
                
    # Regla Formato Antiguo: 2 letras, 4 números (Ej: BKJY33)
    elif corrected[0].isalpha() and corrected[1].isalpha() and corrected[2].isdigit():
        # Los últimos cuatro caracteres DEBEN ser forzados a números
        for i in [2, 3, 4, 5]:
            if corrected[i].isalpha() and corrected[i] in letter_to_number:
                corrected[i] = letter_to_number[corrected[i]]
        
        # Los primeros dos caracteres DEBEN ser forzados a letras
        for i in [0, 1]:
            if corrected[i].isdigit() and corrected[i] in number_to_letter:
                corrected[i] = number_to_letter[corrected[i]]

    return "".join(corrected)

def clean(text):
    """Limpia el texto eliminando cualquier carácter que no sea alfanumérico."""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

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

def encode_plate_crop_base64(image_path, bbox):
    """Recorta la patente y la convierte a un string Base64 para enviarla al Frontend."""
    img = cv2.imread(image_path)
    if img is None:
        return ""
    crop = _get_plate_crop(img, bbox)
    if crop.size == 0:
        return ""
    ok, buffer = cv2.imencode('.jpg', crop)
    if not ok:
        return ""
    return base64.b64encode(buffer.tobytes()).decode('utf-8')

def read_plate(image_path, bbox):
    """
    Ejecuta el OCR sobre el área específica de la patente.
    
    Args:
        image_path (str): Ruta de la imagen completa.
        bbox (dict): Coordenadas de la patente.
        
    Returns:
        dict: Diccionario con la patente limpia, el estado de éxito y un mensaje.
    """

    # Ejecuta el OCR sobre el área específica de la patente
    img = cv2.imread(image_path)
    if img is None:
        return {"plate": "", "success": False, "status": "Imagen no cargada"}

    crop = _get_plate_crop(img, bbox)
    if crop.size == 0:
        return {"plate": "", "success": False, "status": "Recorte de patente inválido"}

    # Ejecutamos el modelo PaddleOCR directamente sobre el recorte a color
    # Al agregar det=False le decimos a al modelo PaddleOCR que no busque donde está el texto
    # ya que este se entrega con el recorte que realiza yolo.
    result = ocr.ocr(crop, det=False, cls=False)

    if not result or not result[0]:
        return {"plate": "", "success": False, "status": "OCR ha fallado"}

    candidates = []

    # Iterar sobre todas las líneas de texto que el OCR logró encontrar en el recorte
    for line in result[0]:
        text_detected = line[0] # Extraemos el texto directamente de la primera posición
        cleaned_text = clean(text_detected)
        
        if not cleaned_text:
            continue

        # Aplicar lógica de reemplazo de caracteres confusos
        corrected_text = correct_common_mistakes(cleaned_text)
        
        # Early Stopping: Si el texto encaja perfectamente en el regex de Chile, 
        # detenemos la búsqueda inmediatamente asumiendo que es la lectura correcta.
        if CHILE_PLATE_PATTERN.match(corrected_text):
            return {"plate": corrected_text, "success": True, "status": "OK"}
            
        candidates.append(corrected_text)

    # PLAN B: Si el OCR leyó texto pero no hizo match perfecto con el regex,
    # filtramos los textos que tienen el largo aproximado de una patente.
    valid = [t for t in candidates if MIN_PLATE_LEN <= len(t) <= MAX_PLATE_LEN]
    
    if not valid:
        return {"plate": "", "success": False, "status": "La detección OCR no es confiable"}

    # Por heurística, asumimos que la lectura más larga dentro del rango válido es la patente
    # ignorando logos pequeños o textos como "CHILE".
    final_text = max(valid, key=len)

    return {"plate": final_text, "success": True, "status": "OK"}
