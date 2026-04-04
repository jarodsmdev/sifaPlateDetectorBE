import cv2
import re
import numpy as np
import base64
import logging
from paddleocr import PaddleOCR


# SILENCIAR LOGS Y CARGAR EL MODELO UNA SOLA VEZ
# Esto es vital para FastAPI: el modelo se carga en memoria al arrancar la API, no en cada petición.
logging.getLogger("ppocr").setLevel(logging.ERROR)
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

PLATE_CROP_PADDING = -5
MIN_PLATE_LEN = 5
MAX_PLATE_LEN = 6

# formato nuevo o antiguo de la patente
CHILE_PLATE_PATTERN = re.compile(r'^([A-Z]{2}[0-9]{4}|[B-DF-HJ-NP-TV-Z]{4}[0-9]{2})$')

def correct_common_mistakes(text):
    if len(text) != 6:
        return text
    
    letter_to_number = {'O': '0', 'I': '1', 'A': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}
    number_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '7': 'T'}
    
    corrected = list(text)
    
    # Regla Formato Nuevo: 4 letras, 2 números (Ej: TZPW11)
    if corrected[0].isalpha() and corrected[1].isalpha() and corrected[2].isalpha():
        for i in [4, 5]: # Los dos últimos DEBEN ser números
            if corrected[i].isalpha() and corrected[i] in letter_to_number:
                corrected[i] = letter_to_number[corrected[i]]
        for i in [0, 1, 2, 3]: # Los 4 primeros DEBEN ser letras
            if corrected[i].isdigit() and corrected[i] in number_to_letter:
                corrected[i] = number_to_letter[corrected[i]]
                
    # Regla Formato Antiguo: 2 letras, 4 números (Ej: BKJY33)
    elif corrected[0].isalpha() and corrected[1].isalpha() and corrected[2].isdigit():
        for i in [2, 3, 4, 5]: # Los últimos 4 DEBEN ser números
            if corrected[i].isalpha() and corrected[i] in letter_to_number:
                corrected[i] = letter_to_number[corrected[i]]
        for i in [0, 1]: # Las primeras 2 DEBEN ser letras
            if corrected[i].isdigit() and corrected[i] in number_to_letter:
                corrected[i] = number_to_letter[corrected[i]]

    return "".join(corrected)

def clean(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def _get_plate_crop(img, bbox, pad=PLATE_CROP_PADDING):
    h_img, w_img = img.shape[:2]
    x1 = max(0, bbox["x1"] - pad)
    y1 = max(0, bbox["y1"] - pad)
    x2 = min(w_img, bbox["x2"] + pad)
    y2 = min(h_img, bbox["y2"] + pad)
    return img[y1:y2, x1:x2]

def encode_plate_crop_base64(image_path, bbox):
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
    img = cv2.imread(image_path)
    if img is None:
        return {"plate": "", "success": False, "status": "Imagen no cargada"}

    crop = _get_plate_crop(img, bbox)
    if crop.size == 0:
        return {"plate": "", "success": False, "status": "Recorte de patente inválido"}

    # EJECUCIÓN DE PADDLEOCR
    # Le pasamos el recorte a color directamente. No necesita filtros de OpenCV.
    result = ocr.ocr(crop, cls=False)

    if not result or not result[0]:
        return {"plate": "", "success": False, "status": "OCR ha fallado"}

    candidates = []

    # result[0] contiene las líneas de texto encontradas
    # Formato: [[[x,y],[x,y],[x,y],[x,y]], ('Texto', confianza)]
    for line in result[0]:
        text_detected = line[1][0]
        cleaned_text = clean(text_detected)
        
        if not cleaned_text:
            continue

        corrected_text = correct_common_mistakes(cleaned_text)
        
        # Early Stopping
        if CHILE_PLATE_PATTERN.match(corrected_text):
            return {"plate": corrected_text, "success": True, "status": "OK"}
            
        candidates.append(corrected_text)

    # PLAN B: Si no hizo match perfecto, buscamos la cadena más lógica
    valid = [t for t in candidates if MIN_PLATE_LEN <= len(t) <= MAX_PLATE_LEN]
    
    if not valid:
        return {"plate": "", "success": False, "status": "La detección OCR no es confiable"}

    # Generalmente la matrícula es el texto más largo de la patente
    final_text = max(valid, key=len)

    return {"plate": final_text, "success": True, "status": "OK"}

    def is_better(a, b):
        if len(a) == 6 and len(b) != 6:
            return True
        if len(b) == 6 and len(a) != 6:
            return False
        return _score_plate_candidate(a) > _score_plate_candidate(b)

    final_text = ""

    for candidate in valid:
        if not final_text:
            final_text = candidate
        elif is_better(candidate, final_text):
            final_text = candidate

    if len(final_text) < MIN_PLATE_LEN:
        return {"plate": "", "success": False, "status": "La detección OCR no es confiable"}

    if len(final_text) > MAX_PLATE_LEN:
        final_text = final_text[:MAX_PLATE_LEN]

    if len(set(final_text)) <= 2:
        return {"plate": "", "success": False, "status": "OCR ha fallado"}

    # Debug
    print("[!] OCR DEBUG:")
    for i, c in enumerate(candidates):
        print(f"  [{i + 1}] {c}")
    print(f"  FINAL: {final_text}")

    return {"plate": final_text, "success": True, "status": "OK"}