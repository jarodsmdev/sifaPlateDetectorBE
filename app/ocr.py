import cv2
import pytesseract
import re
import numpy as np
import os
import shutil
import base64

PLATE_CROP_PADDING = -5
MIN_PLATE_LEN = 5
MAX_PLATE_LEN = 6

#formato nuevo o antiguo de la patente
CHILE_PLATE_PATTERN = re.compile(r'^([A-Z]{2}[0-9]{4}|[B-DF-HJ-NP-TV-Z]{4}[0-9]{2})$')


def _resolve_tesseract_cmd():
    custom_cmd = os.getenv("TESSERACT_CMD")
    if custom_cmd:
        if os.path.isabs(custom_cmd):
            return custom_cmd if os.path.isfile(custom_cmd) else None
        return shutil.which(custom_cmd)
    return shutil.which("tesseract")

def clean(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_correction(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            rect = order_points(approx)
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = int(max(heightA, heightB))
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(crop, M, (maxWidth, maxHeight))
            return warp
    return crop

def _score_plate_candidate(candidate):
    letters = sum(c.isalpha() for c in candidate)
    digits = sum(c.isdigit() for c in candidate)
    value = 0
    if MIN_PLATE_LEN <= len(candidate) <= MAX_PLATE_LEN:
        value += 10
    if len(candidate) == MAX_PLATE_LEN:
        value += 4
    if letters > 0 and digits > 0:
        value += 3
    value += len(set(candidate))
    return value

def _normalize_plate_candidate(text):
    cleaned = clean(text)
    if not cleaned:
        return ""
    if len(cleaned) <= MAX_PLATE_LEN:
        return cleaned
    windows = [cleaned[i:i + MAX_PLATE_LEN] for i in range(len(cleaned) - MAX_PLATE_LEN + 1)]
    return max(windows, key=_score_plate_candidate)

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

def deskew(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return crop
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    (h, w) = crop.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)
    gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)
    inverted = cv2.bitwise_not(thresh)
    return [gray, thresh, inverted]

def read_plate(image_path, bbox):
    img = cv2.imread(image_path)
    if img is None:
        return {"plate": "", "success": False, "status": "Imagen no cargada"}

    tesseract_cmd = _resolve_tesseract_cmd()
    if not tesseract_cmd:
        return {"plate": "", "success": False, "status": "Tesseract no está instalado o no está en PATH."}
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    crop = _get_plate_crop(img, bbox)
    if crop.size == 0:
        return {"plate": "", "success": False, "status": "Recorte de patente inválido"}

    deskewed = deskew(crop) # Arregla si la foto está un poco inclinada (ej: 5 o 10 grados)
    warped = perspective_correction(deskewed) # Arregla si la foto se tomó desde un costado (diagonal)

    variants = [crop, deskewed, warped]
    '''
    variants = [crop, deskewed, warped,
                cv2.rotate(deskewed, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(deskewed, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    '''

    candidates = []

    # la opcion 6 suele fallar o ser lento
    psm_options = [7, 8]
    for img_variant in variants:
        processed_imgs = preprocess_for_ocr(img_variant)
        for processed in processed_imgs:
            for psm in psm_options:
                config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(processed, config=config)
                text = _normalize_plate_candidate(text)
                
                # Plan A: Early Stopping si lee la patente chilena perfecta
                if text and CHILE_PLATE_PATTERN.match(text):
                    return {"plate": text, "success": True, "status": "OK"} 
                
                # Plan B: Guardar lo que leyó por si ninguna variante logra la perfección
                if text:
                    candidates.append(text)

    valid = [t for t in candidates if MIN_PLATE_LEN <= len(t) <= MAX_PLATE_LEN]

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