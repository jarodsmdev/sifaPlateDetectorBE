"""Pruebas unitarias del adaptador OCR (sin cargar PaddleOCR real)."""
from unittest.mock import patch

import numpy as np

from app.ocr import read_plate

IMG = np.zeros((100, 200, 3), dtype=np.uint8)
BBOX = {"x1": 10, "y1": 10, "x2": 100, "y2": 50}


def test_read_plate_captura_texto_y_confianza():
    with patch("app.ocr.ocr.ocr", return_value=[[("TZPW1I", 0.91)]]):
        resultado = read_plate(IMG, BBOX)

    assert resultado["plate"] == "TZPW11"
    assert resultado["plate_valid"] is True
    assert resultado["success"] is True
    assert resultado["ocr_confidence"] == 0.91
    assert resultado["status"] == "OK"


def test_read_plate_elige_candidato_valido():
    with patch(
        "app.ocr.ocr.ocr",
        return_value=[[("CHILE", 0.99), ("BBBB10", 0.80)]],
    ):
        resultado = read_plate(IMG, BBOX)

    assert resultado["plate"] == "BBBB10"
    assert resultado["plate_valid"] is True
    assert resultado["ocr_confidence"] == 0.80


def test_read_plate_sin_resultado():
    with patch("app.ocr.ocr.ocr", return_value=[[]]):
        resultado = read_plate(IMG, BBOX)

    assert resultado["plate"] == ""
    assert resultado["success"] is False
    assert resultado["status"] == "OCR ha fallado"


def test_read_plate_imagen_nula():
    resultado = read_plate(None, BBOX)
    assert resultado["success"] is False
    assert resultado["status"] == "Imagen no cargada"


def test_read_plate_recorte_vacio():
    resultado = read_plate(IMG, {"x1": 10, "y1": 10, "x2": 10, "y2": 10})
    assert resultado["success"] is False
    assert resultado["status"] == "Recorte de patente inválido"
