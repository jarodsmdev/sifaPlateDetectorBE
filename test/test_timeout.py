"""
Prueba unitaria del timeout en el endpoint de detección de patentes.
Verifica que cuando process_image_pipeline supera los 5 segundos,
el endpoint retorna HTTP 408 (Request Timeout) sin depender de
modelos reales (YOLO/PaddleOCR).
"""

import io
import time
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _jpeg_bytes():
    """Genera un JPEG mínimo válido para las peticiones de prueba."""
    ok, buffer = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def test_detect_timeout():
    """
    Mockea process_image_pipeline para que demore >5s y verifica que el
    endpoint retorne 408 con el mensaje de error correspondiente.
    """
    def slow_pipeline(img):
        time.sleep(5.5)
        return []

    with patch("app.main.process_image_pipeline", side_effect=slow_pipeline):
        start = time.time()
        # No necesitamos header Authorization porque el mock sobreescribe
        # la dependencia de autenticación en conftest.py
        response = client.post(
            "/plate/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
        )
        elapsed = time.time() - start

        assert response.status_code == status.HTTP_408_REQUEST_TIMEOUT
        assert "superó el tiempo límite" in response.json()["detail"]
        assert 5.0 <= elapsed
