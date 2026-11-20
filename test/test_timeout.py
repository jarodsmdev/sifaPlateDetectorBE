"""
Prueba unitaria del timeout en el endpoint de detección de patentes.
Verifica que cuando el pipeline supera el tiempo límite, el endpoint retorna
HTTP 408 y activa el evento de cancelación del trabajo.
"""

import io
import time
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import status

from app.jobs import QueueFullError


def _jpeg_bytes():
    """Genera un JPEG mínimo válido para las peticiones de prueba."""
    ok, buffer = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def test_detect_timeout_cancela_el_trabajo(client):
    captured = {}

    def slow_pipeline(img, cancel):
        captured["cancel"] = cancel
        time.sleep(5.5)
        return []

    with patch("app.jobs.process_image_pipeline", side_effect=slow_pipeline):
        start = time.time()
        response = client.post(
            "/plate/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
        )
        elapsed = time.time() - start

    assert response.status_code == status.HTTP_408_REQUEST_TIMEOUT
    assert "superó el tiempo límite" in response.json()["detail"]
    assert 5.0 <= elapsed
    assert captured["cancel"].is_set()


def test_cola_llena_devuelve_503(client):
    with patch("app.main.runner.submit", side_effect=QueueFullError):
        response = client.post(
            "/plate/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
        )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.headers["Retry-After"] == "2"
