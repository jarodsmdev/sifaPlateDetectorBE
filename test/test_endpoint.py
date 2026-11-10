"""Prueba unitaria del esquema de respuesta del endpoint de detección."""
import io
from datetime import datetime
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.config import MAX_FILE_SIZE

client = TestClient(app)


def _jpeg_bytes():
    """Genera un JPEG mínimo válido para las peticiones de prueba."""
    ok, buffer = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return buffer.tobytes()


SAMPLE_JPEG = _jpeg_bytes()

SAMPLE_OUTPUT = [
    {
        "plate": "BBCC11",
        "success": True,
        "plate_valid": True,
        "status": "OK",
        "detection_confidence": 0.9654,
        "ocr_confidence": 0.9123,
        "bbox": {"x1": 145, "y1": 320, "x2": 512, "y2": 410},
    }
]


def _build_response(output):
    """
    Helper que construye una request al endpoint con un token JWT válido.
    El token se genera automáticamente via conftest.py (mock_valid_token).
    """
    with patch("app.main.process_image_pipeline", return_value=output):
        # No necesitamos header Authorization porque el mock sobreescribe
        # la dependencia de autenticación en conftest.py
        return client.post(
            "/plate/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(SAMPLE_JPEG), "image/jpeg")},
        )


def test_devuelve_200_con_lista_vacia():
    response = _build_response([])
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["result"] == []
    assert "timestamp" in body


def test_devuelve_200_con_una_patente():
    response = _build_response(SAMPLE_OUTPUT)
    assert response.status_code == status.HTTP_200_OK
    body = response.json()

    assert body["result"] == SAMPLE_OUTPUT

    timestamp = body["timestamp"]
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def test_devuelve_200_con_varias_patentes():
    multiples = SAMPLE_OUTPUT + [
        {
            "plate": "TZPW11",
            "success": True,
            "plate_valid": True,
            "status": "OK",
            "detection_confidence": 0.8812,
            "ocr_confidence": 0.9001,
            "bbox": {"x1": 10, "y1": 20, "x2": 200, "y2": 80},
        }
    ]
    response = _build_response(multiples)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["result"] == multiples


def test_timestamp_formato_iso_utc():
    response = _build_response(SAMPLE_OUTPUT)
    ts = response.json()["timestamp"]
    assert ts.endswith("Z")
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_imagen_invalida_devuelve_400():
    response = client.post(
        "/plate/api/v1/detect",
        files={"file": ("test.jpg", io.BytesIO(b"no soy una imagen"), "image/jpeg")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Imagen inválida o corrupta"


def test_archivo_demasiado_grande_devuelve_413():
    oversized = b"x" * (MAX_FILE_SIZE + 1)
    response = client.post(
        "/plate/api/v1/detect",
        files={"file": ("test.jpg", io.BytesIO(oversized), "image/jpeg")},
    )
    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert "demasiado grande" in response.json()["detail"]
