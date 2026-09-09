"""Prueba unitaria del esquema de respuesta del endpoint de detección."""
import io
from datetime import datetime
from unittest.mock import patch

from fastapi import status
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_OUTPUT = [
    {
        "plate": "BBCC11",
        "success": True,
        "status": "OK",
        "confidence": 0.9654,
        "bbox": {"x1": 145, "y1": 320, "x2": 512, "y2": 410},
    }
]

SAMPLE_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xdb\x00C\x01\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x15\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x15\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\xff\xc4\x00\x14\x11\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xd2\x1a\x7f\xff\xd9"


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
            "status": "OK",
            "confidence": 0.8812,
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
