"""Pruebas de la guardia del canal interno (app/channel_guard.py).

La guardia se monta en ``app/main.py`` ANTES que CORS (en Starlette lo último
añadido queda por fuera), de modo que el preflight lo resuelve CORS y solo las
peticiones que realmente van al backend pasan por aqui.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.testclient import TestClient

from app.channel_guard import ChannelKeyMiddleware

KEY = "a1b2c3d4" * 8


def _build(channel_key: str) -> FastAPI:
    """Replica el orden de app/main.py: guardia primero, CORS despues."""
    app = FastAPI()

    app.add_middleware(ChannelKeyMiddleware, channel_key=channel_key)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return app


def test_guardia_desactivada_sin_clave_no_bloquea():
    client = TestClient(_build(""))
    assert client.get("/ping").status_code == 200


def test_sin_cabecera_responde_403():
    client = TestClient(_build(KEY))
    response = client.get("/ping")
    assert response.status_code == 403


def test_403_incluye_mensaje_json():
    client = TestClient(_build(KEY))
    response = client.get("/ping")
    assert "Canal interno" in response.text


def test_clave_incorrecta_responde_403():
    client = TestClient(_build(KEY))
    response = client.get("/ping", headers={"X-Internal-Key": "valor-equivocado"})
    assert response.status_code == 403


def test_clave_correcta_continua():
    client = TestClient(_build(KEY))
    response = client.get("/ping", headers={"X-Internal-Key": KEY})
    assert response.status_code == 200


def test_preflight_options_no_lo_bloquea_la_guardia():
    client = TestClient(_build(KEY))
    response = client.options(
        "/ping",
        headers={"Origin": "http://x", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code != 403


def test_condiciones_de_exclusion():
    mw_on = ChannelKeyMiddleware(app=None, channel_key=KEY)
    mw_off = ChannelKeyMiddleware(app=None, channel_key="")

    assert mw_on._should_check({"method": "GET", "client": ("127.0.0.1", 1)}) is False
    assert mw_on._should_check({"method": "GET", "client": ("::1", 1)}) is False
    assert mw_on._should_check({"method": "GET", "client": ("10.0.2.10", 5)}) is True
    assert mw_on._should_check({"method": "OPTIONS", "client": ("10.0.2.10", 5)}) is False
    assert mw_off._should_check({"method": "GET", "client": ("10.0.2.10", 5)}) is False