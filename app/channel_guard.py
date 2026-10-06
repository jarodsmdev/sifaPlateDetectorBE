"""
Guardia del canal interno para el servicio de detección de patentes.

Exige que toda petición HTTP llegue con la cabecera ``X-Internal-Key`` firmada
con ``INTERNAL_CHANNEL_KEY``, que es la que el API Gateway adjunta cuando
reenvía la llamada. De este modo, una petición que llegue directo a la IP
privada de la instancia queda descartada con 403 aunque pase el JWT de sesión.

Excepciones:
- ``INTERNAL_CHANNEL_KEY`` sin definir: guardia desactivada (se avisa al arrancar).
- Tráfico de loopback: healthchecks y comprobaciones locales no pasan por el
  gateway.
- Peticiones ``OPTIONS`` (preflight de CORS): el preflight no viaja al backend
  en condiciones normales, y si llega no lleva credenciales por diseño.
"""

import logging
import os
from typing import Iterable, Optional, Tuple

from starlette.responses import JSONResponse

logger = logging.getLogger("uvicorn.error")

CHANNEL_KEY_HEADER = b"x-internal-key"

_LOOPBACK_HOSTS = {"127.0.0.1", "0.0.0.0", "::1"}


def _header_value(scope: dict, name: bytes) -> Optional[str]:
    for key, value in scope.get("headers") or []:
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _is_loopback(scope: dict) -> bool:
    client: Optional[Tuple[str, int]] = scope.get("client")
    return bool(client) and client[0] in _LOOPBACK_HOSTS


def constant_time_equals(expected: str, actual: str) -> bool:
    """Comparación en tiempo constante para no filtrar el secreto por timing."""
    import hmac

    return hmac.compare_digest(expected.encode(), actual.encode())


class ChannelKeyMiddleware:
    """Middleware ASGI que valida la firma del canal interno."""

    def __init__(self, app, channel_key: str = ""):
        self.app = app
        self.channel_key = (channel_key or "").strip()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self._should_check(scope):
            await self.app(scope, receive, send)
            return

        provided = _header_value(scope, CHANNEL_KEY_HEADER)
        if provided is None or not constant_time_equals(self.channel_key, provided):
            logger.warning(
                "[-] Petición sin cabecera de canal válida desde %s: %s %s",
                scope.get("client", ("?",))[0],
                scope.get("method"),
                scope.get("path"),
            )
            response = JSONResponse(
                {"error": "Forbidden", "message": "Canal interno no valido"},
                status_code=403,
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

    def _should_check(self, scope: dict) -> bool:
        if not self.channel_key:
            return False
        if scope.get("method") == "OPTIONS":
            return False
        if _is_loopback(scope):
            return False
        return True


def channel_key_from_env() -> str:
    return os.getenv("INTERNAL_CHANNEL_KEY", "").strip()


def warn_if_disabled(key: str) -> None:
    if not key:
        logger.warning(
            "[!] INTERNAL_CHANNEL_KEY no definida: el canal interno NO esta protegido."
        )
    else:
        logger.info("[+] Canal interno activo: se exige X-Internal-Key")


__all__: Iterable[str] = (
    "ChannelKeyMiddleware",
    "channel_key_from_env",
    "warn_if_disabled",
)
