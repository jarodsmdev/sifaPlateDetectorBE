"""
Configuración JWT para validación de tokens.

Centraliza la configuración de autenticación en un solo lugar
(Single Source of Truth). Las variables de entorno se cargan
una vez y se reutilizan en todas las dependencies de FastAPI.

Seguridad: No se incluyen valores default para issuer/audience
para evitar exponer información sensible en el código fuente.
"""

import os
import base64
from dataclasses import dataclass, field


def _require_env(name: str) -> str:
    """
    Obtiene una variable de entorno requerida.

    Args:
        name: Nombre de la variable de entorno

    Returns:
        str: Valor de la variable de entorno

    Raises:
        RuntimeError: Si la variable no está definida o está vacía
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} environment variable is required. "
            f"Set it in your .env file or export it in your shell."
        )
    return value


def _decode_secret(raw_secret: str) -> bytes:
    """
    Decodifica el JWT_SECRET replicando la lógica de Java (JJWT).

    En Java, JwtService.getSignInKey() hace:
        1. Intenta Decoders.BASE64.decode(secret)
        2. Si falla (IllegalArgumentException), usa secret.getBytes(UTF_8)

    Esta función replica esa lógica para asegurar compatibilidad:
        1. Intenta decodificar como Base64
        2. Si falla, retorna los bytes UTF-8 del string

    Args:
        raw_secret: String del JWT_SECRET desde variable de entorno

    Returns:
        bytes: Clave decodificada para usar con PyJWT
    """
    try:
        decoded = base64.b64decode(raw_secret, validate=True)
        # Verificar que la decodificación tuvo sentido
        # (Base64 válido podría no ser la clave real)
        if len(decoded) >= 32:
            return decoded
    except Exception:
        pass
    # Si Base64 falla o el resultado es muy corto, usar UTF-8 directamente
    return raw_secret.encode("utf-8")


@dataclass(frozen=True)
class JWTConfig:
    """
    Configuración JWT inmutable.

    Attributes:
        secret_bytes: Clave HMAC decodificada para verificar firmas
        algorithm: Algoritmo de firma (HS256, HS384, HS512)
        issuer: Emisor válido del token (claim 'iss')
        audience: Audiencia válida del token (claim 'aud')
    """
    secret_bytes: bytes = field(repr=False)
    algorithm: str = "HS256"
    issuer: str = ""
    audience: str = ""

    @classmethod
    def from_env(cls) -> "JWTConfig":
        """
        Crea instancia desde variables de entorno.

        Todas las variables JWT son requeridas (sin defaults)
        para evitar exponer valores en código fuente.

        La clave secret se decodifica de Base64 para compatibilidad
        con el servicio de Auth (Java/JJWT).

        Raises:
            RuntimeError: Si JWT_SECRET, JWT_ISSUER o JWT_AUDIENCE
                         no están definidos
        """
        raw_secret = _require_env("JWT_SECRET")
        return cls(
            secret_bytes=_decode_secret(raw_secret),
            issuer=_require_env("JWT_ISSUER"),
            audience=_require_env("JWT_AUDIENCE"),
        )
