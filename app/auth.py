"""
Dependencies de autenticación y autorización para FastAPI.

Este módulo implementa la validación completa de tokens JWT:
- Verificación de firma HMAC
- Validación de expiración (claim 'exp')
- Validación de emisor (claim 'iss')
- Validación de audiencia (claim 'aud')
- Autorización por rol (claim 'roles')

Flujo de una request autenticada:
1. HTTPBearer extrae el token del header Authorization
2. get_current_user valida firma, expiración, issuer, audience
3. require_role verifica que el usuario tenga el rol requerido
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from .auth_config import JWTConfig


# Scheme para Swagger UI - solo extrae el token, NO valida
security = HTTPBearer(
    description="Token JWT válido emitido por el servicio de Autenticación"
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    config: JWTConfig = Depends(JWTConfig.from_env),
) -> dict:
    """
    Valida el token JWT y retorna el payload.

    Verifica:
    - Firma HMAC (token no fue alterado)
    - Expiración (token no está vencido)
    - Issuer (token fue emitido por sifa-auth)
    - Audience (token fue emitido para sifa-clients)

    Args:
        credentials: Token extraído del header Authorization
        config: Configuración JWT desde variables de entorno

    Returns:
        dict: Payload del JWT con claims (sub, roles, etc.)

    Raises:
        HTTPException 401: Para cualquier error de validación
    """
    token = credentials.credentials
    try:
        # Usar secret_bytes (ya decodificado de Base64 en auth_config)
        # para compatibilidad con la firma del Auth Service (Java/JJWT)
        payload = jwt.decode(
            token,
            config.secret_bytes,
            algorithms=[config.algorithm],
            issuer=config.issuer,
            audience=config.audience,
        )
        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expirado",
        )
    except jwt.InvalidSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firma del token inválida",
        )
    except jwt.InvalidIssuerError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Emisor del token inválido",
        )
    except jwt.InvalidAudienceError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Audiencia del token inválida",
        )
    except jwt.DecodeError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token malformado",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
        )


def require_role(role: str):
    """
    Factory de dependency que verifica el rol del usuario.

    Uso:
        @app.get("/endpoint")
        async def endpoint(user: dict = Depends(require_role("USER_APP"))):
            # Solo usuarios con rol USER_APP llegan aquí
            ...

    Args:
        role: Rol requerido para acceder al recurso

    Returns:
        Callable: Dependency que valida el rol
    """

    def role_checker(payload: dict = Depends(get_current_user)) -> dict:
        user_roles = payload.get("roles", [])
        if role not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes permiso para acceder a este recurso",
            )
        return payload

    return role_checker
