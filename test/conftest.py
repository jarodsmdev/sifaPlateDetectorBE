"""Configuración global de pytest: mockea módulos pesados y rutas del sistema."""
import os
import sys
import tempfile
import atexit
import shutil
from unittest.mock import MagicMock

import jwt
import pytest

# Asegurar que el directorio raíz del proyecto esté en sys.path
# para poder importar los módulos de la aplicación (app.config, etc.).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ──────────────────────────────────────────────
# 1. Mockear módulos pesados ANTES de cualquier import del test
#    (PaddlePaddle, PaddleOCR, YOLO). Se insertan en sys.modules
#    para que Python los use en lugar de ejecutar el código real
#    (descarga de modelos, inicialización, etc.).
# ──────────────────────────────────────────────
_MOCK_MODULES = ["paddle", "paddleocr", "open_image_models"]
for mod_name in _MOCK_MODULES:
    sys.modules.setdefault(mod_name, MagicMock())

# Para que "from open_image_models import LicensePlateDetector" funcione
# sobre el mock, LicensePlateDetector debe existir como atributo.
sys.modules["open_image_models"].LicensePlateDetector = MagicMock()

# ──────────────────────────────────────────────
# 2. Redirigir UPLOAD_DIR a un directorio temporal
#    IMPORTANTE: Esto debe ejecutarse ANTES de importar app.main,
#    ya que app.main ejecuta os.makedirs(UPLOAD_DIR) a nivel de módulo.
# ──────────────────────────────────────────────
import app.config  # noqa: E402

_UPLOAD_TMP = tempfile.mkdtemp(prefix="sifa_test_upload_")
app.config.UPLOAD_DIR = _UPLOAD_TMP

# ──────────────────────────────────────────────
# 3. Mockear app.model y app.ocr para que app.services
#    no intente importar los módulos reales pesados.
# ──────────────────────────────────────────────
sys.modules["app.model"] = MagicMock()
sys.modules["app.ocr"] = MagicMock()

# ──────────────────────────────────────────────
# 4. Importar app.main DESPUÉS de configurar UPLOAD_DIR y mocks.
# ──────────────────────────────────────────────
from app.main import app  # noqa: E402
from app.auth import get_current_user  # noqa: E402

# ──────────────────────────────────────────────
# 5. Configuración de JWT para tests
# ──────────────────────────────────────────────
TEST_JWT_SECRET = "test-secret-key-for-unit-tests"
TEST_JWT_ISSUER = "sifa-auth"
TEST_JWT_AUDIENCE = "sifa-clients"


@pytest.fixture
def mock_valid_token():
    """
    Genera un token JWT válido para tests.
    Payload incluye los claims requeridos: sub, roles, iss, aud, exp.
    """
    payload = {
        "sub": "test@example.com",
        "roles": ["USER_APP"],
        "iss": TEST_JWT_ISSUER,
        "aud": TEST_JWT_AUDIENCE,
        "exp": 9999999999,  # Nunca expira para tests
    }
    return jwt.encode(payload, TEST_JWT_SECRET, algorithm="HS256")


@pytest.fixture(autouse=True)
def override_auth_dependency(mock_valid_token):
    """
    Override la dependencia de auth para tests.
    Esto permite que los tests pasen sin necesidad de un token real.
    """

    def mock_get_current_user():
        return jwt.decode(
            mock_valid_token,
            TEST_JWT_SECRET,
            algorithms=["HS256"],
            issuer=TEST_JWT_ISSUER,
            audience=TEST_JWT_AUDIENCE,
        )

    app.dependency_overrides[get_current_user] = mock_get_current_user
    yield
    app.dependency_overrides.clear()


def _cleanup():
    shutil.rmtree(_UPLOAD_TMP, ignore_errors=True)


atexit.register(_cleanup)
