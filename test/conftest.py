"""Configuración global de pytest: mockea módulos pesados y rutas del sistema."""
import os
import sys
import tempfile
import atexit
import shutil
from unittest.mock import MagicMock

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


def _cleanup():
    shutil.rmtree(_UPLOAD_TMP, ignore_errors=True)


atexit.register(_cleanup)
