import os

# Capturamos la URL del API Gateway perimetral
GATEWAY_URL = os.getenv("GATEWAY_PUBLIC_URL", "http://localhost:9000")

# Metadatos globales para Swagger UI
API_TITLE = "Módulo de Reconocimiento de Patentes (Plate Detector) - SIFA"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Servidor de inferencia matemática especializado en visión artificial para el ecosistema SIFA. "
    "Este microservicio asíncrono implementa una tubería de ejecución (Pipeline) compuesta por un modelo "
    "YOLOv9 optimizado en formato ONNX para la detección perimetral de placas patentes en imágenes complejas, "
    "seguido por el motor avanzado PaddleOCR (PP-OCRv4) para la segmentación y reconocimiento óptico de caracteres alfanuméricos."
)

# Definición de servidores para enrutamiento desde el Gateway
API_SERVERS = [
    {"url": "/", "description": "Ruta relativa del contenedor actual"},
    {"url": GATEWAY_URL, "description": "API Gateway Perimetral Centralizado"}
]

# Configuración de CORS
LOCAL_DEV_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

# Configuraciones de Sistema
UPLOAD_DIR = "/data/imagenes_recibidas"
MAX_FILE_SIZE = 5 * 1024 * 1024 # 5MB