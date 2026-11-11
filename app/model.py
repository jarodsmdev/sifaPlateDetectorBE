"""
Módulo de Detección Espacial de Patentes.
Encargado de inicializar y ejecutar el modelo de visión computacional
para encontrar las coordenadas de las matrículas en la imagen completa.
"""

import numpy as np
from open_image_models import LicensePlateDetector

# Se inicializa el modelo YOLO preentrenado globalmente para evitar
# recargas en cada petición a la API.
detector = LicensePlateDetector(
    detection_model="yolo-v9-t-256-license-plate-end2end"
)


def detect_plate(image: np.ndarray):
    """
    Escanea una imagen para encontrar patentes vehiculares.

    Args:
        image (numpy.ndarray): Imagen BGR en memoria.

    Returns:
        list: Una lista de objetos de detección. Cada objeto contiene el
              bounding_box (coordenadas x1, y1, x2, y2) y el nivel de confianza.
    """
    return detector.predict(image)
