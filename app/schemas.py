from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class BBoxSchema(BaseModel):
    """Esquema que define las coordenadas del recuadro delimitador de la patente."""
    x1: int = Field(..., description="Coordenada horizontal inicial (píxel izquierdo)", example=145)
    y1: int = Field(..., description="Coordenada vertical inicial (píxel superior)", example=320)
    x2: int = Field(..., description="Coordenada horizontal final (píxel derecho)", example=512)
    y2: int = Field(..., description="Coordenada vertical final (píxel inferior)", example=410)

class DetectionItemSchema(BaseModel):
    """Representa los datos individuales de una patente detectada y procesada."""
    plate: str = Field(..., description="Texto alfanumérico de la patente extraído por el OCR", example="BBCC11")
    success: bool = Field(..., description="Indica si el motor OCR logró procesar y leer la patente correctamente", example=True)
    status: str = Field(..., description="Mensaje de estado transaccional interno", example="OK")
    confidence: float = Field(..., description="Nivel de certeza del modelo YOLOv9 al localizar el recuadro", example=0.9654)
    bbox: BBoxSchema = Field(..., description="Coordenadas geométricas de la patente")

class PlateDetectionResponseSchema(BaseModel):
    """Esquema global de respuesta para el endpoint de detección."""
    result: List[DetectionItemSchema] = Field(..., description="Colección indexada con todas las patentes identificadas")
    timestamp: datetime = Field(..., description="Marca de tiempo en formato ISO 8601 UTC", example="2026-06-10T14:35:22.094321Z")