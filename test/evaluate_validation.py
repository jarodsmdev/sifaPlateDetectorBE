#!/usr/bin/env python3
"""
SIFA Plate Detector - Evaluación del dataset etiquetado de ``data/``.

Cada imagen se nombra con el formato:
    <dia|noche>_[etiqueta...]_<PATENTE>.<ext>

El último token es la patente real (Ground Truth). El primero puede ser
``dia`` o ``noche``; si falta, se reporta como ``desconocido``.
"""

import argparse
import os
import sys
import time

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.plate_format import detect_plate_format, parse_validation_filename
from app.services import process_image_pipeline

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def levenshtein_distance(s1, s2):
    """Calcula la distancia de edición entre dos cadenas."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def calcular_mediana(valores):
    """Calcula la mediana de una lista de números."""
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    n = len(ordenados)
    mitad = n // 2
    if n % 2 != 0:
        return ordenados[mitad]
    return (ordenados[mitad - 1] + ordenados[mitad]) / 2.0


def calcular_moda(valores, decimales=2):
    """Calcula la moda agrupando por redondeo."""
    if not valores:
        return 0.0
    redondeados = [round(valor, decimales) for valor in valores]
    frecuencias = {}
    for valor in redondeados:
        frecuencias[valor] = frecuencias.get(valor, 0) + 1
    max_frecuencia = max(frecuencias.values())
    return next(valor for valor, freq in frecuencias.items() if freq == max_frecuencia)


def _promedio(valores):
    return sum(valores) / len(valores) if valores else 0.0


def evaluar_patentes(data_dir=None, strict_tags=False):
    data_dir = data_dir or os.path.join(PROJECT_ROOT, "data")
    if not os.path.isdir(data_dir):
        print(f"Error: La carpeta '{data_dir}' no existe.")
        return 1

    image_files = sorted(
        nombre
        for nombre in os.listdir(data_dir)
        if nombre.lower().endswith(VALID_EXTENSIONS)
    )
    if not image_files:
        print(f"No se encontraron imágenes en la carpeta: {data_dir}")
        print("Usa el formato <dia|noche>_[etiqueta...]_<PATENTE>.jpg")
        return 1

    print("=" * 100)
    print("           SIFA PLATE DETECTOR - EVALUACIÓN DE PRECISIÓN DE IA")
    print(f"           Evaluando {len(image_files)} imágenes en carpeta 'data/'...")
    print("=" * 100)
    print(
        f"{'Imagen':<30} | {'Esperada':<9} | {'Detectada':<10} | {'Formato':<10} | "
        f"{'OCR':<13} | {'YOLO':<6} | {'OCR c':<6} | {'Tiempo':<7}"
    )
    print("-" * 100)

    total_images = len(image_files)
    detecciones_yolo = 0
    detecciones_validas = 0
    aciertos_exactos = 0
    suma_precision_caracteres = 0.0
    tiempos_procesamiento = []
    confianzas_yolo = []
    confianzas_ocr = []
    evaluables_ocr = 0
    etiquetas = {}
    periodos = {}

    avisos = []

    for filename in image_files:
        parsed = parse_validation_filename(filename)
        expected_plate = parsed["plate"]
        evaluable = detect_plate_format(expected_plate) is not None
        if evaluable:
            evaluables_ocr += 1
        if parsed["warnings"]:
            avisos.append((filename, parsed["warnings"]))

        periodos[parsed["day_period"]] = periodos.get(parsed["day_period"], 0) + 1
        for tag in parsed["tags"]:
            etiquetas[tag] = etiquetas.get(tag, 0) + 1

        original_path = os.path.join(data_dir, filename)
        img = cv2.imread(original_path)
        if img is None:
            print(f"{filename[:30]:<30} | {expected_plate:<9} | Error al leer la imagen")
            continue

        start_time = time.time()
        try:
            result = process_image_pipeline(img)
        except Exception as exc:
            print(f"{filename[:30]:<30} | {expected_plate:<9} | Error: {str(exc)[:35]}")
            continue
        elapsed = time.time() - start_time
        tiempos_procesamiento.append(elapsed)

        detected_plate = ""
        plate_valid = False
        detection_confidence = 0.0
        ocr_confidence = 0.0

        if result:
            detecciones_yolo += 1
            first = result[0]
            detected_plate = first.get("plate", "")
            plate_valid = bool(first.get("plate_valid", False))
            detection_confidence = float(first.get("detection_confidence", 0.0))
            ocr_confidence = float(first.get("ocr_confidence", 0.0))
            confianzas_yolo.append(detection_confidence)
            confianzas_ocr.append(ocr_confidence)
            if plate_valid:
                detecciones_validas += 1

        match_status = "❌ Fallo"
        char_acc = 0.0

        if not evaluable:
            match_status = "❓ (sin ref)"
        elif not result:
            match_status = "🔍 No Det."
        elif expected_plate == detected_plate:
            match_status = "✅ Exacta"
            if plate_valid:
                aciertos_exactos += 1
            char_acc = 1.0
        else:
            distancia = levenshtein_distance(expected_plate, detected_plate)
            max_len = max(len(expected_plate), len(detected_plate), 1)
            char_acc = max(0.0, 1.0 - (distancia / max_len))
            match_status = f"⚠️ {char_acc * 100:.1f}%"

        if evaluable:
            suma_precision_caracteres += char_acc

        formato = detect_plate_format(detected_plate) or "-"
        disp_detected = detected_plate if result else "N/A"
        print(
            f"{filename[:30]:<30} | {expected_plate:<9} | {disp_detected:<10} | {formato:<10} | "
            f"{match_status:<13} | {detection_confidence:<6.2f} | {ocr_confidence:<6.2f} | {elapsed:.2f}s"
        )

    print("=" * 100)
    print("                     RESUMEN DE MÉTRICAS (MÓDULO IA)")
    print("=" * 100)
    print(f"Total imágenes procesadas    : {total_images}")
    print(f"Detección de Patente (YOLO)  : {(detecciones_yolo / total_images) * 100:.2f}% ({detecciones_yolo}/{total_images})")
    print(f"Patentes con formato válido  : {(detecciones_validas / total_images) * 100:.2f}% ({detecciones_validas}/{total_images})")

    if evaluables_ocr:
        print(f"Acierto Exacto OCR (Completo): {(aciertos_exactos / evaluables_ocr) * 100:.2f}% ({aciertos_exactos}/{evaluables_ocr})")
        print(f"Precisión Promedio Caracter  : {(suma_precision_caracteres / evaluables_ocr) * 100:.2f}%")
    else:
        print("Acierto Exacto OCR (Completo): N/A (sin patentes de referencia válidas)")

    print(f"Confianza YOLO promedio      : {_promedio(confianzas_yolo):.4f}")
    print(f"Confianza OCR promedio       : {_promedio(confianzas_ocr):.4f}")
    print(f"Tiempo de Respuesta (Media)  : {_promedio(tiempos_procesamiento):.2f} segundos")
    print(f"Tiempo de Respuesta (Mediana): {calcular_mediana(tiempos_procesamiento):.2f} segundos")
    print(f"Tiempo de Respuesta (Moda)   : {calcular_moda(tiempos_procesamiento):.2f} segundos")
    print(f"Tiempo Total de Evaluación   : {sum(tiempos_procesamiento):.2f} segundos")

    print("-" * 100)
    print(f"Periodos (día/noche)         : {dict(sorted(periodos.items()))}")
    print(f"Etiquetas                    : {dict(sorted(etiquetas.items()))}")

    if avisos:
        print("-" * 100)
        print("Avisos de nombres de archivo:")
        for filename, warnings in avisos:
            print(f"  - {filename}: {', '.join(warnings)}")
        if strict_tags:
            print("\nModo --strict-tags: corrige los nombres anteriores.")
            return 1

    print("=" * 100)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evalúa el detector contra el dataset etiquetado.")
    parser.add_argument("--data-dir", default=None, help="Carpeta de imágenes (por defecto: data/)")
    parser.add_argument("--strict-tags", action="store_true", help="Falla si un nombre no sigue la convención")
    args = parser.parse_args()
    return evaluar_patentes(args.data_dir, args.strict_tags)


if __name__ == "__main__":
    raise SystemExit(main())
