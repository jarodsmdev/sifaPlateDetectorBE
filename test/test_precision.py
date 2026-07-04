#!/usr/bin/env python3
"""
SIFA Plate Detector - Módulo de Evaluación de Métricas de Precisión.
Diseñado para la presentación del Examen Final (30 de Junio).

Este script evalúa la precisión del pipeline de visión artificial procesando
imágenes de la carpeta 'data/'. Utiliza el nombre de la imagen como la
patente esperada (Ground Truth).
"""

import os
import sys
import time
import shutil
import tempfile
import re

# Forzar codificación UTF-8 en la salida estándar para evitar UnicodeEncodeError en Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Asegurar que el directorio raíz esté en sys.path para poder importar 'app'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services import process_image_pipeline

def levenshtein_distance(s1, s2):
    """Calcula la distancia de edición entre dos cadenas para medir similitud de caracteres."""
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

def limpiar_patente(texto):
    """Limpia el texto quitando espacios, guiones y convirtiendo a mayúsculas."""
    if not texto:
        return ""
    return re.sub(r'[^A-Z0-9]', '', texto.upper().strip())

def calcular_mediana(valores):
    """Calcula la mediana de una lista de números."""
    if not valores:
        return 0.0
    valores_ordenados = sorted(valores)
    n = len(valores_ordenados)
    mitad = n // 2
    if n % 2 != 0:
        return valores_ordenados[mitad]
    else:
        return (valores_ordenados[mitad - 1] + valores_ordenados[mitad]) / 2.0

def calcular_moda(valores, decimales=2):
    """Calcula la moda de una lista de números, redondeando a los decimales indicados para agrupar."""
    if not valores:
        return 0.0
    valores_redondeados = [round(val, decimales) for val in valores]
    frecuencias = {}
    for val in valores_redondeados:
        frecuencias[val] = frecuencias.get(val, 0) + 1
    max_frecuencia = max(frecuencias.values())
    modas = [k for k, v in frecuencias.items() if v == max_frecuencia]
    return modas[0]

def evaluar_patentes():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if not os.path.exists(data_dir):
        print(f"Error: La carpeta '{data_dir}' no existe.")
        return

    # Buscar imágenes soportadas
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No se encontraron imágenes en la carpeta: {data_dir}")
        print("Guarda imágenes en 'data/' con el nombre de la patente (ej: ABCD12.jpg) para evaluarlas.")
        return

    print("=" * 80)
    print("           SIFA PLATE DETECTOR - EVALUACIÓN DE PRECISIÓN DE IA")
    print(f"           Evaluando {len(image_files)} imágenes en carpeta 'data/'...")
    print("=" * 80)
    print(f"{'Imagen':<22} | {'Esperada':<10} | {'Detectada':<10} | {'Acierto OCR':<13} | {'YOLO Conf':<10} | {'Tiempo':<7}")
    print("-" * 80)

    total_images = len(image_files)
    detecciones_yolo = 0
    aciertos_exactos = 0
    suma_precision_caracteres = 0.0
    tiempos_procesamiento = []
    
    # Carpeta temporal para procesar copias y no sobreescribir las fotos originales de 'data/'
    temp_dir = tempfile.mkdtemp(prefix="sifa_eval_")

    try:
        for idx, filename in enumerate(image_files, 1):
            original_path = os.path.join(data_dir, filename)
            
            # Obtener patente esperada del nombre de archivo (ej: "ABCD12_1.jpg" -> "ABCD12")
            base_name, _ = os.path.splitext(filename)
            raw_expected = base_name.split('_')[0]
            expected_plate = limpiar_patente(raw_expected)

            # Es una patente genérica si tiene nombres tipo "patente", "image", etc.
            es_generica = False
            if re.match(r'^(patente|imagen|image|test)\d*$', raw_expected, re.IGNORECASE):
                es_generica = True
                expected_plate = "GENÉRICA"

            # Copiar imagen a la carpeta temporal para evitar sobreescritura de tamaño
            temp_path = os.path.join(temp_dir, filename)
            shutil.copy(original_path, temp_path)

            # Ejecutar el pipeline midiendo tiempo
            start_time = time.time()
            try:
                result = process_image_pipeline(temp_path)
            except Exception as e:
                result = []
                print(f"{filename[:20]:<22} | {expected_plate:<10} | Error: {str(e)[:25]:<22}")
                continue
            
            elapsed = time.time() - start_time
            tiempos_procesamiento.append(elapsed)

            detected_plate = ""
            yolo_conf = 0.0
            yolo_ok = False

            if result:
                yolo_ok = True
                detecciones_yolo += 1
                # Tomamos la primera detección
                detected_plate = limpiar_patente(result[0].get("plate", ""))
                yolo_conf = result[0].get("confidence", 0.0)

            # Evaluar OCR
            match_status = "❌ Fallo"
            char_acc = 0.0

            if es_generica:
                match_status = "❓ (Sin ref)"
                char_acc = 0.0
            elif yolo_ok:
                if expected_plate == detected_plate:
                    match_status = "✅ Exacta"
                    aciertos_exactos += 1
                    char_acc = 1.0
                else:
                    # Calcular porcentaje de similitud por caracteres
                    dist = levenshtein_distance(expected_plate, detected_plate)
                    max_len = max(len(expected_plate), len(detected_plate), 1)
                    char_acc = 1.0 - (dist / max_len)
                    if char_acc > 0.0:
                        match_status = f"⚠️ {char_acc*100:.1f}%"
                    else:
                        match_status = "❌ Fallo"
            else:
                match_status = "🔍 No Det."
                char_acc = 0.0

            if not es_generica:
                suma_precision_caracteres += char_acc

            # Mostrar línea de resultado para la imagen
            disp_detected = detected_plate if yolo_ok else "N/A"
            disp_yolo_conf = f"{yolo_conf:.2f}" if yolo_ok else "0.00"
            disp_name = filename if len(filename) <= 22 else filename[:19] + "..."
            
            print(f"{disp_name:<22} | {expected_plate:<10} | {disp_detected:<10} | {match_status:<13} | {disp_yolo_conf:<10} | {elapsed:.2f}s")

    finally:
        # Limpieza de carpeta temporal
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Calcular estadísticas de tiempos (Media, Mediana y Moda)
    avg_time = sum(tiempos_procesamiento) / len(tiempos_procesamiento) if tiempos_procesamiento else 0.0
    mediana_time = calcular_mediana(tiempos_procesamiento)
    moda_time = calcular_moda(tiempos_procesamiento, decimales=2)
    
    tasa_deteccion_yolo = (detecciones_yolo / total_images) * 100
    
    # Excluir de la métrica de OCR las imágenes genéricas donde no conocemos el Ground Truth
    imagenes_evaluables_ocr = sum(1 for f in image_files if not re.match(r'^(patente|imagen|image|test)\d*$', os.path.splitext(f)[0].split('_')[0], re.IGNORECASE))
    
    if imagenes_evaluables_ocr > 0:
        tasa_acierto_ocr = (aciertos_exactos / imagenes_evaluables_ocr) * 100
        avg_char_accuracy = (suma_precision_caracteres / imagenes_evaluables_ocr) * 100
    else:
        tasa_acierto_ocr = 0.0
        avg_char_accuracy = 0.0

    print("=" * 80)
    print("                     RESUMEN DE MÉTRICAS (MÓDULO IA)")
    print("=" * 80)
    print(f"Total imágenes procesadas    : {total_images}")
    print(f"Detección de Patente (YOLO)  : {tasa_deteccion_yolo:.2f}% ({detecciones_yolo}/{total_images} encontradas)")
    
    if imagenes_evaluables_ocr > 0:
        print(f"Acierto Exacto OCR (Completo): {tasa_acierto_ocr:.2f}% ({aciertos_exactos}/{imagenes_evaluables_ocr} exactas)")
        print(f"Precisión Promedio Caracter  : {avg_char_accuracy:.2f}% (Tolerancia a pequeños errores)")
    else:
        print("Acierto Exacto OCR (Completo): N/A (Renombra tus archivos a la patente real para calcular)")
        print("Precisión Promedio Caracter  : N/A (Renombra tus archivos a la patente real para calcular)")
        
    print(f"Tiempo de Respuesta (Media)  : {avg_time:.2f} segundos")
    print(f"Tiempo de Respuesta (Mediana): {mediana_time:.2f} segundos")
    print(f"Tiempo de Respuesta (Moda)   : {moda_time:.2f} segundos")
    print(f"Tiempo Total de Evaluación   : {sum(tiempos_procesamiento):.2f} segundos")
    print("=" * 80)
    
    if imagenes_evaluables_ocr < total_images:
        print("\n[CONSEJO EXAMEN] Tienes imágenes con nombres genéricos (ej. patente.jpg).")
        print("Renómbralas a su patente real (ej. 'ABCD12.jpg') para que el script evalúe el acierto OCR.")
    print("=" * 80)

if __name__ == "__main__":
    evaluar_patentes()
