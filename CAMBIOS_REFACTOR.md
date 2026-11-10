# Cambios del refactor — sifaPlateDetectorBE

Fecha de referencia: 2026-11-10

Este documento resume el refactor aplicado al módulo de detección de patentes:
procesamiento de imágenes en memoria, validación de formatos chilenos,
separación de confianzas YOLO/OCR, nuevo contrato de respuesta, dataset
etiquetado y actualización de pruebas, consumidores y documentación.

## Resumen ejecutivo

- La API ya no escribe capturas en disco: recibe los bytes, los decodifica con
  `cv2.imdecode` y procesa el `numpy.ndarray` en memoria.
- Se eliminó la carpeta `/data/imagenes_recibidas`, el `UPLOAD_DIR`, los nombres
  temporales con UUID y la limpieza en `finally`.
- Se creó `app/plate_format.py`, que concentra la normalización, los formatos
  chilenos, las confusiones visuales y la selección del mejor candidato OCR.
- La respuesta ahora distingue `detection_confidence` (YOLO) de
  `ocr_confidence` (PaddleOCR) y agrega `plate_valid`.
- `success` ahora es igual a `plate_valid`; se mantiene por compatibilidad.
- El dataset local usa la convención
  `<dia|noche>_[etiqueta...]_<PATENTE>.<ext>` y ya no se versiona.
- Se actualizaron las pruebas, el evaluador de precisión, la app Android y el
  fake de la API de patentes.

## 1. Procesamiento en memoria

### `app/main.py`

Antes:

- Generaba un nombre único (`patente_<timestamp>_<uuid>.jpg`).
- Escribía el archivo en `UPLOAD_DIR`.
- Pasaba la ruta del archivo a `process_image_pipeline`.
- Eliminaba el archivo temporal en un bloque `finally`.
- Leía el cuerpo completo con `await file.read()` antes de validar el tamaño.
- Devolvía el detalle interno de la excepción en el HTTP 500.

Ahora:

- Lee como máximo `MAX_FILE_SIZE + 1` bytes con
  `await file.read(MAX_FILE_SIZE + 1)`.
- Si supera `MAX_FILE_SIZE` (5 MB), responde HTTP 413 sin seguir leyendo.
- Decodifica en memoria con `cv2.imdecode(np.frombuffer(...), cv2.IMREAD_COLOR)`.
- Si `cv2.imdecode` devuelve `None`, responde HTTP 400
  (`"Imagen inválida o corrupta"`), validando el contenido real y no solo el
  `content-type`.
- Pasa el `numpy.ndarray` a `process_image_pipeline` dentro de
  `asyncio.to_thread`, con timeout de 5 segundos.
- Distingue `asyncio.TimeoutError` (HTTP 408), `ValueError` (HTTP 400) y
  errores no controlados (HTTP 500 con detalle genérico).
- Ya no importa `os`, `shutil`, `uuid` ni crea directorios.

### `app/services.py`

Antes:

- `process_image_pipeline(path: str)` cargaba la imagen con `cv2.imread(path)`.
- Si la imagen superaba FullHD, la redimensionaba y **sobrescribía el archivo**
  con `cv2.imwrite(path, img)`.
- Llamaba a `detect_plate(path)`.
- Generaba un recorte Base64 con `encode_plate_crop_base64`.
- Devolvía `success`, `status`, `confidence` y `bbox`.

Ahora:

- `process_image_pipeline(img: np.ndarray)` recibe la imagen ya decodificada.
- Valida `img is None` o `img.size == 0` y lanza `ValueError`.
- Redimensiona en memoria si supera 1920x1080, sin escribir nada en disco.
- Llama a `detect_plate(img)`.
- Ya no genera Base64.
- Devuelve `plate`, `success`, `plate_valid`, `status`,
  `detection_confidence`, `ocr_confidence` y `bbox`.

### `app/model.py`

- `detect_plate(image_path: str)` pasó a `detect_plate(image: np.ndarray)`.
- El detector YOLO recibe directamente el arreglo BGR en memoria.

### `app/config.py`

- Se eliminó `UPLOAD_DIR = "/data/imagenes_recibidas"`.
- Se mantiene `MAX_FILE_SIZE = 5 * 1024 * 1024`.

## 2. Validación de formatos chilenos

### Nuevo módulo `app/plate_format.py`

Concentra toda la lógica de patentes y no depende de OpenCV, PaddleOCR ni
FastAPI. Expone:

- `normalize_plate(text)`: deja solo `A-Z` y `0-9` en mayúsculas.
- `detect_plate_format(plate)`: devuelve el formato que calza exactamente o
  `None`.
- `parse_ocr_candidates(candidates)`: elige el mejor texto OCR y lo valida.
- `parse_validation_filename(filename)`: interpreta los nombres del dataset.

Formatos soportados:

| Formato | Plantilla | Ejemplo | Letras permitidas |
|---|---|---|---|
| Auto nuevo | `LLLLDD` | `TZPW11` | 18 consonantes sin `M`, `N`, `Q` |
| Auto antiguo | `LLDDDD` | `AB1234` | Alfabeto amplio |
| Moto nueva | `LLLDD` | `GZB74` | Consonantes |
| Moto antigua | `LLDDD` | `XG440` | Consonantes |
| Auto futuro | `LLLLLD` | `BBBBB1` | 18 consonantes sin `M`, `N`, `Q` |

Reglas de selección:

1. Prefiere un calce exacto con algún formato.
2. Si no hay calce exacto, prefiere el candidato con menos correcciones.
3. Si empatan, prefiere el de mayor score OCR.
4. `MAX_CORRECTIONS = 2` evita falsos positivos como `CHILE -> CH113`.
5. Si ningún candidato calza pero hay un texto plausible de 5 a 7 caracteres
   con letras y dígitos, lo devuelve como mejor esfuerzo con
   `plate_valid: false`.
6. Si no hay texto plausible, devuelve `plate_valid: false` y estado
   `"La detección OCR no es confiable"`.

Confusiones visuales usadas para corregir posiciones:

| Letra leída | Dígito esperado |
|---|---|
| `O`, `Q`, `D` | `0` |
| `I`, `L` | `1` |
| `Z` | `2` |
| `E` | `3` |
| `A` | `4` |
| `S` | `5` |
| `G` | `6` |
| `T` | `7` |
| `B` | `8` |

La dirección inversa (`0→O`, `1→I`, `2→Z`, `3→E`, `4→A`, `5→S`, `6→G`,
`7→T`, `8→B`) se usa para posiciones que deben ser letras.

### `app/ocr.py`

Se eliminaron:

- `correct_common_mistakes`.
- `clean`.
- `CHILE_PLATE_PATTERN`.
- `MIN_PLATE_LEN` y `MAX_PLATE_LEN`.
- `encode_plate_crop_base64`.

Se agregó:

- `_extract_candidates(result)`: convierte la salida de PaddleOCR en pares
  `(texto, score)`, tolerando distintos formatos de respuesta.
- `read_plate(img, bbox)` ahora devuelve `plate`, `plate_valid`, `success`,
  `status` y `ocr_confidence`.
- La selección y corrección del texto se delega a
  `parse_ocr_candidates()`.

Se mantiene:

- `PLATE_CROP_PADDING = -5`.
- `ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)`.
- `ocr.ocr(crop, det=False, cls=False)`.

## 3. Nuevo contrato de respuesta

### `app/schemas.py`

Cambios:

- Se agregó `plate_valid: bool`.
- `confidence` se reemplazó por `detection_confidence`.
- Se agregó `ocr_confidence`.
- `success` se documenta como igual a `plate_valid`.

Ejemplo de detección:

```json
{
  "plate": "BBCC11",
  "success": true,
  "plate_valid": true,
  "status": "OK",
  "detection_confidence": 0.9654,
  "ocr_confidence": 0.9123,
  "bbox": {"x1": 145, "y1": 320, "x2": 512, "y2": 410}
}
```

Ejemplo de respuesta global:

```json
{
  "result": [
    {
      "plate": "TZPW11",
      "success": true,
      "plate_valid": true,
      "status": "OK",
      "detection_confidence": 0.8812,
      "ocr_confidence": 0.9765,
      "bbox": {"x1": 189, "y1": 398, "x2": 327, "y2": 444}
    }
  ],
  "timestamp": "2026-11-10T17:32:00.000000Z"
}
```

## 4. Dataset y evaluación

### Convención de nombres

Las imágenes de validación se guardan en `data/` con el formato:

```text
<dia|noche>_[etiqueta...]_<PATENTE>.<jpg|jpeg|png>
```

Ejemplos:

- `dia_angulo_BBBB10.jpg`
- `noche_suciedad_KPZX17.jpg`
- `dia_moto_GZB74.jpg`

Etiquetas reconocidas: `suciedad`, `angulo`, `lluvia`, `moto`.

Si falta `dia`/`noche`, el evaluador reporta `desconocido`. Si aparece una
etiqueta no reconocida, la reporta en los avisos.

### `data/README.md`

Nuevo documento con la convención de nombres, los formatos chilenos y las
etiquetas aceptadas.

### `.gitignore`

Se reemplazó la regla de `/data/imagenes_recibidas` por:

```gitignore
/data/*.jpg
/data/*.jpeg
/data/*.png
```

Solo se versiona `data/README.md`; las imágenes reales quedan locales.

### `test/evaluate_validation.py`

Nuevo evaluador que:

- Lee el nombre de cada imagen con `parse_validation_filename`.
- Procesa la imagen con `process_image_pipeline`.
- Calcula detección YOLO, formato válido, acierto exacto OCR, precisión por
  carácter, confianzas promedio y tiempos.
- Agrupa métricas por período (`dia`/`noche`) y por etiqueta.
- Reporta avisos de nombres inválidos.
- Falla con código 1 si se usa `--strict-tags` y hay nombres fuera de la
  convención.
- Acepta `--data-dir` para apuntar a otra carpeta.

### `test/test_precision.py`

Quedó como alias histórico de `evaluate_validation.py` para no romper comandos
existentes.

## 5. Pruebas

### Archivos modificados

- `test/conftest.py`: se eliminaron el directorio temporal y la redirección de
  `UPLOAD_DIR`; se mantiene el mock de `app.model` y se deja `app.ocr` real
  sobre el mock de `paddleocr`.
- `test/test_endpoint.py`: se actualizó al nuevo contrato; agrega pruebas de
  imagen inválida (400) y archivo demasiado grande (413).
- `test/test_timeout.py`: genera un JPEG válido en memoria y mockea el pipeline
  con la nueva firma.
- `test/test_carga.py`: se envolvió la ejecución en `main()` para que pytest no
  dispare peticiones al importar el archivo.

### Archivos nuevos

- `test/test_plate_format.py`: normalización, formatos, correcciones, selección
  de candidatos y parseo de nombres del dataset.
- `test/test_ocr.py`: extracción de candidatos, selección del mejor texto,
  errores de OCR, imagen nula y recorte vacío.
- `test/test_metrics.py`: distancia de Levenshtein, mediana y moda del
  evaluador.

### Resultado

```text
34 passed, 41 warnings in 6.53s
```

Cobertura reportada: 80% total. `app/model.py` y parte de `app/services.py`
aparecen con baja cobertura porque las pruebas unitarias mockean el pipeline
pesado; esos caminos se verificaron con modelos reales.

## 6. Verificación con modelos reales

Se ejecutó el pipeline dentro de un contenedor con los modelos cacheados
(`paddle_cache` y `yolo_cache`) y el repositorio montado en `/app`.

Resultados puntuales:

| Imagen | Patente | Formato | `plate_valid` | OCR conf. |
|---|---|---|---|---|
| `dia_TZPW11.jpg` | `TZPW11` | nuevo | `true` | 0.9765 |
| `dia_moto_GZB74.jpg` | `GZB74` | moto_nuevo | `true` | 0.8933 |
| `noche_AA1234.jpg` | `AA1234` | antiguo | `true` | 0.8540 |
| `dia_angulo_BBBB10.jpg` | `BBBB10` | nuevo | `true` | 0.7620 |

También se verificó que el conjunto de archivos de `data/` no cambia después
de procesar una imagen: la API no escribe capturas en disco.

### Evaluación completa del dataset

```text
Total imágenes procesadas    : 30
Detección de Patente (YOLO)  : 100.00% (30/30)
Patentes con formato válido  : 96.67% (29/30)
Acierto Exacto OCR (Completo): 83.33% (25/30)
Precisión Promedio Caracter  : 97.22%
Confianza YOLO promedio      : 0.8403
Confianza OCR promedio       : 0.9132
Tiempo de Respuesta (Media)  : 0.10 segundos
Tiempo de Respuesta (Mediana): 0.10 segundos
Tiempo Total de Evaluación   : 3.02 segundos
Periodos (día/noche)         : {'dia': 18, 'noche': 12}
Etiquetas                    : {'angulo': 5, 'lluvia': 1, 'moto': 3, 'suciedad': 2}
```

La única patente sin formato válido fue `BVRC8` (leída como 5 caracteres en
lugar de `BVRC87`). Los otros errores fueron lecturas parciales o un carácter
incorrecto, no fallos de detección.

## 7. Consumidores actualizados

### `SIFA_GO`

`app/src/main/java/com/sifa/sifa_go/data/model/PlateResponse.kt`:

- Se agregó `plateValid` mapeado a `plate_valid`.
- `confidence` se reemplazó por `detectionConfidence` y `ocrConfidence`,
  mapeados a `detection_confidence` y `ocr_confidence`.
- Se conserva `image_base64` mapeado a `image`; el backend ya no envía ese
  campo, pero el campo es nullable y no rompe la deserialización.

`app/src/main/java/com/sifa/sifa_go/viewmodel/SifaViewModel.kt`:

- La condición de éxito ahora exige
  `plateResult.success && plateResult.plateValid` además de una patente no
  vacía.

### `FAKE_APIs_SIFA/FAKE_PLATE_DETECT/assets/app.js`

- Se muestra `plate_valid`.
- Se muestran `detection_confidence` y `ocr_confidence` por separado.
- Se eliminó el uso de `confidence`.

## 8. Documentación actualizada

- `README.md`: se reescribió para reflejar el procesamiento en memoria, el
  nuevo contrato, los códigos de error, la convención del dataset y los
  comandos de prueba/evaluación.
- `docu/sifaPlateDetectorBE_FLUJO.md`: se actualizaron el diagrama de
  secuencia y las secciones de validación, decodificación, pipeline, OCR,
  post-procesamiento, respuesta, ausencia de persistencia, errores y estados.

## 9. Archivos afectados

### Modificados

- `.gitignore`
- `README.md`
- `app/config.py`
- `app/main.py`
- `app/model.py`
- `app/ocr.py`
- `app/schemas.py`
- `app/services.py`
- `test/conftest.py`
- `test/test_carga.py`
- `test/test_endpoint.py`
- `test/test_precision.py`
- `test/test_timeout.py`

### Creados

- `app/plate_format.py`
- `data/README.md`
- `test/evaluate_validation.py`
- `test/test_metrics.py`
- `test/test_ocr.py`
- `test/test_plate_format.py`
- `CAMBIOS_REFACTOR.md` (este documento)

### Eliminados del control de versiones

Se eliminaron 20 imágenes antiguas de `data/`. Dieciocho corresponden a las
nuevas imágenes renombradas con la convención `dia_*`/`noche_*`; las otras dos
(`CD4567.jpg` y `ETCH99.jpg`) no tienen equivalente en el dataset actual.

Las 30 imágenes nuevas no aparecen como untracked porque `data/*.jpg` está en
`.gitignore`. Esto es intencional: el dataset queda local y solo se versiona
`data/README.md`.

## 10. Comandos de verificación

Pruebas unitarias (en el contenedor de pruebas usado durante el refactor):

```bash
docker run --rm -v "$PWD:/app" -w /app sifa-plate-test:latest python -m pytest -q
```

Evaluación con modelos reales:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -v "$PWD/paddle_cache:/root/.paddleocr" \
  -v "$PWD/yolo_cache:/root/.cache/open-image-models" \
  -w /app sifa-plate-test:latest \
  python test/evaluate_validation.py
```

Modo estricto de nombres:

```bash
docker run --rm \
  -v "$PWD:/app" \
  -v "$PWD/paddle_cache:/root/.paddleocr" \
  -v "$PWD/yolo_cache:/root/.cache/open-image-models" \
  -w /app sifa-plate-test:latest \
  python test/evaluate_validation.py --strict-tags
```

## 11. Pendientes y riesgos

- No se ejecutó la prueba de carga `test/test_carga.py` porque requiere el
  servidor levantado y peticiones HTTP reales.
- No se probaron múltiples peticiones concurrentes con los modelos reales; la
  verificación con modelos reales fue secuencial.
- El `.venv` local del entorno de desarrollo quedó inutilizable por un
  desajuste entre Python 3.12 y 3.14; las pruebas se ejecutaron en Docker.
- Al ignorar `data/*.jpg`, las imágenes de evaluación no quedan versionadas.
  Para reproducir las métricas hay que disponer del dataset local en `data/`.
- El modelo Android conserva el campo nullable `image` aunque el backend ya no
  lo envía. Es un campo heredado y no afecta la lectura de la respuesta.
