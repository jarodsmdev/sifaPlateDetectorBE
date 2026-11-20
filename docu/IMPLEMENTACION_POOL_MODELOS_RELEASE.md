# Implementación: pool acotado, lockfile con hashes y modelos en GitHub Release

Estado: implementado en la rama `dev`, pendiente de publicar el release
`models-v1` y de validar el despliegue en EC2.

> Nota de estado: el artefacto `plate-models-v1.tar.gz` ya existe y su SHA256 es
> `7814edec45d8cb2b7a38e92639f82ff336bafc4f6f9c9494cd5fc621982dbca6`, pero el
> release `models-v1` todavía no está publicado. Hasta que se suba el archivo, la
> URL usada por el Dockerfile devuelve 404.

Este documento explica, para cualquier desarrollador del equipo, qué se cambió
en `sifaPlateDetectorBE`, por qué se cambió y qué pasos quedan pendientes en GitHub. No
asume experiencia previa en concurrencia, Docker ni gestión de dependencias.

---

## 1. Resumen en una frase

El servicio dejará de aceptar trabajo sin límite, avisará a los trabajos lentos que
se detengan en cuanto puedan, congelará las versiones exactas de sus librerías y
descargará los modelos de IA desde un release de GitHub, verificando su huella
digital antes de usarlos.

En la práctica:

- Se atienden como máximo **N imágenes a la vez** y se encolan hasta **M**.
- Si llegan más, se responde **503** en vez de acumular trabajo invisible.
- Al vencer los 5 segundos, el trabajo recibe un aviso de cancelación y se detiene
  en el próximo punto seguro.
- Las dependencias quedan fijadas con versión y hash en un archivo generado.
- Los modelos salen de Git y pasan a un release versionado; al construir la imagen
  se descargan y se verifica su SHA256.

---

## 2. El problema que se confirmó

La revisión de código encontró cuatro problemas reales en el repositorio (rama
`dev`, commit `15f3c46`):

### 2.1 El timeout no detiene el trabajo

En `app/main.py:134-137` el endpoint hace:

```python
output = await asyncio.wait_for(
    asyncio.to_thread(process_image_pipeline, img),
    timeout=5.0,
)
```

`asyncio.to_thread` ejecuta el pipeline en un hilo aparte. Cuando `wait_for` vence,
el endpoint deja de esperar y responde **408**, pero **el hilo sigue trabajando**:
YOLO y PaddleOCR continúan consumiendo CPU hasta terminar. En Python no existe una
forma segura de matar un hilo desde fuera; solo se puede pedir que se detenga.

Consecuencia: el cliente recibe un error, pero el servidor sigue gastando CPU en un
resultado que ya nadie va a leer.

### 2.2 No hay cola ni límite de concurrencia explícito

`asyncio.to_thread` usa el executor por defecto de Python. Ese executor tiene un
límite implícito de hilos (`min(32, CPUs + 4)`), pero:

- No hay un límite definido por el equipo ni visible en configuración.
- No hay tope para la cola de trabajos pendientes.
- No hay rechazo ni backpressure: si llegan 15 peticiones, todas se encolan.
- Las peticiones que esperan pueden recibir 408 mientras su trabajo sigue en cola o
  en ejecución.

Con `test/test_carga.py` (15 peticiones simultáneas) esto se traduce en CPU
saturada, latencias altas y respuestas 408 que no liberan recursos.

### 2.3 Dependencias sin versión fija

`requirements.txt` deja sin versión a `fastapi`, `uvicorn[standard]`,
`python-multipart`, `open-image-models[onnx]` y `opencv-python-headless`; otras usan
rangos como `>=`. No hay lockfile ni hashes.

Consecuencia: dos builds en fechas distintas pueden instalar versiones distintas,
romper la compatibilidad con PaddleOCR o cambiar el layout de los modelos sin que
nadie haya tocado el código.

### 2.4 Modelos y binarios versionados en Git

`git ls-files` incluye:

- 9 archivos de `paddle_cache/` (~15,8 MB).
- `yolo_cache/.../yolo-v9-t-256-license-plates-end2end.onnx` (~7,75 MB).
- `test/patente5.jpg`.

Consecuencia: el repositorio pesa más de lo necesario, cada clon descarga binarios y
la versión del modelo queda atada a un commit de código en vez de a un artefacto
versionado e identificable.

### 2.5 Hallazgo adicional (fuera de alcance)

`deploy_SIFA-PLATE.sh:42-45` contiene `JWT_SECRET` e `INTERNAL_CHANNEL_KEY` en texto
plano. No se modifica en esta implementación, pero debe rotarse y moverse a un
gestor de secretos (Doppler u otro) en un trabajo aparte.

---

## 3. Glosario mínimo

| Término | Qué significa aquí |
|---|---|
| **Hilo (thread)** | Una línea de ejecución dentro del mismo proceso. Permite que la CPU haga trabajo pesado sin bloquear la recepción de peticiones. |
| **Event loop** | El bucle de FastAPI/uvicorn que recibe peticiones y las reparte. Si se bloquea, el servicio deja de responder. |
| **Pool acotado** | Un número fijo de trabajadores. Si hay 2, solo 2 imágenes se procesan a la vez. |
| **Cola** | Lista de trabajos esperando un trabajador libre. Aquí tendrá un tamaño máximo. |
| **Backpressure** | Decirle al cliente "no puedo ahora" (503) en vez de aceptar trabajo que no se podrá atender. |
| **Timeout** | Tiempo máximo que el cliente espera una respuesta. |
| **Cancelación cooperativa** | Pedir al trabajo que se detenga en el próximo punto de control. No es una interrupción instantánea. |
| **Lockfile** | Archivo generado con las versiones exactas de todas las librerías, directas e indirectas. |
| **Hash (SHA256)** | Huella digital de un archivo. Si el archivo cambia, el hash cambia. Sirve para detectar descargas corruptas o alteradas. |
| **Imagen Docker** | Paquete ejecutable que contiene el código, las librerías y, en esta implementación, los modelos. |
| **GitHub Release** | Sección de GitHub para publicar archivos descargables asociados a una etiqueta (tag), separados del código fuente. |

---

## 4. Parte A: pool acotado, cola y cancelación cooperativa

### 4.1 Qué se construye

Un componente nuevo, `app/jobs.py`, con un `PlateJobRunner`:

- **N trabajadores** (`PLATE_MAX_CONCURRENCY`), cada uno procesa un trabajo a la vez.
- **Una cola con tope** (`PLATE_MAX_QUEUE`) para los trabajos que esperan.
- **Un evento de cancelación por trabajo** (`threading.Event`).

El endpoint ya no llama directamente a `asyncio.to_thread`. Ahora:

1. Pide al runner que acepte el trabajo.
2. Si la cola está llena, responde **503** con `Retry-After`.
3. Si hay espacio, espera el resultado con el timeout de siempre (5 s).
4. Si vence, activa el evento de cancelación y responde **408**.

### 4.2 Cómo funciona, con una analogía

Pensemos en una oficina con **N ventanillas** (trabajadores) y una **sala de espera**
con **M sillas** (cola):

- Si hay una ventanilla libre, el cliente pasa de inmediato.
- Si todas están ocupadas pero hay silla, espera en la sala.
- Si no hay silla, se le dice "vuelve más tarde" (503).
- Si el cliente se cansa de esperar (timeout), se le avisa al funcionario que está
  trabajando en su caso: "cuando termines el paso actual, deja de procesar este
  caso". Eso es la cancelación cooperativa.

La diferencia clave con el código actual es que el número de ventanillas y sillas es
**explícito, configurable y acotado**.

### 4.3 Flujo del endpoint

```text
Petición HTTP
    |
    v
¿Cola llena? -- sí --> 503 + Retry-After
    |
    no
    v
Encolar trabajo (con evento de cancelación)
    |
    v
Esperar resultado con timeout de 5 s
    |
    +-- resultado --> 200
    |
    +-- timeout --> activar evento de cancelación --> 408
```

### 4.4 Cambios por archivo

| Archivo | Cambio |
|---|---|
| `app/jobs.py` (nuevo) | `PlateJob`, `PlateJobRunner`, `QueueFullError`, `_silence_future`. |
| `app/main.py` | `lifespan` que arranca y detiene el runner; `submit`; 503 si la cola está llena; timeout activa la cancelación; `asyncio.shield` para no cancelar el future del worker. |
| `app/services.py` | `PipelineCancelled`, `_ensure_not_cancelled` y `process_image_pipeline(img, cancel=None)` con puntos de control antes del resize, antes de YOLO y antes de cada OCR. |
| `app/config.py` | `PLATE_MAX_CONCURRENCY`, `PLATE_MAX_QUEUE`, `PLATE_TIMEOUT_S`. |
| `test/conftest.py` | Fixture `client` con `TestClient(app)` para que el `lifespan` arranque el runner. |
| `test/test_endpoint.py` | Usa el fixture `client`; parchea `app.jobs.process_image_pipeline`. |
| `test/test_timeout.py` | 408 activa la cancelación; 503 cuando la cola está llena. |
| `test/test_jobs.py` (nuevo) | Éxito, cola llena, cancelación antes de empezar y cancelación antes del OCR. |
| `README.md` | 503, variables del pool, modelos horneados, entorno de pruebas y uso de los lockfiles. |

### 4.5 Forma implementada del runner

El código completo está en `app/jobs.py`. Este es el resumen de las piezas que
importan:

```python
class PlateJobRunner:
    def __init__(self, max_concurrency: int, max_queue: int):
        self._max_concurrency = max(1, max_concurrency)
        self._max_queue = max(1, max_queue)
        self._queue = None
        self._workers = []

    async def start(self):
        if self._queue is not None:
            return
        self._queue = asyncio.Queue(maxsize=self._max_queue)
        self._workers = [
            asyncio.create_task(self._worker()) for _ in range(self._max_concurrency)
        ]

    async def submit(self, img):
        if self._queue is None:
            raise RuntimeError("PlateJobRunner no está iniciado")
        if self._queue.full():
            raise QueueFullError()
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(_silence_future)
        job = PlateJob(img=img, future=future)
        self._queue.put_nowait(job)
        return job

    async def _worker(self):
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
                if job.cancel.is_set():
                    if not job.future.done():
                        job.future.set_result(None)
                    continue
                try:
                    result = await asyncio.to_thread(
                        process_image_pipeline, job.img, job.cancel
                    )
                except PipelineCancelled:
                    if not job.future.done():
                        job.future.set_result(None)
                else:
                    if not job.future.done():
                        job.future.set_result(result)
            finally:
                self._queue.task_done()
```

Puntos finos que respeta el código final:

- `asyncio.shield` al esperar el future, para que `wait_for` no cancele el future
  que el worker necesita para publicar el resultado.
- Si el cliente ya se fue por timeout, `_silence_future` consume la excepción y
  evita el aviso "Future exception was never retrieved".
- Al apagar la aplicación, `stop()` encola centinelas (`None`) y espera a los
  workers.

### 4.6 Puntos de control de cancelación

En `app/services.py`, antes de cada etapa pesada:

```python
def _ensure_not_cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise PipelineCancelled()

def process_image_pipeline(img, cancel=None):
    _ensure_not_cancelled(cancel)
    # resize...
    _ensure_not_cancelled(cancel)
    detections = detect_plate(img)
    for det in detections:
        _ensure_not_cancelled(cancel)
        plate_result = read_plate(img, bbox)
    ...
```

Con `cancel=None` el comportamiento es idéntico al actual, así que
`test/evaluate_validation.py` y los scripts locales siguen funcionando.

### 4.7 Límite conocido (importante)

La cancelación es **cooperativa**. Si una llamada individual a `detect_plate` o
`ocr.ocr` ya empezó, no se puede interrumpir a mitad de camino; se detendrá al
terminar esa etapa. Es una mejora real (evita seguir con las siguientes etapas y
libera el trabajador antes), pero no es cancelación dura.

Si en el futuro se necesita cancelación dura, habría que ejecutar la inferencia en
subprocesos y terminarlos; eso obliga a recargar o mantener modelos por proceso y es
mucho más complejo. No se incluye ahora.

### 4.8 Valores iniciales recomendados

| Variable | Valor inicial | Motivo |
|---|---|---|
| `PLATE_MAX_CONCURRENCY` | `1` | La instancia es pequeña y las librerías de inferencia no garantizan ser seguras entre hilos. Se sube a 2 solo después de probar carga real. |
| `PLATE_MAX_QUEUE` | `8` | Absorbe ráfagas cortas sin crecer sin control. |
| `PLATE_TIMEOUT_S` | `5` | Mantiene el contrato actual del endpoint. |

Advertencia técnica: los modelos se inicializan una sola vez al importar
(`app/ocr.py:17`, `app/model.py:12`). ONNX Runtime permite llamadas concurrentes,
pero PaddleOCR 2.x no garantiza que su predictor sea seguro entre hilos. Por eso el
valor seguro para empezar es 1; aumentar concurrencia requiere una prueba con modelos
reales.

### 4.9 Impacto en consumidores

Aparece un código nuevo: **503** cuando el servicio está saturado. El gateway y la
app móvil deberían reintentar después de `Retry-After`. El **408** sigue existiendo
para el caso "tu trabajo tardó demasiado". Esta parte del contrato debe comunicarse
al equipo de `BEsifaAPIGateway` y `SIFA_GO`.

---

## 5. Parte B: lockfile con hashes

### 5.1 Qué es y por qué se necesita

Hoy `requirements.txt` mezcla dependencias con y sin versión. Un lockfile es un
archivo **generado** que contiene:

- Todas las librerías, incluidas las que instalan otras librerías.
- La versión exacta de cada una.
- El hash de cada archivo descargable.

Así, dos personas (o el CI, o la EC2) instalan exactamente lo mismo. Si alguien
publica una versión alterada de un paquete, el hash no coincide y la instalación
falla en vez de ejecutar código distinto.

### 5.2 Archivos

| Archivo | Rol | ¿Se edita a mano? |
|---|---|---|
| `requirements.in` | Dependencias directas con versión exacta. | Sí. |
| `requirements.txt` | Lockfile generado con versiones y hashes de todo el árbol. | No. |
| `requirements-dev.in` | Dependencias mínimas de las pruebas unitarias. | Sí. |
| `requirements-dev.txt` | Lockfile de pruebas (sin PaddleOCR/YOLO). | No. |
| `Dockerfile.test` | Imagen mínima que instala `requirements-dev.txt`. | Sí. |

`requirements.in` implementado:

```text
fastapi==0.137.1
uvicorn[standard]==0.49.0
python-multipart==0.0.32
PyJWT[crypto]==2.13.0
python-dotenv==1.2.2
open-image-models[onnx]==0.5.1
paddlepaddle==2.6.2
paddleocr==2.7.3
opencv-python-headless==4.6.0.66
numpy==1.26.4
```

`requirements-dev.in` implementado (runtime mínimo para importar la app con
mocks; no incluye PaddleOCR, PaddlePaddle ni open-image-models):

```text
-c requirements.txt

fastapi==0.137.1
python-multipart==0.0.32
PyJWT[crypto]==2.13.0
python-dotenv==1.2.2
opencv-python-headless==4.6.0.66
numpy==1.26.4

pytest==9.1.0
pytest-cov==7.1.0
httpx==0.28.1
requests==2.34.2
```

`requests` queda para `test/test_carga.py`. La evaluación de precisión
(`test/evaluate_validation.py`) sí requiere el runtime completo y los modelos.

### 5.3 Cómo se genera

Se usa `pip-tools`. Los locks se generaron en **Linux x86_64 y Python 3.10**
(la misma plataforma de la imagen). En una máquina sin Python 3.10 se puede usar
un contenedor:

```bash
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD":/w -w /w python:3.10-slim bash -c '
    python -m venv /tmp/venv &&
    /tmp/venv/bin/pip install pip-tools==7.7.0 &&
    /tmp/venv/bin/pip-compile --generate-hashes \
      --output-file requirements.txt requirements.in &&
    /tmp/venv/bin/pip-compile --generate-hashes \
      --constraint requirements.txt \
      --output-file requirements-dev.txt requirements-dev.in
  '
```

El lock de pruebas ya no usa `--allow-unsafe`: no necesita fijar `pip`,
`setuptools` ni `wheel` porque la imagen base de Python 3.10 ya los trae. La
cabecera de cada `.txt` registra el comando exacto usado.

El resultado se versiona en Git. Cada vez que se cambie una versión en los `.in`, se
regeneran los `.txt` y se revisa el diff como cualquier otro cambio de código.

### 5.4 Cambio en el Dockerfile

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes --prefix=/install -r requirements.txt
```

`--require-hashes` obliga a que **todas** las líneas tengan hash y a que las
versiones estén fijas. Si falta una, el build falla: es intencional.

### 5.5 Reglas de oro

- Nunca editar `requirements.txt` a mano.
- No agregar dependencias solo en `requirements.txt`: van en el `.in` y se regenera.
- Regenerar el lock en Linux x86_64/Python 3.10. En otra plataforma pueden faltar
  hashes de wheels específicos y el build fallará.
- Un cambio de dependencia sin lockfile actualizado es un cambio incompleto.

---

## 6. Parte C: modelos en GitHub Release y horneados en la imagen

### 6.1 Decisión tomada

Se eligió la **Opción 1**: los modelos se descargan desde un GitHub Release **durante
el build de la imagen Docker** y quedan dentro de ella. No se descargan en tiempo de
arranque.

Ventajas:

- `docker compose up --build` sigue siendo el único comando.
- El contenedor arranca sin depender de GitHub.
- La versión del modelo queda atada al digest de la imagen.
- No hace falta mantener cachés en la EC2.

Desventajas asumidas:

- La imagen crece ~24 MB.
- Cambiar de modelo exige reconstruir y redistribuir la imagen.
- El build necesita internet (igual que ya ocurre con `pip install`).

### 6.2 Crear el release en GitHub (paso a paso)

El repositorio `jarodsmdev/sifaPlateDetectorBE` es **público**, así que la descarga no
necesita token ni credenciales.

#### Paso 1: empaquetar los modelos actuales

Desde la raíz del repositorio, crear un artefacto con una estructura fija. El
artefacto implementado se generó con:

```bash
rm -rf /tmp/plate-models-v1
mkdir -p /tmp/plate-models-v1
cp -a paddle_cache yolo_cache /tmp/plate-models-v1/

tar --sort=name --mtime='UTC 2020-01-01' --owner=0 --group=0 --numeric-owner \
    -czf plate-models-v1.tar.gz -C /tmp/plate-models-v1 paddle_cache yolo_cache
sha256sum plate-models-v1.tar.gz
```

Resultado actual:

```text
7814edec45d8cb2b7a38e92639f82ff336bafc4f6f9c9494cd5fc621982dbca6  plate-models-v1.tar.gz
```

El artefacto pesa ~16 MB comprimido y contiene 22 entradas: los modelos de
PaddleOCR bajo `paddle_cache/whl/...` y el ONNX de YOLO bajo `yolo_cache/...`.
El `tar` reproducible garantiza que el mismo contenido produzca siempre el mismo
SHA256; se verificó regenerándolo desde las carpetas locales.

#### Paso 2: publicar el release

Desde la interfaz web:

1. Ir a `https://github.com/jarodsmdev/sifaPlateDetectorBE/releases`.
2. "Draft a new release".
3. Tag: `models-v1` (crear el tag en el momento).
4. Título: `Modelos v1`.
5. Adjuntar `plate-models-v1.tar.gz`.
6. En las notas, pegar el SHA256.
7. "Publish release".

Con GitHub CLI (requiere `gh auth login`):

```bash
gh release create models-v1 plate-models-v1.tar.gz \
  --title "Modelos v1" \
  --notes "SHA256: 7814edec45d8cb2b7a38e92639f82ff336bafc4f6f9c9494cd5fc621982dbca6"
```

#### Paso 3: registrar URL y hash

La URL queda con este patrón:

```text
https://github.com/jarodsmdev/sifaPlateDetectorBE/releases/download/models-v1/plate-models-v1.tar.gz
```

Verificar que se descarga y que el hash coincide:

```bash
curl -fL -o /tmp/plate-models-v1.tar.gz \
  https://github.com/jarodsmdev/sifaPlateDetectorBE/releases/download/models-v1/plate-models-v1.tar.gz

sha256sum /tmp/plate-models-v1.tar.gz
# debe coincidir con el hash publicado
```

#### Paso 4: dejar los valores en el repositorio

Agregar en `Dockerfile` (como `ARG` con valor por defecto) y documentar en
`.envExample`. No son secretos; pueden estar en el repositorio.

```dockerfile
ARG PLATE_MODELS_URL="https://github.com/jarodsmdev/sifaPlateDetectorBE/releases/download/models-v1/plate-models-v1.tar.gz"
ARG PLATE_MODELS_SHA256="7814edec45d8cb2b7a38e92639f82ff336bafc4f6f9c9494cd5fc621982dbca6"
```

### 6.3 El script `scripts/fetch_models.sh`

Este script es el que descarga, verifica y coloca los modelos. Se ejecuta durante el
build de la imagen.

```bash
#!/usr/bin/env bash
set -euo pipefail

URL="${PLATE_MODELS_URL:?falta PLATE_MODELS_URL}"
SHA="${PLATE_MODELS_SHA256:?falta PLATE_MODELS_SHA256}"
PADDLE_DIR="${PADDLE_CACHE_DIR:-/root/.paddleocr}"
YOLO_DIR="${YOLO_CACHE_DIR:-/root/.cache/open-image-models}"

if [ -f "$PADDLE_DIR/.models-version" ] &&
   [ "$(cat "$PADDLE_DIR/.models-version")" = "$SHA" ] &&
   [ -f "$YOLO_DIR/.models-version" ] &&
   [ "$(cat "$YOLO_DIR/.models-version")" = "$SHA" ]; then
  echo "Modelos ya presentes y verificados ($SHA). Nada que hacer."
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Descargando modelos desde $URL"
curl --fail --location --retry 3 --retry-delay 2 \
     --connect-timeout 10 --max-time 300 \
     --output "$tmp/models.tar.gz" "$URL"

echo "$SHA  $tmp/models.tar.gz" | sha256sum --check --status || {
  echo "ERROR: el SHA256 del artefacto no coincide. Se aborta." >&2
  exit 1
}

tar -xzf "$tmp/models.tar.gz" -C "$tmp"

rm -rf "$PADDLE_DIR" "$YOLO_DIR"
mkdir -p "$PADDLE_DIR" "$YOLO_DIR"
cp -a "$tmp/paddle_cache/." "$PADDLE_DIR/"
cp -a "$tmp/yolo_cache/." "$YOLO_DIR/"

printf '%s' "$SHA" > "$PADDLE_DIR/.models-version"
printf '%s' "$SHA" > "$YOLO_DIR/.models-version"

echo "Modelos instalados y verificados ($SHA)."
```

Qué hace, línea por línea:

| Parte | Qué hace |
|---|---|
| `set -euo pipefail` | Corta el script ante cualquier error, variable sin definir o fallo dentro de una tubería. |
| `URL=...:?` | Exige que la URL exista; si falta, el build falla con un mensaje claro. |
| `PADDLE_DIR` / `YOLO_DIR` | Carpetas donde las librerías buscan los modelos. Se pueden sobrescribir para uso local. |
| Bloque `if` con `.models-version` | Si ya están los modelos de esa versión, no descarga de nuevo (idempotente). |
| `mktemp` + `trap` | Trabaja en un directorio temporal y lo borra siempre, incluso si falla. |
| `curl --fail --location --retry --connect-timeout --max-time` | Descarga siguiendo redirecciones, falla ante HTTP 4xx/5xx, reintenta y corta si la conexión no responde en 10 s (máximo 5 min por intento). |
| `sha256sum --check --status` | Verifica la huella digital; si no coincide, aborta. |
| `tar -xzf` | Descomprime el artefacto. |
| `cp -a` | Copia los modelos a las rutas que esperan PaddleOCR y open-image-models. |
| `.models-version` | Marcador para no repetir la descarga. |

El script es idempotente: se puede ejecutar varias veces sin efectos secundarios si
la versión ya está instalada. La verificación SHA256 es su autocomprobación: si el
archivo descargado está corrupto o cambió, el build no continúa.

### 6.4 Cambios en el Dockerfile

Además del `ARG`, hay que instalar `curl` y `ca-certificates` en la imagen (para
descargar por HTTPS) y copiar el script:

```dockerfile
# STAGE 1: system deps
FROM base AS system-deps

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# STAGE 3: runtime
FROM system-deps AS runtime

WORKDIR /app

ARG PLATE_MODELS_URL="https://github.com/jarodsmdev/sifaPlateDetectorBE/releases/download/models-v1/plate-models-v1.tar.gz"
ARG PLATE_MODELS_SHA256="7814edec45d8cb2b7a38e92639f82ff336bafc4f6f9c9494cd5fc621982dbca6"
LABEL org.sifa.models.sha256="${PLATE_MODELS_SHA256}"

COPY --from=builder /install /usr/local
COPY scripts/fetch_models.sh /usr/local/bin/fetch_models.sh

RUN chmod +x /usr/local/bin/fetch_models.sh \
    && /usr/local/bin/fetch_models.sh

COPY app ./app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

En logs del build debe aparecer `Modelos instalados y verificados (<hash>)` o
`Modelos ya presentes...`.

También conviene fijar la imagen base por digest, no solo por etiqueta:

```dockerfile
FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d AS base
```

El digest se obtiene con:

```bash
docker buildx imagetools inspect python:3.10-slim
```

### 6.5 Cambios en Docker Compose y en el deploy

Con los modelos dentro de la imagen, los volúmenes de caché **deben eliminarse**.
Si se dejan, un directorio vacío del host tapa los modelos horneados.

En `docker-compose.yml` ya se eliminaron:

```yaml
volumes:
  - ./data:/data
```

y se agregaron los argumentos de build y las variables del pool:

```yaml
build:
  context: .
  args:
    PLATE_MODELS_URL: ${PLATE_MODELS_URL:-https://github.com/jarodsmdev/sifaPlateDetectorBE/releases/download/models-v1/plate-models-v1.tar.gz}
    PLATE_MODELS_SHA256: ${PLATE_MODELS_SHA256:-7814edec45d8cb2b7a38e92639f82ff336bafc4f6f9c9494cd5fc621982dbca6}
...
environment:
  - PLATE_MAX_CONCURRENCY=${PLATE_MAX_CONCURRENCY:-1}
  - PLATE_MAX_QUEUE=${PLATE_MAX_QUEUE:-8}
  - PLATE_TIMEOUT_S=${PLATE_TIMEOUT_S:-5.0}
```

`deploy_SIFA-PLATE.sh` y `.github/workflows/deploy.yml` ya no montan
`paddle_cache` ni `yolo_cache`. El workflow agrega `PLATE_MAX_CONCURRENCY=1`,
`PLATE_MAX_QUEUE=8` y `PLATE_TIMEOUT_S=5.0` a `/tmp/plate.env`.

Las carpetas `paddle_cache/` y `yolo_cache/` en la EC2 quedan sin uso y se pueden
borrar más adelante.

### 6.6 Sacar los modelos de Git

Ya se ejecutó `git rm -r --cached paddle_cache yolo_cache` (los archivos locales se
conservan). Ambas rutas están en `.gitignore` y en `.dockerignore`, junto con
`plate-models-*.tar.gz`.

Verificación:

```bash
git ls-files | grep -E 'paddle_cache|yolo_cache'
# no devuelve nada
```

Nota importante: `git rm --cached` no borra los modelos del historial. Un clon nuevo
seguirá descargando los commits antiguos con los binarios. Para eliminarlos del todo
hay que reescribir el historial con `git filter-repo` y coordinar un `git push
--force` con todo el equipo. Es opcional y no forma parte de esta implementación;
sin hacerlo, el ahorro es solo de aquí en adelante.

### 6.7 Actualizaciones futuras de modelos

Procedimiento cada vez que cambie un modelo:

1. Generar un artefacto nuevo (`plate-models-v2.tar.gz`).
2. Verificar su SHA256.
3. Crear un release nuevo (`models-v2`) y adjuntar el archivo.
4. **No sobrescribir** `models-v1`.
5. Actualizar `PLATE_MODELS_URL` y `PLATE_MODELS_SHA256` en el Dockerfile (o en
   Compose/CI) en un commit revisable.
6. Reconstruir y redistribuir la imagen.
7. Mantener los releases antiguos: las imágenes viejas los necesitan para ser
   reproducibles.

Como los binarios ya no están en el repositorio, el release se publica a mano o con
un workflow manual (`workflow_dispatch`). No se puede generar desde el código.

### 6.8 Qué significa para el flujo diario

- **`docker compose up --build`**: sigue funcionando igual. La primera vez (o si se
  limpia la caché de capas de Docker) descarga ~24 MB desde GitHub.
- **Sin internet**: el build falla si no hay capa cacheada. Alternativa: usar una
  imagen ya construida localmente. Es el costo de la Opción 1.
- **Tests unitarios (`pytest`)**: no usan modelos reales (hay mocks en
  `test/conftest.py`), así que no cambian.
- **Evaluación de precisión (`test/evaluate_validation.py`)**: necesita modelos. Se
  puede ejecutar el script en el host para poblar `./paddle_cache` y
  `./yolo_cache`, o correr dentro de la imagen.
- **EC2**: no necesita descargar modelos; ya vienen en la imagen. Sigue necesitando
  internet para `docker pull`.

Para uso local sin Docker, el script permite rutas alternativas:

```bash
PADDLE_CACHE_DIR="$PWD/paddle_cache" \
YOLO_CACHE_DIR="$PWD/yolo_cache" \
PLATE_MODELS_URL="https://github.com/jarodsmdev/sifaPlateDetectorBE/releases/download/models-v1/plate-models-v1.tar.gz" \
PLATE_MODELS_SHA256="7814edec45d8cb2b7a38e92639f82ff336bafc4f6f9c9494cd5fc621982dbca6" \
./scripts/fetch_models.sh
```

---

## 7. Verificación: lista de comprobación

Estado de la verificación al cerrar la implementación:

- [x] `pytest` pasa completo: 39 pruebas en `python:3.10-slim` con
      `Dockerfile.test` y las dependencias de `requirements-dev.txt`.
- [x] `test/test_jobs.py` cubre éxito, cola llena (503) y cancelación antes de
      empezar y antes del OCR.
- [x] `test/test_timeout.py` devuelve 408 y activa el evento de cancelación.
- [ ] Prueba de carga con 15 peticiones: se observa 503 al saturar, sin crecimiento
      descontrolado de CPU. (Pendiente; requiere el servicio levantado.)
- [x] `git ls-files` no lista ningún archivo de `paddle_cache/` ni `yolo_cache/`.
- [x] `curl` de la URL del release devuelve el artefacto y el SHA256 coincide
      (`15.95 MB`, `7814edec...dbca6`).
- [x] Build de la imagen validado con un servidor HTTP local como origen del
      artefacto; el log mostró `Modelos instalados y verificados (...)`. Con el
      release real será el mismo paso, solo cambia la URL.
- [x] La imagen resultante carga YOLO y PaddleOCR con `docker run --network none`,
      lo que demuestra que los modelos quedaron horneados y no se descargan al
      arrancar.
- [ ] `docker compose up --build` contra el release real y detección de una patente.
- [x] El contenedor no tiene montados `paddle_cache` ni `yolo_cache` en Compose ni
      en los scripts de deploy.
- [x] `requirements.txt` se instala con `pip install --require-hashes` en
      `python:3.10-slim`. No se comprobó que regenerarlo produzca un diff vacío.
- [ ] El deploy en EC2 usa la nueva imagen sin volúmenes de modelos y responde.

---

## 8. Riesgos y mitigaciones

| Riesgo | Qué pasaría | Mitigación |
|---|---|---|
| **Asset mutable** | Alguien reemplaza el archivo del release con el mismo nombre; el hash deja de coincidir o se actualiza sin revisión. | Nombres inmutables por versión (`models-v1`, `models-v2`); hash versionado en un commit revisable; nunca reutilizar una versión. |
| **Volumen tapa los modelos** | Si se hornean modelos y se deja el bind mount, el directorio vacío del host oculta la imagen y el servicio falla o descarga. | Quitar los volúmenes en Compose y en los scripts de deploy (parte 6.5). |
| **Historial de Git** | Los ~23 MB siguen en commits antiguos; un clon nuevo los descarga. | Aceptar el costo o reescribir historia con `git filter-repo` en un trabajo coordinado. |
| **GitHub no disponible** | Falla el build de la imagen (no el runtime). | Usar imagen ya construida; reintentar; el asset es pequeño y está en CDN de GitHub. |
| **Librería vs. artefacto** | Un upgrade de PaddleOCR/open-image-models cambia el layout de caché y el artefacto v1 deja de servir. | Regenerar el artefacto y bumpear versión junto con el upgrade del lockfile. |
| **Lock específico de plataforma** | Hashes generados en macOS no cubren wheels de Linux y el build falla. | Generar el lock en Linux x86_64/Python 3.10, igual que la imagen. |
| **Hashes y mirror privado** | Si se usa un índice distinto de PyPI, los hashes no coinciden. | Usar PyPI o regenerar con el índice real; documentar el índice. |
| **Concurrencia de PaddleOCR** | Varias inferencias simultáneas sobre el mismo predictor podrían corromper estado. | Empezar con `PLATE_MAX_CONCURRENCY=1`; subir solo tras prueba real. |
| **Timeout cooperativo** | Una inferencia ya iniciada sigue hasta el próximo punto de control. | Aceptado; el pool acotado evita crecimiento infinito y el 503 protege al servidor. |
| **503 en consumidores** | Gateway o app móvil pueden no reintentar y mostrar error al usuario. | Coordinar contrato con `BEsifaAPIGateway` y `SIFA_GO`; reintento con `Retry-After`. |
| **Licencias de modelos** | Redistribuir modelos públicos puede tener restricciones. | Confirmar licencias antes de publicar el release. |

---

## 9. Fuera de alcance de esta implementación

- Cancelación dura con subprocesos.
- Rotación de los secretos hardcodeados en `deploy_SIFA-PLATE.sh`.
- Reescritura del historial de Git para eliminar los modelos de commits antiguos.
- Cambios en el gateway o en la app móvil para manejar el 503 (solo se documenta la
  necesidad).
- Pin de paquetes del sistema operativo (`libgl1`, etc.) más allá de la imagen base.

---

## 10. Valores a registrar en el repositorio

Al terminar la migración deben quedar visibles en el código (no en un chat ni en la
cabeza de una persona):

| Dato | Valor / ubicación |
|---|---|
| URL del release de modelos | `https://github.com/jarodsmdev/sifaPlateDetectorBE/releases/download/models-v1/plate-models-v1.tar.gz` en `Dockerfile`, `docker-compose.yml` y `.envExample` |
| SHA256 del artefacto | `7814edec45d8cb2b7a38e92639f82ff336bafc4f6f9c9494cd5fc621982dbca6` en `Dockerfile`, `docker-compose.yml` y `.envExample` |
| Versión de Python de la imagen | `Dockerfile`: `python:3.10-slim` |
| Digest de la imagen base | `python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d` |
| Imagen de pruebas | `Dockerfile.test` (instala solo `requirements-dev.txt`) |
| Versiones directas de librerías | `requirements.in` y `requirements-dev.in` |
| Lockfiles completos | `requirements.txt` y `requirements-dev.txt` (generados) |
| Límites de concurrencia y timeout | `app/config.py`, `docker-compose.yml`, `.envExample` y `.github/workflows/deploy.yml` |

### 10.1 Pendientes fuera del código

1. Ejecutar la prueba de carga de 15 peticiones y ajustar los límites si hace falta.
2. Validar `docker compose up --build` contra el release real.
3. Desplegar en EC2 y comprobar que responde sin volúmenes de modelos.
4. Comunicar el 503 y `Retry-After` a los equipos de `BEsifaAPIGateway` y
   `SIFA_GO`.

---

## 11. Orden de ejecución

1. [x] Parte A (pool, cola y cancelación) con sus pruebas.
2. [x] Parte B (lockfile) y build con `--require-hashes`.
3. [x] Parte C en código (script, Dockerfile, Compose, deploy, `.gitignore`,
   `.dockerignore` y sacar los binarios de Git).
4. [ ] Release publicado y SHA verificado; faltan la prueba de carga y el
   `docker compose up --build` contra el release real (sección 7).
5. [ ] Comunicar el nuevo 503 a los equipos de gateway y app móvil.
