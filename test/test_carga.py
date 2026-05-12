import concurrent.futures
import requests
import time

# Cambia esto por la IP de tu EC2 si ya está en la nube, o déjalo así si pruebas en local
URL = "http://localhost:8000/plate/api/v1/detect"
IMAGE_PATH = "test/patente5.jpg" # La ruta a tu imagen de prueba local

def simular_fiscalizador(id_fiscalizador):
    print(f"[Fiscalizador {id_fiscalizador}] Disparando foto...")
    start_time = time.time()
    
    try:
        # Abrimos la imagen y la enviamos por POST
        with open(IMAGE_PATH, 'rb') as f:
            archivos = {'file': (IMAGE_PATH, f, 'image/jpeg')}
            respuesta = requests.post(URL, files=archivos)
        
        end_time = time.time()
        tiempo_total = end_time - start_time
        
        if respuesta.status_code == 200:
            print(f"✅ [Fiscalizador {id_fiscalizador}] Patente leída en {tiempo_total:.2f} segundos.")
        else:
            print(f"❌ [Fiscalizador {id_fiscalizador}] Error {respuesta.status_code}: {respuesta.text}")
            
    except Exception as e:
        print(f"⚠️ [Fiscalizador {id_fiscalizador}] Fallo de conexión: {e}")

# Aquí definimos cuántos fiscalizadores disparan a la vez
CANTIDAD_FISCALIZADORES = 15

print(f"Iniciando prueba de estrés con {CANTIDAD_FISCALIZADORES} peticiones simultáneas...\n")

# Usamos ThreadPoolExecutor para lanzar las peticiones en paralelo real
with concurrent.futures.ThreadPoolExecutor(max_workers=CANTIDAD_FISCALIZADORES) as executor:
    # Mapeamos la función a la cantidad de fiscalizadores
    executor.map(simular_fiscalizador, range(1, CANTIDAD_FISCALIZADORES + 1))

print("\nPrueba finalizada.")