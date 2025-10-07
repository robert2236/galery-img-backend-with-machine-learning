import os
import shutil
from pathlib import Path

# Configuración
carpeta_origen = "Objects"
carpeta_destino = "datasets_objects"
total_imagenes = 1500

# Crear carpeta de destino
Path(carpeta_destino).mkdir(exist_ok=True)

# Extensiones de imagen válidas
extensiones_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

print(f"Buscando imágenes en: {carpeta_origen}")

# Obtener todas las imágenes de la carpeta
imagenes = []
for archivo in os.listdir(carpeta_origen):
    if os.path.splitext(archivo)[1].lower() in extensiones_validas:
        imagenes.append(archivo)

print(f"Se encontraron {len(imagenes)} imágenes")

# Verificar si hay suficientes imágenes
if len(imagenes) < total_imagenes:
    print(f"⚠️  Solo hay {len(imagenes)} imágenes. Se copiarán todas.")
    total_imagenes = len(imagenes)

# Copiar las imágenes
contador = 0
for i, imagen in enumerate(imagenes[:total_imagenes]):
    origen = os.path.join(carpeta_origen, imagen)
    destino = os.path.join(carpeta_destino, f"img_{i+1:04d}{os.path.splitext(imagen)[1]}")
    
    shutil.copy2(origen, destino)
    contador += 1
    
    # Mostrar progreso cada 100 imágenes
    if (i + 1) % 100 == 0:
        print(f"Copiadas {i + 1} imágenes...")

print(f"✅ Listo! Se copiaron {contador} imágenes a la carpeta '{carpeta_destino}'")