import os
import shutil
from pathlib import Path

def extract_images_to_dataset():
    # Configuración de rutas
    animals_folder = "animals"  # Carpeta principal con las carpetas de animales
    output_folder = "datasets_animals"  # Carpeta de destino
    
    # Crear carpeta de destino si no existe
    Path(output_folder).mkdir(exist_ok=True)
    
    # Lista de carpetas de animales (puedes ajustar según tus necesidades)
    animal_folders = ["cat", "cow", "deer", "dog", "goat", "hen", "nightvision", "rabbit", "sheep"]
    
    # Extensiones de imagen válidas
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    total_images = 0
    target_count = 1500
    
    print("Iniciando extracción de imágenes...")
    
    for animal in animal_folders:
        animal_path = os.path.join(animals_folder, animal)
        
        if not os.path.exists(animal_path):
            print(f"⚠️  Carpeta {animal_path} no encontrada, saltando...")
            continue
        
        print(f"\n📁 Procesando: {animal}")
        
        # Contar imágenes disponibles en esta carpeta
        available_images = []
        for file in os.listdir(animal_path):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in valid_extensions:
                available_images.append(file)
        
        print(f"   Encontradas {len(available_images)} imágenes")
        
        # Calcular cuántas imágenes tomar de esta carpeta
        # Distribuir proporcionalmente para llegar a 1000
        remaining_needed = target_count - total_images
        if remaining_needed <= 0:
            break
        
        # Tomar todas las imágenes si hay pocas, o una cantidad proporcional
        images_to_take = min(len(available_images), remaining_needed)
        
        # Copiar las imágenes
        copied_count = 0
        for i, image_file in enumerate(available_images):
            if copied_count >= images_to_take:
                break
                
            source_path = os.path.join(animal_path, image_file)
            
            # Crear nombre único para evitar sobreescrituras
            new_filename = f"{animal}_{i+1:03d}{os.path.splitext(image_file)[1]}"
            destination_path = os.path.join(output_folder, new_filename)
            
            try:
                shutil.copy2(source_path, destination_path)
                copied_count += 1
                total_images += 1
                
                if total_images >= target_count:
                    break
                    
            except Exception as e:
                print(f"   ❌ Error copiando {image_file}: {e}")
        
        print(f"   ✅ Copiadas {copied_count} imágenes de {animal}")
        
        if total_images >= target_count:
            print("\n🎯 ¡Meta de 1500 imágenes alcanzada!")
            break
    
    print(f"\n📊 Resumen final:")
    print(f"   Total de imágenes copiadas: {total_images}")
    print(f"   Carpeta destino: {output_folder}")
    
    # Verificar contenido de la carpeta destino
    if os.path.exists(output_folder):
        final_count = len([f for f in os.listdir(output_folder) 
                          if os.path.isfile(os.path.join(output_folder, f))])
        print(f"   Imágenes en carpeta destino: {final_count}")

# Versión alternativa que distribuye equitativamente entre carpetas
def extract_images_balanced():
    # Configuración de rutas
    animals_folder = "animals"
    output_folder = "datasets_animals_balanced"
    
    # Crear carpeta de destino
    Path(output_folder).mkdir(exist_ok=True)
    
    animal_folders = ["cat", "cow", "deer", "dog", "goat", "hen", "nightvision", "rabbit", "sheep"]
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    target_count = 1500
    num_folders = len(animal_folders)
    images_per_folder = target_count // num_folders
    remainder = target_count % num_folders
    
    print("Iniciando extracción balanceada...")
    print(f"Objetivo: ~{images_per_folder} imágenes por carpeta")
    
    total_images = 0
    
    for i, animal in enumerate(animal_folders):
        animal_path = os.path.join(animals_folder, animal)
        
        if not os.path.exists(animal_path):
            print(f"⚠️  Carpeta {animal_path} no encontrada, saltando...")
            continue
        
        # Calcular cuántas imágenes tomar de esta carpeta
        images_to_take = images_per_folder
        if i < remainder:  # Distribuir el resto equitativamente
            images_to_take += 1
        
        print(f"\n📁 Procesando: {animal} (objetivo: {images_to_take} imágenes)")
        
        # Encontrar todas las imágenes en la carpeta
        available_images = []
        for file in os.listdir(animal_path):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in valid_extensions:
                available_images.append(file)
        
        if not available_images:
            print(f"   ❌ No se encontraron imágenes en {animal}")
            continue
        
        # Tomar las imágenes necesarias
        images_to_copy = min(images_to_take, len(available_images))
        
        copied_count = 0
        for j, image_file in enumerate(available_images):
            if copied_count >= images_to_copy:
                break
                
            source_path = os.path.join(animal_path, image_file)
            new_filename = f"{animal}_{j+1:03d}{os.path.splitext(image_file)[1]}"
            destination_path = os.path.join(output_folder, new_filename)
            
            try:
                shutil.copy2(source_path, destination_path)
                copied_count += 1
                total_images += 1
            except Exception as e:
                print(f"   ❌ Error copiando {image_file}: {e}")
        
        print(f"   ✅ Copiadas {copied_count} imágenes de {animal}")
    
    print(f"\n📊 Resumen final (balanceado):")
    print(f"   Total de imágenes copiadas: {total_images}")
    print(f"   Carpeta destino: {output_folder}")

if __name__ == "__main__":
    print("Selecciona el método de extracción:")
    print("1. Extracción secuencial (llena hasta 1000 imágenes)")
    print("2. Extracción balanceada (distribuye equitativamente)")
    
    choice = input("Ingresa 1 o 2: ").strip()
    
    if choice == "1":
        extract_images_to_dataset()
    elif choice == "2":
        extract_images_balanced()
    else:
        print("Opción no válida, usando extracción secuencial por defecto")
        extract_images_to_dataset()