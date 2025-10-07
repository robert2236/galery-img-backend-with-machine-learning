import os
import shutil
from pathlib import Path

def extract_human_faces():
    # Configuración de rutas
    humans_folder = "humans"  # Carpeta principal
    female_folder = os.path.join(humans_folder, "female_faces")
    male_folder = os.path.join(humans_folder, "male_faces")
    output_folder = "datasets_humans"  # Carpeta de destino
    
    # Crear carpeta de destino si no existe
    Path(output_folder).mkdir(exist_ok=True)
    
    # Configuración de objetivos
    target_per_category = 750
    total_target = target_per_category * 2  # 1500 imágenes total
    
    # Extensiones de imagen válidas
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    
    print("=== EXTRACCIÓN DE ROSTROS HUMANOS ===")
    print(f"Objetivo: {target_per_category} imágenes por categoría")
    print(f"Total: {total_target} imágenes\n")
    
    total_images = 0
    results = {}
    
    # Procesar rostros femeninos
    print("👩 Procesando female_faces...")
    if not os.path.exists(female_folder):
        print(f"❌ Carpeta no encontrada: {female_folder}")
        results['female'] = 0
    else:
        female_images = [f for f in os.listdir(female_folder) 
                        if os.path.isfile(os.path.join(female_folder, f)) and 
                        os.path.splitext(f)[1].lower() in valid_extensions]
        
        print(f"   Encontradas {len(female_images)} imágenes disponibles")
        
        images_to_take = min(target_per_category, len(female_images))
        copied_female = 0
        
        for i, image_file in enumerate(female_images[:images_to_take]):
            source_path = os.path.join(female_folder, image_file)
            new_filename = f"female_{i+1:04d}{os.path.splitext(image_file)[1]}"
            destination_path = os.path.join(output_folder, new_filename)
            
            try:
                shutil.copy2(source_path, destination_path)
                copied_female += 1
                total_images += 1
                
                # Mostrar progreso cada 50 imágenes
                if (i + 1) % 50 == 0:
                    print(f"   Progreso: {i + 1}/{images_to_take}")
                    
            except Exception as e:
                print(f"   ❌ Error copiando {image_file}: {e}")
        
        results['female'] = copied_female
        print(f"   ✅ Copiadas {copied_female} imágenes femeninas")
    
    # Procesar rostros masculinos
    print("\n👨 Procesando male_faces...")
    if not os.path.exists(male_folder):
        print(f"❌ Carpeta no encontrada: {male_folder}")
        results['male'] = 0
    else:
        male_images = [f for f in os.listdir(male_folder) 
                      if os.path.isfile(os.path.join(male_folder, f)) and 
                      os.path.splitext(f)[1].lower() in valid_extensions]
        
        print(f"   Encontradas {len(male_images)} imágenes disponibles")
        
        images_to_take = min(target_per_category, len(male_images))
        copied_male = 0
        
        for i, image_file in enumerate(male_images[:images_to_take]):
            source_path = os.path.join(male_folder, image_file)
            new_filename = f"male_{i+1:04d}{os.path.splitext(image_file)[1]}"
            destination_path = os.path.join(output_folder, new_filename)
            
            try:
                shutil.copy2(source_path, destination_path)
                copied_male += 1
                total_images += 1
                
                # Mostrar progreso cada 50 imágenes
                if (i + 1) % 50 == 0:
                    print(f"   Progreso: {i + 1}/{images_to_take}")
                    
            except Exception as e:
                print(f"   ❌ Error copiando {image_file}: {e}")
        
        results['male'] = copied_male
        print(f"   ✅ Copiadas {copied_male} imágenes masculinas")
    
    # Generar reporte final
    print("\n" + "="*50)
    print("📊 REPORTE FINAL")
    print("="*50)
    
    female_copied = results.get('female', 0)
    male_copied = results.get('male', 0)
    
    print(f"👩 Rostros femeninos: {female_copied}/{target_per_category}")
    print(f"👨 Rostros masculinos: {male_copied}/{target_per_category}")
    print(f"📦 Total general: {total_images}/{total_target}")
    print(f"📁 Carpeta destino: {output_folder}")
    
    # Verificar archivos copiados
    if os.path.exists(output_folder):
        final_files = [f for f in os.listdir(output_folder) 
                      if os.path.isfile(os.path.join(output_folder, f))]
        print(f"📋 Archivos en destino: {len(final_files)}")
        
        # Mostrar algunos ejemplos
        if final_files:
            print("\n🔍 Primeros 5 archivos como ejemplo:")
            for file in sorted(final_files)[:5]:
                print(f"   - {file}")

def extract_with_random_selection():
    """Versión que selecciona imágenes aleatoriamente en lugar de secuencialmente"""
    import random
    
    humans_folder = "humans"
    female_folder = os.path.join(humans_folder, "female_faces")
    male_folder = os.path.join(humans_folder, "male_faces")
    output_folder = "datasets_humans_random"
    
    Path(output_folder).mkdir(exist_ok=True)
    
    target_per_category = 750
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    
    print("=== EXTRACCIÓN ALEATORIA ===")
    print("Seleccionando imágenes de forma aleatoria...\n")
    
    results = {}
    
    # Procesar female_faces con selección aleatoria
    print("👩 Procesando female_faces (aleatorio)...")
    if os.path.exists(female_folder):
        female_images = [f for f in os.listdir(female_folder) 
                        if os.path.isfile(os.path.join(female_folder, f)) and 
                        os.path.splitext(f)[1].lower() in valid_extensions]
        
        random.shuffle(female_images)  # Mezclar aleatoriamente
        
        images_to_take = min(target_per_category, len(female_images))
        copied_female = 0
        
        for i, image_file in enumerate(female_images[:images_to_take]):
            source_path = os.path.join(female_folder, image_file)
            new_filename = f"female_rand_{i+1:04d}{os.path.splitext(image_file)[1]}"
            destination_path = os.path.join(output_folder, new_filename)
            
            try:
                shutil.copy2(source_path, destination_path)
                copied_female += 1
            except Exception as e:
                print(f"   ❌ Error copiando {image_file}: {e}")
        
        results['female'] = copied_female
        print(f"   ✅ Copiadas {copied_female} imágenes femeninas (aleatorio)")
    
    # Procesar male_faces con selección aleatoria
    print("\n👨 Procesando male_faces (aleatorio)...")
    if os.path.exists(male_folder):
        male_images = [f for f in os.listdir(male_folder) 
                      if os.path.isfile(os.path.join(male_folder, f)) and 
                      os.path.splitext(f)[1].lower() in valid_extensions]
        
        random.shuffle(male_images)  # Mezclar aleatoriamente
        
        images_to_take = min(target_per_category, len(male_images))
        copied_male = 0
        
        for i, image_file in enumerate(male_images[:images_to_take]):
            source_path = os.path.join(male_folder, image_file)
            new_filename = f"male_rand_{i+1:04d}{os.path.splitext(image_file)[1]}"
            destination_path = os.path.join(output_folder, new_filename)
            
            try:
                shutil.copy2(source_path, destination_path)
                copied_male += 1
            except Exception as e:
                print(f"   ❌ Error copiando {image_file}: {e}")
        
        results['male'] = copied_male
        print(f"   ✅ Copiadas {copied_male} imágenes masculinas (aleatorio)")
    
    print(f"\n📊 Total aleatorio: {sum(results.values())} imágenes")
    print(f"📁 Carpeta: {output_folder}")

if __name__ == "__main__":
    print("Selecciona el método de extracción para rostros humanos:")
    print("1. Extracción secuencial (recomendado)")
    print("2. Extracción aleatoria")
    print("3. Ejecutar ambos métodos")
    
    choice = input("Ingresa 1, 2 o 3: ").strip()
    
    if choice == "1":
        extract_human_faces()
    elif choice == "2":
        extract_with_random_selection()
    elif choice == "3":
        print("\n" + "="*60)
        extract_human_faces()
        print("\n" + "="*60)
        extract_with_random_selection()
    else:
        print("Opción no válida, usando extracción secuencial")
        extract_human_faces()