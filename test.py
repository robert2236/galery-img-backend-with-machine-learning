import os
import pickle

def quick_test():
    print("🚀 PRUEBA RÁPIDA DE CARGA DEL MODELO")
    print("="*40)
    
    target_file = 'image_classifier_model.pkl'
    
    print(f"Buscando: {target_file}")
    print(f"Directorio: {os.getcwd()}")
    6
    
    if os.path.exists(target_file):
        print(f"✅ Archivo encontrado!")
        print(f"📏 Tamaño: {os.path.getsize(target_file)} bytes")
        
        try:
            with open(target_file, 'rb') as f:
                data = pickle.load(f)
            print("🎉 ¡Modelo cargado correctamente!")
            print(f"Tipo: {type(data)}")
            if isinstance(data, dict):
                print("Keys disponibles:", list(data.keys()))
        except Exception as e:
            print(f"❌ Error al cargar: {e}")
    else:
        print("❌ Archivo no encontrado")
        print("Archivos .pkl disponibles:")
        for f in os.listdir('.'):
            if f.endswith('.pkl'):
                print(f"  - {f}")

# Ejecuta la prueba
quick_test()