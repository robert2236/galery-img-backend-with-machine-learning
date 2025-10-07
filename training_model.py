import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from PIL import Image
from scipy.stats import skew, kurtosis

# Importaciones para procesamiento de imágenes
from skimage.io import imread
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from skimage import filters, exposure, feature
from skimage.feature import hog, canny, corner_harris, corner_peaks
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte
from skimage.exposure import equalize_adapthist
from skimage.measure import regionprops, label

# Importaciones para machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight

class HighResolutionPreprocessor:
    """Preprocesador optimizado para alta resolución 400x400"""
    
    def __init__(self, target_size=(400, 400)):
        self.target_size = target_size
        print(f"🎯 Configurado para alta resolución: {target_size}")
        
    def preprocess_image(self, img, apply_augmentation=False, feature_level='high'):
        """
        Preprocesamiento optimizado para 400x400
        """
        try:
            # Si es una ruta, cargar la imagen
            if isinstance(img, str):
                img = imread(img)
            
            # Aplicar data augmentation si se solicita
            if apply_augmentation:
                img = self.apply_highres_augmentation(img)
            
            # 1. Conversión a escala de grises
            if len(img.shape) == 3:
                img = rgb2gray(img)
            
            # 2. Redimensionar a alta resolución
            original_shape = img.shape
            img = resize(img, self.target_size, anti_aliasing=True, order=1)
            
            # 3. Mejora de contraste adaptativo
            img = equalize_adapthist(img, clip_limit=0.02)
            
            # 4. Filtrado optimizado
            img = filters.gaussian(img, sigma=1.0)
            
            # Extraer características
            if feature_level == 'high':
                features = self.extract_highres_features(img)
            else:
                features = img.flatten()
                
            return features, img
            
        except Exception as e:
            print(f"❌ Error en preprocesamiento: {str(e)}")
            default_size = self.target_size[0] * self.target_size[1] + 600
            default_features = np.zeros(default_size)
            default_img = np.zeros(self.target_size)
            return default_features, default_img
    
    def apply_highres_augmentation(self, img):
        """Aumentación optimizada para alta resolución"""
        # Flip horizontal
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        
        # Rotación leve
        if np.random.random() > 0.6:
            angle = np.random.uniform(-15, 15)
            img = rotate(img, angle, mode='reflect')
        
        # Zoom sutil
        if np.random.random() > 0.5:
            zoom_factor = np.random.uniform(0.95, 1.05)
            new_size = int(self.target_size[0] * zoom_factor)
            img_resized = resize(img, (new_size, new_size), anti_aliasing=True)
            
            if new_size > self.target_size[0]:
                start = (new_size - self.target_size[0]) // 2
                img = img_resized[start:start+self.target_size[0], start:start+self.target_size[1]]
            else:
                pad = (self.target_size[0] - new_size) // 2
                img = np.pad(img_resized, ((pad, pad), (pad, pad)), mode='reflect')
        
        return img
    
    def extract_highres_features(self, img):
        """Características optimizadas para alta resolución"""
        features_list = []
        
        # 1. Características de píxeles (submuestreadas)
        pixel_features = resize(img, (200, 200), anti_aliasing=True).flatten()
        features_list.append(pixel_features)
        
        # 2. HOG multi-escala
        hog_features = self.extract_highres_hog(img)
        features_list.append(hog_features)
        
        # 3. Características de textura en regiones
        texture_features = self.extract_region_texture(img)
        features_list.append(texture_features)
        
        # 4. Características de bordes multi-escala
        edge_features = self.extract_multi_scale_edges(img)
        features_list.append(edge_features)
        
        # 5. Características estadísticas
        stats_features = self.extract_advanced_statistics(img)
        features_list.append(stats_features)
        
        # 6. Características de forma
        shape_features = self.extract_shape_features(img)
        features_list.append(shape_features)
        
        return np.concatenate(features_list)
    
    def extract_highres_hog(self, img):
        """HOG optimizado para alta resolución"""
        hog_features = []
        
        # HOG en resolución completa
        hog1 = hog(img, orientations=9, pixels_per_cell=(32, 32),
                  cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(hog1)
        
        # HOG en escala media
        img_medium = resize(img, (200, 200), anti_aliasing=True)
        hog2 = hog(img_medium, orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(hog2)
        
        return np.concatenate(hog_features)
    
    def extract_region_texture(self, img):
        """Extrae textura de diferentes regiones"""
        texture_features = []
        img_uint8 = img_as_ubyte(img)
        
        # Dividir imagen en 4 regiones
        h, w = img.shape
        regions = [
            img_uint8[:h//2, :w//2],  # Superior izquierda
            img_uint8[:h//2, w//2:],  # Superior derecha
            img_uint8[h//2:, :w//2],  # Inferior izquierda
            img_uint8[h//2:, w//2:]   # Inferior derecha
        ]
        
        for region in regions:
            # LBP para cada región
            lbp = local_binary_pattern(region, 8, 1, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            if np.sum(hist) > 0:
                hist = hist / np.sum(hist)
            texture_features.append(hist)
        
        return np.concatenate(texture_features)
    
    def extract_multi_scale_edges(self, img):
        """Características de bordes en múltiples escalas"""
        edge_features = []
        
        scales = [1.0, 1.5, 2.0]
        for sigma in scales:
            edges = canny(img, sigma=sigma)
            if np.any(edges):
                edge_density = np.sum(edges) / edges.size
                edge_features.append(edge_density)
            else:
                edge_features.append(0.0)
        
        # Estadísticas de gradientes
        grad_y, grad_x = np.gradient(img)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        edge_features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.max(gradient_magnitude)
        ])
        
        return np.array(edge_features)
    
    def extract_advanced_statistics(self, img):
        """Estadísticas avanzadas de la imagen - CORREGIDO"""
        # Estadísticas globales - USANDO scipy.stats en lugar de numpy
        global_stats = [
            np.mean(img),
            np.std(img),
            np.median(img),
            skew(img.flatten()),        # Desde scipy.stats
            kurtosis(img.flatten())     # Desde scipy.stats
        ]
        
        # Estadísticas por quadrantes
        h, w = img.shape
        quadrants = [
            img[:h//2, :w//2],
            img[:h//2, w//2:],
            img[h//2:, :w//2],
            img[h//2:, w//2:]
        ]
        
        quadrant_stats = []
        for quadrant in quadrants:
            quadrant_stats.extend([np.mean(quadrant), np.std(quadrant)])
        
        return np.array(global_stats + quadrant_stats)
    
    def extract_shape_features(self, img):
        """Características de forma adicionales"""
        shape_features = []
        
        # Relación de aspecto de regiones con alta intensidad
        threshold = np.percentile(img, 70)
        bright_regions = img > threshold
        
        if np.any(bright_regions):
            labeled_regions = label(bright_regions)
            regions = regionprops(labeled_regions)
            
            if regions:
                largest_region = max(regions, key=lambda x: x.area)
                shape_features.extend([
                    largest_region.eccentricity,
                    largest_region.solidity,
                    largest_region.extent
                ])
            else:
                shape_features.extend([0, 0, 0])
        else:
            shape_features.extend([0, 0, 0])
        
        return np.array(shape_features)

class BalancedHighResolutionClassifier:
    """Clasificador optimizado para balancear precision y recall"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_selector = None
        self.categories = []
        self.class_weights = None
    
    def train_balanced_ensemble(self, X_train, y_train, categories):
        """
        Entrenamiento con balanceo de clases y optimización
        """
        self.categories = categories
        
        print("🔧 Balanceando clases y optimizando hiperparámetros...")
        
        # Calcular pesos de clases para balancear recall
        self.class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(self.class_weights)}
        
        print(f"📊 Pesos de clases calculados: {class_weight_dict}")
        
        # Preprocesamiento de características
        X_train_scaled = self.scaler.fit_transform(X_train)
        print(f"📈 Dimensionalidad original: {X_train_scaled.shape[1]}")
        
        # Reducción de dimensionalidad para alta resolución
        if X_train_scaled.shape[1] > 1000:
            self.pca = PCA(n_components=0.95)
            X_train_processed = self.pca.fit_transform(X_train_scaled)
            print(f"📉 Reducido a {X_train_processed.shape[1]} componentes (95% varianza)")
        else:
            X_train_processed = X_train_scaled
        
        # SVM optimizado para balance precision/recall
        print("\n🎯 Optimizando SVM...")
        svm_param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'class_weight': [class_weight_dict, 'balanced']
        }
        
        svm_model = GridSearchCV(
            SVC(kernel='rbf', probability=True, random_state=42),
            svm_param_grid, cv=3, n_jobs=-1, scoring='f1_weighted', verbose=0
        )
        
        # Random Forest optimizado
        print("🌲 Optimizando Random Forest...")
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [20, 30, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf_model = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            rf_param_grid, cv=3, n_jobs=-1, scoring='f1_weighted', verbose=0
        )
        
        # KNN optimizado
        print("📐 Optimizando KNN...")
        knn_param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        knn_model = GridSearchCV(
            KNeighborsClassifier(n_jobs=-1),
            knn_param_grid, cv=3, n_jobs=-1, scoring='f1_weighted', verbose=0
        )
        
        # Gradient Boosting
        print("🚀 Configurando Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Entrenar todos los modelos
        models_to_train = [
            ('svm', svm_model),
            ('rf', rf_model), 
            ('knn', knn_model),
            ('gb', gb_model)
        ]
        
        best_models = {}
        for name, model in models_to_train:
            print(f"🚀 Entrenando {name.upper()}...")
            start_time = time()
            model.fit(X_train_processed, y_train)
            training_time = time() - start_time
            
            best_models[name] = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
            self.models[name] = model
            
            best_score = model.best_score_ if hasattr(model, 'best_score_') else model.score(X_train_processed, y_train)
            print(f"   ✅ Mejor score: {best_score:.3f} ({training_time:.1f}s)")
            
            if hasattr(model, 'best_params_'):
                print(f"   ⚙️  Mejores parámetros: {model.best_params_}")
        
        # Ensemble final con los mejores modelos
        print("\n🤝 Creando ensemble final...")
        ensemble = VotingClassifier(
            estimators=[(name, best_models[name]) for name in best_models.keys()],
            voting='soft',
            weights=[2, 2, 1, 1]  # Dar más peso a SVM y RF
        )
        
        ensemble.fit(X_train_processed, y_train)
        self.models['ensemble'] = ensemble
        
        # Evaluación rápida del ensemble
        ensemble_score = ensemble.score(X_train_processed, y_train)
        print(f"📊 Score del ensemble en entrenamiento: {ensemble_score:.3f}")
        
        return class_weight_dict
    
    def predict(self, X, algorithm='ensemble'):
        """Predicción optimizada"""
        if algorithm not in self.models:
            raise ValueError(f"Modelo {algorithm} no entrenado.")
        
        # Preprocesamiento consistente
        X_scaled = self.scaler.transform(X)
        
        if self.pca:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled
        
        # Predicción
        model = self.models[algorithm]
        
        # Para GridSearchCV, usar el mejor estimador
        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
        
        predictions = model.predict(X_processed)
        
        try:
            probabilities = model.predict_proba(X_processed)
        except:
            probabilities = np.ones((len(predictions), len(self.categories))) / len(self.categories)
        
        return predictions, probabilities
    
    def evaluate_detailed(self, X_test, y_test, algorithm='ensemble'):
        """Evaluación detallada con análisis por clase"""
        print(f"\n🧪 EVALUACIÓN DETALLADA - {algorithm.upper()}")
        print("="*50)
        
        predictions, probabilities = self.predict(X_test, algorithm)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f'🎯 Precisión General: {accuracy * 100:.2f}%')
        
        # Reporte completo
        print("\n📊 Reporte de Clasificación:")
        print(classification_report(y_test, predictions, target_names=self.categories))
        
        # Métricas por clase
        precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions)
        print("\n🔍 Métricas por Clase:")
        for i, category in enumerate(self.categories):
            print(f"  {category}:")
            print(f"    - Precision: {precision[i]:.3f}")
            print(f"    - Recall:    {recall[i]:.3f}")
            print(f"    - F1-score:  {f1[i]:.3f}")
            print(f"    - Support:   {support[i]}")
        
        # Matriz de confusión
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.categories, yticklabels=self.categories)
        plt.title(f'Matriz de Confusión - {algorithm.upper()}\n(Precisión: {accuracy*100:.1f}%)')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.show()
        
        return accuracy

def load_highres_dataset(dataset_paths, samples_per_category=400):
    """
    Carga dataset optimizado para alta resolución
    """
    preprocessor = HighResolutionPreprocessor(target_size=(400, 400))
    
    data = []
    labels = []
    categories = list(dataset_paths.keys())
    
    print("📥 CARGANDO DATASET DE ALTA RESOLUCIÓN 400x400")
    print("="*50)
    
    for category_idx, (category_name, dataset_path) in enumerate(dataset_paths.items()):
        if not os.path.exists(dataset_path):
            print(f"❌ Directorio no encontrado: {dataset_path}")
            continue
            
        image_files = [f for f in os.listdir(dataset_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"⚠️ No hay imágenes en: {dataset_path}")
            continue
        
        # Limitar número de imágenes
        if len(image_files) > samples_per_category:
            image_files = image_files[:samples_per_category]
        
        print(f"\n📁 Procesando {category_name}: {len(image_files)} imágenes")
        
        successful = 0
        for i, file in enumerate(image_files):
            if i % 50 == 0 and i > 0:
                print(f"  🔄 Procesadas {i}/{len(image_files)}...")
                
            img_path = os.path.join(dataset_path, file)
            try:
                features, _ = preprocessor.preprocess_image(img_path, apply_augmentation=False)
                data.append(features)
                labels.append(category_idx)
                successful += 1
            except Exception as e:
                continue
        
        print(f"  ✅ Exitosa: {successful}/{len(image_files)}")
    
    if not data:
        raise ValueError("❌ No se pudieron cargar imágenes")
    
    data_array = np.array(data)
    labels_array = np.array(labels)
    
    print(f"\n📊 DATASET FINAL:")
    print(f"  - Total imágenes: {len(data_array)}")
    print(f"  - Características por imagen: {data_array.shape[1]}")
    
    # Distribución
    unique, counts = np.unique(labels_array, return_counts=True)
    for category_idx, count in zip(unique, counts):
        print(f"  - {categories[category_idx]}: {count} imágenes")
    
    return data_array, labels_array, categories

def load_trained_model(model_path='image_classifier_model.pkl'):
    """Carga un modelo entrenado desde un archivo"""
    print(f"🔍 Intentando cargar modelo desde: {model_path}")
    
    try: 
        # Verificar si el archivo existe
        if not os.path.exists(model_path):
            print(f"❌ No se encontró el modelo en: {model_path}")
            print("📋 Archivos .pkl disponibles:")
            for file in os.listdir('.'):
                if file.endswith('.pkl'):
                    print(f"   - {file}")
            return None
        
        print(f"✅ Archivo encontrado. Tamaño: {os.path.getsize(model_path)} bytes")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"🎉 Modelo cargado exitosamente!")
        print(f"📊 Categorías: {model_data['categories']}")
        print(f"🎯 Precisión del modelo: {model_data['accuracy'] * 100:.2f}%")
        print(f"🤖 Mejor algoritmo: {model_data['best_algorithm']}")
        print(f"📏 Resolución: {model_data['resolution']}")
        
        return model_data
        
    except FileNotFoundError:
        print(f"❌ No se encontró el modelo en: {model_path}")
        return None
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return None

def predict_single_image(model_data, image_path, algorithm='ensemble'):
    """Predice una sola imagen usando el modelo cargado"""
    try:
        classifier = model_data['classifier']
        categories = model_data['categories']
        
        # Preprocesar la imagen
        preprocessor = HighResolutionPreprocessor(target_size=(400, 400))
        features, processed_img = preprocessor.preprocess_image(image_path)
        
        # Hacer predicción
        features_array = features.reshape(1, -1)
        prediction, probabilities = classifier.predict(features_array, algorithm)
        
        # Mostrar resultados
        predicted_class = categories[prediction[0]]
        confidence = probabilities[0][prediction[0]]
        
        print(f"\n🎯 RESULTADO DE LA PREDICCIÓN:")
        print(f"📁 Imagen: {os.path.basename(image_path)}")
        print(f"🏷️  Clase predicha: {predicted_class}")
        print(f"📊 Confianza: {confidence * 100:.2f}%")
        
        # Mostrar todas las probabilidades
        print(f"\n📈 PROBABILIDADES POR CLASE:")
        for i, category in enumerate(categories):
            print(f"  {category}: {probabilities[0][i] * 100:.2f}%")
        
        # Mostrar la imagen procesada
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        original_img = imread(image_path)
        if len(original_img.shape) == 3:
            original_img = rgb2gray(original_img)
        plt.imshow(original_img, cmap='gray')
        plt.title(f'Imagen Original\n{original_img.shape}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(processed_img, cmap='gray')
        plt.title(f'Imagen Procesada\n{processed_img.shape}')
        plt.axis('off')
        
        plt.suptitle(f'Predicción: {predicted_class} ({confidence*100:.1f}%)', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"❌ Error en la predicción: {e}")
        return None, 0

def predict_multiple_images(model_data, image_folder, algorithm='ensemble'):
    """Predice múltiples imágenes desde una carpeta"""
    try:
        # Buscar imágenes en la carpeta
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"❌ No se encontraron imágenes en: {image_folder}")
            return
        
        print(f"\n📁 Prediciendo {len(image_files)} imágenes de: {image_folder}")
        print("="*50)
        
        results = []
        classifier = model_data['classifier']
        categories = model_data['categories']
        preprocessor = HighResolutionPreprocessor(target_size=(400, 400))
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_folder, image_file)
            
            try:
                # Preprocesar y predecir
                features, _ = preprocessor.preprocess_image(image_path)
                features_array = features.reshape(1, -1)
                prediction, probabilities = classifier.predict(features_array, algorithm)
                
                predicted_class = categories[prediction[0]]
                confidence = probabilities[0][prediction[0]]
                
                results.append({
                    'image': image_file,
                    'prediction': predicted_class,
                    'confidence': confidence
                })
                
                print(f"{i+1:2d}. {image_file:<30} -> {predicted_class:<15} ({confidence*100:5.1f}%)")
                
            except Exception as e:
                print(f"{i+1:2d}. {image_file:<30} -> ERROR: {str(e)}")
                results.append({
                    'image': image_file,
                    'prediction': 'ERROR',
                    'confidence': 0
                })
        
        # Resumen
        print(f"\n📊 RESUMEN DE PREDICCIONES:")
        for category in categories:
            count = sum(1 for r in results if r['prediction'] == category)
            print(f"  {category}: {count} imágenes")
        
        return results
        
    except Exception as e:
        print(f"❌ Error procesando múltiples imágenes: {e}")
        return []
    
def verify_model_files():
    """Verifica los archivos del modelo en el directorio"""
    print(f"📁 Directorio actual: {os.getcwd()}")
    print("\n🔍 Buscando archivos del modelo...")
    
    # Buscar todos los archivos .pkl
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    
    if not pkl_files:
        print("❌ No se encontraron archivos .pkl en el directorio")
        print("💡 Ejecuta la Opción 1 para entrenar un modelo")
        return
    
    print(f"✅ Se encontraron {len(pkl_files)} archivo(s) .pkl:")
    
    for pkl_file in pkl_files:
        file_size = os.path.getsize(pkl_file)
        print(f"\n📄 {pkl_file}:")
        print(f"   Tamaño: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
        
        # Intentar cargar el archivo
        try:
            with open(pkl_file, 'rb') as f:
                model_data = pickle.load(f)
            
            print("   ✅ ARCHIVO CARGABLE")
            
            if isinstance(model_data, dict):
                print(f"   📊 Contenido:")
                for key, value in model_data.items():
                    if key == 'categories':
                        print(f"     - {key}: {value}")
                    elif key == 'accuracy':
                        print(f"     - {key}: {value * 100:.2f}%")
                    elif key in ['classifier', 'feature_dim', 'training_samples']:
                        print(f"     - {key}: {value}")
                    else:
                        print(f"     - {key}: {type(value)}")
            else:
                print(f"   ℹ️  Tipo de datos: {type(model_data)}")
                
        except Exception as e:
            print(f"   ❌ ERROR al cargar: {e}")

def improved_highres_training():
    """Entrenamiento mejorado con todas las optimizaciones"""
    
    dataset_paths = {
        'animal': 'datasets_animals',
        'human': 'datasets_humans', 
    }
    
    print("🎯 ENTRENAMIENTO MEJORADO - OBJETIVO 80%+ PRECISIÓN")
    print("="*60)
    
    # Verificar datasets
    available_datasets = {}
    for category_name, dataset_path in dataset_paths.items():
        if os.path.exists(dataset_path):
            image_files = [f for f in os.listdir(dataset_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            available_datasets[category_name] = dataset_path
            print(f"✅ {category_name}: {len(image_files)} imágenes")
        else:
            print(f"❌ No encontrado: {category_name}")
    
    if len(available_datasets) < 2:
        print("❌ Se necesitan al menos 2 datasets")
        return None, 0
    
    # Configurar tamaño de muestra
    try:
        user_input = input(f"\n📝 Imágenes por categoría (ENTER para 400): ").strip()
        samples = int(user_input) if user_input else 400
    except:
        samples = 400
    
    # Cargar dataset
    print(f"\n📥 Cargando {samples} imágenes por categoría en 400x400...")
    try:
        data, labels, categories = load_highres_dataset(available_datasets, samples_per_category=samples)
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return None, 0
    
    # División de datos
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"\n📊 DIVISIÓN DE DATOS:")
    print(f"  - Entrenamiento: {len(x_train)} imágenes")
    print(f"  - Prueba: {len(x_test)} imágenes")
    
    # Entrenar clasificador balanceado
    classifier = BalancedHighResolutionClassifier()
    class_weights = classifier.train_balanced_ensemble(x_train, y_train, categories)
    
    # Evaluación completa
    print("\n" + "="*60)
    print("📈 EVALUACIÓN FINAL COMPLETA")
    print("="*60)
    
    best_accuracy = 0
    best_algorithm = ''
    
    for algorithm in ['ensemble', 'svm', 'rf', 'knn', 'gb']:
        if algorithm in classifier.models:
            accuracy = classifier.evaluate_detailed(x_test, y_test, algorithm)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_algorithm = algorithm
    
    # Resultado final
    print(f"\n🏆 MEJOR MODELO: {best_algorithm.upper()}")
    print(f"🎯 PRECISIÓN FINAL: {best_accuracy * 100:.2f}%")
    
    # Guardar modelo - NOMBRE CORREGIDO
    model_path = 'image_classifier_model.pkl'
    try:
        with open(model_path, 'wb') as f:
            pickle.dump({
                'classifier': classifier,
                'categories': categories,
                'accuracy': best_accuracy,
                'best_algorithm': best_algorithm,
                'resolution': '400x400',
                'feature_dim': data.shape[1],
                'training_samples': len(x_train),
                'class_weights': class_weights
            }, f)
        
        print(f"💾 Modelo guardado exitosamente en: {model_path}")
        print(f"📏 Tamaño del archivo: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error al guardar el modelo: {e}")
        return classifier, best_accuracy
    
    return classifier, best_accuracy

def main():
    """Función principal"""
    print("🔥 CLASIFICADOR OPTIMIZADO DE ALTA RESOLUCIÓN")
    print("="*55)
    
    while True:
        print("\n🎯 OPCIONES PRINCIPALES:")
        print("1. 🧠 Entrenar modelo optimizado")
        print("2. 🔍 Cargar modelo y predecir una imagen")
        print("3. 📁 Cargar modelo y predecir múltiples imágenes")
        print("4. 🔧 Verificar archivos del modelo")
        print("5. ❌ Salir")
        
        option = input("\nSelecciona una opción (1-5): ").strip()
        
        if option == '1':
            print("🚀 Iniciando entrenamiento del modelo...")
            classifier, accuracy = improved_highres_training()
            
            if classifier is not None:
                print(f"🎊 ¡Entrenamiento completado! Precisión: {accuracy * 100:.2f}%")
            else:
                print("❌ El entrenamiento falló")
            
        elif option == '2':
            print("\n" + "="*50)
            print("🔍 CARGANDO MODELO PARA PREDICCIÓN INDIVIDUAL")
            print("="*50)
            
            model_data = load_trained_model('image_classifier_model.pkl')
            
            if model_data is None:
                print("❌ No se pudo cargar el modelo. Razones posibles:")
                print("   - El modelo no existe (ejecuta Opción 1 primero)")
                print("   - El archivo está corrupto")
                print("   - Problema de permisos")
                continue
                
            print("\n📸 LISTO PARA PREDECIR IMAGEN")
            image_path = input("📁 Ruta de la imagen a predecir: ").strip()
            
            if os.path.exists(image_path):
                predict_single_image(model_data, image_path)
            else:
                print(f"❌ La imagen no existe: {image_path}")
                    
        elif option == '3':
            print("\n" + "="*50)
            print("📁 CARGANDO MODELO PARA PREDICCIÓN MÚLTIPLE")
            print("="*50)
            
            model_data = load_trained_model('image_classifier_model.pkl')
            
            if model_data is None:
                print("❌ No se pudo cargar el modelo.")
                continue
                
            print("\n📸 LISTO PARA PREDECIR MÚLTIPLES IMÁGENES")
            folder_path = input("📁 Ruta de la carpeta con imágenes: ").strip()
            
            if os.path.exists(folder_path):
                predict_multiple_images(model_data, folder_path)
            else:
                print(f"❌ La carpeta no existe: {folder_path}")
                    
        elif option == '4':
            print("\n" + "="*50)
            print("🔍 VERIFICACIÓN DE ARCHIVOS DEL MODELO")
            print("="*50)
            verify_model_files()
            
        elif option == '5':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida")

if __name__ == "__main__":
    main()