🖼️ Clasificador de Imágenes Inteligente
Un sistema de clasificación de imágenes que utiliza técnicas avanzadas de machine learning para distinguir entre animales, humanos y objetos.

📁 Estructura del Proyecto
📊 Datasets
datasets_animals/ - Imágenes de animales para entrenamiento

datasets_humans/ - Imágenes de humanos para entrenamiento

🔧 Scripts de Extracción
extract_animals.py - Procesa y extrae características de imágenes de animales

extract_humans.py - Procesa y extrae características de imágenes de humanos

extract_objects.py - Procesa y extrae características de imágenes de objetos

🧠 Entrenamiento y Modelo
training_model.py - ARCHIVO PRINCIPAL - Entrena el clasificador

image_classifier_model.pkl - Modelo entrenado guardado

🧪 Testing
test.py - Script de pruebas general

individual_test/ - Pruebas con imágenes individuales

multiple_test/ - Pruebas con múltiples imágenes

🚀 Características Principales
Preprocesamiento Avanzado
Alta resolución (400x400 píxeles)

Mejora de contraste (CLAHE)

Reducción de ruido (filtro gaussiano)

Extracción de características multi-modal

Modelo Ensemble
Combina 4 algoritmos diferentes:

SVM - Para límites de decisión complejos

Random Forest - Robustez mediante promediado

KNN - Aprendizaje basado en similitudes

Gradient Boosting - Mejora secuencial de errores

Métricas de Rendimiento
Precisión >80%

Balance automático de clases

Validación rigurosa con cross-validation

🛠️ Instalación y Uso
1. Preparar el entorno
bash
# Instalar dependencias (ejemplo)
pip install numpy pandas scikit-learn opencv-python matplotlib
2. Organizar los datos
Coloca tus imágenes en las carpetas correspondientes:

datasets_animals/

datasets_humans/

datasets_objects/

3. Entrenar el modelo
bash
python training_model.py
4. Probar el modelo
bash
python test.py
📈 Resultados Esperados
El modelo está optimizado para:

Alta precisión en clasificación triple

Balance entre recall y precisión

Robustez ante variaciones en las imágenes

Eficiencia en tiempo de predicción

🔍 Métricas de Evaluación
Accuracy: Porcentaje total de aciertos

Precision: Calidad de las predicciones positivas

Recall: Capacidad de detectar casos reales

F1-Score: Balance entre precision y recall

💾 Archivos de Salida
image_classifier_model.pkl - Modelo entrenado serializado

Logs de entrenamiento - Métricas y progreso

Matrices de confusión - Análisis de errores

🎯 Aplicaciones
Clasificación automática de imágenes

Organización de archivos multimedia

Filtrado de contenido por categorías

Análisis de datasets visuales
