# Sistema de Entrenamiento ML para Análisis de Confiabilidad Facial
# Archivo: train_trust_model.py
# Ejecutar una sola vez para entrenar el modelo

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import face_recognition
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import requests
import zipfile
from tqdm import tqdm
import json

class TrustModelTrainer:
    def __init__(self, model_path="./trust_model.h5"):
        self.model_path = model_path
        self.dataset_path = "./training_dataset"
        self.model = None
        
        # Crear directorios
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(f"{self.dataset_path}/trustworthy", exist_ok=True)
        os.makedirs(f"{self.dataset_path}/untrustworthy", exist_ok=True)
    
    def download_sample_dataset(self):
        """Descarga un dataset de muestra para entrenamiento"""
        print("🔄 Descargando dataset de muestra...")
        
        # URLs de ejemplo (estos son datasets públicos reales)
        datasets = {
            "trustworthy": [
                "https://github.com/NVlabs/ffhq-dataset",  # Rostros de alta calidad
                "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"
            ],
            "untrustworthy": [
                "https://www.kaggle.com/datasets/dansbecker/5-celebrity-faces-dataset"
            ]
        }
        
        print("""
📋 INSTRUCCIONES PARA CONFIGURAR EL DATASET:

1. Descarga manualmente estos datasets:
   - CelebA: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
   - FFHQ: https://github.com/NVlabs/ffhq-dataset
   - UTKFace: https://susanqq.github.io/UTKFace/

2. Organiza las imágenes:
   - Personas que consideres CONFIABLES → ./training_dataset/trustworthy/
   - Personas que consideres NO CONFIABLES → ./training_dataset/untrustworthy/

3. Recomendación: Mínimo 500 imágenes por categoría

4. Una vez organizadas, ejecuta este script nuevamente.
        """)
        
        # Crear algunas imágenes sintéticas para demo
        self.create_synthetic_samples()
    
    def create_synthetic_samples(self):
        """Crea muestras sintéticas para demostración"""
        print("🎨 Creando muestras sintéticas para demostración...")
        
        # Generar rostros sintéticos básicos para prueba
        for category in ["trustworthy", "untrustworthy"]:
            for i in range(50):  # 50 muestras por categoría
                # Crear imagen sintética (en un caso real, usar dataset real)
                img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                
                # Agregar un "rostro" básico (rectángulo)
                if category == "trustworthy":
                    cv2.rectangle(img, (30, 30), (98, 98), (100, 150, 200), -1)
                else:
                    cv2.rectangle(img, (30, 30), (98, 98), (50, 50, 150), -1)
                
                # Agregar características faciales básicas
                cv2.circle(img, (50, 50), 5, (0, 0, 0), -1)  # Ojo izquierdo
                cv2.circle(img, (78, 50), 5, (0, 0, 0), -1)  # Ojo derecho
                cv2.circle(img, (64, 70), 3, (0, 0, 0), -1)  # Nariz
                cv2.ellipse(img, (64, 85), (10, 5), 0, 0, 180, (0, 0, 0), 2)  # Boca
                
                # Guardar imagen
                img_path = f"{self.dataset_path}/{category}/synthetic_{i:03d}.jpg"
                cv2.imwrite(img_path, img)
        
        print("✅ Muestras sintéticas creadas (reemplaza con dataset real para mejores resultados)")
    
    def create_model(self):
        """Crea la arquitectura del modelo de confiabilidad"""
        print("🏗️ Creando arquitectura del modelo...")
        
        model = keras.Sequential([
            # Capas de preprocesamiento
            layers.Rescaling(1./255, input_shape=(128, 128, 3)),
            
            # Capas convolucionales - Extracción de características
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Capas densas - Clasificación
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            # Capa de salida
            layers.Dense(1, activation='sigmoid')  # 0=No confiable, 1=Confiable
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        print("✅ Modelo creado exitosamente")
        
        # Mostrar resumen
        model.summary()
        return model
    
    def preprocess_image(self, img_path):
        """Preprocesa una imagen para el entrenamiento"""
        try:
            # Cargar imagen
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Convertir BGR a RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detectar rostros
            face_locations = face_recognition.face_locations(img, model="hog")
            
            if not face_locations:
                # Si no detecta rostro, usar imagen completa redimensionada
                img = cv2.resize(img, (128, 128))
                return img
            
            # Usar el primer rostro detectado
            top, right, bottom, left = face_locations[0]
            face = img[top:bottom, left:right]
            
            # Redimensionar a tamaño estándar
            face = cv2.resize(face, (128, 128))
            
            return face
            
        except Exception as e:
            print(f"❌ Error procesando {img_path}: {e}")
            return None
    
    def load_dataset(self):
        """Carga y preprocesa el dataset completo"""
        print("📂 Cargando dataset...")
        
        X, y = [], []
        
        # Contar archivos totales
        total_files = 0
        for category in ["trustworthy", "untrustworthy"]:
            category_path = f"{self.dataset_path}/{category}"
            if os.path.exists(category_path):
                total_files += len([f for f in os.listdir(category_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if total_files == 0:
            print("❌ No se encontraron imágenes en el dataset")
            return None, None, None, None
        
        print(f"📊 Procesando {total_files} imágenes...")
        
        with tqdm(total=total_files, desc="Procesando imágenes") as pbar:
            # Cargar imágenes confiables (etiqueta = 1)
            trustworthy_path = f"{self.dataset_path}/trustworthy"
            if os.path.exists(trustworthy_path):
                for img_name in os.listdir(trustworthy_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(trustworthy_path, img_name)
                        img = self.preprocess_image(img_path)
                        if img is not None:
                            X.append(img)
                            y.append(1)  # Confiable
                        pbar.update(1)
            
            # Cargar imágenes no confiables (etiqueta = 0)
            untrustworthy_path = f"{self.dataset_path}/untrustworthy"
            if os.path.exists(untrustworthy_path):
                for img_name in os.listdir(untrustworthy_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(untrustworthy_path, img_name)
                        img = self.preprocess_image(img_path)
                        if img is not None:
                            X.append(img)
                            y.append(0)  # No confiable
                        pbar.update(1)
        
        if not X:
            print("❌ No se pudieron procesar imágenes válidas")
            return None, None, None, None
        
        # Convertir a arrays numpy
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Dataset cargado:")
        print(f"   - Entrenamiento: {len(X_train)} imágenes")
        print(f"   - Prueba: {len(X_test)} imágenes")
        print(f"   - Confiables: {np.sum(y == 1)} imágenes")
        print(f"   - No confiables: {np.sum(y == 0)} imágenes")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, epochs=100, batch_size=32):
        """Entrena el modelo con el dataset"""
        print("🚀 Iniciando entrenamiento del modelo...")
        
        # Crear modelo si no existe
        if self.model is None:
            self.create_model()
        
        # Cargar datos
        X_train, X_test, y_train, y_test = self.load_dataset()
        
        if X_train is None:
            print("❌ No se pudo cargar el dataset")
            return None
        
        # Callbacks para mejorar entrenamiento
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar modelo
        print(f"🎯 Entrenando por máximo {epochs} épocas...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluación final
        print("\n📊 Evaluación final del modelo:")
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"✅ Resultados finales:")
        print(f"   - Precisión: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   - Precisión (Precision): {test_precision:.4f}")
        print(f"   - Recall: {test_recall:.4f}")
        print(f"   - F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")
        
        # Predicciones detalladas
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        
        print("\n📋 Reporte de clasificación:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['No Confiable', 'Confiable']))
        
        print("\n🔄 Matriz de confusión:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"         Predicho")
        print(f"Real     No-Conf  Confiable")
        print(f"No-Conf    {cm[0,0]:4d}     {cm[0,1]:4d}")
        print(f"Confiable  {cm[1,0]:4d}     {cm[1,1]:4d}")
        
        # Guardar métricas
        metrics = {
            "accuracy": float(test_accuracy),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1_score": float(2 * (test_precision * test_recall) / (test_precision + test_recall)),
            "confusion_matrix": cm.tolist(),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        with open("model_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n💾 Modelo guardado en: {self.model_path}")
        print(f"📊 Métricas guardadas en: model_metrics.json")
        
        return history
    
    def test_model(self, test_image_path):
        """Prueba el modelo con una imagen específica"""
        if not os.path.exists(self.model_path):
            print("❌ No se encontró modelo entrenado")
            return None
        
        # Cargar modelo
        model = keras.models.load_model(self.model_path)
        
        # Procesar imagen
        img = self.preprocess_image(test_image_path)
        if img is None:
            print("❌ No se pudo procesar la imagen de prueba")
            return None
        
        # Predecir
        img_batch = np.expand_dims(img, axis=0)
        prediction = model.predict(img_batch)[0][0]
        
        is_trustworthy = prediction > 0.5
        confidence = prediction if is_trustworthy else 1 - prediction
        
        print(f"\n🔍 Resultado de prueba:")
        print(f"   - Imagen: {test_image_path}")
        print(f"   - Confiable: {'SÍ' if is_trustworthy else 'NO'}")
        print(f"   - Confianza: {confidence:.2%}")
        
        return {
            "trustworthy": is_trustworthy,
            "confidence": float(confidence),
            "raw_prediction": float(prediction)
        }

def main():
    print("🤖 SISTEMA DE ENTRENAMIENTO - ANÁLISIS DE CONFIABILIDAD FACIAL\n")
    
    trainer = TrustModelTrainer()
    
    print("Opciones disponibles:")
    print("1. Configurar dataset (descargar/organizar)")
    print("2. Entrenar modelo")
    print("3. Probar modelo con imagen")
    print("4. Proceso completo (configurar + entrenar)")
    
    choice = input("\n👉 Selecciona una opción (1-4): ").strip()
    
    if choice == "1":
        trainer.download_sample_dataset()
        
    elif choice == "2":
        if not os.path.exists(f"{trainer.dataset_path}/trustworthy") or \
           not os.path.exists(f"{trainer.dataset_path}/untrustworthy"):
            print("❌ Dataset no configurado. Ejecuta opción 1 primero.")
            return
        
        epochs = input("Número de épocas (default: 100): ").strip()
        epochs = int(epochs) if epochs else 100
        
        trainer.train_model(epochs=epochs)
        
    elif choice == "3":
        img_path = input("Ruta de la imagen de prueba: ").strip()
        if os.path.exists(img_path):
            trainer.test_model(img_path)
        else:
            print("❌ Imagen no encontrada")
            
    elif choice == "4":
        print("🔄 Ejecutando proceso completo...")
        trainer.download_sample_dataset()
        
        input("\n⏸️ Configura tu dataset manualmente y presiona ENTER para continuar...")
        
        trainer.train_model()
        
    else:
        print("❌ Opción no válida")

if __name__ == "__main__":
    main()