"""
Script para hacer predicciones usando el modelo ya entrenado.
Carga el modelo guardado sin reentrenar.
"""

import pickle
import os
from train import detect_products, draw_boxes

def predict(image_path, model_path="../models/model_logistic_regression.pkl", output_dir="../results"):
    """
    Detecta productos en una imagen usando el modelo entrenado.
    
    Args:
        image_path: Ruta a la imagen a procesar
        model_path: Ruta al archivo del modelo guardado
        output_dir: Directorio donde guardar la imagen con detecciones
    """
    
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        print(f"✗ Error: Modelo no encontrado en {model_path}")
        print("  Ejecuta main.py primero para entrenar el modelo")
        return
    
    # Verificar que la imagen existe
    if not os.path.exists(image_path):
        print(f"✗ Error: Imagen no encontrada en {image_path}")
        return
    
    # Crear directorio de resultados si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar el modelo
    print(f"Cargando modelo desde {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✓ Modelo cargado correctamente")
    
    # Hacer predicción
    print(f"Procesando imagen: {image_path}")
    img, boxes = detect_products(model, image_path)
    print(f"✓ Detección completada: {len(boxes)} productos encontrados")
    
    # Guardar resultado
    filename = os.path.basename(image_path).replace('.jpg', '_detections.jpg')
    output_path = os.path.join(output_dir, filename)
    draw_boxes(img, boxes, output_path=output_path)


if __name__ == "__main__":
    # Ejemplo: procesar imagen de prueba
    test_img = "../SKU110K/images/test/test_0.jpg"
    predict(test_img)
