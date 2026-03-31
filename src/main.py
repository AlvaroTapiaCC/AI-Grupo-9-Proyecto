"""
Main script para orquestar el pipeline completo de detección de productos en gondolas.
"""

import pickle
import os
from train import prepare_data, train_model, evaluate_model, detect_products, draw_boxes

def main():
    """Pipeline principal: preparar datos, entrenar y evaluar modelo."""
    
    IMG_DIR = "../SKU110K/images/train"
    LBL_DIR = "../SKU110K/labels/train"
    MODELS_DIR = "../models"
    RESULTS_DIR = "../results"
    MODEL_PATH = os.path.join(MODELS_DIR, "model_logistic_regression.pkl")
    
    # Crear directorios si no existen
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Preparar datos
    print("=" * 50)
    print("PASO 1: Preparando dataset...")
    print("=" * 50)
    X_train, X_test, Y_train, Y_test = prepare_data(IMG_DIR, LBL_DIR)
    print(f"✓ Dataset listo: {X_train.shape[0]} muestras de entrenamiento")
    print(f"✓ Conjunto de prueba: {X_test.shape[0]} muestras")
    
    # 2. Entrenar modelo
    print("\n" + "=" * 50)
    print("PASO 2: Entrenando modelo...")
    print("=" * 50)
    model = train_model(X_train, Y_train)
    print("✓ Modelo entrenado correctamente")
    
    # 3. Guardar modelo
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Modelo guardado en: {MODEL_PATH}")
    
    # 4. Evaluar modelo
    print("\n" + "=" * 50)
    print("PASO 3: Evaluando modelo...")
    print("=" * 50)
    evaluate_model(model, X_test, Y_test)
    
    # 5. Detección en imagen de prueba
    print("\n" + "=" * 50)
    print("PASO 4: Detección en imagen de prueba...")
    print("=" * 50)
    test_img = "../SKU110K/images/test/test_0.jpg"
    
    if os.path.exists(test_img):
        img, boxes = detect_products(model, test_img)
        print(f"✓ Detección completada: {len(boxes)} productos encontrados")
        
        output_img = os.path.join(RESULTS_DIR, "detections_test_0.jpg")
        draw_boxes(img, boxes, output_path=output_img)
    else:
        print(f"✗ Imagen de prueba no encontrada: {test_img}")
    
    print("\n" + "=" * 50)
    print("Pipeline completado exitosamente")
    print("=" * 50)


if __name__ == "__main__":
    main()
