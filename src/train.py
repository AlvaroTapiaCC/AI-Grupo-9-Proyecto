import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from format_dataset import build_dataset


# -------------------------------
# 1. Preparar datos
# -------------------------------
def prepare_data(img_dir, lbl_dir):
    X, Y = build_dataset(img_dir, lbl_dir)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    return X_train, X_test, Y_train, Y_test


# -------------------------------
# 2. Entrenar modelo
# -------------------------------
def train_model(X_train, Y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    return model


# -------------------------------
# 3. Evaluar modelo
# -------------------------------
def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred)
    rec = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)

    print("\n--- Resultados ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nMatriz de confusión:")
    print(cm)


# -------------------------------
# 4. Sliding Window Detection
# -------------------------------
def detect_products(model, img_path, patch_size=32, step=16, threshold=0.7):
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    boxes = []

    for y in range(0, h - patch_size, step):
        for x in range(0, w - patch_size, step):

            patch = img[y:y+patch_size, x:x+patch_size]
            patch = cv2.resize(patch, (patch_size, patch_size))

            X = patch.flatten().reshape(1, -1) / 255.0

            prob = model.predict_proba(X)[0][1]

            if prob > threshold:
                boxes.append((x, y, x+patch_size, y+patch_size, prob))

    # Aplicar NMS para eliminar duplicados
    boxes = non_maximum_suppression(boxes, iou_threshold=0.3)
    
    return img, boxes


# -------------------------------
# 4b. Non-Maximum Suppression
# -------------------------------
def non_maximum_suppression(boxes, iou_threshold=0.3):
    """Elimina detecciones superpuestas manteniendo las de mayor confianza."""
    if len(boxes) == 0:
        return []
    
    # Ordenar por probabilidad descendente
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    selected = []
    
    while len(boxes) > 0:
        current = boxes.pop(0)
        selected.append(current)
        
        # Eliminar boxes con IOU > threshold
        boxes = [b for b in boxes if iou(current, b) < iou_threshold]
    
    return selected


def iou(box1, box2):
    """Calcula Intersection over Union entre dos boxes."""
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]
    
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


# -------------------------------
# 5. Dibujar resultados
# -------------------------------
def draw_boxes(img, boxes, output_path=None):
    img_copy = img.copy()
    for box in boxes:
        if len(box) == 5:
            x1, y1, x2, y2, prob = box
            label = f"{prob:.2f}"
        else:
            x1, y1, x2, y2 = box
            label = ""
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if label:
            cv2.putText(img_copy, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, img_copy)
        print(f"Imagen guardada en: {output_path}")
    
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    #plt.show()


# -------------------------------
# 6. MAIN
# -------------------------------
def main():
    IMG_DIR = "../SKU110K/images/train"
    LBL_DIR = "../SKU110K/labels/train"
    MODEL_PATH = "../models/model_logistic_regression.pkl"

    print("Construyendo dataset...")
    X_train, X_test, Y_train, Y_test = prepare_data(IMG_DIR, LBL_DIR)

    print("Entrenando modelo...")
    model = train_model(X_train, Y_train)
    
    print(f"Guardando modelo en {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print("Evaluando modelo...")
    evaluate_model(model, X_test, Y_test)

    # -------- DETECCIÓN --------
    test_img = "../SKU110K/images/test/test_0.jpg"

    print("\nDetectando productos en imagen...")
    img, boxes = detect_products(model, test_img)

    print(f"Boxes detectadas (después de NMS): {len(boxes)}")
    draw_boxes(img, boxes)


if __name__ == "__main__":
    main()