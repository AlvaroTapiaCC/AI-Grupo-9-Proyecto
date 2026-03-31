import cv2
import numpy as np
import matplotlib.pyplot as plt

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
                boxes.append((x, y, x+patch_size, y+patch_size))

    return img, boxes


# -------------------------------
# 5. Dibujar resultados
# -------------------------------
def draw_boxes(img, boxes):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# -------------------------------
# 6. MAIN
# -------------------------------
def main():
    IMG_DIR = "../SKU110K/images/train"
    LBL_DIR = "../SKU110K/labels/train"

    print("Construyendo dataset...")
    X_train, X_test, Y_train, Y_test = prepare_data(IMG_DIR, LBL_DIR)

    print("Entrenando modelo...")
    model = train_model(X_train, Y_train)

    print("Evaluando modelo...")
    evaluate_model(model, X_test, Y_test)

    # -------- DETECCIÓN --------
    test_img = "../SKU110K/images/test/test_0.jpg"

    print("\nDetectando productos en imagen...")
    img, boxes = detect_products(model, test_img)

    print("Boxes detectadas:", len(boxes))
    draw_boxes(img, boxes)


if __name__ == "__main__":
    main()