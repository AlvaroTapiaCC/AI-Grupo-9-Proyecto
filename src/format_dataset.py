import os
import cv2
import numpy as np
import random
from utils import load_image, load_labels, yolo_to_bbox

random.seed(42)
np.random.seed(42)

def build_dataset(img_dir, lbl_dir, patch_size=32, max_images=50):
    X = []
    Y = []

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Directorio de imágenes no encontrado: {img_dir}")
    if not os.path.exists(lbl_dir):
        raise FileNotFoundError(f"Directorio de etiquetas no encontrado: {lbl_dir}")

    image_files = os.listdir(img_dir)[:max_images]
    processed = 0
    skipped = 0

    for img_name in image_files:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt"))

        if not os.path.exists(lbl_path):
            skipped += 1
            continue

        img = load_image(img_path)
        if img is None:
            skipped += 1
            continue
        
        processed += 1

        h, w, _ = img.shape
        boxes = load_labels(lbl_path)

        for box in boxes:
            x1, y1, x2, y2 = yolo_to_bbox(box, w, h)
            
            # Validar límites
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            patch = img[y1:y2, x1:x2]

            if patch.size == 0:
                continue

            patch = cv2.resize(patch, (patch_size, patch_size))
            X.append(patch.flatten())
            Y.append(1)

        for _ in range(len(boxes) * 2):
            x = random.randint(0, max(0, w - patch_size))
            y = random.randint(0, max(0, h - patch_size))

            patch = img[y:y+patch_size, x:x+patch_size]

            if patch.size == 0:
                continue

            patch = cv2.resize(patch, (patch_size, patch_size))
            X.append(patch.flatten())
            Y.append(0)

    X = np.array(X) / 255.0
    Y = np.array(Y)

    print(f"[Dataset] Procesadas {processed} imágenes, {skipped} saltadas")
    print(f"[Dataset] Total muestras: {len(X)} (positivas: {np.sum(Y)}, negativas: {len(Y)-np.sum(Y)})")

    return X, Y