import os
import cv2
import numpy as np
import random
from utils import load_image, load_labels, yolo_to_bbox


def build_dataset(img_dir, lbl_dir, patch_size=32, max_images=50):
    X = []
    Y = []

    image_files = os.listdir(img_dir)[:max_images]

    for img_name in image_files:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt"))

        if not os.path.exists(lbl_path):
            continue

        img = load_image(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        boxes = load_labels(lbl_path)

        for box in boxes:
            x1, y1, x2, y2 = yolo_to_bbox(box, w, h)

            patch = img[y1:y2, x1:x2]

            if patch.size == 0:
                continue

            patch = cv2.resize(patch, (patch_size, patch_size))
            X.append(patch.flatten())
            Y.append(1)

        for _ in range(len(boxes)):
            x = random.randint(0, w - patch_size)
            y = random.randint(0, h - patch_size)

            patch = img[y:y+patch_size, x:x+patch_size]

            if patch.size == 0:
                continue

            patch = cv2.resize(patch, (patch_size, patch_size))
            X.append(patch.flatten())
            Y.append(0)

    X = np.array(X) / 255.0
    Y = np.array(Y)

    return X, Y