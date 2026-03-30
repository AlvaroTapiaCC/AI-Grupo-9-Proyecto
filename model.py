import cv2
import os
import matplotlib.pyplot as plt

DATASET_PATH = "SKU110K/images/train"

files = os.listdir(DATASET_PATH)
print("Total images:", len(files))
print("First 5:", files[:5])

img_path = os.path.join(DATASET_PATH, files[0])
img = cv2.imread(img_path)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(files[0])
plt.axis("off")
plt.show()
