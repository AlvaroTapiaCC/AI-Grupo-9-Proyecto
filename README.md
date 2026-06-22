# Retail Product Checkout — AI Pipeline

Pipeline de detección y clasificación de productos de supermercado usando DINOv2 y CLIP.

**Integrantes:** Nicolas Leighton · Alvaro Tapia

**Dataset:** [Retail Product Checkout Dataset (Kaggle)](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)

---

## Estructura del repositorio

```
AI-Grupo-9-Proyecto/
│
├── main.py                  # Punto de entrada: entrenamiento, evaluación y pipeline
├── src/
│   ├── config.py            # Parámetros globales (modo, device, epochs, lr, etc.)
│   ├── paths.py             # Todas las rutas del proyecto centralizadas
│   │
│   ├── data/                # Carga y preprocesamiento de datos
│   │   ├── datasets/        # Datasets de PyTorch (embeddings, imágenes, features)
│   │   ├── label_encoder.py # Mapeo supercat_id ↔ índice de clase
│   │   └── data_utils.py    # Utilidades para leer anotaciones COCO
│   │
│   ├── encoders/
│   │   ├── dinov2.py        # DINOv2Encoder + precomputo de embeddings y features
│   │   └── clip.py          # CLIPEncoder + precomputo de embeddings para el clasificador
│   │
│   ├── models/
│   │   ├── mlp.py           # MLPClassifier (clasificador sobre embeddings CLIP)
│   │   └── detector.py      # RetailDetector (cabeza de conteo + cabeza de bboxes)
│   │
│   ├── training/
│   │   ├── loop.py          # Loops de entrenamiento/evaluación por epoch
│   │   ├── losses.py        # Hungarian matching + pérdida GIoU + SmoothL1
│   │   └── metrics.py       # Métricas de clasificador, detector y pipeline
│   │
│   ├── stages/              # Lógica de alto nivel de cada etapa
│   │   ├── classifier/      # train.py + evaluate.py del clasificador
│   │   ├── detector/        # train.py + evaluate.py del detector
│   │   └── pipeline/        # evaluate.py del pipeline completo
│   │
│   ├── inference/
│   │   └── pipeline.py      # Inferencia end-to-end sobre una imagen
│   │
│   ├── visualization/
│   │   ├── plots.py         # Curvas de entrenamiento, scatter latente, métricas
│   │   └── predictions.py   # Visualización de predicciones ejemplo
│   │
│   └── utils/
│       ├── box_ops.py       # Conversiones cxcywh ↔ xyxy
│       ├── io.py            # load_json / save_json
│       └── model_io.py      # save_model / load_model
│
├── dataset/                 # Dataset COCO-style (no incluido en el repo)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── precomputed/             # Embeddings precomputados del clasificador (generados con encode=True)
│   ├── dino_embeddings/     # Embeddings CLS de DINOv2 por crop de bbox (clasificador alternativo)
│   └── clip_embeddings/     # Embeddings de CLIP por crop de bbox (clasificador principal)
│
├── checkpoints/             # Pesos guardados de los modelos entrenados
│   ├── classifier/
│   └── detector/
│
└── results/                 # Métricas, gráficos e imágenes de ejemplo por corrida
    ├── classifier/
    │   ├── last/            # Resultados de la última corrida
    │   └── best/            # Copia de last/ si mejoró la métrica principal
    ├── detector/
    │   ├── last/
    │   └── best/
    └── pipeline/
```

> **dataset/**, **precomputed/**, **checkpoints/** y **results/** no se incluyen en el repositorio (ver `.gitignore`).

---

## Desarrollo del proyecto

### Etapa 1 — Detección binaria por patches (dataset original)

El punto de partida fue un dataset de imágenes de productos sin etiquetas de categoría. Se implementó un clasificador lineal que dividía cada imagen en patches y predecía si cada patch contenía un producto o no (detección binaria). Los resultados fueron malos: el modelo no lograba distinguir producto de fondo de forma consistente, y la ausencia de labels de categoría impedía ir más allá de esa detección. Se decidió cambiar de dataset.

---

### Etapa 2 — Clasificador de productos (dataset actual)

Se adoptó el [Retail Product Checkout Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset), que incluye anotaciones COCO con bounding boxes y categorías para cada producto. Esto permitió construir un clasificador real usando los crops de cada bbox como entrada.

Se compararon dos enfoques:

- **CNN propia** — red convolucional entrenada desde cero sobre los crops. Para acelerar el entrenamiento se precomputaban las activaciones de las capas convolucionales, de modo que en cada epoch solo se entrenaban las capas finales. Aun así, el rendimiento fue limitado por el tamaño del dataset.

- **MLP + CLIP** — se usó CLIP (`ViT-B/32`) como encoder fijo para obtener embeddings de 512 dimensiones por crop, y un MLP pequeño como cabeza clasificadora. Al aprovechar un modelo preentrenado en cientos de millones de pares imagen-texto, el MLP superó ampliamente a la CNN con mucho menos cómputo.

Se eligió MLP+CLIP como clasificador definitivo. El MLP alcanzó ~98% de accuracy en validación.

---

### Etapa 3 — Detector de bboxes

Con el clasificador resuelto, se reestructuró el repositorio: se eliminó todo el código de la CNN y se organizó el proyecto en módulos separados por responsabilidad (`encoders/`, `models/`, `stages/`, `training/`, etc.), dejando `main.py` como único punto de entrada controlado por `config.py`.

El detector tiene como objetivo predecir cuántos productos hay en una imagen y dónde están. Se probaron varias aproximaciones en orden de complejidad:

**1. MLP sobre CLS token de DINOv2**
El token CLS de DINOv2 (representación global de 768 dims) se pasaba directamente por un MLP para predecir las bboxes. IoU promedio ~0.022. El problema es que el CLS token no contiene información espacial — sabe qué hay en la imagen pero no dónde.

**2. Cabeza DETR-style con patch tokens**
Se añadió una segunda cabeza que usa los patch tokens de DINOv2 (256 tokens × 768 dims, uno por región de 14×14 px) mediante cross-attention con queries aprendidas, una por bbox posible. Al tener información espacial explícita, el IoU subió a ~0.079.

**3. Hungarian matching**
En lugar de asignar predicciones a slots GT en orden fijo, se adoptó el algoritmo húngaro para encontrar la asignación óptima minimizando un costo combinado de L1 + GIoU. Esto evita penalizar predicciones correctas solo por estar en el slot equivocado. IoU ~0.096.

**4. Fine-tuning de DINOv2**
Con DINOv2 completamente congelado, las features no estaban optimizadas para localización. Se descongelaron las últimas capas del transformer para que aprendieran representaciones más útiles para detección. IoU ~0.099.

**5. Optimizaciones finales**
Para mejorar la estabilidad y calidad del fine-tuning se aplicaron varias mejoras simultáneas:
- Formato de bboxes cambiado de `(x1,y1,x2,y2)` a `(cx,cy,w,h)` — más adecuado para regresión
- `Adam` → `AdamW` con weight decay para regularización
- Warmup lineal + CosineAnnealingLR para estabilizar el inicio del fine-tuning
- Gradient clipping (`max_norm=1.0`) para evitar explosión de gradientes en los bloques descongelados
- 4 bloques del transformer descongelados en lugar de 2
- Label smoothing (`0.1`) en la pérdida de conteo para reducir overfitting de la cabeza de clasificación de conteo

---

### Etapa 4 — Pipeline completo y resultados

Se conectaron el detector y el clasificador en un flujo end-to-end: dado una imagen, el detector predice el número de productos y sus bboxes, luego cada crop se clasifica con CLIP+MLP. Las métricas del pipeline combinan localización y clasificación: `loc_recall`, `clf_accuracy` y `end_to_end`.

Se implementó un sistema completo de visualización y logging para el informe:
- Curvas de pérdida y accuracy del clasificador
- Curvas de pérdida y MAE del detector
- Histogramas de distribución de IoU y error de conteo
- Visualización de predicciones GT vs predichas para detector y pipeline
- Análisis del espacio latente CLIP con PCA y UMAP, y clustering K-Means con métricas ARI/NMI
- Métricas de pipeline por clase y globales

---
