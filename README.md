# Categorizacion de productos "checkout"

## Integrantes:

-Nicolas Leighton   
-Alvaro Tapia   


### Dataset:

https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset

---

## Documentación Técnica: Funcionamiento del Modelo de Categorizacion de Productos

*Inserte documentacion*

### Requerimientos

-Se deben instalar todas las dependencias de requirements.txt   
-Adicionalmente se debe instalar torch y torchvision correspondiente al SO y hardware de su PC   
-Si tiene GPU NVDIA instale pytorch con opcion para CUDA (version correspondiente) 

### Ejecucion

-El programa `main.py` ejecuta el entrenamiento y metricas para un modelo y "nivel" indicado   
-Antes de ejecutar ajuste las variables encontradas en `src/config.py`   
-Aqui debe indicar si el modelo es `"mlp"` o `"cnn"`, guardando la variable `model`   
-Ademas debe especificar la dificultad de las imagenes a utilizar, ya sea `"easy"`, `"medium"` o `"hard"` , guardando la variable `level`   
-Lo mas importantees que se debe indicar si se necesita precomputar tensores/embeddings, indicando `True` o `False` en la variable `encode`   
-El encoding solo se necesita realizar una vez para cada dificultad, luego se puede entrenar tomando los embeddings guardados   
-Ademas hay que indicar si se quiere utilizar un modelo preentrenado, o se quiere entrenar uno nuevo, indicando `True` o `False` en la variable `train_new`   
-La variable `compare` indica si se quieren realizar graficos comparativos entre el mejor modelo MLP y el mejor modelo CNN (`True` o `False`)   
-Por ultimo se deben indicar los parametros de entrenamiento: `epochs`, `batch_size`, `lr` (learning rate) y para CNN el `image_size`
-Los resultados y metricas seran guardados en la carpeta `results/`   

### Estructura del programa

*Inserte estructura*