# Conteo de productos en gondolas

## Integrantes:

-Nicolas Leighton   
-Alvaro Tapia   


### Dataset:

https://www.kaggle.com/datasets/thedatasith/sku110k-annotations

---

## Documentación Técnica: Funcionamiento del Modelo de Detección de Productos

### Visión General del Sistema

El sistema implementa un pipeline completo de machine learning para detectar instancias individuales de productos en imágenes de góndolas. El enfoque utiliza regresión logística como clasificador base, combinado con una estrategia de ventana deslizante (sliding window) para localizar productos a nivel de píxeles. El proceso se divide en tres fases principales: formateo del dataset, entrenamiento del modelo y detección en imágenes nuevas.

### Fase 1: Formateo del Dataset

El formateo del dataset es fundamental para convertir las anotaciones en bruto del dataset SKU110K en muestras de entrenamiento utilizables. El dataset proporciona imágenes de góndolas junto con etiquetas en formato YOLO, donde cada etiqueta contiene coordenadas normalizadas del centro del producto (xc, yc) y dimensiones normalizadas (ancho y alto). La función `build_dataset` en `format_dataset.py` es responsable de transformar esta información en vectores de características.

El proceso comienza cargando las imágenes y sus correspondientes archivos de etiquetas. Para cada imagen, el sistema convierte las coordenadas YOLO normalizadas a coordenadas de píxeles reales mediante la función `yolo_to_bbox` en `utils.py`. Esta conversión es crítica: multiplica las coordenadas normalizadas por las dimensiones reales de la imagen para obtener las esquinas del rectángulo delimitador (bounding box) que rodea cada producto. Con estos rectángulos, se extraen parches de la imagen de tamaño 32x32 píxeles, que luego se redimensionan a un tamaño estándar para mantener consistencia. Estos parches positivos representan ejemplos de productos que el modelo debe aprender a identificar.

Paralelamente, el sistema genera muestras negativas para entrenar al modelo a distinguir entre productos y fondo. Por cada producto detectado en una imagen, se generan aleatoriamente dos parches del mismo tamaño desde regiones que no contienen productos (fondo o espacios vacíos). Esta proporción 1:2 (producto a no-producto) es deliberada: crea un desbalance controlado que refleja mejor la realidad en imágenes reales, donde los píxeles sin productos son generalmente más abundantes que aquellos con productos.

Un aspecto importante del formateo es la validación de límites. Durante la extracción de parches, el sistema verifica que las coordenadas calculadas no excedan los límites de la imagen, evitando errores de indexación. Todos los parches, positivos y negativos, se normalizan dividiendo por 255 para escalar los valores de píxeles al rango [0, 1], lo que mejora la convergencia durante el entrenamiento. El resultado final es una matriz X que contiene vectores aplanados de parches (cada parche de 32x32x3 se convierte en un vector de 3072 dimensiones) y un vector Y con etiquetas binarias (1 para producto, 0 para fondo).

### Fase 2: Entrenamiento del Modelo

Con el dataset preparado, la siguiente fase es entrenar el clasificador. El sistema divide el dataset en conjuntos de entrenamiento y prueba mediante validación estratificada con una proporción 80-20, asegurando que ambos conjuntos tengan una distribución similar de clases positivas y negativas. Esto es crucial para obtener evaluaciones confiables.

El modelo utilizado es Regresión Logística, un clasificador lineal que aprende a separar productos de fondo en el espacio de características de 3072 dimensiones. Durante el entrenamiento, el modelo ajusta sus pesos internos para minimizar el error de clasificación en las muestras de entrenamiento. La regresión logística es apropiada para este problema porque es rápida de entrenar, interpretable y proporciona probabilidades de pertenencia a clase mediante su función sigmoide, lo que es valioso para establecer umbrales de confianza.

Después del entrenamiento, el modelo se evalúa en el conjunto de prueba utilizando múltiples métricas: precisión (proporción de detecciones correctas entre todas las detecciones), recall (proporción de productos detectados entre todos los productos reales), F1-score (promedio armónico de precisión y recall) y una matriz de confusión que muestra el desempeño detallado. El modelo entrenado se persiste en disco utilizando `pickle`, permitiendo su reutilización posterior sin necesidad de reentrenamiento.

### Fase 3: Detección en Imágenes Nuevas

La detección utiliza una estrategia de ventana deslizante (sliding window) para escanear toda una imagen en busca de productos. El modelo divide la imagen en parches pequeños (32x32 píxeles) con un paso de desplazamiento de 16 píxeles, lo que crea una superposición del 50% entre ventanas consecutivas. Esta superposición es intencional: aumenta la probabilidad de detectar productos completamente incluso si los límites de los parches no coinciden perfectamente con los límites reales del producto.

Para cada parche, el sistema lo normaliza y lo pasa al modelo entrenado, obteniendo una probabilidad de que contenga un producto. Si esta probabilidad supera un umbral configurable (por defecto 0.7), el parche se registra como una detección potencial. Sin embargo, debido a la superposición de ventanas, es común obtener múltiples detecciones del mismo producto en ubicaciones ligeramente diferentes.

Para resolver esto, se aplica Non-Maximum Suppression (NMS), un algoritmo de post-procesamiento que elimina detecciones redundantes. El NMS ordena todas las detecciones por confianza (probabilidad) y mantiene la detección con mayor confianza, mientras elimina otras detecciones que se superponen significativamente (determinado por un umbral de Intersection over Union). Esto produce un conjunto final de detecciones limpias y no redundantes que representa la mejor estimación de las ubicaciones de productos en la imagen.

### Consideraciones Técnicas Importantes

La normalización de píxeles es esencial: garantiza que los valores de entrada al modelo estén en un rango consistente, mejorando la estabilidad numérica y la convergencia. La reproducibilidad se asegura estableciendo semillas aleatorias para NumPy y Python, permitiendo obtener resultados consistentes en múltiples ejecuciones. La validación de límites previene comportamientos inesperados cuando se procesan imágenes con dimensiones variables o cuando se extraen parches cerca de los bordes.

El tamaño de parche de 32x32 píxeles es un equilibrio entre capturar suficiente contexto visual para distinguir productos de fondo y mantener un número manejable de parches por imagen. El paso de 16 píxeles proporciona suficiente superposición para garantizar cobertura completa sin generar un número excesivo de parches redundantes. El umbral de confianza de 0.7 puede ajustarse según se requiera: valores más altos producen detecciones más confiables pero pueden perder algunos productos (menor recall), mientras que valores más bajos detectan más productos pero pueden incluir falsos positivos (menor precisión).

### Integración del Pipeline

El sistema está organizado en módulos específicos: `format_dataset.py` maneja la preparación de datos, `train.py` contiene la lógica de entrenamiento y detección, `utils.py` proporciona funciones utilitarias de conversión de coordenadas, y `main.py` orquesta todo el flujo integrando estas componentes. Esta modularidad permite reutilizar componentes, modificar fácilmente parámetros y extender el sistema con nuevas funcionalidades como diferentes modelos o estrategias de detección.