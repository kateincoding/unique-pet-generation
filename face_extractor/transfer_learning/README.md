# Unique Pet Generation

Genera mascotas que se parecen a sus dueños humanos, extrayendo rasgos faciales y mapeándolos a características de mascota.

## Fase 1: Extracción de Rasgos Faciales

Dos enfoques complementarios extraen atributos faciales a partir de fotos de personas:

1. **Modelos de deep learning** — un baseline ResNet-18 y un modelo ViT-B/16 multi-head, ambos producen un embedding de 256 dimensiones que codifica características como color de pelo, forma de cara y estructura facial. Este embedding es el puente hacia la Fase 2 de generación de mascotas.
2. **Pipeline geométrico** — MediaPipe Face Mesh (478 landmarks) más DeepFace, que extrae rasgos interpretables (forma de cara, ojos, tono de piel, expresión, edad) y los traduce a prompt tokens de Stable Diffusion para la generación de la mascota.

### Modelos

| | ResNet-18 | ViT Multi-Head |
|---|---|---|
| Backbone | ResNet-18 (ImageNet-1k) | ViT-B/16 (ImageNet-21k) |
| Salida | 15 atributos binarios (sigmoide) | 17 categorías multi-clase (softmax) |
| Parámetros | 11.3M | 86.0M |
| Checkpoint | 136 MB | 344 MB |
| Inferencia (batch 1) | 3.5 ms | 14.5 ms |
| Embedding | 256-dim | 256-dim |
| Mejor epoch | 19 | 9 |
| Métrica de validación | mAP 0.7530 | acc media 0.883 |

Los dos modelos usan esquemas de etiquetas distintos (atributos binarios de CelebA vs categorías de `flags.txt`), así que sus métricas no son directamente comparables. El ViT es ~4x más lento pero parte de un preentrenamiento más rico y alcanza mayor accuracy media; el ResNet queda como la opción liviana cuando la velocidad importa.

## Marco Teórico y Técnicas Implementadas

El proyecto combina dos familias de técnicas: **transfer learning** sobre backbones que yo entreno (fine-tuning), y **modelos preentrenados que uso tal cual** sin entrenar (inferencia directa). A continuación el marco teórico de cada modelo.

### Modelos entrenados por transfer learning

El transfer learning reutiliza una red ya entrenada sobre un dataset enorme y la adapta a mi tarea, en lugar de entrenar desde cero. La intuición: las primeras capas aprenden features visuales genéricas (bordes, texturas, formas) que sirven para casi cualquier problema de imágenes, así que solo necesito re-entrenar las capas finales y ajustar suavemente el resto. Esto me permite entrenar con relativamente pocos datos y en poco tiempo.

#### ResNet-18 (CNN residual)

Red convolucional de 18 capas preentrenada en ImageNet-1k. Su aporte teórico son las **conexiones residuales** (skip connections): cada bloque aprende un residuo que se suma a su entrada, lo que evita el problema del gradiente que se desvanece y permite entrenar redes profundas de forma estable. Las convoluciones aplican filtros locales que detectan patrones espaciales con invariancia de traslación. La uso como extractor liviano: 11.3M de parámetros, 256-dim de embedding, salida con 15 atributos binarios y activación sigmoide (cada atributo es una decisión independiente).

#### ViT-B/16 (Vision Transformer)

Transformer de visión preentrenado en ImageNet-21k. En lugar de convoluciones, **divide la imagen en parches de 16x16**, los proyecta a vectores (tokens) y los procesa con **auto-atención** (self-attention), que pondera la relación entre todos los parches a la vez. Esto captura dependencias de largo alcance en la cara (relacionar ojos con mandíbula, por ejemplo) mejor que el campo receptivo local de una CNN. El costo es cuadrático en el número de parches y necesita más datos de preentrenamiento, por eso parte de ImageNet-21k. Aquí es **multi-task / multi-head**: un tronco común genera el embedding de 256-dim y 17 cabezas softmax independientes, una por categoría, predicen en paralelo compartiendo la representación.

### Modelos preentrenados usados sin entrenar (inferencia directa)

Estos modelos ya vienen entrenados por terceros. No los entreno: solo les paso la imagen y consumo su salida.

#### MediaPipe Face Mesh — sí, es un modelo entrenado

Es un modelo preentrenado de Google (no una librería de geometría manual). Internamente corre en dos etapas: primero un detector de rostro liviano tipo **BlazeFace** que localiza la cara, y después una **red de regresión** que predice las coordenadas de **478 landmarks** faciales (468 de la malla base + 10 de iris con refinamiento). Fue entrenado sobre miles de caras anotadas. Yo no ajusto sus pesos: tomo los landmarks que devuelve y, sobre esas coordenadas, calculo rasgos con **morfometría geométrica** (proporciones entre puntos normalizadas por el ancho de la cara, invariantes a escala y distancia a la cámara).

#### DeepFace — redes preentrenadas para emoción y edad

Librería que envuelve varias redes preentrenadas. Para **emoción** usa una red liviana tipo **mini-Xception** entrenada sobre FER, que clasifica entre 7 emociones básicas; para **edad** usa una red basada en **VGG-Face** que la estima por regresión. A diferencia de la morfometría geométrica, estas redes aprenden patrones de textura y forma directamente de los píxeles.

#### MTCNN — detector de rostro para preprocesamiento

Red en cascada (Multi-task Cascaded CNN) usada en el pipeline de inferencia de los modelos de deep learning para detectar y alinear la cara antes de pasarla al backbone. También es preentrenada.

### Técnicas implementadas

- **Transfer learning + fine-tuning**: backbones preentrenados (ResNet-18, ViT) adaptados a la tarea.
- **Multi-task learning**: un embedding compartido alimenta múltiples cabezas (17 en el ViT) que se entrenan juntas.
- **Backbone freezing escalonado**: las primeras 5 epochs congelo el backbone y entreno solo las cabezas; después descongelo y hago fine-tune completo, evitando destruir los pesos preentrenados al inicio.
- **Learning rates discriminativas**: tasa alta para las cabezas nuevas y ~10x menor para el backbone, para no arruinar lo que ya sabía.
- **Precisión mixta (AMP)**: cómputo en float16 donde es seguro, ~2x más rápido y menos memoria en GPU.
- **Cosine annealing scheduler**: baja la learning rate de forma suave a lo largo del entrenamiento.
- **Data augmentation** (Albumentations): flip horizontal, jitter de color, ruido gaussiano, para mejorar generalización.
- **Funciones de pérdida**: BCEWithLogitsLoss con `pos_weight` para el desbalance de clases en el ResNet; suma de cross-entropy de las 17 cabezas en el ViT.
- **Early stopping + checkpointing**: corto cuando la validación deja de mejorar y guardo el mejor modelo.
- **Gradient clipping**: recorto la norma del gradiente a 1.0, habitual para estabilizar transformers.
- **Morfometría geométrica**: ratios entre landmarks normalizados por ancho de cara para derivar rasgos interpretables.
- **Clasificación de color en espacios HSV/Lab**: mediana del iris en HSV para color de ojos, robusta frente a reflejos.
- **Condicionamiento por texto (CLIP / Stable Diffusion)**: los rasgos se traducen a prompt tokens en inglés que el encoder CLIP convierte en vectores de condición para la generación.
- **Meta-learning / stacking** (en curso): un metalearner combinará las salidas de todos los extractores (atributos ResNet, cabezas ViT, rasgos geométricos, color de ojos) en una predicción final.

## Estructura del Proyecto

```
unique-pet-generation/
├── configs/
│   └── base.yaml                 # Hiperparámetros y configuración de entrenamiento
├── data/
│   └── celeba/                   # Dataset CelebA (descargado por script)
├── scripts/
│   ├── download_celeba.py        # Descarga CelebA desde Kaggle usando kagglehub
│   ├── train.py                  # Punto de entrada principal de entrenamiento
│   └── inference.py              # Predicciones por webcam o archivo de imagen
├── src/pet_gen/
│   ├── data/
│   │   ├── celeba_dataset.py     # Dataset PyTorch para CelebA (soporta formato CSV de Kaggle)
│   │   ├── transforms.py         # Pipelines de aumentos con Albumentations
│   │   └── preprocessing.py      # Detección de cara con MTCNN para inferencia
│   ├── models/
│   │   ├── backbone.py           # Extractores de features ResNet-18 / MobileNetV2
│   │   ├── heads.py              # Proyección del embedding y clasificador de atributos
│   │   └── feature_model.py      # Modelo completo: backbone + embedding 256-dim + cabeza de atributos
│   └── training/
│       ├── trainer.py            # Loop de entrenamiento con early stopping y checkpointing
│       ├── losses.py             # BCEWithLogitsLoss con ponderación por desbalance de clases
│       └── metrics.py            # Accuracy, F1 y mAP por atributo
├── notebooks/
│   ├── 00_s3_setup.ipynb                # Sube el dataset a S3 para entrenar en SageMaker
│   ├── 01_explore_celeba.ipynb          # Exploración de datos y distribución de atributos
│   ├── 02_celeba_to_flags_mapping.ipynb # Mapea atributos de CelebA a categorías de flags.txt
│   ├── 03_sagemaker_vit_training.ipynb  # Entrenamiento ViT-B/16 multi-head en SageMaker (anotado)
│   ├── 04_comparison_resnet_vit.ipynb   # ResNet vs ViT: params, velocidad, métricas, embeddings (anotado)
│   ├── 05_normalizacion_de_outputs.ipynb # Normaliza las salidas de los modelos a un esquema común
│   └── 06_facemesh.ipynb                # Pipeline de extracción de rasgos con MediaPipe + DeepFace (anotado)
├── tests/
│   ├── test_dataset.py           # Tests de carga del dataset, splits y transforms
│   └── test_model.py             # Tests de forward, flujo de gradientes y freeze/unfreeze
├── checkpoints/                  # Pesos guardados del modelo (best.pt, latest.pt)
├── doc/
│   └── index.html                # Reporte del proyecto para GitHub Pages
└── document.txt                  # Notas de entrenamiento y resultados
```

## Instalación

### Requisitos previos

- Python 3.12
- [Credenciales de la API de Kaggle](https://www.kaggle.com/docs/api) configuradas (`~/.kaggle/kaggle.json`)

### Paso 1: Clonar y preparar el entorno

```bash
git clone git@github.com:kateincoding/unique-pet-generation.git
cd unique-pet-generation
python -m venv .venv
source .venv/bin/activate
```

### Paso 2: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 3: Descargar el dataset CelebA

```bash
python scripts/download_celeba.py
```

Esto descarga el dataset CelebA (~1.4 GB) desde Kaggle y prepara el directorio de datos.

### Paso 4: Explorar los datos (opcional)

```bash
jupyter notebook notebooks/01_explore_celeba.ipynb
```

### Paso 5: Entrenar el modelo

```bash
caffeinate -i python scripts/train.py
```

El entrenamiento corre hasta 30 epochs con early stopping (patience=5). En Apple Silicon (MPS) tarda aproximadamente 1-2 horas.

### Paso 6: Ejecutar inferencia

```bash
# Captura por webcam (ESPACIO para capturar, Q para salir)
python scripts/inference.py

# O desde un archivo de imagen
python scripts/inference.py path/to/photo.jpg
```

## Ejecutar Tests

```bash
pytest tests/ -v
```

## Pipeline de Notebooks

El directorio `notebooks/` documenta todo el flujo de la Fase 1 de principio a fin. Los notebooks 03, 04 y 06 llevan anotaciones celda por celda (teoría e interpretación de resultados) en primera persona:

- **03 — Entrenamiento ViT (SageMaker)**: ViT-B/16 multi-head con 17 cabezas softmax que comparten un embedding de 256-dim. Backbone congelado 5 epochs, después fine-tune completo con learning rates discriminativas y precisión mixta. Early stopping en el epoch 12, mejor en el epoch 9, ~84 min en una A10G.
- **04 — ResNet vs ViT**: comparación lado a lado de arquitectura, velocidad de inferencia, métricas guardadas, espacios de embedding (PCA / coseno) y acuerdo en conceptos solapados como el color de pelo.
- **06 — Pipeline Face Mesh**: webcam o imagen → MediaPipe Face Mesh (478 landmarks) → rasgos geométricos (forma de cara, ojos, tono de piel, simetría, cejas vía ratios de landmarks) + DeepFace (emoción, edad) → prompt tokens de Stable Diffusion → `rasgos_checklist.json` para el paso de generación.

## Resultados

### Baseline ResNet-18

- **Mejor epoch**: 19 (early stopping en 24)
- **mAP de validación**: 0.7530
- **Accuracy de validación**: 83.22%
- **Dimensión del embedding**: 256

### ViT multi-head

- **Mejor epoch**: 9 (early stopping en 12)
- **Accuracy media de validación**: 0.883
- **Rasgos fuertes**: barbilla, pecas, gafas, tono de piel, forma de cara (>0.95)
- **Rasgos débiles**: tamaño/forma de nariz, forma de ojos (~0.68–0.79) — sutiles, dependientes del ángulo y la iluminación, con etiquetas más ruidosas

### Pipeline Face Mesh

Este pipeline no se entrena, así que no tiene métricas de validación; usa modelos preentrenados (MediaPipe + DeepFace) en inferencia directa y es determinista sobre una imagen dada. Sus resultados son cualitativos: el conjunto de rasgos extraídos y su traducción a prompt. Sobre la imagen de ejemplo (`agus.jpg`) extrae 9 rasgos y 6 prompt tokens:

| Rasgo | Valor | Fuente | Confianza |
|---|---|---|---|
| Forma de cara | redonda (ratio 0.827) | geométrico | alta |
| Tamaño de ojos | pequeños | geométrico | alta |
| Color de ojos | negro | iris HSV | media |
| Cejas | arqueadas | geométrico | media |
| Sonrisa | 1.0 | geométrico | media |
| Simetría | 0.0 | geométrico | alta |
| Expresión | neutral (67%) | DeepFace | media |
| Edad estimada | ~23 años | DeepFace | media |
| Tono de piel | oscuro | luminosidad | media |

**Salida**: `outputs/rasgos_checklist.json` con los rasgos, sus confianzas y el prompt completo concatenado, listo para el paso de generación.

**Limitaciones observadas**: la simetría dio 0.0 (piso) porque la cara no está perfectamente frontal respecto al eje central que asume la fórmula; el tono de piel se calcula de luminosidad cruda, así que una foto con poca luz lo empuja hacia oscuro. Son las limitaciones típicas de un método geométrico con umbrales fijos: rápido y explicable, pero sensible al encuadre y la iluminación.

Ver `doc/index.html` para el reporte completo del proyecto.

## Licencia

Licencia MIT - ver [LICENSE](LICENSE) para más detalles.
