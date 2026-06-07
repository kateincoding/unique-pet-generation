<div style="display: flex; align-items: center; gap: 12px;">
  <img width="70" height="70" src="https://github.com/user-attachments/assets/ada07bce-4ea6-4926-a5e2-43dee01829f1" />
  <h1>Petly - Unique Pet Generation 🐾</h1>
</div>

Genera mascotas virtuales que se parecen a sus dueños humanos, extrayendo rasgos faciales y mapeándolos a características de mascota mediante un modelo generativo.

## Fase 1: Extracción de Rasgos Faciales

Cuatro aproximaciones complementarias extraen atributos faciales a partir de fotos de personas:

1. **CNN entrenada desde cero** ([`face_extractor/cnn_from_scratch/`](face_extractor/cnn_from_scratch/README.md)) — backbone convolucional propio con 17 cabezas, entrenado sobre CelebA y fine-tuneado con FairFace, más dos cabezas extra de visión clásica para color de ojos y tono de piel. Sirve como baseline ligero (~1.6M parámetros) frente a los enfoques preentrenados.
2. **Transfer learning con ResNet-18** ([`face_extractor/resnet/`](face_extractor/resnet/README.md)) — produce un embedding de 256 dimensiones y predice 15 atributos binarios.
3. **Vision Transformer multi-head** ([`face_extractor/vit/`](face_extractor/vit/README.md)) — ViT-B/16 con 17 cabezas softmax, entrenado en SageMaker.
4. **Pipeline geométrico con MediaPipe Face Mesh** ([`face_extractor/clip_facemesh/`](face_extractor/clip_facemesh/README.md)) — MediaPipe (478 landmarks) + DeepFace, extrae rasgos interpretables y los traduce a prompt tokens para Stable Diffusion.

### Comparativa de los modelos entrenados

| | CNN from scratch | ResNet-18 | ViT Multi-Head |
|---|---|---|---|
| Backbone | Custom (5 conv blocks) | ResNet-18 (ImageNet-1k) | ViT-B/16 (ImageNet-21k) |
| Tipo de entrenamiento | Desde cero | Transfer learning | Transfer learning |
| Datasets | CelebA + FairFace + pseudo-labels | CelebA | CelebA |
| Salida | 19 atributos multi-clase | 15 atributos binarios (sigmoide) | 17 categorías multi-clase (softmax) |
| Parámetros | 1.6M | 11.3M | 86.0M |
| Embedding / descriptor | 512-dim | 256-dim | 256-dim |
| Mejor epoch | 8 | 19 | 9 |
| Métrica de validación | acc media 0.872 | mAP 0.7530 | acc media 0.883 |

Los tres modelos usan esquemas de etiquetas y métricas distintos, así que las cifras no son directamente comparables. La CNN desde cero queda como baseline más ligero y autocontenido; el ViT alcanza la mayor accuracy media pero es ~4x más lento que el ResNet; el ResNet queda como la opción equilibrada cuando la velocidad importa.

## Marco Teórico y Técnicas Implementadas

El proyecto combina tres familias de técnicas: **entrenamiento desde cero** sobre una CNN propia; **transfer learning** sobre backbones preentrenados con fine-tuning; y **modelos preentrenados usados sin entrenar** en inferencia directa. A continuación el marco teórico de cada bloque.

### CNN entrenada desde cero

Backbone convolucional propio de cinco bloques (Conv 3x3 + BatchNorm + ReLU + MaxPool) que recorre la secuencia 3→32→64→128→256→512 canales mientras la resolución espacial desciende de 224 a 7 píxeles. Un *adaptive average pooling* final produce un descriptor compacto de 512 dimensiones que alimenta 19 cabezas lineales independientes para atributos como color y textura de pelo, forma de ojos, tono de piel y rango de edad. La pérdida es multi-tarea (suma ponderada de cross-entropies) con `ignore_index=-1` para muestras sin etiqueta en una cabeza concreta, lo que permite aprovechar CelebA aunque la señal por atributo sea heterogénea. Tras el entrenamiento principal sobre CelebA (20 epochs), se hace fine-tuning de las cabezas de tono de piel y rango de edad sobre FairFace, y se añaden dos cabezas adicionales (color de ojos y tono de piel LAB) entrenadas sobre pseudo-etiquetas generadas con visión clásica.

### Transfer learning sobre backbones preentrenados

El transfer learning reutiliza una red ya entrenada sobre un dataset enorme y la adapta a la tarea actual, en lugar de entrenar desde cero. Las primeras capas aprenden features visuales genéricas (bordes, texturas, formas) que sirven para casi cualquier problema de imágenes, así que solo es necesario re-entrenar las capas finales y ajustar suavemente el resto. Esto permite entrenar con relativamente pocos datos y en poco tiempo.

**ResNet-18 (CNN residual)**. Red convolucional de 18 capas preentrenada en ImageNet-1k. Su aporte teórico son las **conexiones residuales** (skip connections): cada bloque aprende un residuo que se suma a su entrada, lo que evita el problema del gradiente que se desvanece y permite entrenar redes profundas de forma estable. Las convoluciones aplican filtros locales que detectan patrones espaciales con invariancia de traslación. Se utiliza como extractor liviano: 11.3M de parámetros, 256-dim de embedding, salida con 15 atributos binarios y activación sigmoide (cada atributo es una decisión independiente).

**ViT-B/16 (Vision Transformer)**. Transformer de visión preentrenado en ImageNet-21k. En lugar de convoluciones, **divide la imagen en parches de 16x16**, los proyecta a vectores (tokens) y los procesa con **auto-atención** (self-attention), que pondera la relación entre todos los parches a la vez. Esto captura dependencias de largo alcance en la cara (relacionar ojos con mandíbula, por ejemplo) mejor que el campo receptivo local de una CNN. El coste es cuadrático en el número de parches y necesita más datos de preentrenamiento, por eso parte de ImageNet-21k. Aquí se configura como **multi-task / multi-head**: un tronco común genera el embedding de 256-dim y 17 cabezas softmax independientes, una por categoría, predicen en paralelo compartiendo la representación.

### Modelos preentrenados usados sin entrenar (inferencia directa)

Estos modelos ya vienen entrenados por terceros. No se entrenan: solo se les pasa la imagen y se consume su salida.

**MediaPipe Face Mesh** es un modelo preentrenado de Google (no una librería de geometría manual). Internamente corre en dos etapas: primero un detector de rostro liviano tipo **BlazeFace** que localiza la cara, y después una **red de regresión** que predice las coordenadas de **478 landmarks** faciales (468 de la malla base + 10 de iris con refinamiento). Fue entrenado sobre miles de caras anotadas. Sobre esas coordenadas se calculan rasgos con **morfometría geométrica** (proporciones entre puntos normalizadas por el ancho de la cara, invariantes a escala y distancia a la cámara).

**DeepFace** es una librería que envuelve varias redes preentrenadas. Para **emoción** usa una red liviana tipo **mini-Xception** entrenada sobre FER, que clasifica entre 7 emociones básicas; para **edad** usa una red basada en **VGG-Face** que la estima por regresión. A diferencia de la morfometría geométrica, estas redes aprenden patrones de textura y forma directamente de los píxeles.

**MTCNN** es una red en cascada (Multi-task Cascaded CNN) usada en el pipeline de inferencia de los modelos de deep learning para detectar y alinear la cara antes de pasarla al backbone. También es preentrenada.

### Técnicas implementadas

- **Transfer learning + fine-tuning**: backbones preentrenados (ResNet-18, ViT-B/16) adaptados a la tarea.
- **Multi-task learning**: un embedding compartido alimenta múltiples cabezas (19 en la CNN propia, 17 en el ViT, 15 en el ResNet) que se entrenan juntas.
- **Backbone freezing escalonado**: las primeras epochs el backbone se congela y solo se entrenan las cabezas; después se descongela para fine-tune completo, evitando destruir los pesos preentrenados al inicio.
- **Learning rates discriminativas**: tasa alta para las cabezas nuevas y ~10x menor para el backbone, para no arruinar lo aprendido.
- **Precisión mixta (AMP)**: cómputo en float16 donde es seguro, ~2x más rápido y con menos consumo de memoria en GPU.
- **Cosine annealing scheduler**: la learning rate baja suavemente a lo largo del entrenamiento.
- **Data augmentation** (Albumentations): flip horizontal, jitter de color, ruido gaussiano, para mejorar generalización.
- **Funciones de pérdida**: BCEWithLogitsLoss con `pos_weight` para el desbalance de clases en el ResNet; suma ponderada de cross-entropies con `ignore_index=-1` en la CNN from scratch; suma de cross-entropy de las 17 cabezas en el ViT.
- **Pseudo-etiquetado con visión clásica**: en CelebA se generan pseudo-labels de color de ojos (HSV con balance de blancos por esclera y máscara anular del iris) y tono de piel (canal L del espacio LAB), que después se usan para entrenar cabezas adicionales en la CNN from scratch.
- **Early stopping + checkpointing**: corte cuando la validación deja de mejorar y guardado del mejor modelo.
- **Gradient clipping**: norma del gradiente recortada a 1.0, habitual para estabilizar transformers.
- **Morfometría geométrica**: ratios entre landmarks normalizados por ancho de cara para derivar rasgos interpretables.
- **Clasificación de color en espacios HSV/LAB**: mediana del iris en HSV para color de ojos; canal L de LAB para tono de piel; ambas robustas frente a variaciones de iluminación.
- **Condicionamiento por texto (CLIP / Stable Diffusion)**: los rasgos se traducen a prompt tokens en inglés que el encoder CLIP convierte en vectores de condición para la generación.
- **Meta-learning / stacking** (en curso): un metalearner combinará las salidas de todos los extractores (atributos ResNet, cabezas ViT, rasgos geométricos, color de ojos) en una predicción final.

## Fase 2: Generación de la mascota

La fase de generación reside en [`pet_generation/pet_generation.ipynb`](pet_generation/README.md). Toma los rasgos extraídos en la Fase 1 y genera un *familiar mágico* — un animalito real y tierno (zorrito, búho, gatito, panda rojo, cervatillo...) — condicionado por la cara del usuario. El notebook articula el pipeline en tres bloques:

1. **Ensemble de cinco fuentes**. Los tres modelos entrenados (`final_model-agus.pt`, `resnet-18-kat.pt`, `vit_multihead_best-kat.pt`) predicen probabilidades sobre 19 rasgos. A esas tres distribuciones se suman dos votantes adicionales: una extracción geométrica con MediaPipe Face Mesh (478 landmarks, 11 rasgos derivados de ratios) y un verificador zero-shot con CLIP (`openai/clip-vit-base-patch32`). Las cinco se combinan por voto suave ponderado por rasgo, con una capa final de corrección manual (`ipywidgets`) que permite ajustar cualquier predicción antes de pasar a generación.
2. **Selección determinista de la criatura**. Un mapper convierte los rasgos consolidados en una especie, una paleta de colores y un conjunto de *traits* concretos. La función es determinista, así que la misma combinación de rasgos siempre desemboca en la misma especie.
3. **Generación con difusión**. Stable Diffusion 1.5 (`Yntec/CuteFurry`) genera la imagen final, condicionada por (a) un prompt enfático construido a partir de los rasgos y la especie elegida con `compel` para ponderar tokens, (b) IP-Adapter *plus-face* (`h94/IP-Adapter`) que recibe la cara recortada del usuario para inyectar coloración y expresión, y (c) LCM-LoRA (`latent-consistency/lcm-lora-sdv1-5`) que reduce la inferencia a ~8 pasos en lugar de los 25-30 habituales.

Los checkpoints de los modelos de la Fase 1 que el notebook consume (`final_model-agus.pt`, `resnet-18-kat.pt`, `vit_multihead_best-kat.pt`) y las imágenes de prueba (`rostro_agus.jpeg`, `rostro_prueba.jpg`) no están en el repositorio porque superan el límite de tamaño de GitHub. Se distribuyen aparte y deben colocarse al mismo nivel que el `.ipynb`. Ver [`pet_generation/README.md`](pet_generation/README.md) para la lista completa de archivos externos.

## Aplicación web: Petly

[`petly/`](petly/README.md) es la aplicación web que envuelve todo el pipeline anterior en un producto utilizable: el usuario sube una foto desde el navegador y recibe una mascota generada. Internamente reutiliza la misma lógica que el notebook de la Fase 2 (Stable Diffusion 1.5 + IP-Adapter *plus-face* + LCM-LoRA para acelerar la inferencia) más una capa determinista que traduce los 18 rasgos extraídos a una especie, una paleta de colores y un conjunto de *traits* concretos.

- **Backend** (FastAPI, [`petly/backend/`](petly/backend/)): expone dos endpoints, `POST /api/analyze` (recibe la foto y devuelve la mascota propuesta con la PNG generada) y `GET /api/pets` (lista la colección). El pipeline de análisis facial es el mismo que el del notebook, portado a módulos Python (`pipeline/core.py`, `mapper.py`).
- **Frontend** (Vite + React, [`petly/frontend/`](petly/frontend/)): port hifi del prototipo Petly con una *state machine* de cinco pantallas (intro → captura → análisis → reveal → galería). Si la generación con Stable Diffusion está apagada o falla, cae a un SVG vectorial procedural construido a partir de los mismos *traits*.
- **Requisitos**: Python 3.11 con los checkpoints `.pt` del notebook, Node 18+, y opcionalmente GPU para que la generación con SD vaya en segundos en vez de minutos.

Los detalles de arranque (`uvicorn`, `npm run dev`), variables de entorno (`PETLY_MODELS_DIR`, `PETLY_GEN`, `PETLY_GEN_STEPS`...) y el contrato de la API están en el [README de petly](petly/README.md).

## Estructura del Proyecto

```
unique-pet-generation/
├── README.md                               # este archivo
├── LICENSE
├── face_extractor/
│   ├── README.md                           # índice de las 4 aproximaciones
│   ├── cnn_from_scratch/                   # 5.1 CNN multi-cabeza desde cero
│   │   ├── facial_attributes_extractor.ipynb
│   │   ├── inference.py
│   │   ├── finetune_extensions.py
│   │   ├── generate_pseudolabels.py
│   │   ├── generate_skin_pseudolabels.py
│   │   ├── models/                         # backbone + heads
│   │   ├── utils/                          # análisis HSV/LAB y label maps
│   │   ├── checkpoints/                    # final_model.pt
│   │   ├── test_photos/
│   │   └── requirements.txt
│   ├── resnet/                             # 5.2 Transfer learning con ResNet
│   │   ├── configs/base.yaml               # hiperparámetros y configuración
│   │   ├── data/                           # CelebA (descargado por script)
│   │   ├── scripts/
│   │   │   ├── download_celeba.py          # descarga CelebA desde Kaggle (kagglehub)
│   │   │   ├── train.py                    # punto de entrada de entrenamiento
│   │   │   └── inference.py                # webcam o archivo de imagen
│   │   ├── src/pet_gen/
│   │   │   ├── data/                       # dataset CelebA, transforms, preprocessing
│   │   │   ├── models/                     # backbone, heads, feature_model
│   │   │   └── training/                   # trainer, losses, metrics
│   │   ├── notebooks/
│   │   │   ├── 01_explore_celeba.ipynb
│   │   │   ├── 04_comparison_resnet_vit.ipynb
│   │   │   └── 05_normalizacion_de_outputs.ipynb
│   │   ├── tests/
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   └── uv.lock
│   ├── vit/                                # 5.3 ViT multi-head en SageMaker
│   │   └── notebooks/
│   │       ├── 00_s3_setup.ipynb           # sube CelebA a S3 para SageMaker
│   │       ├── 02_celeba_to_flags_mapping.ipynb
│   │       └── 03_sagemaker_vit_training.ipynb
│   └── clip_facemesh/                      # 5.4 MediaPipe + DeepFace + CLIP
│       ├── notebooks/
│       │   └── 06_facemesh.ipynb
│       └── agus.jpg                        # imagen de ejemplo
├── pet_generation/                         # Fase 2: generación con Stable Diffusion
│   ├── README.md
│   └── pet_generation.ipynb
└── petly/                                  # Aplicación web (FastAPI + React)
    ├── README.md
    ├── backend/                            # FastAPI + pipeline portado del notebook
    │   ├── app/
    │   │   ├── main.py                     # app FastAPI, CORS, routers
    │   │   ├── api/                        # /api/analyze, /api/pets
    │   │   ├── pipeline/core.py            # lógica de análisis facial
    │   │   ├── mapper.py                   # 18 rasgos → Pet (especie, color, traits)
    │   │   ├── db.py                       # colección en SQLite
    │   │   └── schemas.py                  # contrato Pet / Trait / AnalyzeResponse
    │   └── requirements.txt
    └── frontend/                           # Vite + React
        ├── src/
        │   ├── App.jsx                     # state machine + cliente
        │   ├── api.js                      # llamadas al backend
        │   ├── screens.jsx                 # intro / capture / analyze / reveal / gallery
        │   └── pets.jsx                    # 6 mascotas SVG + registro
        └── package.json
```

## Instalación y uso

Cada subcarpeta de `face_extractor/` y la de `pet_generation/` es autocontenida con sus propias dependencias y notebooks. Lo que sigue son los pasos rápidos para cada aproximación; el README específico de cada subcarpeta da más detalle.

### Requisitos previos

- Python 3.12.
- Para los pipelines basados en CelebA: [credenciales de la API de Kaggle](https://www.kaggle.com/docs/api) configuradas en `~/.kaggle/kaggle.json`.
- Para el ViT: cuenta de AWS con permisos para SageMaker y S3.

### Modelos preentrenados (entrega aparte)

Los checkpoints `.pt` no están en el repositorio porque cada uno supera el límite de GitHub. Se distribuyen en un zip aparte que contiene una carpeta `models/` con tres archivos:

```
models/
├── cnn_scratch.pt              (~6 MB)   — CNN multi-cabeza desde cero
├── resnet-18.pt                (~136 MB) — ResNet-18 transfer learning
└── vit_multihead_best.pt       (~344 MB) — ViT-B/16 multi-head
```

Cada subproyecto del repo espera los `.pt` en una carpeta y con un nombre concretos. La tabla siguiente indica dónde copiar cada archivo según qué se quiera ejecutar:

| Si quieres ejecutar... | Copia desde `models/` | a esta ruta del repo |
|---|---|---|
| **Demo principal de generación de mascotas** (`pet_generation/pet_generation.ipynb`) | `cnn_scratch.pt` | `pet_generation/final_model-agus.pt` |
| ↑ | `resnet-18.pt` | `pet_generation/resnet-18-kat.pt` |
| ↑ | `vit_multihead_best.pt` | `pet_generation/vit_multihead_best-kat.pt` |
| **Análisis CNN from scratch** (`face_extractor/cnn_from_scratch/facial_attributes_extractor.ipynb`) | `cnn_scratch.pt` | `face_extractor/cnn_from_scratch/checkpoints/final_model.pt` |
| **Comparativa ResNet vs ViT** (`face_extractor/resnet/notebooks/04_comparison_resnet_vit.ipynb`) | `resnet-18.pt` | `face_extractor/resnet/checkpoints/resnet-18.pt` |
| ↑ | `vit_multihead_best.pt` | `face_extractor/resnet/checkpoints/vit_multihead_best.pt` |

> **Para el caso más común (sólo el demo principal)**: basta con copiar los tres archivos a `pet_generation/` renombrándolos como indica la tabla. Los notebooks de análisis individuales sólo son necesarios si se quieren reproducir los experimentos por separado.

Adicionalmente, el demo necesita dos imágenes de prueba (`rostro_agus.jpeg` y `rostro_prueba.jpg`) que también se distribuyen en el zip. Se colocan al lado del notebook, en `pet_generation/`.

### CNN from scratch

```bash
cd face_extractor/cnn_from_scratch
pip install -r requirements.txt
```

Inferencia rápida sobre una imagen:

```python
from inference import extract_face_attributes
attrs = extract_face_attributes("test_photos/foto1.jpeg",
                                checkpoint_path="checkpoints/final_model.pt")
print(attrs)
```

El notebook `facial_attributes_extractor.ipynb` contiene el entrenamiento completo y los análisis post entrenamiento (distribuciones reales vs predichas, matrices de confusión, ejemplos cualitativos).

### ResNet (paquete pet_gen)

```bash
cd face_extractor/resnet
python -m venv .venv
source .venv/bin/activate            # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_celeba.py    # descarga CelebA (~1.4 GB)
python scripts/train.py              # entrenamiento hasta 30 epochs con early stopping
```

Inferencia:

```bash
python scripts/inference.py                  # webcam (SPACE para capturar, Q para salir)
python scripts/inference.py path/to/photo.jpg
```

En Apple Silicon (MPS) el entrenamiento tarda aproximadamente 1-2 horas.

### ViT (entrenado en SageMaker)

Los notebooks de la subcarpeta `vit/` están pensados para ejecutarse sobre AWS SageMaker. El flujo es: `00_s3_setup` sube CelebA a S3, `02_celeba_to_flags_mapping` define las 17 categorías softmax (`flags.txt`), y `03_sagemaker_vit_training` lanza el job de entrenamiento.

### CLIP + Face Mesh

```bash
cd face_extractor/clip_facemesh
jupyter notebook notebooks/06_facemesh.ipynb
```

El notebook carga MediaPipe Face Mesh y DeepFace, procesa una imagen (webcam o archivo) y produce un `rasgos_checklist.json` con los atributos y el prompt completo para Stable Diffusion.

### Generación de mascotas

Ver [`pet_generation/README.md`](pet_generation/README.md). Requiere descargar los checkpoints externos y las imágenes de prueba listadas allí.

## Tests

```bash
cd face_extractor/resnet
pytest tests/ -v
```

Cubren la carga del dataset, los splits y transforms, los forwards de los backbones y el flujo de gradientes con freeze/unfreeze.

## Notebooks documentados

Los siguientes notebooks llevan anotaciones celda por celda con teoría e interpretación de resultados:

- `face_extractor/cnn_from_scratch/facial_attributes_extractor.ipynb` — diseño, entrenamiento, extensión con pseudo-etiquetas y análisis de sesgo del baseline desde cero.
- `face_extractor/vit/notebooks/03_sagemaker_vit_training.ipynb` — ViT-B/16 multi-head con 17 cabezas softmax sobre un embedding de 256-dim. Backbone congelado 5 epochs, luego fine-tune completo con learning rates discriminativas y precisión mixta. Early stopping en el epoch 12, mejor en el 9, ~84 min en una A10G.
- `face_extractor/resnet/notebooks/04_comparison_resnet_vit.ipynb` — comparación lado a lado de arquitectura, velocidad de inferencia, métricas guardadas, espacios de embedding (PCA / coseno) y acuerdo en conceptos solapados como el color de pelo.
- `face_extractor/clip_facemesh/notebooks/06_facemesh.ipynb` — webcam o imagen → MediaPipe Face Mesh (478 landmarks) → rasgos geométricos (forma de cara, ojos, tono de piel, simetría, cejas vía ratios de landmarks) + DeepFace (emoción, edad) → prompt tokens de Stable Diffusion → `rasgos_checklist.json` para el paso de generación.

## Resultados

### CNN from scratch

- **Parámetros**: 1.6M (1.57M backbone + 37K cabezas).
- **Mejor epoch**: 8.
- **Accuracy media de validación**: 0.872 sobre 15 cabezas activas de CelebA.
- **Cabezas finales**: 19 (17 entrenadas por el CNN + 2 entrenadas sobre pseudo-etiquetas de visión clásica).
- **Comportamiento observado**: predictor centrado en la moda del dataset en imágenes ambiguas; la cabeza CNN de tono de piel queda por debajo de la versión LAB, por lo que el sistema final se apoya en la versión clásica para esa decisión.

### Baseline ResNet-18

- **Mejor epoch**: 19 (early stopping en 24).
- **mAP de validación**: 0.7530.
- **Accuracy de validación**: 83.22%.
- **Dimensión del embedding**: 256.

### ViT multi-head

- **Mejor epoch**: 9 (early stopping en 12).
- **Accuracy media de validación**: 0.883.
- **Rasgos fuertes**: barbilla, pecas, gafas, tono de piel, forma de cara (>0.95).
- **Rasgos débiles**: tamaño/forma de nariz, forma de ojos (~0.68–0.79), sutiles, dependientes del ángulo y la iluminación, con etiquetas más ruidosas.

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

Salida: `outputs/rasgos_checklist.json` con los rasgos, sus confianzas y el prompt completo concatenado, listo para el paso de generación.

**Limitaciones observadas**: la simetría dio 0.0 porque la cara no estaba perfectamente frontal respecto al eje central que asume la fórmula; el tono de piel se calcula de luminosidad cruda, así que una foto con poca luz lo empuja hacia oscuro. Son las limitaciones típicas de un método geométrico con umbrales fijos: rápido y explicable, pero sensible al encuadre y la iluminación.

## Licencia

MIT — ver [LICENSE](LICENSE) para más detalles.
