# Face Extractor

Fase 1 del pipeline: dada una foto de una persona, extraer atributos faciales que sirvan como entrada al generador de mascotas. Cuatro aproximaciones independientes, una por sección de la memoria.

| Subcarpeta | Sección | Aproximación |
|---|---|---|
| [`cnn_from_scratch/`](cnn_from_scratch/README.md) | 5.1 | CNN multi-head entrenada **desde cero** sobre CelebA + fine-tuning FairFace. Color de ojos y tono de piel con visión clásica (Haar + HoughCircles + LAB). |
| [`resnet/`](resnet/README.md) | 5.2 | **Transfer learning con ResNet-18** preentrenada en ImageNet-1k. Embedding 256-dim + 15 cabezas binarias. Paquete `pet_gen` con dataset, training y métricas. |
| [`vit/`](vit/README.md) | 5.3 | **Vision Transformer (ViT-B/16)** multi-head preentrenado en ImageNet-21k, entrenado en SageMaker. 17 cabezas softmax. |
| [`clip_facemesh/`](clip_facemesh/README.md) | 5.4 | **MediaPipe Face Mesh + DeepFace + CLIP** sin entrenamiento. Pipeline geométrico que produce prompt tokens para Stable Diffusion. |

Las cuatro aproximaciones producen atributos que se consumen en la fase de generación ([`../pet_generation/`](../pet_generation/README.md)).

El paquete `pet_gen` (`resnet/src/pet_gen/`) lo importan los notebooks de la carpeta `resnet/` y se instala vía `pip install -e .` desde `resnet/`.
