# Face Extractor

Fase 1 del pipeline: dada una foto de una persona, extraer atributos faciales que sirvan como entrada al generador de mascotas. Hay dos aproximaciones independientes, cada una con su propio entorno y dependencias.

| Subcarpeta | Aproximación |
|---|---|
| [`cnn_from_scratch/`](cnn_from_scratch/README.md) | CNN multi-head entrenada **desde cero** sobre CelebA, con fine-tuning de las cabezas de tono de piel y edad en FairFace. Atributos extra de color de ojos y tono de piel con visión clásica (Haar + HoughCircles + LAB). |
| [`transfer_learning/`](transfer_learning/README.md) | Combina **transfer learning** (ResNet-18 sobre ImageNet-1k, ViT-B/16 sobre ImageNet-21k) con **modelos preentrenados usados en inferencia directa** (MediaPipe Face Mesh para morfometría geométrica, DeepFace para emoción y edad). |

Cada subcarpeta es autocontenida: tiene su propio README, requirements, modelos, notebooks y carpetas auxiliares. Las dos producen un conjunto de atributos que se consume en la fase de generación (`../pet_generation/`).
