# Unique Pet Generation

Sistema que genera mascotas virtuales que se parecen a sus dueños. El pipeline tiene dos fases: extraer rasgos faciales de una foto de la persona y generar una mascota condicionada por esos rasgos.

## Estructura

```
unique-pet-generation/
├── face_extractor/          # Fase 1 — extracción de rasgos faciales
│   ├── cnn_from_scratch/    #   5.1 CNN multi-cabeza desde cero
│   ├── resnet/              #   5.2 Transfer learning con ResNet-18
│   ├── vit/                 #   5.3 Vision Transformer (ViT)
│   └── clip_facemesh/       #   5.4 CLIP + MediaPipe Face Mesh
├── pet_generation/          # Fase 2 — generación con Stable Diffusion
└── docs/                    # Reporte del proyecto (GitHub Pages)
```

| Carpeta | Sección | Técnica |
|---|---|---|
| [`face_extractor/cnn_from_scratch/`](face_extractor/cnn_from_scratch/README.md) | 5.1 | CNN multi-head entrenada desde cero (CelebA + FairFace) |
| [`face_extractor/resnet/`](face_extractor/resnet/README.md) | 5.2 | ResNet-18 preentrenada en ImageNet-1k, fine-tune sobre CelebA |
| [`face_extractor/vit/`](face_extractor/vit/README.md) | 5.3 | ViT-B/16 multi-head preentrenado en ImageNet-21k (SageMaker) |
| [`face_extractor/clip_facemesh/`](face_extractor/clip_facemesh/README.md) | 5.4 | MediaPipe Face Mesh + DeepFace + CLIP, sin entrenamiento |
| [`pet_generation/`](pet_generation/README.md) | 6 | Stable Diffusion / CuteFurry condicionado por los rasgos extraídos |

El reporte web está en [`docs/index.html`](docs/index.html) (GitHub Pages).
