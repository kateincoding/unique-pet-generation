# Unique Pet Generation

Sistema que genera mascotas virtuales que se parecen a sus dueños. El pipeline tiene dos fases: extraer rasgos faciales de una foto de la persona y generar una mascota condicionada por esos rasgos.

## Estructura

```
unique-pet-generation/
├── face_extractor/          # Fase 1 — extracción de rasgos faciales
│   ├── cnn_from_scratch/    #   CNN multi-cabeza entrenada desde cero
│   └── transfer_learning/   #   ResNet-18, ViT y MediaPipe Face Mesh
├── pet_generation/          # Fase 2 — generación de la mascota con Stable Diffusion
└── docs/                    # Reporte del proyecto (GitHub Pages)
```

| Carpeta | Aproximación | Datos / Modelo |
|---|---|---|
| [`face_extractor/cnn_from_scratch/`](face_extractor/cnn_from_scratch/README.md) | CNN multi-head entrenada desde cero | CelebA + fine-tuning FairFace, color de ojos por visión clásica |
| [`face_extractor/transfer_learning/`](face_extractor/transfer_learning/README.md) | Transfer learning + modelos preentrenados | ResNet-18 (ImageNet-1k), ViT-B/16 (ImageNet-21k), MediaPipe Face Mesh + DeepFace |
| [`pet_generation/`](pet_generation/README.md) | Modelo generativo condicionado | Stable Diffusion / CuteFurry, prompt engineering sobre los rasgos extraídos |

El reporte web está en [`docs/index.html`](docs/index.html) (GitHub Pages).
