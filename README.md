# Unique Pet Generation

Sistema que genera mascotas virtuales que se parecen a sus dueños. El pipeline tiene dos fases: extraer rasgos faciales de una foto de la persona y generar una mascota condicionada por esos rasgos.

## Estructura del repositorio

```
unique-pet-generation/
├── face_extractor/          # Fase 1 — extracción de rasgos faciales
│   ├── agus/                # CNN multi-cabeza desde cero (CelebA + FairFace)
│   └── kat/                 # Transfer learning (ResNet-18, ViT) + MediaPipe Face Mesh
├── pet_generation/          # Fase 2 — generación de la mascota (notebooks de Paulina)
└── docs/                    # Reporte del proyecto (GitHub Pages)
```

## Contribuciones

| Carpeta | Responsable | Contenido |
|---|---|---|
| `face_extractor/agus/` | Agus | CNN from-scratch multi-head sobre CelebA + fine-tuning FairFace, color de ojos por visión clásica, análisis de sesgo. Ver [README](face_extractor/agus/README.md). |
| `face_extractor/kat/` | Kat | Paquete `pet_gen` con baseline ResNet-18, ViT multi-head entrenado en SageMaker y pipeline de Face Mesh + DeepFace. Ver [README](face_extractor/kat/README.md). |
| `pet_generation/` | Paulina | Notebooks que toman los embeddings/rasgos y generan la mascota con Stable Diffusion. Ver [README](pet_generation/README.md). |

## Reporte

El reporte web del proyecto está en [`docs/index.html`](docs/index.html) (servido vía GitHub Pages).
