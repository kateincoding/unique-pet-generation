# Generación de mascotas

Notebook de generación de mascotas (parte 6 de la memoria):

- `pet_generation.ipynb` — versión 15 del pipeline. Genera un *familiar mágico* (zorrito, búho, gatito, panda rojo, cervatillo...) a partir de una foto de la cara del usuario.

## Cómo funciona

El notebook encadena nueve etapas, organizadas en tres bloques:

1. **Extracción de rasgos** (etapas 1–4): captura de la imagen (webcam o archivo), detección facial con MediaPipe Face Mesh (478 landmarks con refinamiento de iris), extracción geométrica de 11 de los 19 rasgos a partir de ratios de landmarks, y carga de los tres modelos preentrenados de los compañeros (`resnet-18-kat.pt`, `vit_multihead_best-kat.pt`, `final_model-agus.pt`) más un verificador zero-shot con CLIP.
2. **Ensemble** (etapa 5): voto suave de cinco fuentes (los tres modelos + geometría + CLIP) ponderado por rasgo, con una capa final de corrección manual (*human-in-the-loop*) mediante `ipywidgets`.
3. **Generación** (etapas 6–9): selección determinista de la especie y la paleta a partir de los rasgos, construcción de un prompt enfático con seed única, y generación con Stable Diffusion 1.5 (`Yntec/CuteFurry`) + IP-Adapter *plus-face* (para que la cara real aporte color y expresión) + LCM-LoRA (8 pasos en vez de 25-30).

El pipeline completo está encapsulado en una función `end_to_end()` en la etapa 9, que es la que consume después la [aplicación web Petly](../petly/README.md).

## Archivos externos necesarios

Los modelos preentrenados y las imágenes de prueba no están en el repo porque superan el límite de tamaño de GitHub (100 MB por archivo). Para correr el notebook hay que descargarlos aparte y ubicarlos al mismo nivel que el `.ipynb`.

### Checkpoints `.pt` (locales)

| Archivo | Tamaño | Proviene de |
|---|---|---|
| `final_model-agus.pt` | ~6.5 MB | `face_extractor/cnn_from_scratch/` |
| `resnet-18-kat.pt` | ~136 MB | `face_extractor/resnet/` |
| `vit_multihead_best-kat.pt` | ~344 MB | `face_extractor/vit/` |

### Imágenes de prueba

| Archivo |
|---|
| `rostro_agus.jpeg` |
| `rostro_prueba.jpg` |

### Link de descarga

> Completar con el link de Drive o WeTransfer.

### Modelos descargados automáticamente (HuggingFace)

La primera ejecución descarga ~4-5 GB en caché:

| Modelo | Función |
|---|---|
| `Yntec/CuteFurry` | Stable Diffusion 1.5 con prior de animalitos peludos |
| `h94/IP-Adapter` (plus-face) | inyecta identidad de la cara recortada |
| `latent-consistency/lcm-lora-sdv1-5` | LCM-LoRA para reducir a ~8 pasos |
| `openai/clip-vit-base-patch32` | verificador zero-shot en el ensemble |

## Estructura esperada

```
pet_generation/
├── pet_generation.ipynb
├── final_model-agus.pt
├── resnet-18-kat.pt
├── vit_multihead_best-kat.pt
├── rostro_agus.jpeg
└── rostro_prueba.jpg
```

## Entorno

Notebook desarrollado con un entorno conda llamado `mascota`. Las dependencias principales son:

- **Core**: `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `pillow`, `opencv-python`
- **Visión**: `mediapipe` (Face Mesh con 478 landmarks)
- **Modelos preentrenados**: `transformers` (CLIP), `diffusers` (Stable Diffusion + LCM scheduler), `compel` (prompt weighting)
- **UI**: `ipywidgets` (corrección manual de rasgos)

En CPU, una imagen tarda 1-3 minutos en generarse (8 pasos LCM). Con GPU NVIDIA se reduce a segundos.
