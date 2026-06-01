# Generación de mascotas

Notebook de generación de mascotas (parte 6 de la memoria):

- `pet_generation.ipynb`

## Archivos externos necesarios

Los modelos preentrenados y las imágenes de prueba no están en el repo porque superan el límite de tamaño de GitHub (100 MB por archivo). Para correr el notebook hay que descargarlos aparte y ubicarlos al mismo nivel que el `.ipynb`.

### Modelos

| Archivo | Tamaño |
|---|---|
| `final_model-agus.pt` | ~6.5 MB |
| `resnet-18-kat.pt` | ~136 MB |
| `vit_multihead_best-kat.pt` | ~344 MB |

### Imágenes de prueba

| Archivo |
|---|
| `rostro_agus.jpeg` |
| `rostro_prueba.jpg` |

### Link de descarga

> Completar con el link de Drive o WeTransfer.

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

El notebook se desarrolló con un entorno conda llamado `mascota`. Las dependencias principales son `torch`, `torchvision` y las librerías estándar de cómputo científico (numpy, pandas, matplotlib, PIL).
