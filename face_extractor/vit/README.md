# Vision Transformer (ViT)

Sección 5.3 de la memoria. Entrena un **ViT-B/16 multi-head** preentrenado en ImageNet-21k sobre CelebA en SageMaker, con 17 cabezas softmax que comparten un embedding de 256-dim.

## Notebooks

| Notebook | Qué hace |
|---|---|
| `00_s3_setup.ipynb` | Sube el dataset CelebA a S3 para entrenar en SageMaker |
| `02_celeba_to_flags_mapping.ipynb` | Mapea los 40 atributos binarios de CelebA a las 17 categorías softmax (`flags.txt`) usadas por el ViT |
| `03_sagemaker_vit_training.ipynb` | Entrenamiento ViT multi-head en SageMaker, anotado celda por celda (teoría + interpretación) |

## Resultados

- Backbone: ViT-B/16 (ImageNet-21k), 86M parámetros
- Mejor epoch: 9 (early stopping en 12)
- Accuracy media de validación: **0.883**
- Tiempo de entrenamiento: ~84 min en una A10G

El checkpoint (`vit_multihead_best.pt`, ~344 MB) no está en el repo; se distribuye aparte y se usa desde `../../pet_generation/`. La comparación cuantitativa contra el baseline ResNet está en [`../resnet/notebooks/04_comparison_resnet_vit.ipynb`](../resnet/notebooks/04_comparison_resnet_vit.ipynb).
