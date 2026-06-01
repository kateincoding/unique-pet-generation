# Unique Pet Generation

Generate pets that resemble their human owners by extracting facial features and mapping them to pet characteristics.

## Phase 1: Facial Feature Extraction

Two complementary approaches extract facial attributes from human photos:

1. **Deep learning models** — a ResNet-18 baseline and a ViT-B/16 multi-head model, both producing a 256-dimensional embedding that encodes characteristics like hair color, face shape, and facial structure. This embedding is the bridge to Phase 2 pet generation.
2. **Geometric pipeline** — MediaPipe Face Mesh (478 landmarks) plus DeepFace, extracting interpretable traits (face shape, eyes, skin tone, expression, age) and translating them into Stable Diffusion prompt tokens for pet generation.

### Models

| | ResNet-18 | ViT Multi-Head |
|---|---|---|
| Backbone | ResNet-18 (ImageNet-1k) | ViT-B/16 (ImageNet-21k) |
| Output | 15 binary attributes (sigmoid) | 17 multi-class categories (softmax) |
| Params | 11.3M | 86.0M |
| Checkpoint | 136 MB | 344 MB |
| Inference (batch 1) | 3.5 ms | 14.5 ms |
| Embedding | 256-dim | 256-dim |
| Best epoch | 19 | 9 |
| Validation metric | mAP 0.7530 | mean acc 0.883 |

The two models use different label schemas (CelebA binary attributes vs `flags.txt` categories), so their metrics are not directly comparable. The ViT is ~4x slower but starts from a richer pretraining and reaches higher mean accuracy; ResNet stays the lightweight option when speed matters.

## Project Structure

```
unique-pet-generation/
├── configs/
│   └── base.yaml                 # Hyperparameters and training configuration
├── data/
│   └── celeba/                   # CelebA dataset (downloaded via script)
├── scripts/
│   ├── download_celeba.py        # Download CelebA from Kaggle using kagglehub
│   ├── train.py                  # Main training entry point
│   └── inference.py              # Run predictions via webcam or image file
├── src/pet_gen/
│   ├── data/
│   │   ├── celeba_dataset.py     # PyTorch Dataset for CelebA (supports Kaggle CSV format)
│   │   ├── transforms.py         # Albumentations augmentation pipelines
│   │   └── preprocessing.py      # MTCNN face detection for inference
│   ├── models/
│   │   ├── backbone.py           # ResNet-18 / MobileNetV2 feature extractors
│   │   ├── heads.py              # Embedding projection and attribute classifier
│   │   └── feature_model.py      # Full model: backbone + 256-dim embedding + attribute head
│   └── training/
│       ├── trainer.py            # Training loop with early stopping and checkpointing
│       ├── losses.py             # BCEWithLogitsLoss with class imbalance weighting
│       └── metrics.py            # Per-attribute accuracy, F1, and mAP
├── notebooks/
│   ├── 00_s3_setup.ipynb                # Upload dataset to S3 for SageMaker training
│   ├── 01_explore_celeba.ipynb          # Data exploration and attribute distribution analysis
│   ├── 02_celeba_to_flags_mapping.ipynb # Map CelebA attributes to flags.txt categories
│   ├── 03_sagemaker_vit_training.ipynb  # ViT-B/16 multi-head training on SageMaker (annotated)
│   ├── 04_comparison_resnet_vit.ipynb   # ResNet vs ViT: params, speed, metrics, embeddings (annotated)
│   ├── 05_normalizacion_de_outputs.ipynb # Normalize model outputs to a common schema
│   └── 06_facemesh.ipynb                # MediaPipe + DeepFace feature extraction pipeline (annotated)
├── tests/
│   ├── test_dataset.py           # Dataset loading, splits, and transform tests
│   └── test_model.py             # Model forward pass, gradient flow, freeze/unfreeze tests
├── checkpoints/                  # Saved model weights (best.pt, latest.pt)
├── doc/
│   └── index.html                # Project report for GitHub Pages
└── document.txt                  # Training notes and results
```

## Installation

### Prerequisites

- Python 3.12
- [Kaggle API credentials](https://www.kaggle.com/docs/api) configured (`~/.kaggle/kaggle.json`)

### Step 1: Clone and set up environment

```bash
git clone git@github.com:kateincoding/unique-pet-generation.git
cd unique-pet-generation
python -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download CelebA dataset

```bash
python scripts/download_celeba.py
```

This downloads the CelebA dataset (~1.4 GB) from Kaggle and sets up the data directory.

### Step 4: Explore the data (optional)

```bash
jupyter notebook notebooks/01_explore_celeba.ipynb
```

### Step 5: Train the model

```bash
caffeinate -i python scripts/train.py
```

Training runs for up to 30 epochs with early stopping (patience=5). On Apple Silicon (MPS) it takes approximately 1-2 hours.

### Step 6: Run inference

```bash
# Webcam capture (press SPACE to snap, Q to quit)
python scripts/inference.py

# Or from an image file
python scripts/inference.py path/to/photo.jpg
```

## Running Tests

```bash
pytest tests/ -v
```

## Notebooks Pipeline

The `notebooks/` directory documents the full Phase 1 workflow end to end. Notebooks 03, 04, and 06 carry cell-by-cell annotations (theory and result interpretation) in first person:

- **03 — ViT training (SageMaker)**: ViT-B/16 multi-head with 17 softmax heads sharing a 256-dim embedding. Backbone frozen 5 epochs, then full fine-tune with discriminative learning rates and mixed precision. Early stopped at epoch 12, best at epoch 9, ~84 min on an A10G.
- **04 — ResNet vs ViT**: side-by-side comparison of architecture, inference speed, saved metrics, embedding spaces (PCA / cosine), and agreement on overlapping concepts like hair color.
- **06 — Face Mesh pipeline**: webcam or image → MediaPipe Face Mesh (478 landmarks) → geometric traits (face shape, eyes, skin tone, symmetry, eyebrows via landmark ratios) + DeepFace (emotion, age) → Stable Diffusion prompt tokens → `rasgos_checklist.json` for the generation step.

## Results

### ResNet-18 baseline

- **Best epoch**: 19 (early stopped at 24)
- **Validation mAP**: 0.7530
- **Validation accuracy**: 83.22%
- **Embedding dimension**: 256

### ViT multi-head

- **Best epoch**: 9 (early stopped at 12)
- **Validation mean accuracy**: 0.883
- **Strong traits**: chin, freckles, glasses, skin tone, face shape (>0.95)
- **Weak traits**: nose size/shape, eye shape (~0.68–0.79) — subtle, angle- and lighting-dependent, noisier labels

See `doc/index.html` for the full project report.

## License

MIT License - see [LICENSE](LICENSE) for details.
