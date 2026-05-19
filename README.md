# Unique Pet Generation

Generate pets that resemble their human owners by extracting facial features and mapping them to pet characteristics.

## Phase 1: Facial Feature Extraction Baseline

A multi-task neural network that extracts facial attributes from human photos, producing a 256-dimensional embedding that encodes characteristics like hair color, face shape, and facial structure.

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
│   └── 01_explore_celeba.ipynb   # Data exploration and attribute distribution analysis
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

## Results

- **Best epoch**: 19 (early stopped at 24)
- **Validation mAP**: 0.7530
- **Validation accuracy**: 83.22%
- **Embedding dimension**: 256

See `doc/index.html` for the full project report.

## License

MIT License - see [LICENSE](LICENSE) for details.
