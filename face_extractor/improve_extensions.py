"""
Mejora del modelo extendido:

  1. Renombra la cabeza tono_piel original (colapsada) a tono_piel_legacy.
     Se mantiene para no romper la API pero queda explícitamente marcada
     como deprecada.

  2. Re-entrena color_ojos con class weights inversos a la frecuencia,
     para mitigar el sesgo masivo hacia 'marron' (69% de los pseudo-labels).

  3. Re-entrena tono_piel_lab reducido a 5 clases (fusiona los extremos):
        muy_oscuro+oscuro -> oscuro
        bronceado         -> bronceado
        oliva             -> oliva
        medio             -> medio
        claro+muy_claro   -> claro
     Esta granularidad menor es más alcanzable desde features de cara
     y refleja mejor lo que necesita el generador de mascotas.

Usa features cacheados (una sola pasada por el backbone) para que el
entrenamiento de los heads dure segundos.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

CELEBA_ROOT    = './data/celeba'
CHECKPOINT_DIR = './checkpoints'
BATCH_SIZE     = 32
LR             = 1e-3
EPOCHS         = 15
FEATURE_DIM    = 512
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SUBSAMPLE    = 8000
VAL_FRACTION   = 0.15

EYE_COLOR_LABELS = ['marron_oscuro', 'gris', 'marron', 'avellana', 'ambar', 'verde', 'azul']
EYE_COLOR_IDX    = {c: i for i, c in enumerate(EYE_COLOR_LABELS)}

# 5 clases: fusionar extremos
SKIN_TONE_LABELS_5 = ['oscuro', 'bronceado', 'oliva', 'medio', 'claro']
SKIN_TONE_MAP_7_TO_5 = {
    'muy_oscuro': 0, 'oscuro': 0,
    'bronceado':  1,
    'oliva':      2,
    'medio':      3,
    'claro':      4, 'muy_claro': 4,
}

EYE_CSV  = os.path.join(CELEBA_ROOT, 'celeba_eye_color_pseudolabels.csv')
SKIN_CSV = os.path.join(CELEBA_ROOT, 'celeba_skin_tone_pseudolabels.csv')

HEADS_CONFIG = {
    'color_pelo': 6, 'textura_pelo': 4, 'longitud_pelo': 4,
    'cejas': 5, 'forma_ojos': 5, 'tamano_nariz': 3, 'forma_nariz': 4,
    'grosor_labios': 3, 'pomulos': 4, 'mandibula': 4, 'barbilla': 4,
    'forma_cara': 6, 'vello_facial': 4, 'gafas': 2, 'pecas': 2,
    'tono_piel': 7, 'rango_edad': 5,
}


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)


class FaceBackbone(nn.Module):
    def __init__(self, feature_dim=512, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32), ConvBlock(32, 64), ConvBlock(64, 128),
            ConvBlock(128, 256), ConvBlock(256, feature_dim),
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.features(x); x = self.pool(x)
        return self.dropout(x.view(x.size(0), -1))


class AttributeHead(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    def forward(self, x): return self.fc(x)


VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


print(f'Dispositivo: {DEVICE}')

# ── Carga del modelo base ─────────────────────────────────────────────
backbone = FaceBackbone(FEATURE_DIM).to(DEVICE)
heads = nn.ModuleDict({
    name: AttributeHead(FEATURE_DIM, n).to(DEVICE) for name, n in HEADS_CONFIG.items()
})
heads.add_module('color_ojos',    AttributeHead(FEATURE_DIM, len(EYE_COLOR_LABELS)).to(DEVICE))
heads.add_module('tono_piel_lab', AttributeHead(FEATURE_DIM, 7).to(DEVICE))

ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'final_model.pt'), map_location=DEVICE)
backbone.load_state_dict(ckpt['backbone'])
heads.load_state_dict(ckpt['heads'])
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False
print('Modelo cargado.')

# ── 1. Renombrar tono_piel -> tono_piel_legacy ────────────────────────
print('\n=== Paso 1: renombrar cabeza tono_piel a tono_piel_legacy ===')
legacy_head = heads['tono_piel']
heads.add_module('tono_piel_legacy', legacy_head)
del heads._modules['tono_piel']
print('Cabeza renombrada. La predicción original sigue accesible como tono_piel_legacy.')

# ── 2. Recalcular features (una sola pasada) ──────────────────────────
print('\n=== Paso 2: cachear features del backbone ===')

eye_df  = pd.read_csv(EYE_CSV,  index_col='image_id')
skin_df = pd.read_csv(SKIN_CSV, index_col='image_id')
common  = eye_df.index.intersection(skin_df.index)
print(f'Imágenes con ambas pseudo-etiquetas: {len(common):,}')

rng = np.random.default_rng(seed=42)
sample = list(rng.choice(common, size=min(N_SUBSAMPLE, len(common)), replace=False))
print(f'Sub-muestra para fine-tune: {len(sample):,}')

img_dir = os.path.join(CELEBA_ROOT, 'img_align_celeba', 'img_align_celeba')


class ImageOnlyDataset(Dataset):
    def __init__(self, file_names): self.file_names = file_names
    def __len__(self): return len(self.file_names)
    def __getitem__(self, idx):
        fname = self.file_names[idx]
        img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
        return VAL_TF(img), fname


ds = ImageOnlyDataset(sample)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
all_feats, all_fnames = [], []
with torch.no_grad():
    for imgs, fnames in tqdm(dl, desc='Features'):
        feats = backbone(imgs.to(DEVICE))
        all_feats.append(feats.cpu())
        all_fnames.extend(fnames)
features_tensor = torch.cat(all_feats, dim=0)
print(f'Features cacheados: {features_tensor.shape}')

n_total = len(all_fnames)
n_val   = int(n_total * VAL_FRACTION)
perm    = rng.permutation(n_total)
val_ids   = perm[:n_val]
train_ids = perm[n_val:]
print(f'Train: {len(train_ids):,}  Val: {len(val_ids):,}')


def finetune_with_weights(head_name, label_arr, labels_meta, class_weights=None):
    print(f'\n--- Fine-tune {head_name} ({len(labels_meta)} clases) ---')

    # Distribución
    unique, counts = np.unique(label_arr, return_counts=True)
    print('  Distribución de clases:')
    for u, c in zip(unique, counts):
        print(f'    {labels_meta[u]:<15} {c:>6} ({c/len(label_arr)*100:.1f}%)')

    if class_weights is not None:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
        print(f'  Class weights: {[round(w, 3) for w in class_weights]}')
    else:
        cw = None

    n_classes = len(labels_meta)
    # Reset de la cabeza
    head = AttributeHead(FEATURE_DIM, n_classes).to(DEVICE)
    heads.add_module(head_name, head)

    label_tensor = torch.from_numpy(label_arr.astype(np.int64))

    train_x = features_tensor[train_ids].to(DEVICE)
    train_y = label_tensor[train_ids].to(DEVICE)
    val_x   = features_tensor[val_ids].to(DEVICE)
    val_y   = label_tensor[val_ids].to(DEVICE)

    for p in head.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)

    best_acc = 0.0
    best_macro_f1 = 0.0
    n_train = len(train_x)

    for epoch in range(1, EPOCHS + 1):
        head.train()
        perm_ep = torch.randperm(n_train, device=DEVICE)
        tr_loss, n_b = 0.0, 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm_ep[start:start + BATCH_SIZE]
            xb, yb = train_x[idx], train_y[idx]
            loss = F.cross_entropy(head(xb), yb, weight=cw)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr_loss += loss.item(); n_b += 1

        head.eval()
        with torch.no_grad():
            preds = head(val_x).argmax(1)
            acc = (preds == val_y).float().mean().item()
            # Macro F1 aproximado: media de recall por clase
            recalls = []
            for k in range(n_classes):
                mask = (val_y == k)
                if mask.sum() > 0:
                    recalls.append(((preds[mask] == k).float().mean()).item())
            macro_recall = float(np.mean(recalls)) if recalls else 0.0
        best_acc = max(best_acc, acc)
        best_macro_f1 = max(best_macro_f1, macro_recall)
        print(f'  Época {epoch:2d}/{EPOCHS}  loss={tr_loss/n_b:.4f}  '
              f'val_acc={acc:.3f}  macro_recall={macro_recall:.3f}')

    print(f'  Mejor val_acc: {best_acc:.3f}   Mejor macro_recall: {best_macro_f1:.3f}')


# ── 3. Re-entrenar color_ojos con class weights ──────────────────────
print('\n=== Paso 3: re-entrenar color_ojos con balanceo de clases ===')
eye_labels = np.array([EYE_COLOR_IDX[eye_df.loc[f, 'color_ojos']] for f in all_fnames])
unique_eye, counts_eye = np.unique(eye_labels, return_counts=True)
# class weight inverso a la frecuencia, normalizado
freq = np.zeros(len(EYE_COLOR_LABELS))
for u, c in zip(unique_eye, counts_eye):
    freq[u] = c
freq = np.where(freq == 0, 1, freq)
eye_weights = (1.0 / freq)
eye_weights = eye_weights / eye_weights.sum() * len(EYE_COLOR_LABELS)
finetune_with_weights('color_ojos', eye_labels, EYE_COLOR_LABELS, class_weights=eye_weights.tolist())

# ── 4. Re-entrenar tono_piel_lab reducido a 5 clases ─────────────────
print('\n=== Paso 4: re-entrenar tono_piel_lab con 5 clases ===')
skin_labels = np.array([
    SKIN_TONE_MAP_7_TO_5[skin_df.loc[f, 'tono_piel_lab']] for f in all_fnames
])
finetune_with_weights('tono_piel_lab', skin_labels, SKIN_TONE_LABELS_5)

# ── 5. Guardar checkpoint mejorado ────────────────────────────────────
out_path = os.path.join(CHECKPOINT_DIR, 'final_model_v3.pt')
torch.save({'backbone': backbone.state_dict(), 'heads': heads.state_dict()}, out_path)
torch.save({'backbone': backbone.state_dict(), 'heads': heads.state_dict()},
           os.path.join(CHECKPOINT_DIR, 'final_model.pt'))
print(f'\nGuardado en {out_path} y final_model.pt actualizado.')
print(f'Total de cabezas: {len(heads)}')
print(f'Cabezas: {list(heads.keys())}')
