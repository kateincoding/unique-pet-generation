"""
Fine-tune de las cabezas color_ojos y tono_piel_lab usando los pseudo-labels
generados sobre CelebA. Estrategia optimizada:

  1. Subsamplea N imágenes de CelebA train para acotar el tiempo en CPU.
  2. Calcula los features del backbone una sola vez y los cachea en memoria.
  3. Entrena el linear head sobre los features cacheados (muy rápido).

Al final actualiza final_model.pt con las dos cabezas incorporadas y guarda
una copia versionada como final_model_full.pt. El modelo base se conserva
en final_model_base.pt.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

CELEBA_ROOT    = './data/celeba'
CHECKPOINT_DIR = './checkpoints'
BATCH_SIZE     = 64
LR             = 1e-3
EPOCHS         = 12
FEATURE_DIM    = 512
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SUBSAMPLE    = 20000   # número de imágenes para el fine-tune
VAL_FRACTION   = 0.15

EYE_COLOR_LABELS = ['marron_oscuro', 'gris', 'marron', 'avellana', 'ambar', 'verde', 'azul']
EYE_COLOR_IDX    = {c: i for i, c in enumerate(EYE_COLOR_LABELS)}
SKIN_TONE_LABELS = ['muy_oscuro', 'oscuro', 'bronceado', 'oliva', 'medio', 'claro', 'muy_claro']
SKIN_TONE_IDX    = {c: i for i, c in enumerate(SKIN_TONE_LABELS)}

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

backbone = FaceBackbone(FEATURE_DIM).to(DEVICE)
heads = nn.ModuleDict({
    name: AttributeHead(FEATURE_DIM, n).to(DEVICE) for name, n in HEADS_CONFIG.items()
})

ckpt_path = os.path.join(CHECKPOINT_DIR, 'final_model.pt')
ckpt = torch.load(ckpt_path, map_location=DEVICE)
backbone.load_state_dict(ckpt['backbone'])
heads.load_state_dict(ckpt['heads'])
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False
print(f'Checkpoint base cargado desde {ckpt_path}')

# Backup versión base si no existe ya
base_path = os.path.join(CHECKPOINT_DIR, 'final_model_base.pt')
if not os.path.exists(base_path):
    torch.save({'backbone': backbone.state_dict(), 'heads': heads.state_dict()}, base_path)
    print(f'Backup guardado en {base_path}')


# ── Selección común de imágenes para cachear features ────────────────
eye_df  = pd.read_csv(EYE_CSV,  index_col='image_id')
skin_df = pd.read_csv(SKIN_CSV, index_col='image_id')
common  = eye_df.index.intersection(skin_df.index)
print(f'Imágenes con ambas pseudo-etiquetas: {len(common):,}')

rng = np.random.default_rng(seed=42)
sample = rng.choice(common, size=min(N_SUBSAMPLE, len(common)), replace=False)
print(f'Sub-muestra para fine-tune: {len(sample):,}')

img_dir = os.path.join(CELEBA_ROOT, 'img_align_celeba', 'img_align_celeba')


class ImageOnlyDataset(Dataset):
    def __init__(self, file_names):
        self.file_names = list(file_names)
    def __len__(self): return len(self.file_names)
    def __getitem__(self, idx):
        fname = self.file_names[idx]
        img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
        return VAL_TF(img), fname


print('\n=== Calculando features del backbone (una sola pasada) ===')
ds = ImageOnlyDataset(sample)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

all_feats  = []
all_fnames = []
with torch.no_grad():
    for imgs, fnames in tqdm(dl, desc='Features'):
        imgs = imgs.to(DEVICE)
        feats = backbone(imgs)
        all_feats.append(feats.cpu())
        all_fnames.extend(fnames)

features_tensor = torch.cat(all_feats, dim=0)
print(f'Features tensor: {features_tensor.shape}')

# Mapeo fname -> idx en features_tensor
fname_to_idx = {f: i for i, f in enumerate(all_fnames)}

# Split train/val determinista
n_total = len(all_fnames)
n_val   = int(n_total * VAL_FRACTION)
perm    = rng.permutation(n_total)
val_ids   = perm[:n_val]
train_ids = perm[n_val:]
print(f'Train: {len(train_ids):,}  Val: {len(val_ids):,}')


def finetune_head(head_name, csv_path, labels, label_idx, label_col):
    print(f'\n=== Fine-tune {head_name} ===')
    df = pd.read_csv(csv_path, index_col='image_id')

    # Vector de etiquetas alineado con features_tensor
    labels_arr = np.array([
        label_idx.get(df.loc[fname, label_col], 0) for fname in all_fnames
    ], dtype=np.int64)
    label_tensor = torch.from_numpy(labels_arr)

    # Distribución de clases
    unique, counts = np.unique(labels_arr, return_counts=True)
    print('  Distribución de clases:')
    for u, c in zip(unique, counts):
        print(f'    {labels[u]:<15} {c:>6} ({c/len(labels_arr)*100:.1f}%)')

    if head_name not in heads:
        heads.add_module(head_name, AttributeHead(FEATURE_DIM, len(labels)).to(DEVICE))
        print(f'  Cabeza {head_name} añadida')

    train_x = features_tensor[train_ids].to(DEVICE)
    train_y = label_tensor[train_ids].to(DEVICE)
    val_x   = features_tensor[val_ids].to(DEVICE)
    val_y   = label_tensor[val_ids].to(DEVICE)

    head = heads[head_name]
    for p in head.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)

    n_train = len(train_x)
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        head.train()
        # Mini-batches sobre features ya cacheados (rapidísimo)
        perm_ep = torch.randperm(n_train, device=DEVICE)
        tr_loss, n_batches = 0.0, 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm_ep[start:start + BATCH_SIZE]
            xb, yb = train_x[idx], train_y[idx]
            logits = head(xb)
            loss   = F.cross_entropy(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr_loss += loss.item(); n_batches += 1

        head.eval()
        with torch.no_grad():
            preds = head(val_x).argmax(1)
            acc   = (preds == val_y).float().mean().item()
        best_acc = max(best_acc, acc)
        print(f'  Epoch {epoch:2d}/{EPOCHS}  loss={tr_loss/n_batches:.4f}  val_acc={acc:.3f}')

    print(f'  Mejor val_acc {head_name}: {best_acc:.3f}')


finetune_head('color_ojos',    EYE_CSV,  EYE_COLOR_LABELS,  EYE_COLOR_IDX,  'color_ojos')
finetune_head('tono_piel_lab', SKIN_CSV, SKIN_TONE_LABELS, SKIN_TONE_IDX, 'tono_piel_lab')

full_path = os.path.join(CHECKPOINT_DIR, 'final_model_full.pt')
torch.save({'backbone': backbone.state_dict(), 'heads': heads.state_dict()}, full_path)
print(f'\nfinal_model_full.pt guardado con {len(heads)} cabezas')

# Sobreescribir también final_model.pt
torch.save({'backbone': backbone.state_dict(), 'heads': heads.state_dict()},
           os.path.join(CHECKPOINT_DIR, 'final_model.pt'))
print('final_model.pt actualizado (versión full)')
