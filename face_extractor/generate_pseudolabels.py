"""
Genera pseudo-labels de color de ojos para CelebA usando
Landmarks + Sclera WB + HSV (analyze_eye_color_landmarks_wb).

Salida: celeba_eye_color_pseudolabels.csv
  image_id, color_ojos, n_valid

Solo se procesa el split de train (partition == 0).
"""
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

CELEBA_ROOT = './data/celeba'
IMG_DIR     = os.path.join(CELEBA_ROOT, 'img_align_celeba', 'img_align_celeba')
OUT_CSV     = './data/celeba/celeba_eye_color_pseudolabels.csv'
TARGET_SIZE = 224   # tamaño al que se redimensiona para el análisis

# ── Funciones de color de ojos (copiadas del notebook) ───────────────

def _classify_iris_hue(hm, sm, vm):
    if vm < 40:              return 'marron_oscuro'
    if sm < 20:              return 'gris'
    if hm < 16 or hm > 165: return 'marron'
    if 16 <= hm < 25:       return 'avellana'
    if 25 <= hm < 40:       return 'ambar'
    if 40 <= hm < 80:       return 'verde'
    if 80 <= hm < 140:      return 'azul'
    return 'marron'


def white_balance_sclera(face_bgr, eye_boxes):
    if not eye_boxes:
        return face_bgr.copy()
    sclera_pixels = []
    for (ex, ey, ew, eh) in eye_boxes:
        crop = face_bgr[ey:ey+eh, ex:ex+ew].astype(np.float32)
        if crop.size == 0:
            continue
        hsv_c = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2HSV)
        mask  = (hsv_c[:,:,2] > 175) & (hsv_c[:,:,1] < 45)
        if mask.sum() > 5:
            sclera_pixels.append(crop[mask])
    if not sclera_pixels:
        return face_bgr.copy()
    all_sc = np.concatenate(sclera_pixels, axis=0)
    avg    = all_sc.mean(axis=0)
    scale  = np.clip(avg.mean() / (avg + 1e-6), 0.6, 1.8)
    return np.clip(face_bgr.astype(np.float32) * scale[None, None, :], 0, 255).astype(np.uint8)


def analyze_eye_color_landmarks_wb(face_bgr, eye_landmarks):
    """Devuelve (color_str, n_valid_pixels)."""
    lx, ly = eye_landmarks['left_eye']
    rx, ry = eye_landmarks['right_eye']
    iod    = np.sqrt((lx - rx)**2 + (ly - ry)**2)
    ir     = max(6, int(iod / 11))
    pr     = max(3, int(ir * 0.42))
    h, w   = face_bgr.shape[:2]
    margin = int(ir * 1.8)

    eye_boxes_raw = []
    for (cx, cy) in [eye_landmarks['left_eye'], eye_landmarks['right_eye']]:
        x1 = max(0, cx - margin); y1 = max(0, cy - margin)
        x2 = min(w, cx + margin); y2 = min(h, cy + margin)
        if x2 > x1 and y2 > y1:
            eye_boxes_raw.append((x1, y1, x2-x1, y2-y1, cx, cy))

    if not eye_boxes_raw:
        return 'marron', 0

    face_wb = white_balance_sclera(face_bgr,
                                   [(x,y,ew,eh) for (x,y,ew,eh,_,__) in eye_boxes_raw])

    all_valid = []
    for (ex, ey, ew, eh, lm_cx, lm_cy) in eye_boxes_raw:
        crop = face_wb[ey:ey+eh, ex:ex+ew]
        if crop.size == 0:
            continue
        px, py = lm_cx - ex, lm_cy - ey
        mask   = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (px, py), max(3, int(ir*0.85)), 255, -1)
        cv2.circle(mask, (px, py), max(2, int(pr*1.10)),   0, -1)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hh, ss, vv = hsv[:,:,0].flatten(), hsv[:,:,1].flatten(), hsv[:,:,2].flatten()
        in_mask = mask.flatten() > 0
        valid   = (in_mask & (vv>30) & (vv<220) & (ss>15) & (hh>13)
                   & ~((vv>210) & (ss<25)))
        if valid.sum() > 0:
            all_valid.append(np.stack([hh[valid], ss[valid], vv[valid]], axis=1))

    if not all_valid:
        return 'marron', 0

    combined = np.concatenate(all_valid, axis=0)
    if len(combined) < 20:
        return 'marron', len(combined)

    weights = combined[:,1].astype(float) + 1.0
    hm = float(np.average(combined[:,0], weights=weights))
    sm = float(combined[:,1].mean())
    vm = float(combined[:,2].mean())
    return _classify_iris_hue(hm, sm, vm), len(combined)


# ── Carga de datos ────────────────────────────────────────────────────

partitions = pd.read_csv(os.path.join(CELEBA_ROOT, 'list_eval_partition.csv'),
                         index_col='image_id')
landmarks  = pd.read_csv(os.path.join(CELEBA_ROOT, 'list_landmarks_align_celeba.csv'),
                         index_col='image_id')

train_imgs = partitions[partitions['partition'] == 0].index.tolist()
print(f'Imagenes train: {len(train_imgs):,}')

# Escala para convertir coordenadas originales (178x218) -> 224x224
ORIG_W, ORIG_H = 178, 218
SX = TARGET_SIZE / ORIG_W
SY = TARGET_SIZE / ORIG_H

# ── Generar pseudo-labels ─────────────────────────────────────────────

results = []
errors  = 0

for fname in tqdm(train_imgs, desc='Generando pseudo-labels'):
    img_path = os.path.join(IMG_DIR, fname)
    if not os.path.exists(img_path):
        errors += 1
        continue

    try:
        img     = Image.open(img_path).convert('RGB').resize((TARGET_SIZE, TARGET_SIZE))
        face_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        lm  = landmarks.loc[fname]
        eye_landmarks = {
            'left_eye':  (int(lm['lefteye_x']  * SX), int(lm['lefteye_y']  * SY)),
            'right_eye': (int(lm['righteye_x'] * SX), int(lm['righteye_y'] * SY)),
        }

        color, n_valid = analyze_eye_color_landmarks_wb(face_bgr, eye_landmarks)
        results.append({'image_id': fname, 'color_ojos': color, 'n_valid': n_valid})

    except Exception:
        errors += 1
        results.append({'image_id': fname, 'color_ojos': 'marron', 'n_valid': 0})

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_CSV, index=False)

print(f'\nGuardado: {OUT_CSV}')
print(f'Errores: {errors}')
print(df_out['color_ojos'].value_counts())
