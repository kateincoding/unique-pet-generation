"""
Generates skin tone pseudo-labels for CelebA using analyze_skin_tone (LAB).
Output: celeba_skin_tone_pseudolabels.csv  ->  image_id, tono_piel_lab
Only processes the train split (partition == 0).
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

CELEBA_ROOT = './data/celeba'
IMG_DIR     = os.path.join(CELEBA_ROOT, 'img_align_celeba', 'img_align_celeba')
OUT_CSV     = './data/celeba/celeba_skin_tone_pseudolabels.csv'
TARGET_SIZE = 224

def analyze_skin_tone(face_bgr):
    h, w   = face_bgr.shape[:2]
    roi    = face_bgr[h//4:3*h//4, w//4:3*w//4]
    lab    = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L_mean = float(lab[:,:,0].mean())
    thresholds = [60, 85, 105, 125, 150, 185]
    labels     = ['muy_oscuro', 'oscuro', 'bronceado', 'oliva', 'medio', 'claro', 'muy_claro']
    for i, t in enumerate(thresholds):
        if L_mean < t:
            return labels[i]
    return labels[-1]

partitions = pd.read_csv(os.path.join(CELEBA_ROOT, 'list_eval_partition.csv'),
                         index_col='image_id')
train_imgs = partitions[partitions['partition'] == 0].index.tolist()
print(f'Train images: {len(train_imgs):,}')

results = []
errors  = 0

for fname in tqdm(train_imgs, desc='Generating skin tone pseudo-labels'):
    img_path = os.path.join(IMG_DIR, fname)
    if not os.path.exists(img_path):
        errors += 1
        continue
    try:
        img      = Image.open(img_path).convert('RGB').resize((TARGET_SIZE, TARGET_SIZE))
        face_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        tone     = analyze_skin_tone(face_bgr)
        results.append({'image_id': fname, 'tono_piel_lab': tone})
    except Exception:
        errors += 1
        results.append({'image_id': fname, 'tono_piel_lab': 'medio'})

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_CSV, index=False)

print(f'\nSaved: {OUT_CSV}')
print(f'Errors: {errors}')
print(df_out['tono_piel_lab'].value_counts())
