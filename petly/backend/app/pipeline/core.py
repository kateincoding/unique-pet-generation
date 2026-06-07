# -*- coding: utf-8 -*-
"""Petly: pipeline de analisis facial (portado, verbatim, del notebook verificado).

Cara RGB -> 18 rasgos faciales (color de pelo/ojos/piel, forma, gafas, etc.).
Modelos (ViT timm + CNN de Agus) + geometria MediaPipe + CLIP zero-shot, por voto suave.
Los pesos de los modelos se cargan de forma perezosa en la primera llamada a analizar().
"""
import os
import math
import json
import hashlib

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tv_models

import mediapipe as mp
import timm

# --- configuracion (rutas y device por variables de entorno) ---
device = os.environ.get("PETLY_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
_DEFAULT_MODELS_DIR = r"c:/Users/Paulina Peralta/Desktop/gen-pet"
MODELOS_DIR = os.environ.get("PETLY_MODELS_DIR", _DEFAULT_MODELS_DIR)
MODELOS_PT = {
    "resnet_kat": os.path.join(MODELOS_DIR, "resnet-18-kat.pt"),
    "vit_kat":    os.path.join(MODELOS_DIR, "vit_multihead_best-kat.pt"),
    "agus":       os.path.join(MODELOS_DIR, "final_model-agus.pt"),
}

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# ===================== definiciones verbatim del notebook =====================


# ---- CLASS_LABELS = { ----

CLASS_LABELS = {
    'color_pelo':    ['negro', 'castano', 'rubio', 'pelirrojo', 'gris', 'calvo'],
    'textura_pelo':  ['liso', 'ondulado', 'rizado', 'muy_rizado'],
    'longitud_pelo': ['corto', 'medio', 'largo', 'calvo'],
    'cejas':         ['normales', 'arqueadas', 'pobladas', 'finas', 'rectas'],
    'forma_ojos':    ['almendrada', 'redonda', 'rasgada', 'caida', 'prominente'],
    'tamano_nariz':  ['pequena', 'mediana', 'grande'],
    'forma_nariz':   ['recta', 'aguilena', 'respingona', 'ancha'],
    'grosor_labios': ['finos', 'medianos', 'carnosos'],
    'pomulos':       ['planos', 'normales', 'altos', 'prominentes'],
    'mandibula':     ['suave', 'marcada', 'ancha', 'estrecha'],
    'barbilla':      ['redonda', 'puntiaguda', 'cuadrada', 'hendida'],
    'forma_cara':    ['oval', 'redonda', 'cuadrada', 'corazon', 'diamante', 'oblonga'],
    'vello_facial':  ['sin_barba', 'barba_corta', 'barba_larga', 'bigote'],
    'gafas':         [False, True],
    'pecas':         [False, True],
    'tono_piel':     ['muy_claro', 'claro', 'medio', 'oliva', 'bronceado', 'oscuro', 'muy_oscuro'],
    'rango_edad':    ['nino', 'joven', 'adulto', 'maduro', 'mayor'],
    'color_ojos':    ['azul', 'verde', 'avellana', 'marron', 'marron_oscuro', 'gris', 'negro'],
}

# ---- def muestrear_color ----

def punto_a_xy(landmarks, idx, ancho, alto):
    return np.array([landmarks[idx].x * ancho, landmarks[idx].y * alto])

def distancia_landmarks(landmarks, idx_a, idx_b, ancho, alto):
    a = punto_a_xy(landmarks, idx_a, ancho, alto)
    b = punto_a_xy(landmarks, idx_b, ancho, alto)
    return np.linalg.norm(a - b)

def muestrear_color(imagen_rgb, landmarks, indices, radio: int = 4):
    h, w = imagen_rgb.shape[:2]
    pixeles = []
    for idx in indices:
        x = min(w - 1, max(0, int(landmarks[idx].x * w)))
        y = min(h - 1, max(0, int(landmarks[idx].y * h)))
        x1, y1 = max(0, x - radio), max(0, y - radio)
        x2, y2 = min(w, x + radio), min(h, y + radio)
        parche = imagen_rgb[y1:y2, x1:x2]
        if parche.size > 0:
            # mediana: descarta outliers (esclerotica blanca, reflejos, pelo de fondo)
            pixeles.append(np.median(parche.reshape(-1, 3), axis=0))
    if not pixeles:
        return None
    return np.median(pixeles, axis=0)

def _a_lab(rgb):
    arr = np.clip(np.array(rgb), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)

def color_a_nombre(rgb, paleta):
    # Distancia perceptual (Delta-E en LAB), mas estable ante iluminacion que el RGB crudo.
    muestra = _a_lab(rgb)
    distancias = {nombre: np.linalg.norm(muestra - _a_lab(ref)) for nombre, ref in paleta.items()}
    return min(distancias, key=distancias.get)

# ---- IRIS_DERECHO = [ ----

PALETA_OJOS = {
    'azul':          (70, 130, 180),
    'verde':         (90, 140, 90),
    'avellana':      (140, 110, 70),
    'marron':        (90, 60, 40),
    'marron_oscuro': (50, 30, 20),
    'gris':          (130, 130, 130),
    'negro':         (25, 25, 25),
}

PALETA_PIEL = {
    'muy_claro':  (240, 220, 200),
    'claro':      (220, 190, 170),
    'medio':      (200, 160, 130),
    'oliva':      (180, 150, 110),
    'bronceado':  (170, 120, 90),
    'oscuro':     (110, 70, 50),
    'muy_oscuro': (70, 40, 30),
}

PALETA_PELO = {
    'negro':       (30, 25, 25),
    'castano':     (90, 65, 50),
    'rubio':       (200, 170, 110),
    'pelirrojo':   (160, 80, 50),
    'gris':        (150, 150, 150),
}

IRIS_DERECHO = [468, 469, 470, 471, 472]

IRIS_IZQUIERDO = [473, 474, 475, 476, 477]

MEJILLA_DERECHA = [50, 101, 36, 205]

MEJILLA_IZQUIERDA = [280, 330, 266, 425]

# ---- def extraer_colores_base ----

def muestrear_iris_robusto(imagen_rgb, landmarks, indices_iris, radio: int = 3):
    # Muestrea el iris evitando que el verde/azul colapse a marron: junta pixeles
    # alrededor de los landmarks del iris, descarta los muy oscuros (pupila/sombra)
    # y los muy brillantes (reflejo especular) y toma la mediana del resto (v15).
    h, w = imagen_rgb.shape[:2]
    pix = []
    for idx in indices_iris:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        x1, y1 = max(0, x - radio), max(0, y - radio)
        x2, y2 = min(w, x + radio), min(h, y + radio)
        parche = imagen_rgb[y1:y2, x1:x2].reshape(-1, 3)
        if parche.size:
            pix.append(parche)
    if not pix:
        return None
    pix = np.concatenate(pix, axis=0).astype(np.float32)
    vmax = pix.max(axis=1) / 255.0
    mask = (vmax > 0.18) & (vmax < 0.92)
    filtrados = pix[mask] if int(mask.sum()) >= 5 else pix
    return np.median(filtrados, axis=0)


def muestrear_pelo_robusto(imagen_rgb, landmarks, color_piel_rgb):
    # Varios parches en la zona del pelo (arriba y a los costados de la frente),
    # descarta los que se parecen a la piel y toma la mediana (robusto, v15).
    h, w = imagen_rgb.shape[:2]
    frente = landmarks[10]
    menton = landmarks[152]
    altura_cara = abs(menton.y - frente.y) * h
    cx, cy = frente.x * w, frente.y * h
    offsets = [
        (0.00, -0.12), (0.00, -0.18), (0.00, -0.25),
        (-0.10, -0.15), (0.10, -0.15),
        (-0.15, -0.08), (0.15, -0.08),
    ]
    parches = []
    for dx, dy in offsets:
        x = int(cx + dx * altura_cara)
        y = int(cy + dy * altura_cara)
        if not (0 <= x < w and 0 <= y < h):
            continue
        x1, y1 = max(0, x - 10), max(0, y - 10)
        x2, y2 = min(w, x + 10), min(h, y + 10)
        parche = imagen_rgb[y1:y2, x1:x2]
        if parche.size == 0:
            continue
        parches.append(parche.reshape(-1, 3).mean(axis=0))
    if not parches:
        return None
    parches = np.array(parches)
    if color_piel_rgb is not None:
        dist = np.linalg.norm(parches - np.array(color_piel_rgb), axis=1)
        no_piel = parches[dist > 40]
        if len(no_piel) >= 2:
            parches = no_piel
    return np.median(parches, axis=0)


def extraer_colores_base(imagen_rgb, landmarks):
    # Iris: muestreo robusto (descarta pupila y reflejo) para no perder verdes/azules.
    color_ojos_rgb = muestrear_iris_robusto(imagen_rgb, landmarks, IRIS_DERECHO + IRIS_IZQUIERDO, radio=3)
    color_piel_rgb = muestrear_color(imagen_rgb, landmarks, MEJILLA_DERECHA + MEJILLA_IZQUIERDA, radio=8)
    # Pelo: varios parches, descarta piel, mediana.
    color_pelo_rgb = muestrear_pelo_robusto(imagen_rgb, landmarks, color_piel_rgb)

    return {
        'color_ojos': color_a_nombre(color_ojos_rgb, PALETA_OJOS) if color_ojos_rgb is not None else 'marron',
        'tono_piel':  color_a_nombre(color_piel_rgb, PALETA_PIEL) if color_piel_rgb is not None else 'medio',
        'color_pelo': color_a_nombre(color_pelo_rgb, PALETA_PELO) if color_pelo_rgb is not None else 'castano',
        '_color_ojos_rgb': tuple(int(v) for v in color_ojos_rgb) if color_ojos_rgb is not None else None,
        '_color_piel_rgb': tuple(int(v) for v in color_piel_rgb) if color_piel_rgb is not None else None,
        '_color_pelo_rgb': tuple(int(v) for v in color_pelo_rgb) if color_pelo_rgb is not None else None,
    }

# ---- def calcular_forma_cara ----

def calcular_forma_cara(imagen_rgb, landmarks):
    h, w = imagen_rgb.shape[:2]
    altura       = distancia_landmarks(landmarks, 10, 152, w, h)
    ancho_pom    = distancia_landmarks(landmarks, 234, 454, w, h)
    ancho_mand   = distancia_landmarks(landmarks, 172, 397, w, h)
    ancho_frente = distancia_landmarks(landmarks, 21, 251, w, h)

    denom = max(ancho_pom, 1e-6)  # evita division por cero con landmarks degenerados
    ratio_aspecto = altura / denom
    ratio_mand    = ancho_mand / denom
    ratio_frente  = ancho_frente / denom

    if ratio_aspecto > 1.5:
        forma = 'oblonga'
    elif ratio_aspecto < 1.05 and ratio_mand > 0.85:
        forma = 'redonda'
    elif ratio_mand > 0.95 and ratio_aspecto < 1.25:
        forma = 'cuadrada'
    elif ratio_frente > 0.9 and ratio_mand < 0.75:
        forma = 'corazon'
    elif ratio_frente < 0.75 and ratio_mand < 0.75:
        forma = 'diamante'
    else:
        forma = 'oval'

    return {
        'forma_cara': forma,
        '_ratio_aspecto':  round(float(ratio_aspecto), 3),
        '_ratio_mandibula': round(float(ratio_mand), 3),
        '_ratio_frente':   round(float(ratio_frente), 3),
    }

# ---- def calcular_forma_ojos ----

def calcular_forma_ojos(imagen_rgb, landmarks):
    h, w = imagen_rgb.shape[:2]

    ancho_d = distancia_landmarks(landmarks, 33, 133, w, h)
    alto_d  = distancia_landmarks(landmarks, 159, 145, w, h)
    ancho_i = distancia_landmarks(landmarks, 362, 263, w, h)
    alto_i  = distancia_landmarks(landmarks, 386, 374, w, h)

    ratio_ar_d = ancho_d / max(alto_d, 1e-6)
    ratio_ar_i = ancho_i / max(alto_i, 1e-6)
    ratio_ar = (ratio_ar_d + ratio_ar_i) / 2

    # 33 = canto externo, 133 = canto interno (ojo derecho). Mantenemos el mismo
    # vector/angulo de antes; solo corregimos la asignacion de etiquetas, que estaba al reves.
    p_ext = punto_a_xy(landmarks, 33, w, h)
    p_int = punto_a_xy(landmarks, 133, w, h)
    delta = p_ext - p_int
    angulo = math.degrees(math.atan2(delta[1], delta[0]))

    if ratio_ar > 3.2 and angulo > 3:
        forma = 'rasgada'
    elif ratio_ar > 3.2 and angulo < -3:
        forma = 'caida'
    elif ratio_ar < 2.4:
        forma = 'redonda'
    elif alto_d / max(ancho_d, 1e-6) > 0.45:
        forma = 'prominente'
    else:
        forma = 'almendrada'

    return {
        'forma_ojos': forma,
        '_ratio_ojo_ar': round(float(ratio_ar), 3),
        '_angulo_canto': round(float(angulo), 2),
    }

# ---- def calcular_forma_nariz ----

def calcular_forma_nariz(imagen_rgb, landmarks):
    h, w = imagen_rgb.shape[:2]

    altura_nariz = distancia_landmarks(landmarks, 8, 2, w, h)
    altura_cara  = distancia_landmarks(landmarks, 10, 152, w, h)
    ancho_nariz  = distancia_landmarks(landmarks, 49, 279, w, h)
    ancho_cara   = distancia_landmarks(landmarks, 234, 454, w, h)

    ratio_altura = altura_nariz / max(altura_cara, 1e-6)
    ratio_ancho  = ancho_nariz / max(ancho_cara, 1e-6)

    if ratio_altura < 0.27:
        tamano = 'pequena'
    elif ratio_altura > 0.34:
        tamano = 'grande'
    else:
        tamano = 'mediana'

    # 4 = punta real de la nariz, 2 = subnasal (debajo), 6 = puente.
    # Normalizamos por la escala facial -> invariante a la resolucion. Umbrales aproximados.
    puente   = punto_a_xy(landmarks, 6, w, h)
    tip      = punto_a_xy(landmarks, 4, w, h)
    subnasal = punto_a_xy(landmarks, 2, w, h)
    escala = max(altura_cara, 1e-6)
    levanta_norm = (subnasal[1] - tip[1]) / escala   # >0: punta por encima del subnasal (respingona)
    desplaz_norm = (tip[0] - puente[0]) / escala

    if ratio_ancho > 0.32:
        forma = 'ancha'
    elif levanta_norm > 0.012:
        forma = 'respingona'
    elif levanta_norm < -0.012:
        forma = 'aguilena'
    else:
        forma = 'recta'

    return {
        'tamano_nariz': tamano,
        'forma_nariz':  forma,
        '_ratio_altura_nariz': round(float(ratio_altura), 3),
        '_ratio_ancho_nariz':  round(float(ratio_ancho), 3),
        '_levanta_norm': round(float(levanta_norm), 4),
        '_desplaz_norm': round(float(desplaz_norm), 4),
    }

# ---- def calcular_grosor_labios ----

def calcular_grosor_labios(imagen_rgb, landmarks):
    h, w = imagen_rgb.shape[:2]

    alto_sup = distancia_landmarks(landmarks, 0, 13, w, h)
    alto_inf = distancia_landmarks(landmarks, 14, 17, w, h)
    ancho_boca = distancia_landmarks(landmarks, 61, 291, w, h)

    grosor_norm = (alto_sup + alto_inf) / max(ancho_boca, 1e-6)

    if grosor_norm < 0.22:
        grosor = 'finos'
    elif grosor_norm > 0.34:
        grosor = 'carnosos'
    else:
        grosor = 'medianos'

    return {
        'grosor_labios': grosor,
        '_grosor_norm': round(float(grosor_norm), 3),
    }

# ---- def calcular_estructura_inferior ----

def calcular_estructura_inferior(imagen_rgb, landmarks):
    h, w = imagen_rgb.shape[:2]

    frente = punto_a_xy(landmarks, 10, w, h)
    menton = punto_a_xy(landmarks, 152, w, h)
    altura_cara = max(menton[1] - frente[1], 1e-6)

    # Posicion vertical del pomulo: promediamos ambos lados (robusto a yaw).
    pomulo_der = punto_a_xy(landmarks, 234, w, h)
    pomulo_izq = punto_a_xy(landmarks, 454, w, h)
    y_pomulo = (pomulo_der[1] + pomulo_izq[1]) / 2.0
    pos_pomulo = (y_pomulo - frente[1]) / altura_cara

    ancho_pom  = distancia_landmarks(landmarks, 234, 454, w, h)
    ancho_mand = distancia_landmarks(landmarks, 172, 397, w, h)
    prom_pomulo = ancho_pom / max(ancho_mand, 1e-6)   # pomulos anchos vs mandibula -> mas marcados

    if prom_pomulo > 1.4 and pos_pomulo < 0.42:
        pomulos = 'prominentes'
    elif pos_pomulo < 0.35:
        pomulos = 'altos'
    elif pos_pomulo > 0.48:
        pomulos = 'planos'
    else:
        pomulos = 'normales'

    ratio_mand = ancho_mand / max(ancho_pom, 1e-6)
    if ratio_mand < 0.72:
        mandibula = 'estrecha'
    elif ratio_mand > 0.92:
        mandibula = 'ancha'
    elif ratio_mand > 0.85:
        mandibula = 'marcada'
    else:
        mandibula = 'suave'

    p_izq = punto_a_xy(landmarks, 176, w, h)
    p_med = punto_a_xy(landmarks, 152, w, h)
    p_der = punto_a_xy(landmarks, 400, w, h)
    v1 = p_izq - p_med
    v2 = p_der - p_med
    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angulo_barbilla = math.degrees(math.acos(np.clip(cos_ang, -1.0, 1.0)))

    # 'hendida' (barbilla partida) es un rasgo de profundidad: lo cubre CLIP, no la geometria 2D.
    if angulo_barbilla < 110:
        barbilla = 'puntiaguda'
    elif angulo_barbilla > 150:
        barbilla = 'cuadrada'
    else:
        barbilla = 'redonda'

    return {
        'pomulos':   pomulos,
        'mandibula': mandibula,
        'barbilla':  barbilla,
        '_pos_pomulo': round(float(pos_pomulo), 3),
        '_prom_pomulo': round(float(prom_pomulo), 3),
        '_ratio_mand': round(float(ratio_mand), 3),
        '_angulo_barbilla': round(float(angulo_barbilla), 2),
    }

# ---- def extraer_rasgos_geometricos ----

def extraer_rasgos_geometricos(imagen_rgb, landmarks):
    rasgos = {}
    rasgos.update(extraer_colores_base(imagen_rgb, landmarks))
    rasgos.update(calcular_forma_cara(imagen_rgb, landmarks))
    rasgos.update(calcular_forma_ojos(imagen_rgb, landmarks))
    rasgos.update(calcular_forma_nariz(imagen_rgb, landmarks))
    rasgos.update(calcular_grosor_labios(imagen_rgb, landmarks))
    rasgos.update(calcular_estructura_inferior(imagen_rgb, landmarks))
    return rasgos

# ---- class ViTMultiHead ----

class ViTMultiHead(nn.Module):
    """Backbone ViT-B/16 (timm) -> embedding(256) -> una head Linear por atributo."""
    def __init__(self, heads_config):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.embedding = nn.Sequential(nn.Linear(768, 256))
        self.heads = nn.ModuleDict({n: nn.Linear(256, c) for n, c in heads_config.items()})

    def forward(self, x):
        e = self.embedding(self.backbone(x))
        return {n: h(e) for n, h in self.heads.items()}

# ---- class _BloqueConv ----

class _BloqueConv(nn.Module):
    def __init__(self, ent, sal):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ent, sal, 3, padding=1, bias=False),
            nn.BatchNorm2d(sal),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)

class AgusBackbone(nn.Module):
    """CNN custom de 5 bloques conv+BN + pooling global adaptativo (salida 512-dim)."""
    def __init__(self):
        super().__init__()
        canales = [(3, 32), (32, 64), (64, 128), (128, 256), (256, 512)]
        self.features = nn.ModuleList([_BloqueConv(i, o) for i, o in canales])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        for bloque in self.features:
            x = bloque(x)
        return self.pool(x).flatten(1)

# ---- class AgusMultiHead ----

class AgusMultiHead(nn.Module):
    """Backbone custom de Agus + una head Linear por atributo."""
    def __init__(self, heads_config):
        super().__init__()
        self.backbone = AgusBackbone()
        self.heads = nn.ModuleDict({n: nn.Linear(512, c) for n, c in heads_config.items()})

    def forward(self, x):
        f = self.backbone(x)
        return {n: h(f) for n, h in self.heads.items()}

# ---- def _strip_module ----

def _strip_module(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

def _remap_keys(checkpoint, fmt):
    if fmt == "vit":
        sd = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        return _strip_module(sd)
    if fmt == "agus":
        sd = {}
        for k, v in checkpoint["backbone"].items():
            sd[f"backbone.{k}"] = v
        for k, v in checkpoint["heads"].items():          # "color_pelo.fc.weight" -> "heads.color_pelo.weight"
            attr, _fc, wb = k.split(".")
            sd[f"heads.{attr}.{wb}"] = v
        return sd
    raise ValueError(f"Formato desconocido: {fmt}")

# ---- def cargar_modelo_estricto ----

def cargar_modelo_estricto(modelo, ruta_pt, fmt):
    if not os.path.exists(ruta_pt):
        raise FileNotFoundError(f"No existe el checkpoint: {ruta_pt}")
    ck = torch.load(ruta_pt, map_location=device, weights_only=False)
    sd = _remap_keys(ck, fmt)
    missing, unexpected = modelo.load_state_dict(sd, strict=False)
    heads_missing = [m for m in missing if m.startswith("heads.")]
    if heads_missing:
        raise RuntimeError(
            f"{os.path.basename(ruta_pt)}: {len(heads_missing)} heads NO cargaron "
            f"(pesos aleatorios). Ej: {heads_missing[:3]}"
        )
    backbone_missing = [m for m in missing if not m.startswith("heads.")]
    print(f"  OK {os.path.basename(ruta_pt)}: heads cargadas, "
          f"backbone_missing={len(backbone_missing)}, unexpected={len(unexpected)}")
    return modelo.to(device).eval()

# ---- TRANSFORM_IMAGENET = T.Compose ----

TRANSFORM_IMAGENET = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predecir_probabilidades(modelo, imagen_rgb, transform=TRANSFORM_IMAGENET):
    if modelo is None:
        return {}
    pil_img = Image.fromarray(imagen_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        salidas = modelo(tensor)
    probs = {}
    for nombre_head, logits in salidas.items():
        p = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        probs[nombre_head] = p
    return probs

# ---- VERIFICACIONES_CLIP = { ----

VERIFICACIONES_CLIP = {
    # === RASGOS BINARIOS ===
    'gafas': {
        'queries': [
            'a photo of a person clearly wearing eyeglasses',
            'a photo of a person without any eyeglasses',
        ],
        'clases_objetivo': [True, False],
    },
    'pecas': {
        'queries': [
            'a photo of a face with visible freckles on the skin',
            'a photo of a face with clear skin without freckles',
        ],
        'clases_objetivo': [True, False],
    },

    # === VELLO FACIAL Y EDAD ===
    'vello_facial': {
        'queries': [
            'a photo of a clean-shaven face without any beard or mustache',
            'a photo of a face with a short beard or stubble',
            'a photo of a face with a long thick beard',
            'a photo of a face with only a mustache and no beard',
        ],
        'clases_objetivo': ['sin_barba', 'barba_corta', 'barba_larga', 'bigote'],
    },
    'rango_edad': {
        'queries': [
            'a photo of a young child face, age between 0 and 12 years old',
            'a photo of a young adult face, age between 13 and 25 years old',
            'a photo of an adult face, age between 26 and 45 years old',
            'a photo of a mature adult face, age between 46 and 65 years old',
            'a photo of an elderly person face, age above 65 years old',
        ],
        'clases_objetivo': ['nino', 'joven', 'adulto', 'maduro', 'mayor'],
    },

    # === PELO ===
    'color_pelo': {
        'queries': [
            'a photo of a person with black hair',
            'a photo of a person with brown hair',
            'a photo of a person with blonde hair',
            'a photo of a person with red hair',
            'a photo of a person with gray hair',
            'a photo of a bald person without hair',
        ],
        'clases_objetivo': ['negro', 'castano', 'rubio', 'pelirrojo', 'gris', 'calvo'],
    },
    'textura_pelo': {
        'queries': [
            'a photo of a person with straight smooth hair',
            'a photo of a person with wavy hair',
            'a photo of a person with curly hair',
            'a photo of a person with very tight curly hair',
        ],
        'clases_objetivo': ['liso', 'ondulado', 'rizado', 'muy_rizado'],
    },
    'longitud_pelo': {
        'queries': [
            'a photo of a person with very short hair',
            'a photo of a person with medium length hair to the shoulders',
            'a photo of a person with long hair below the shoulders',
            'a photo of a bald person without hair',
        ],
        'clases_objetivo': ['corto', 'medio', 'largo', 'calvo'],
    },

    # === CEJAS ===
    'cejas': {
        'queries': [
            'a photo of a face with normal regular eyebrows',
            'a photo of a face with strongly arched curved eyebrows',
            'a photo of a face with very thick bushy eyebrows',
            'a photo of a face with very thin sparse eyebrows',
            'a photo of a face with straight horizontal eyebrows',
        ],
        'clases_objetivo': ['normales', 'arqueadas', 'pobladas', 'finas', 'rectas'],
    },

    # === FORMAS ANATOMICAS ===
    'forma_cara': {
        'queries': [
            'a photo of a person with an oval shaped face',
            'a photo of a person with a round shaped face',
            'a photo of a person with a square shaped face',
            'a photo of a person with a heart shaped face',
            'a photo of a person with a diamond shaped face',
            'a photo of a person with a long oblong shaped face',
        ],
        'clases_objetivo': ['oval', 'redonda', 'cuadrada', 'corazon', 'diamante', 'oblonga'],
    },
    'forma_ojos': {
        'queries': [
            'a photo of a face with almond shaped eyes',
            'a photo of a face with round large eyes',
            'a photo of a face with slanted upward eyes',
            'a photo of a face with downturned eyes',
            'a photo of a face with very prominent bulging eyes',
        ],
        'clases_objetivo': ['almendrada', 'redonda', 'rasgada', 'caida', 'prominente'],
    },
    'tamano_nariz': {
        'queries': [
            'a photo of a face with a small petite nose',
            'a photo of a face with a medium sized nose',
            'a photo of a face with a large big nose',
        ],
        'clases_objetivo': ['pequena', 'mediana', 'grande'],
    },
    'forma_nariz': {
        'queries': [
            'a photo of a face with a straight nose',
            'a photo of a face with a hooked aquiline nose',
            'a photo of a face with an upturned button nose',
            'a photo of a face with a wide flat nose',
        ],
        'clases_objetivo': ['recta', 'aguilena', 'respingona', 'ancha'],
    },
    'grosor_labios': {
        'queries': [
            'a photo of a face with thin narrow lips',
            'a photo of a face with medium average lips',
            'a photo of a face with thick plump lips',
        ],
        'clases_objetivo': ['finos', 'medianos', 'carnosos'],
    },
    'pomulos': {
        'queries': [
            'a photo of a face with flat low cheekbones',
            'a photo of a face with normal cheekbones',
            'a photo of a face with high cheekbones',
            'a photo of a face with very prominent sharp cheekbones',
        ],
        'clases_objetivo': ['planos', 'normales', 'altos', 'prominentes'],
    },
    'mandibula': {
        'queries': [
            'a photo of a face with a soft round jawline',
            'a photo of a face with a defined sharp jawline',
            'a photo of a face with a wide broad jaw',
            'a photo of a face with a narrow thin jaw',
        ],
        'clases_objetivo': ['suave', 'marcada', 'ancha', 'estrecha'],
    },
    'barbilla': {
        'queries': [
            'a photo of a face with a round soft chin',
            'a photo of a face with a pointed sharp chin',
            'a photo of a face with a square wide chin',
            'a photo of a face with a cleft dimpled chin',
        ],
        'clases_objetivo': ['redonda', 'puntiaguda', 'cuadrada', 'hendida'],
    },
}

# ---- def predecir_con_clip ----

def predecir_con_clip(imagen_rgb):
    pil_img = Image.fromarray(imagen_rgb)
    probs_por_rasgo = {}
    for nombre_rasgo, config_rasgo in VERIFICACIONES_CLIP.items():
        clases_vocab = CLASS_LABELS.get(nombre_rasgo)
        if clases_vocab is None:
            print(f"[AVISO] '{nombre_rasgo}' esta en VERIFICACIONES_CLIP pero no en CLASS_LABELS; se omite.")
            continue
        queries = config_rasgo['queries']
        clases_objetivo = config_rasgo['clases_objetivo']

        inputs = clip_processor(text=queries, images=pil_img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            probs_queries = clip_model(**inputs).logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy()

        vector_aligned = np.zeros(len(clases_vocab), dtype=np.float32)
        for idx_query, clase_objetivo in enumerate(clases_objetivo):
            if clase_objetivo in clases_vocab:
                vector_aligned[clases_vocab.index(clase_objetivo)] = probs_queries[idx_query]

        suma = vector_aligned.sum()
        if suma == 0:
            vector_aligned = np.ones(len(clases_vocab), dtype=np.float32) / len(clases_vocab)
        else:
            vector_aligned = vector_aligned / suma
        probs_por_rasgo[nombre_rasgo] = vector_aligned

    return probs_por_rasgo

# ---- def geometria_a_probabilidades ----

def geometria_a_probabilidades(rasgos_geometria, confianza_base: float = 0.7):
    probs = {}
    for nombre_rasgo, clases in CLASS_LABELS.items():
        valor = rasgos_geometria.get(nombre_rasgo)
        if valor is None or valor not in clases:
            continue
        idx = clases.index(valor)
        n = len(clases)
        prob_resto = (1.0 - confianza_base) / (n - 1)
        vector = np.full(n, prob_resto, dtype=np.float32)
        vector[idx] = confianza_base
        probs[nombre_rasgo] = vector
    return probs

# ---- PESOS_DEFAULT = { ----

PESOS_DEFAULT = {
    'vit_kat':    1.0,
    'agus':       0.8,
    'geometria':  1.2,
    'clip':       0.0,
}

PESOS_POR_RASGO = {
    'color_ojos':    {'vit_kat': 0.8, 'agus': 0.6, 'geometria': 2.5, 'clip': 0.0},
    'tono_piel':     {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 2.0, 'clip': 0.0},
    'gafas':         {'vit_kat': 0.7, 'agus': 0.7, 'geometria': 0.0, 'clip': 2.5},
    'pecas':         {'vit_kat': 0.7, 'agus': 0.7, 'geometria': 0.0, 'clip': 2.5},
    'vello_facial':  {'vit_kat': 0.7, 'agus': 0.7, 'geometria': 0.0, 'clip': 2.5},
    'rango_edad':    {'vit_kat': 0.8, 'agus': 0.8, 'geometria': 0.0, 'clip': 2.0},
    'color_pelo':    {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 1.0, 'clip': 1.8},
    'textura_pelo':  {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 0.0, 'clip': 1.8},
    'longitud_pelo': {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 0.0, 'clip': 1.8},
    'cejas':         {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 0.0, 'clip': 1.5},
    'forma_cara':    {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 1.0, 'clip': 1.3},
    'forma_ojos':    {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 1.0, 'clip': 1.3},
    'tamano_nariz':  {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 1.0, 'clip': 1.3},
    'forma_nariz':   {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 1.0, 'clip': 1.3},
    'grosor_labios': {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 1.0, 'clip': 1.3},
    'pomulos':       {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 1.0, 'clip': 1.3},
    'mandibula':     {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 1.0, 'clip': 1.3},
    'barbilla':      {'vit_kat': 1.0, 'agus': 0.8, 'geometria': 1.0, 'clip': 1.3},
}

def obtener_pesos(nombre_rasgo: str) -> dict:
    return PESOS_POR_RASGO.get(nombre_rasgo, PESOS_DEFAULT)

# ---- def voto_suave_unificado ----

DEFAULTS = {
    'textura_pelo':   'liso',
    'longitud_pelo':  'medio',
    'cejas':          'normales',
    'vello_facial':   'sin_barba',
    'gafas':          False,
    'pecas':          False,
    'rango_edad':     'adulto',
}

def voto_suave_unificado(predicciones_probs: dict, rasgos_geometria: dict,
                         predicciones_clip: dict = None,
                         confianza_geom: float = 0.7):
    probs_geom = geometria_a_probabilidades(rasgos_geometria, confianza_base=confianza_geom)
    todas_fuentes = {**predicciones_probs, 'geometria': probs_geom}
    if predicciones_clip is not None:
        todas_fuentes['clip'] = predicciones_clip

    rasgos = {}
    confianzas = {}
    contribuciones = {}

    for nombre_rasgo, clases in CLASS_LABELS.items():
        pesos = obtener_pesos(nombre_rasgo)
        probs_promedio = np.zeros(len(clases), dtype=np.float32)
        peso_total = 0.0
        fuentes_activas = []

        for nombre_fuente, preds in todas_fuentes.items():
            if nombre_rasgo not in preds:
                continue
            peso = pesos.get(nombre_fuente, 0.0)
            if peso == 0.0:
                continue
            probs_promedio += peso * preds[nombre_rasgo]
            peso_total += peso
            fuentes_activas.append(nombre_fuente)

        if peso_total > 0:
            probs_promedio /= peso_total
            idx = int(probs_promedio.argmax())
            # clases[idx] ya es del tipo correcto (bool para gafas/pecas, str para el resto).
            rasgos[nombre_rasgo] = clases[idx]
            confianzas[nombre_rasgo] = float(probs_promedio[idx])
            contribuciones[nombre_rasgo] = fuentes_activas
        else:
            rasgos[nombre_rasgo] = DEFAULTS.get(nombre_rasgo)
            confianzas[nombre_rasgo] = 0.0
            contribuciones[nombre_rasgo] = ['default']

    return rasgos, confianzas, contribuciones



# ===================== carga perezosa + orquestador =====================
_estado = {"loaded": False}
modelos = {}
clip_model = None
clip_processor = None
face_mesh = None
HEADS_CONFIG = {}
ATRIBUTOS_MODELO = []
VAL_ACCS_VIT = {}


def cargar_pipeline():
    """Carga modelos, CLIP y MediaPipe una sola vez (perezoso)."""
    global modelos, clip_model, clip_processor, face_mesh
    global HEADS_CONFIG, ATRIBUTOS_MODELO, VAL_ACCS_VIT
    if _estado["loaded"]:
        return

    meta = torch.load(MODELOS_PT["vit_kat"], map_location="cpu", weights_only=False)
    ATRIBUTOS_MODELO = list(meta["class_labels"].keys())
    VAL_ACCS_VIT = {k: float(v) for k, v in meta["val_accs"].items()}
    HEADS_CONFIG = {k: len(CLASS_LABELS[k]) for k in ATRIBUTOS_MODELO}
    del meta

    modelos = {}
    for nombre, ctor, fmt in [("vit_kat", ViTMultiHead, "vit"), ("agus", AgusMultiHead, "agus")]:
        try:
            modelos[nombre] = cargar_modelo_estricto(ctor(HEADS_CONFIG), MODELOS_PT[nombre], fmt)
        except Exception as e:
            print(f"[pipeline] EXCLUIDO {nombre}: {e}")

    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )
    _estado["loaded"] = True


def hash_a_seed(rasgos: dict) -> int:
    serializado = json.dumps(rasgos, sort_keys=True, default=str)
    return int(hashlib.md5(serializado.encode()).hexdigest()[:8], 16)


def seed_unica(rasgos: dict, cara_ref) -> int:
    """Seed determinista por PERSONA: mezcla los rasgos categoricos con los PIXELES
    reales de la cara recortada. Misma foto -> misma seed (reproducible); personas
    distintas (aunque compartan rasgos) -> seeds distintas -> NO se repite la imagen.
    """
    base = json.dumps(rasgos, sort_keys=True, default=str).encode()
    pix = np.asarray(cara_ref).tobytes()
    return int(hashlib.md5(base + pix).hexdigest()[:8], 16)


def recortar_cara(imagen_rgb, landmarks, margen: float = 0.35):
    """Recorta la cara con el bounding box de los landmarks + margen (incluye pelo y
    menton). Devuelve un PIL.Image listo para el IP-Adapter (preservar identidad)."""
    h, w = imagen_rgb.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    mx, my = (x2 - x1) * margen, (y2 - y1) * margen
    x1, x2 = int(max(0, x1 - mx)), int(min(w, x2 + mx))
    y1, y2 = int(max(0, y1 - my)), int(min(h, y2 + my))
    return Image.fromarray(imagen_rgb[y1:y2, x1:x2])


def analizar(image_rgb: np.ndarray) -> dict:
    """Imagen RGB (np.uint8) -> rasgos faciales + confianzas + seed + recorte de cara.

    Incluye `rasgos_geom` (con el color RGB real del pixel) y `cara_ref` (PIL) para
    que la generacion SD (IP-Adapter) preserve la identidad, como en v14.
    """
    cargar_pipeline()
    resultado = face_mesh.process(image_rgb)
    if not resultado.multi_face_landmarks:
        raise ValueError("No se detecto ninguna cara en la imagen.")
    landmarks = resultado.multi_face_landmarks[0].landmark

    rasgos_geom = extraer_rasgos_geometricos(image_rgb, landmarks)
    preds_modelos = {m: predecir_probabilidades(modelos[m], image_rgb) for m in modelos}
    preds_clip = predecir_con_clip(image_rgb)

    rasgos, confianzas, contribuciones = voto_suave_unificado(
        preds_modelos, rasgos_geom, predicciones_clip=preds_clip,
    )
    cara_ref = recortar_cara(image_rgb, landmarks)
    return {
        "rasgos": rasgos,
        "confianzas": {k: round(float(v), 4) for k, v in confianzas.items()},
        "contribuciones": contribuciones,
        # Seed por PERSONA (rasgos + pixeles de la cara): evita imagenes repetidas.
        "seed": seed_unica(rasgos, cara_ref),
        "rasgos_geom": rasgos_geom,
        "cara_ref": cara_ref,
    }
