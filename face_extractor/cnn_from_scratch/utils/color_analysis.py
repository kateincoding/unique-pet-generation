"""HSV / LAB color analysis for eye color and skin tone (no training needed)."""

import cv2
import numpy as np

SKIN_TONE_LABELS = [
    "muy_oscuro", "oscuro", "bronceado", "oliva", "medio", "claro", "muy_claro"
]

EYE_COLOR_LABELS = [
    "marron_oscuro", "marron", "avellana", "verde", "azul", "gris", "ambar"
]


def analyze_skin_tone(face_bgr: np.ndarray) -> str:
    """
    Estimate skin tone from the LAB L-channel of the face centre.
    face_bgr: numpy array (H, W, 3), BGR, any size.
    Returns one of SKIN_TONE_LABELS.
    """
    h, w = face_bgr.shape[:2]
    roi = face_bgr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L_mean = float(lab[:, :, 0].mean())

    thresholds = [60, 85, 105, 125, 150, 185]
    for i, t in enumerate(thresholds):
        if L_mean < t:
            return SKIN_TONE_LABELS[i]
    return SKIN_TONE_LABELS[-1]


def analyze_eye_color(face_bgr: np.ndarray, landmarks: dict | None = None) -> str:
    """
    Estimate iris color from HSV of the eye region.
    face_bgr: numpy array (H, W, 3), BGR.
    landmarks: optional {"left_eye": (x, y), "right_eye": (x, y)}.
    Returns one of EYE_COLOR_LABELS.
    """
    h, w = face_bgr.shape[:2]

    if landmarks and "left_eye" in landmarks and "right_eye" in landmarks:
        patches = []
        for key in ("left_eye", "right_eye"):
            ex, ey = landmarks[key]
            r = max(10, h // 15)
            x1, y1 = max(0, ex - r), max(0, ey - r)
            x2, y2 = min(w, ex + r), min(h, ey + r)
            patch = face_bgr[y1:y2, x1:x2]
            if patch.size > 0:
                patches.append(patch)
        roi = np.concatenate(patches, axis=1) if patches else None
    else:
        roi = face_bgr[int(h * 0.25):int(h * 0.45), int(w * 0.2):int(w * 0.8)]

    if roi is None or roi.size == 0:
        return "marron"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hv  = hsv[:, :, 0].flatten()
    sv  = hsv[:, :, 1].flatten()
    vv  = hsv[:, :, 2].flatten()

    # Exclude pupil (very dark) and specular reflections (very bright)
    mask = (vv > 40) & (vv < 220)
    if mask.sum() < 10:
        return "marron"

    h_mean = float(hv[mask].mean())
    s_mean = float(sv[mask].mean())
    v_mean = float(vv[mask].mean())

    if v_mean < 60:
        return "marron_oscuro"
    if s_mean < 25:
        return "gris"
    if h_mean < 15 or h_mean > 165:
        return "marron"
    if 15 <= h_mean < 30:
        return "ambar"
    if 30 <= h_mean < 55:
        return "avellana"
    if 55 <= h_mean < 90:
        return "verde"
    if 90 <= h_mean < 135:
        return "azul"
    return "marron"


def analyze_hair_color_precise(face_bgr: np.ndarray) -> str:
    """
    Estimate hair color from the top strip of the image (above-forehead region).
    face_bgr: numpy array (H, W, 3), BGR.
    Returns a hair color string.
    """
    h, w = face_bgr.shape[:2]
    roi = face_bgr[0:int(h * 0.2), int(w * 0.2):int(w * 0.8)]

    if roi.size == 0:
        return "castano"

    hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = float(hsv[:, :, 0].mean())
    s_mean = float(hsv[:, :, 1].mean())
    v_mean = float(hsv[:, :, 2].mean())

    if v_mean < 40:
        return "negro"
    if s_mean < 30:
        return "gris" if v_mean < 180 else "blanco"
    if 15 <= h_mean < 35 and v_mean > 150:
        return "rubio"
    if (h_mean < 20 or h_mean > 160) and s_mean > 100:
        return "pelirrojo"
    return "castano"
