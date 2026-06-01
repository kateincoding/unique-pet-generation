# CLIP + MediaPipe Face Mesh

Sección 5.4 de la memoria. Pipeline **sin entrenamiento** que combina:

- **MediaPipe Face Mesh** — modelo preentrenado de Google que detecta 478 landmarks faciales. Sobre esos puntos se calcula morfometría geométrica (proporciones normalizadas por el ancho de la cara) para extraer rasgos interpretables (forma de cara, ojos, cejas, simetría).
- **DeepFace** — librería que envuelve redes preentrenadas para estimación de emoción (mini-Xception) y edad (VGG-Face).
- **CLIP** — los rasgos extraídos se traducen a prompt tokens en inglés que el encoder CLIP convierte en vectores de condición para Stable Diffusion, alimentando la fase de generación.

## Notebook

- `06_facemesh.ipynb`: webcam o imagen → 478 landmarks → rasgos geométricos + DeepFace → prompt tokens → `rasgos_checklist.json`.

## Test image

`agus.jpg` es la imagen de ejemplo que usa el notebook para extraer 9 rasgos y 6 prompt tokens.

## Limitaciones observadas

- **Simetría**: tiende a 0 si la cara no está perfectamente frontal respecto al eje central que asume la fórmula.
- **Tono de piel**: se calcula a partir de la luminosidad cruda, así que con poca luz se sesga hacia oscuro.

Son las limitaciones típicas de un método geométrico con umbrales fijos: rápido y explicable, pero sensible al encuadre y la iluminación.
