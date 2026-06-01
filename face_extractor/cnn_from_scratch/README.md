# Face Extractor - Iteración 1

Módulo que, dada una foto de una persona, devuelve un diccionario con sus atributos faciales. Ese diccionario es la entrada del generador de mascotas: la idea es que la mascota generada refleje los rasgos del dueño.

---

## Qué hace

Dado un archivo de imagen, el módulo:

1. Detecta y recorta la cara con **MTCNN** (resize a 224x224).
2. Pasa el recorte por una **CNN multi-cabeza** que predice 17 atributos de clasificación.
3. Calcula **2 atributos extra con visión clásica** (sin red neuronal):
   - `color_ojos`: Haar Cascade + HoughCircles + análisis HSV del iris aislado.
   - `tono_piel_lab`: canal L del espacio LAB sobre la región central de la cara (escala Fitzpatrick).

El resultado es un `dict` con 19 atributos listos para el generador.

---

## Atributos que se extraen

| Atributo | Clases | Fuente |
|---|---|---|
| `color_pelo` | negro, castaño, rubio, pelirrojo, gris, calvo | CNN (CelebA) |
| `textura_pelo` | liso, ondulado, rizado | CNN (CelebA) |
| `longitud_pelo` | corto, medio, largo | CNN (CelebA) |
| `cejas` | normales, arqueadas, pobladas, finas, rectas | CNN (CelebA) |
| `forma_ojos` | almendrada, redonda, monolid, caída, otros | CNN (CelebA) |
| `tamano_nariz` | pequeña, media, grande | CNN (CelebA) |
| `forma_nariz` | recta, respingona, ancha | CNN (CelebA) |
| `grosor_labios` | finos, medios, gruesos | CNN (CelebA) |
| `pomulos` | bajos, medios, altos | CNN (CelebA) |
| `mandibula` | estrecha, media, ancha | CNN (CelebA) |
| `barbilla` | puntiaguda, normal, cuadrada | CNN (CelebA) |
| `forma_cara` | oval, redonda, cuadrada, corazón, diamante, oblonga | CNN (CelebA) |
| `vello_facial` | sin_barba, barba_corta, barba_larga, bigote | CNN (CelebA) |
| `gafas` | True / False | CNN (CelebA) |
| `pecas` | True / False | CNN (CelebA) |
| `tono_piel` | muy_claro ... muy_oscuro (7 niveles) | CNN (FairFace) |
| `rango_edad` | niño, joven, adulto, maduro, mayor | CNN (FairFace) |
| `color_ojos` | marrón, azul, verde, avellana, ámbar, gris, marrón_oscuro | Visión clásica |
| `tono_piel_lab` | muy_oscuro ... muy_claro (7 niveles) | Visión clásica (LAB) |

> **Nota:** `tono_piel_lab` es más fiable que `tono_piel` en esta iteración. `tono_piel` (CNN) tiene precisión baja porque CelebA solo etiqueta "pale_skin" vs resto, lo que genera etiquetas muy ruidosas para 7 clases.

---

## Arquitectura del modelo

```
Input 224x224x3
    |
5x ConvBlock (Conv -> BN -> ReLU -> MaxPool)
    3->32->64->128->256->512   (224->112->56->28->14->7)
    |
AdaptiveAvgPool2d(1)  ->  vector 512-dim
    |
17 cabezas independientes (Linear + Softmax)
    cada cabeza = un atributo
```

**Parámetros totales:** ~6.5M

---

## Datasets y entrenamiento

### CelebA
- ~200k imágenes de caras de famosos con 40 atributos binarios.
- Se mapean los 40 atributos binarios a las 15 clases multi-clase del modelo (ver función `celeba_labels`).
- **Problema conocido:** algunas clases tienen muy pocas muestras (n=1 en el split de validación). No es un bug del modelo, es desbalance de datos en el mapping.

### FairFace
- ~100k imágenes con anotaciones de raza y edad.
- Se usa en **fine-tuning** de las 2 cabezas `tono_piel` y `rango_edad` (las únicas que CelebA etiqueta mal).
- Se congela el backbone y solo se entrenan esas 2 cabezas durante 5 epochs.

### Resultados de entrenamiento
- **CelebA:** 20 epochs, mejor checkpoint en epoch 8 -- `avg_acc = 0.872`
- **FairFace fine-tuning:** 5 epochs adicionales para `tono_piel` y `rango_edad`

---

## Color de ojos (visión clásica)

Por qué no la CNN: los ojos son una región muy pequeña (< 2% de la imagen) y CelebA no tiene etiquetas de color de ojos suficientemente detalladas.

Pipeline:
1. **Haar Cascade** (`haarcascade_eye.xml`) detecta las cajas de los ojos en la mitad superior de la cara.
2. **HoughCircles** localiza el círculo del iris dentro de cada caja.
3. Se crea una **máscara anular** que excluye la pupila (40% interior) y el limbal ring marrón exterior (15% exterior). Esto deja solo el color puro del iris.
4. Se filtran píxeles por saturación (S > 25) y brillo (40 < V < 220) para eliminar pestañas y reflejos.
5. La **media de H ponderada por saturación** decide el color final.

> Workaround técnico: OpenCV en Windows no puede abrir rutas con caracteres Unicode. El XML del cascade se copia a `%TEMP%` (ruta 8.3 ASCII) antes de cargarlo.

---

## Análisis de sesgo

En el notebook hay 3 secciones de análisis post-entrenamiento:

1. **Distribución real vs predicha** (500 muestras de validación): gráficos de barras para los 8 atributos principales mostrando si el modelo replica la distribución del dataset o se sesga hacia la clase dominante.

2. **Matrices de confusión** (normalizadas por fila = recall por clase): visualización en heatmap + texto para cada atributo. Permite ver qué clases se confunden entre sí.

3. **Ejemplos buenos y malos**: escanea 2000 imágenes de validación y muestra las que el modelo acierta y falla para un atributo elegido (`TARGET_ATTR`).

---

## Cómo probar con una foto nueva

Pon la imagen en `test_photos/` y ejecuta la celda de inferencia:

```python
result = extract_face_attributes('test_photos/tu_foto.jpg')
print(json.dumps(result, indent=2, ensure_ascii=False))
```

---

## Archivos

```
face_extractor/
    facial_attributes_extractor.ipynb  -- notebook principal
    checkpoints/
        best_celeba.pt                 -- mejor checkpoint CelebA (epoch 8)
        final_model.pt                 -- modelo tras fine-tuning FairFace
    test_photos/                       -- fotos para probar manualmente
    face_attributes_results.json       -- ejemplo de output para 3 fotos
    data/
        celeba/                        -- dataset CelebA (no subir a git)
        fairface/                      -- dataset FairFace (no subir a git)
```

---

## Limitaciones conocidas y próximos pasos

| Limitación | Causa | Propuesta siguiente iteración |
|---|---|---|
| `tono_piel` CNN poco preciso | Labels CelebA solo "pale" vs resto | Usar solo `tono_piel_lab` o reentrenar con FairFace completo |
| `color_ojos` a veces falla en ojos claros | Limbal ring domina en iris pequeños | Ajustar parámetros HoughCircles, probar con más fotos |
| Clases minoritarias con n=1 en validación | Desbalance en el mapping de CelebA | Oversampling o eliminar clases con < 50 muestras |
| Solo detecta una cara por imagen | Diseño actual | Adaptar para multi-cara si el generador lo necesita |
