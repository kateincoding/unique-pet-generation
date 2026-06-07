# Petly 🐾

Convierte una **foto** en una **mascota kawaii** única. La foto se analiza con un
pipeline de visión (detección facial + clasificadores + CLIP) que extrae 18 rasgos
faciales; con esos rasgos **se genera una imagen real** de la criatura con Stable
Diffusion (igual que en `mascota_generativa_v15`): se elige un **animalito real** (zorrito,
búho, gatito, panda rojo…) desde los rasgos, IP-Adapter usa la **cara recortada** (escala 0.45)
para aportar color y expresión, y LCM-LoRA la genera rápido (~8 pasos). Los rasgos también eligen de
forma **determinista** la especie, los colores y los "traits" del SVG de respaldo.

```
foto ──> [backend: análisis facial] ──> 18 rasgos ──┬─> SD + IP-Adapter (cara real) ──> imagen PNG
                                                     └─> mapper ──> Pet {especie, color, traits}
                                                                          │
                                      [frontend: muestra la imagen generada; SVG si SD no corre]
```

> **Generación real (como v15):** el backend devuelve `image` (PNG en data URL) creado
> con SD 1.5 (`Yntec/CuteFurry`) + IP-Adapter *plus-face* (identidad) + LCM-LoRA (rápido).
> El frontend lo muestra en el reveal y la galería. Si la generación está apagada
> (`PETLY_GEN=0`) o falla, cae al **SVG** vectorial al instante.
>
> ⚠️ La primera llamada a `/api/analyze` descarga los pesos (~4-5 GB: modelo base +
> IP-Adapter + image encoder + LCM-LoRA) y los cachea. En **CPU** una imagen tarda
> ~1-3 min (8 pasos LCM); con **GPU** es de segundos. Ajustables por entorno:
> `PETLY_GEN_STEPS` (8), `PETLY_GEN_GUIDANCE` (2.0), `PETLY_IP_SCALE` (0.75).

## Estructura

```
petly/
├── backend/                 # FastAPI + pipeline de análisis facial
│   ├── app/
│   │   ├── main.py          # app FastAPI, CORS, routers
│   │   ├── config.py        # rutas/CORS por env
│   │   ├── schemas.py       # contrato (Pet, Trait, AnalyzeResponse)
│   │   ├── mapper.py        # 18 rasgos -> Pet (especie/color/traits, determinista)
│   │   ├── db.py            # colección en SQLite (sin auth)
│   │   ├── api/             # /api/analyze, /api/pets
│   │   └── pipeline/core.py # lógica del notebook, portada y verificada
│   └── requirements.txt
└── frontend/                # Vite + React (port del prototipo Petly, hifi)
    ├── src/
    │   ├── App.jsx          # state machine + llamadas a la API
    │   ├── api.js           # cliente del backend
    │   ├── screens.jsx      # intro/capture/analyze/reveal/gallery
    │   ├── pets.jsx         # 6 mascotas SVG + registro
    │   └── styles.css       # tokens/animaciones (verbatim del handoff)
    └── package.json
```

## Requisitos
- **Python 3.11** con los modelos `.pt` (vit/agus/resnet) del proyecto del notebook.
- **Node 18+** para el frontend.

## Backend
```powershell
cd backend
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# apuntar a la carpeta con los .pt (vit/agus/resnet):
$env:PETLY_MODELS_DIR = "c:/Users/Paulina Peralta/Desktop/gen-pet"
uvicorn app.main:app --reload --port 8000
```
- API: http://localhost:8000 · docs: http://localhost:8000/docs
- Primera llamada a `/api/analyze` carga los modelos de análisis (~10-20 s) y, la
  primera vez, descarga los pesos de Stable Diffusion (~4-5 GB; ver nota arriba).

## Frontend
```powershell
cd frontend
npm install
npm run dev      # http://localhost:5173
```

## API (contrato)
| Método | Ruta | Qué hace |
|---|---|---|
| `POST` | `/api/analyze` | `multipart photo` → `Pet` propuesta (especie, color, traits, rasgos + `image` PNG generada por SD) |
| `GET`  | `/api/pets` | lista la colección |
| `POST` | `/api/pets` | guarda una mascota |
| `DELETE` | `/api/pets/{id}` | borra |
| `GET`  | `/api/health` | estado |

`Pet`:
```jsonc
{
  "id": "p-…",                // solo al guardar
  "name": "Mochi",
  "speciesKey": "fox",        // dog|fox|cat|bunny|hamster|panda
  "tint": "#9C6B43",          // ← color de pelo detectado
  "accent": "#B5612F",        // ← color de ojos detectado
  "traits": [{"label":"Pelaje","value":"Castano","color":"#…"}, …],
  "seed": 3908307091,         // hash determinista de los rasgos
  "rasgos": { … 18 rasgos … }
}
```

## Notas de diseño (decisiones)
- **Imagen real con SD + IP-Adapter** (como v15): la cara recortada preserva la
  identidad y LCM-LoRA la genera en pocos pasos. El **SVG por rasgos** queda como
  respaldo instantáneo (si SD está apagado o falla). La especie/color/nombre son
  **deterministas** del hash de los rasgos (misma cara → misma mascota → misma seed).
- Privacidad de la generación: la **cara recortada** se usa solo en memoria para
  condicionar el IP-Adapter; no se almacena (solo se guarda la mascota generada).
- **Sin auth**: la colección vive en SQLite local. Migrar a usuarios = añadir auth +
  `user_id` en `pets`.
- El panel de *Tweaks* del prototipo era herramienta de diseño → **no portado**.
- Privacidad: hoy la foto se procesa en memoria y **no se almacena**. Si se guarda,
  actualizar el copy de privacidad en `CaptureScreen`.
