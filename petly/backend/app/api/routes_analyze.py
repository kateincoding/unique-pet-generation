# -*- coding: utf-8 -*-
"""POST /api/analyze: foto -> rasgos faciales -> mascota propuesta."""
import io

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from .. import config
from ..mapper import rasgos_a_pet
from ..schemas import AnalyzeResponse

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(photo: UploadFile = File(...)):
    data = await photo.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen invalida.")
    image_rgb = np.array(img)

    # Import perezoso: cargar el pipeline pesado solo cuando se usa.
    from ..pipeline import analizar

    try:
        resultado = analizar(image_rgb)
    except ValueError as e:
        # tipico: no se detecto cara
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo el analisis: {e}")

    pet = rasgos_a_pet(resultado["rasgos"], resultado["seed"])

    # Animal REAL generado + elemento magico (ES) para el encabezado (nada inventado).
    from ..pipeline import generate
    pet["animal"] = generate.animal_es(resultado["rasgos"])
    pet["element"] = generate.elemento_es(resultado["rasgos"])

    # Imagen REAL del modelo (Stable Diffusion + IP-Adapter, como en v15): la cara
    # recortada preserva la identidad. Lento en CPU; si falla, cae al SVG.
    if config.GEN_ENABLED:
        from ..pipeline import generate
        try:
            pet["image"] = generate.generar_imagen(
                resultado["rasgos"], resultado["rasgos_geom"],
                resultado["cara_ref"], pet["seed"],
            )
        except Exception as e:
            print(f"[analyze] generacion SD fallo, se usa SVG: {e}")
            pet["image"] = None

    return pet
