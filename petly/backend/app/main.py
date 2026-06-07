# -*- coding: utf-8 -*-
"""Petly API — FastAPI. Envuelve el pipeline de analisis facial + coleccion."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config, db
from .api import routes_analyze, routes_pets

app = FastAPI(title="Petly API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_analyze.router)
app.include_router(routes_pets.router)


@app.on_event("startup")
def _startup():
    db.init_db()


@app.get("/api/health", tags=["health"])
def health():
    # Reporta si la generacion SD esta lista (para diagnosticar por que cae al SVG).
    gen = {"enabled": config.GEN_ENABLED}
    for mod in ("diffusers", "peft", "accelerate"):
        try:
            m = __import__(mod)
            gen[mod] = getattr(m, "__version__", "ok")
        except Exception as e:
            gen[mod] = f"FALTA: {type(e).__name__}"
    return {"status": "ok", "models_dir": config.MODELS_DIR, "generation": gen}
