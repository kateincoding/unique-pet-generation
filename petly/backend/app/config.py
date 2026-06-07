# -*- coding: utf-8 -*-
"""Configuracion via variables de entorno."""
import os

# Carpeta con los checkpoints .pt de los modelos (vit/agus/resnet).
# Por defecto apunta al proyecto del notebook; override con PETLY_MODELS_DIR.
MODELS_DIR = os.environ.get(
    "PETLY_MODELS_DIR", r"c:/Users/Paulina Peralta/Desktop/gen-pet"
)

# Base de datos SQLite (coleccion de mascotas).
DB_PATH = os.environ.get(
    "PETLY_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "petly.db")
)

# Origenes permitidos para CORS (frontend Vite).
CORS_ORIGINS = os.environ.get(
    "PETLY_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

# Aseguramos que el pipeline lea los modelos del mismo MODELS_DIR.
os.environ.setdefault("PETLY_MODELS_DIR", MODELS_DIR)

# Generacion de la imagen real con Stable Diffusion (lento en CPU). 1=on, 0=off (cae al SVG).
GEN_ENABLED = os.environ.get("PETLY_GEN", "1") not in ("0", "false", "False")
