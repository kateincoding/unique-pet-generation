# -*- coding: utf-8 -*-
"""Modelos de datos (contrato API). Pet = lo que el frontend pinta en SVG."""
from typing import Optional
from pydantic import BaseModel, Field


class Trait(BaseModel):
    label: str
    value: str
    color: str


class PetBase(BaseModel):
    name: str
    speciesKey: str = Field(..., description="dog|fox|cat|bunny|hamster|panda")
    tint: str
    accent: str
    traits: list[Trait]
    seed: Optional[int] = None
    date: Optional[str] = None
    image: Optional[str] = Field(None, description="imagen generada por SD (data URL PNG)")
    photo: Optional[str] = Field(None, description="foto de entrada (data URL) linkeada a su mascota generada")
    animal: Optional[str] = Field(None, description="animal real generado, ES (ej. 'Mapache bebe')")
    element: Optional[str] = Field(None, description="elemento magico, ES (ej. 'familiar de brasas')")


class AnalyzeResponse(PetBase):
    """Resultado de analizar una foto (mascota propuesta, aun sin guardar)."""
    rasgos: dict = Field(default_factory=dict, description="18 rasgos faciales detectados")


class PetCreate(PetBase):
    rasgos: Optional[dict] = None


class Pet(PetBase):
    id: str
    rasgos: Optional[dict] = None
