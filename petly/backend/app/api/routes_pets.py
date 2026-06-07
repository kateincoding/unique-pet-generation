# -*- coding: utf-8 -*-
"""CRUD de la coleccion de mascotas (SQLite, sin auth)."""
from fastapi import APIRouter, HTTPException

from .. import db
from ..schemas import Pet, PetCreate

router = APIRouter(prefix="/api/pets", tags=["pets"])


@router.get("", response_model=list[Pet])
def list_pets():
    return db.list_pets()


@router.post("", response_model=Pet, status_code=201)
def create_pet(pet: PetCreate):
    return db.create_pet(pet.model_dump())


@router.delete("/{pet_id}", status_code=204)
def delete_pet(pet_id: str):
    if not db.delete_pet(pet_id):
        raise HTTPException(status_code=404, detail="Mascota no encontrada.")
    return None
