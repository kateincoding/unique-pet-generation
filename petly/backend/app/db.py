# -*- coding: utf-8 -*-
"""Persistencia simple de la coleccion en SQLite (sin auth)."""
import json
import sqlite3
import uuid
from contextlib import contextmanager

from . import config


@contextmanager
def _conn():
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS pets (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                speciesKey TEXT NOT NULL,
                tint       TEXT NOT NULL,
                accent     TEXT NOT NULL,
                traits     TEXT NOT NULL,
                seed       INTEGER,
                rasgos     TEXT,
                date       TEXT,
                image      TEXT,
                photo      TEXT,
                animal     TEXT,
                element    TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Migracion: agrega columnas nuevas a tablas viejas (idempotente).
        for col in ("animal", "element", "photo"):
            try:
                con.execute(f"ALTER TABLE pets ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass  # ya existe


def _row_to_pet(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "speciesKey": row["speciesKey"],
        "tint": row["tint"],
        "accent": row["accent"],
        "traits": json.loads(row["traits"]),
        "seed": row["seed"],
        "rasgos": json.loads(row["rasgos"]) if row["rasgos"] else None,
        "date": row["date"],
        "image": row["image"],
        "photo": row["photo"],
        "animal": row["animal"],
        "element": row["element"],
    }


def list_pets() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM pets ORDER BY created_at DESC").fetchall()
        return [_row_to_pet(r) for r in rows]


def create_pet(pet: dict) -> dict:
    pet_id = "p-" + uuid.uuid4().hex[:10]
    with _conn() as con:
        con.execute(
            """INSERT INTO pets (id, name, speciesKey, tint, accent, traits, seed, rasgos, date, image, photo, animal, element)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pet_id, pet["name"], pet["speciesKey"], pet["tint"], pet["accent"],
                json.dumps(pet["traits"], ensure_ascii=False),
                pet.get("seed"),
                json.dumps(pet.get("rasgos"), ensure_ascii=False) if pet.get("rasgos") else None,
                pet.get("date"),
                pet.get("image"),
                pet.get("photo"),
                pet.get("animal"),
                pet.get("element"),
            ),
        )
        row = con.execute("SELECT * FROM pets WHERE id = ?", (pet_id,)).fetchone()
        return _row_to_pet(row)


def delete_pet(pet_id: str) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM pets WHERE id = ?", (pet_id,))
        return cur.rowcount > 0


# Garantizamos la tabla al importar el modulo (idempotente, ademas del startup event).
init_db()
