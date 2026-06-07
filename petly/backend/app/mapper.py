# -*- coding: utf-8 -*-
"""Mapea los 18 rasgos faciales detectados -> una mascota Pet del frontend.

Determinista a partir del `seed` (hash de los rasgos): la misma cara da siempre
la misma especie, nombre y colores. La especie y los colores salen de los rasgos
reales (color de pelo -> tinte del pelaje, color de ojos -> acento).
"""
from datetime import datetime

# Especies del frontend (deben coincidir con SPECIES en pets.jsx).
SPECIES_KEYS = ["unicorn", "dog", "fox", "cat", "bunny", "hamster", "panda"]

NAMES = ["Mochi", "Tofu", "Galleta", "Canela", "Miel", "Brioche", "Coco", "Avena",
         "Maple", "Nube", "Pancake", "Bombon", "Caramelo", "Almendra", "Trufa", "Cookie"]

# color_pelo -> (hex tinte del pelaje, etiqueta ES)
PELO_HEX = {
    "negro":     ("#4A3B33", "Negro"),
    "castano":   ("#9C6B43", "Castano"),
    "rubio":     ("#E6C07B", "Rubio"),
    "pelirrojo": ("#C0653B", "Pelirrojo"),
    "gris":      ("#A7A29C", "Gris"),
    "calvo":     ("#E6C9A8", "Sin pelo"),
}

# color_ojos -> (hex acento, etiqueta ES)
OJOS_HEX = {
    "azul":          ("#6CA8D9", "Azules"),
    "verde":         ("#7FB582", "Verdes"),
    "avellana":      ("#B5612F", "Avellana"),
    "marron":        ("#8B5E3C", "Marrones"),
    "marron_oscuro": ("#5A3A22", "Marron oscuro"),
    "gris":          ("#9A9A9A", "Grises"),
    "negro":         ("#3A2B22", "Negros"),
}

# forma_cara / forma_ojos -> etiqueta ES (rasgos REALES detectados, no ficcion)
FORMA_CARA_ES = {
    "oval": "Ovalada", "redonda": "Redonda", "cuadrada": "Cuadrada",
    "corazon": "Corazon", "diamante": "Diamante", "oblonga": "Oblonga",
}
FORMA_OJOS_ES = {
    "almendrada": "Almendrados", "redonda": "Redondos", "rasgada": "Rasgados",
    "caida": "Caidos", "prominente": "Prominentes",
}
CARA_COLOR = "#8FB99A"   # verde suave
OJOSF_COLOR = "#7C8BE0"  # azul suave


def rasgos_a_pet(rasgos: dict, seed: int) -> dict:
    species_key = SPECIES_KEYS[seed % len(SPECIES_KEYS)]
    name = NAMES[seed % len(NAMES)]

    tint, pelo_es = PELO_HEX.get(rasgos.get("color_pelo"), ("#9C6B43", "Castano"))
    accent, ojos_es = OJOS_HEX.get(rasgos.get("color_ojos"), ("#8B5E3C", "Marrones"))
    cara_es = FORMA_CARA_ES.get(rasgos.get("forma_cara"), "—")
    ojosf_es = FORMA_OJOS_ES.get(rasgos.get("forma_ojos"), "—")

    # Rasgos faciales REALES detectados por el analisis.
    traits = [
        {"label": "Pelaje", "value": pelo_es, "color": tint},
        {"label": "Ojos",   "value": ojos_es, "color": accent},
        {"label": "Forma de cara", "value": cara_es,  "color": CARA_COLOR},
        {"label": "Forma de ojos", "value": ojosf_es, "color": OJOSF_COLOR},
    ]

    return {
        "name": name,
        "speciesKey": species_key,
        "tint": tint,
        "accent": accent,
        "traits": traits,
        "seed": seed,
        "date": datetime.now().strftime("%d %b"),
        "rasgos": rasgos,
    }
