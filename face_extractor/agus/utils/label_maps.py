"""Mappings from CelebA/FairFace raw labels to our attribute classes."""

CELEBA_ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive",    "Bags_Under_Eyes",
    "Bald",             "Bangs",            "Big_Lips",      "Big_Nose",
    "Black_Hair",       "Blond_Hair",       "Blurry",        "Brown_Hair",
    "Bushy_Eyebrows",   "Chubby",           "Double_Chin",   "Eyeglasses",
    "Goatee",           "Gray_Hair",        "Heavy_Makeup",  "High_Cheekbones",
    "Male",             "Mouth_Slightly_Open", "Mustache",   "Narrow_Eyes",
    "No_Beard",         "Oval_Face",        "Pale_Skin",     "Pointy_Nose",
    "Receding_Hairline","Rosy_Cheeks",      "Sideburns",     "Smiling",
    "Straight_Hair",    "Wavy_Hair",        "Wearing_Earrings","Wearing_Hat",
    "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie","Young",
]

_IDX = {name: i for i, name in enumerate(CELEBA_ATTR_NAMES)}


def celeba_to_labels(attrs) -> dict:
    """
    Convert a 40-element CelebA attribute vector to our label dict.
    attrs: list or tensor of ints (+1/-1 or 1/0 — both work).
    Returns dict {attr_name: class_index}.  -1 means unknown (skip in loss).
    """
    a = [int(v) > 0 for v in attrs]

    def has(name: str) -> bool:
        return a[_IDX[name]]

    labels: dict = {}

    # ── color_pelo ──────────────────────────────────────────────
    if has("Bald"):
        labels["color_pelo"] = 5
    elif has("Black_Hair"):
        labels["color_pelo"] = 0
    elif has("Brown_Hair"):
        labels["color_pelo"] = 1
    elif has("Blond_Hair"):
        labels["color_pelo"] = 2
    elif has("Gray_Hair"):
        labels["color_pelo"] = 4
    else:
        labels["color_pelo"] = 1  # default castano

    # ── textura_pelo ────────────────────────────────────────────
    if has("Bald"):
        labels["textura_pelo"] = -1
    elif has("Straight_Hair"):
        labels["textura_pelo"] = 0
    elif has("Wavy_Hair"):
        labels["textura_pelo"] = 1
    else:
        labels["textura_pelo"] = 2  # rizado (neither annotated)

    # ── longitud_pelo ───────────────────────────────────────────
    if has("Bald"):
        labels["longitud_pelo"] = 3
    elif has("Receding_Hairline"):
        labels["longitud_pelo"] = 0
    elif not has("Male"):
        labels["longitud_pelo"] = 2  # female proxy → long
    else:
        labels["longitud_pelo"] = 0  # male proxy → short

    # ── cejas ───────────────────────────────────────────────────
    arched = has("Arched_Eyebrows")
    bushy  = has("Bushy_Eyebrows")
    if arched and bushy:
        labels["cejas"] = 2   # pobladas
    elif arched:
        labels["cejas"] = 1   # arqueadas
    elif bushy:
        labels["cejas"] = 2   # pobladas
    else:
        labels["cejas"] = 0   # normales

    # ── forma_ojos ──────────────────────────────────────────────
    labels["forma_ojos"] = 2 if has("Narrow_Eyes") else 0

    # ── tamano_nariz ────────────────────────────────────────────
    labels["tamano_nariz"] = 2 if has("Big_Nose") else 1

    # ── forma_nariz ─────────────────────────────────────────────
    labels["forma_nariz"] = 2 if has("Pointy_Nose") else 0

    # ── grosor_labios ───────────────────────────────────────────
    labels["grosor_labios"] = 2 if has("Big_Lips") else 1

    # ── pomulos ─────────────────────────────────────────────────
    labels["pomulos"] = 2 if has("High_Cheekbones") else 1

    # ── mandibula ───────────────────────────────────────────────
    if has("Chubby"):
        labels["mandibula"] = 0   # suave
    elif has("Male"):
        labels["mandibula"] = 1   # marcada
    else:
        labels["mandibula"] = 0   # suave

    # ── barbilla — CelebA has no direct label ───────────────────
    labels["barbilla"] = -1

    # ── forma_cara ──────────────────────────────────────────────
    if has("Oval_Face"):
        labels["forma_cara"] = 0
    elif has("Chubby"):
        labels["forma_cara"] = 1   # redonda
    else:
        labels["forma_cara"] = -1  # unknown

    # ── vello_facial ────────────────────────────────────────────
    if has("Mustache") and not has("Goatee"):
        labels["vello_facial"] = 3
    elif has("Goatee") or has("Sideburns") or has("5_o_Clock_Shadow"):
        labels["vello_facial"] = 1
    elif has("No_Beard"):
        labels["vello_facial"] = 0
    else:
        labels["vello_facial"] = 0

    # ── binarios ────────────────────────────────────────────────
    labels["gafas"] = int(has("Eyeglasses"))
    labels["pecas"]  = int(has("Rosy_Cheeks"))  # proxy

    # ── tono_piel — only Pale_Skin available in CelebA ──────────
    # -1 for most samples; FairFace fine-tune provides real labels
    labels["tono_piel"] = 1 if has("Pale_Skin") else -1

    # ── rango_edad — not in CelebA, comes from FairFace ─────────
    labels["rango_edad"] = 0 if has("Young") else 2  # rough proxy

    return labels


# ── FairFace mappings ────────────────────────────────────────────────────────

FAIRFACE_AGE_MAP: dict[str, int] = {
    "0-2":  0, "3-9":  0,
    "10-19": 1, "20-29": 1,
    "30-39": 2, "40-49": 2,
    "50-59": 3, "60-69": 3,
    "70+":   4,
}

FAIRFACE_RACE_TO_TONE: dict[str, int] = {
    "White":            1,   # claro
    "Black":            5,   # oscuro
    "Latino_Hispanic":  3,   # oliva
    "East Asian":       2,   # medio
    "Southeast Asian":  3,   # oliva
    "Indian":           4,   # bronceado
    "Middle Eastern":   2,   # medio
}
