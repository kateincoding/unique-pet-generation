import torch.nn as nn

# Number of output classes per attribute
HEADS_CONFIG: dict[str, int] = {
    # ── derived from CelebA ──────────────────────────────────────
    "color_pelo":    6,   # negro castano rubio pelirrojo gris calvo
    "textura_pelo":  4,   # liso ondulado rizado muy_rizado
    "longitud_pelo": 4,   # corto medio largo calvo
    "cejas":         5,   # normales arqueadas pobladas finas rectas
    "forma_ojos":    5,   # almendrada redonda rasgada caida prominente
    "tamano_nariz":  3,   # pequena mediana grande
    "forma_nariz":   4,   # recta aguileña respingona ancha
    "grosor_labios": 3,   # finos medianos carnosos
    "pomulos":       4,   # planos normales altos prominentes
    "mandibula":     4,   # suave marcada ancha estrecha
    "barbilla":      4,   # redonda puntiaguda cuadrada hendida
    "forma_cara":    6,   # oval redonda cuadrada corazon diamante oblonga
    "vello_facial":  4,   # sin_barba barba_corta barba_larga bigote
    "gafas":         2,   # False True
    "pecas":         2,   # False True
    # ── derived from FairFace ────────────────────────────────────
    "tono_piel":     7,   # muy_claro claro medio oliva bronceado oscuro muy_oscuro
    "rango_edad":    5,   # nino joven adulto maduro mayor
}

CLASS_LABELS: dict[str, list] = {
    "color_pelo":    ["negro", "castano", "rubio", "pelirrojo", "gris", "calvo"],
    "textura_pelo":  ["liso", "ondulado", "rizado", "muy_rizado"],
    "longitud_pelo": ["corto", "medio", "largo", "calvo"],
    "cejas":         ["normales", "arqueadas", "pobladas", "finas", "rectas"],
    "forma_ojos":    ["almendrada", "redonda", "rasgada", "caida", "prominente"],
    "tamano_nariz":  ["pequena", "mediana", "grande"],
    "forma_nariz":   ["recta", "aguileña", "respingona", "ancha"],
    "grosor_labios": ["finos", "medianos", "carnosos"],
    "pomulos":       ["planos", "normales", "altos", "prominentes"],
    "mandibula":     ["suave", "marcada", "ancha", "estrecha"],
    "barbilla":      ["redonda", "puntiaguda", "cuadrada", "hendida"],
    "forma_cara":    ["oval", "redonda", "cuadrada", "corazon", "diamante", "oblonga"],
    "vello_facial":  ["sin_barba", "barba_corta", "barba_larga", "bigote"],
    "gafas":         [False, True],
    "pecas":         [False, True],
    "tono_piel":     ["muy_claro", "claro", "medio", "oliva", "bronceado", "oscuro", "muy_oscuro"],
    "rango_edad":    ["nino", "joven", "adulto", "maduro", "mayor"],
}

# Higher weight for harder or more impactful attributes
LOSS_WEIGHTS: dict[str, float] = {
    "color_pelo":    1.0,
    "textura_pelo":  1.0,
    "longitud_pelo": 1.0,
    "cejas":         0.8,
    "forma_ojos":    1.0,
    "tamano_nariz":  0.8,
    "forma_nariz":   0.8,
    "grosor_labios": 0.8,
    "pomulos":       0.8,
    "mandibula":     0.8,
    "barbilla":      0.6,
    "forma_cara":    1.2,
    "vello_facial":  1.0,
    "gafas":         1.0,
    "pecas":         0.6,
    "tono_piel":     1.2,
    "rango_edad":    1.0,
}


class AttributeHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def build_heads(feature_dim: int = 512) -> nn.ModuleDict:
    return nn.ModuleDict({
        name: AttributeHead(feature_dim, n_classes)
        for name, n_classes in HEADS_CONFIG.items()
    })
