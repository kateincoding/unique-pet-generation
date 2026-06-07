# -*- coding: utf-8 -*-
"""Generacion de la imagen REAL de la mascota (como en mascota_generativa_v15).

Portado del notebook v15 (familiar magico = animalito real):
  - Modelo base SD 1.5 cute: Yntec/CuteFurry (fallback Dreamshaper-8).
  - IP-Adapter plus-face a escala BAJA (0.45): la cara real aporta coloracion y
    expresion, pero deja emerger la silueta del animal (no cara humanoide).
  - LCM-LoRA: generacion rapida (~8 pasos, guidance ~2.5).
  - Prompt: elige una ESPECIE de animalito real (arquetipo_animal) desde los rasgos,
    mete las FORMAS detectadas (describir_anatomia) y una magia integrada al pelaje
    (firma_magica), + color real del pixel (HSV). Negative anti-humanoide/monstruo.

Carga perezosa (la primera llamada baja los pesos, ~4-5 GB). En CPU es lento
(1-3 min con LCM); GPU recomendado. Devuelve la imagen como data URL PNG.
"""
import base64
import colorsys
import hashlib
import io
import json
import os

import torch

from .core import device

dtype = torch.float16 if device == "cuda" else torch.float32

# ---- modelo base + ajustes (overridables por entorno) ----
# Yntec/CuteFurry primero (prior de animalitos peludos y tiernos, decision de v15).
_OPCIONES = ["Yntec/CuteFurry", "Lykon/dreamshaper-8"]
# IP scale bajo (v15): para que salga ANIMAL y no cara peluda frontal.
IP_ADAPTER_SCALE = float(os.environ.get("PETLY_IP_SCALE", "0.45"))
GEN_STEPS = int(os.environ.get("PETLY_GEN_STEPS", "8"))
GEN_GUIDANCE = float(os.environ.get("PETLY_GEN_GUIDANCE", "2.5"))

FONDO = "in a cozy magical forest with soft floating lights and dappled sunlight"

_pipe = {"obj": None, "id": None}


# ===================== prompt (portado de v15) =====================

TRADUCCION = {
    # color_pelo
    "negro": "black", "castano": "brown", "rubio": "blonde",
    "pelirrojo": "red", "gris": "gray", "calvo": "bald",
    # textura_pelo
    "liso": "straight", "ondulado": "wavy", "rizado": "curly",
    "muy_rizado": "very curly",
    # longitud_pelo
    "corto": "short", "medio": "medium length", "largo": "long",
    # cejas
    "normales": "normal", "arqueadas": "arched", "pobladas": "thick",
    "finas": "thin", "rectas": "straight",
    # forma_ojos
    "almendrada": "almond shaped", "redonda": "round",
    "rasgada": "slanted upward", "caida": "downturned",
    "prominente": "prominent",
    # tamano_nariz
    "pequena": "small", "mediana": "medium", "grande": "big",
    # forma_nariz
    "recta_nariz": "straight", "aguilena": "aquiline",
    "respingona": "upturned", "ancha": "wide",
    # grosor_labios
    "finos": "thin", "medianos": "medium", "carnosos": "plump",
    # pomulos
    "planos": "flat", "altos": "high", "prominentes": "prominent",
    # mandibula
    "suave": "soft", "marcada": "defined", "estrecha": "narrow",
    # barbilla
    "puntiaguda": "pointed", "cuadrada": "square", "hendida": "cleft",
    # forma_cara
    "oval": "oval", "corazon": "heart-shaped",
    "diamante": "diamond-shaped", "oblonga": "oblong",
    # vello_facial
    "sin_barba": "no beard", "barba_corta": "short beard",
    "barba_larga": "long beard", "bigote": "mustache",
    # tono_piel
    "muy_claro": "very fair", "claro": "fair",
    "oliva": "olive", "bronceado": "tan",
    "oscuro": "dark", "muy_oscuro": "very dark",
    # rango_edad
    "nino": "child-like", "joven": "young", "adulto": "adult",
    "maduro": "mature", "mayor": "elder",
    # color_ojos
    "azul": "blue", "verde": "green", "avellana": "hazel",
    "marron": "brown", "marron_oscuro": "dark brown",
}


def traducir(valor):
    if isinstance(valor, str):
        return TRADUCCION.get(valor, valor.replace("_", " "))
    return valor


def _w(frase, peso):
    # Envuelve la frase con peso compel (se pasa literal al pipeline, como en v14).
    return f"({frase}){peso}"


def describir_color_pelo(rgb):
    # Descripcion rica del color de pelo desde el RGB real (HSV). None -> sin RGB.
    if rgb is None:
        return None
    r, g, b = [c / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360
    if v < 0.15:
        return "black"
    claridad = "light " if v > 0.55 else ("dark " if v < 0.32 else "")
    if s < 0.12:
        return (claridad + "gray").strip()
    calido = "golden " if s > 0.45 else ("warm " if s > 0.28 else "ash ")
    if hue < 18 or hue >= 345:
        base = "auburn"
    elif 30 <= hue < 70 and v > 0.6 and s > 0.3:
        base = "blonde"
    else:
        base = "brown"
    return (claridad + calido + base).strip()


def describir_color_ojos(rgb):
    # Descripcion rica del color de ojos desde el RGB del iris (HSV).
    if rgb is None:
        return None
    r, g, b = [c / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360
    if v < 0.18:
        return "dark brown"
    claridad = "light " if v > 0.6 else ("deep " if v < 0.32 else "")
    if s < 0.12:
        return (claridad + "gray").strip()
    if 180 <= hue < 260:
        return (claridad + "blue").strip()
    if 80 <= hue < 170:
        return (claridad + "green").strip()
    if 60 <= hue < 80:
        return "hazel green"
    if 35 <= hue < 60:
        return "amber hazel"
    if 18 <= hue < 35:
        return (claridad + ("hazel brown" if v > 0.45 else "brown")).strip()
    return (claridad + "brown").strip()


def describir_anatomia(rasgos: dict) -> list:
    # Lleva los rasgos de FORMA (no color) al prompt como descriptores de animalito.
    # En la v14 estos se detectaban y se tiraban; aca SI moldean al familiar.
    frases = []
    mapa_ojos = {
        "almendrada": "almond-shaped eyes", "redonda": "big round eyes",
        "rasgada": "slanted upturned eyes", "caida": "gentle downturned eyes",
        "prominente": "large expressive eyes",
    }
    if rasgos.get("forma_ojos") in mapa_ojos:
        frases.append(_w(mapa_ojos[rasgos["forma_ojos"]], 1.0))

    mapa_cara = {
        "redonda": "round chubby face", "corazon": "heart-shaped face with a pointy little chin",
        "diamante": "delicate narrow face", "oblonga": "slightly long face",
        "cuadrada": "broad sturdy face", "oval": "soft oval face",
    }
    if rasgos.get("forma_cara") in mapa_cara:
        frases.append(_w(mapa_cara[rasgos["forma_cara"]], 0.9))

    mapa_nariz_forma = {"respingona": "upturned button nose", "ancha": "wide soft snout"}
    mapa_nariz_tam = {"pequena": "tiny nose", "grande": "prominent little snout"}
    if rasgos.get("forma_nariz") in mapa_nariz_forma:
        frases.append(_w(mapa_nariz_forma[rasgos["forma_nariz"]], 0.9))
    elif rasgos.get("tamano_nariz") in mapa_nariz_tam:
        frases.append(_w(mapa_nariz_tam[rasgos["tamano_nariz"]], 0.9))

    if rasgos.get("cejas") in ("pobladas", "arqueadas"):
        frases.append("expressive fluffy brow tufts")
    if rasgos.get("rango_edad") in ("maduro", "mayor"):
        frases.append("calm gentle expression")
    return frases


def arquetipo_animal(rasgos: dict) -> tuple:
    # Elige una ESPECIE de animalito real y bebe desde los rasgos. La forma de ojos
    # marca el caracter, la forma de cara afina la silueta, la textura define lo peludo.
    ojos = rasgos.get("forma_ojos")
    cara = rasgos.get("forma_cara")
    textura = rasgos.get("textura_pelo")
    edad = rasgos.get("rango_edad")

    if ojos == "prominente":
        animal = "baby owl owlet"
    elif ojos == "redonda":
        animal = "kitten"
    elif ojos == "rasgada":
        animal = "baby red fox kit"
    elif ojos == "caida":
        animal = "baby otter pup"
    else:
        animal = "baby deer fawn"      # almendrada / default

    if cara == "redonda":
        animal = "baby red panda cub"
    elif cara in ("diamante", "corazon") and ojos != "caida":
        animal = "baby fennec fox"
    elif cara in ("cuadrada", "oblonga"):
        animal = "baby raccoon kit"

    if textura in ("rizado", "muy_rizado"):
        pelaje_extra = "extremely fluffy round and fuzzy"
    elif textura == "ondulado":
        pelaje_extra = "soft wavy fluffy coat"
    else:
        pelaje_extra = "sleek soft coat"

    if edad == "nino":
        animal = "tiny " + animal + ", extra round and chubby"

    descripcion = f"a real {animal}, {pelaje_extra}, real animal anatomy, big gentle eyes, tiny paws"
    return animal, descripcion


def firma_magica(rasgos: dict) -> tuple:
    # Firma magica determinista (misma cara -> misma magia), INTEGRADA al pelaje del
    # animal en vez de chispas pegoteadas. El elemento sale del hash de los rasgos.
    n = int(hashlib.md5(json.dumps(rasgos, sort_keys=True, default=str).encode()).hexdigest()[8:12], 16)
    firmas = [
        ("ember",  "softly glowing warm-orange markings on its fur, a few faint ember motes around it"),
        ("frost",  "delicate glowing pale-blue markings, a soft frosty shimmer on its coat"),
        ("forest", "gentle glowing leaf-green markings, tiny drifting spores of light"),
        ("dusk",   "soft glowing violet markings, a faint starry shimmer in its fur"),
        ("star",   "tiny glowing golden freckles like little stars, a soft warm halo"),
        ("moon",   "a faint silver crescent glow on its forehead, calm moonlit aura"),
    ]
    return firmas[n % len(firmas)]


# arquetipo_animal (EN) -> nombre ES, para mostrar el animal REAL generado en la UI.
ANIMAL_ES = {
    "baby owl owlet": "Buho bebe",
    "kitten": "Gatito",
    "baby red fox kit": "Zorro rojo bebe",
    "baby otter pup": "Nutria bebe",
    "baby deer fawn": "Cervatillo",
    "baby red panda cub": "Panda rojo bebe",
    "baby fennec fox": "Zorro fennec bebe",
    "baby raccoon kit": "Mapache bebe",
}
ELEMENTO_ES = {
    "ember": "familiar de brasas",
    "frost": "familiar de escarcha",
    "forest": "familiar del bosque",
    "dusk": "familiar del crepusculo",
    "star": "familiar estelar",
    "moon": "familiar lunar",
}


def animal_es(rasgos: dict) -> str:
    # Nombre ES del animal REAL que se genera (mismo arquetipo que el prompt).
    animal_en, _ = arquetipo_animal(rasgos)
    core = animal_en.replace("tiny ", "").split(", extra")[0].strip()
    base = ANIMAL_ES.get(core, "Familiar magico")
    return base + " (mini)" if animal_en.startswith("tiny ") else base


def elemento_es(rasgos: dict) -> str:
    nombre, _ = firma_magica(rasgos)
    return ELEMENTO_ES.get(nombre, "familiar magico")


def construir_prompt(rasgos: dict, rgb_pelo=None, rgb_ojos=None) -> tuple[str, str]:
    material = "fur"

    # Color desde el PIXEL real (rico); si no hay RGB, caemos a la etiqueta traducida.
    pelo_desc = describir_color_pelo(rgb_pelo) or traducir(rasgos.get("color_pelo", "castano"))
    ojos_desc = describir_color_ojos(rgb_ojos) or traducir(rasgos.get("color_ojos", "marron"))
    textura = traducir(rasgos.get("textura_pelo", "liso"))

    longitud_val = rasgos.get("longitud_pelo")
    longitud = "" if longitud_val in ("medio", "calvo", None) else traducir(longitud_val)

    if rasgos.get("color_pelo") == "calvo":
        pelaje = f"short smooth {material}"
    else:
        pelaje = f"{longitud} {pelo_desc} {textura} {material}"
    pelaje = " ".join(pelaje.split())

    accesorios = []
    if rasgos.get("gafas") is True:
        accesorios.append(_w("tiny round glasses", 0.9))
    if rasgos.get("pecas") is True:
        accesorios.append(_w("small freckle-like spots on its fur", 0.9))
    if rasgos.get("vello_facial") not in ("sin_barba", None):
        accesorios.append("extra fluffy cheek fur")
    accesorios_str = (", " + ", ".join(accesorios)) if accesorios else ""

    # Especie del animalito (peso alto), formas detectadas y magia integrada.
    _criatura_animal, criatura_desc = arquetipo_animal(rasgos)
    _, magia = firma_magica(rasgos)
    anatomia = describir_anatomia(rasgos)
    anatomia_str = (", " + ", ".join(anatomia)) if anatomia else ""

    partes = [
        _w(criatura_desc, 1.2),                  # ES un animalito real (zorrito, buho, gatito...)
        _w(pelaje, 1.0),                         # color real del pelaje
        _w(f"{ojos_desc} eyes", 1.3),            # color de ojos: el rasgo identitario que mas se perdia
        _w(magia, 1.0),                          # magia gentil integrada al cuerpo
    ]
    prompt = (
        ", ".join(partes) + anatomia_str + accesorios_str +
        ", a single small magical animal companion, gentle forest spirit familiar"
        ", full body, sitting, cute and huggable, big soft expressive eyes"
        ", soft rounded fluffy body, fluffy cheeks"
        f", {FONDO}"
        ", soft painterly storybook illustration, gentle watercolor shading, warm dreamy light, soft focus background"
    )

    # Negative: anti-humanoide (de v14) + anti-monstruo + anti-retrato (v15).
    negative_prompt = (
        _w("photorealistic, realistic fur texture, 3d render, hyperdetailed", 1.2) + ", " +
        _w("anthropomorphic, humanoid, bipedal, standing upright, two legs, human body", 1.3) + ", " +
        "human, person, human face, human facial features, portrait, bust, frontal face, facing camera, "
        "monster, creepy, scary, fangs, sharp teeth, evil grin, video game creature, mascot logo, "
        "human hands, wearing clothes, extra limbs, extra ears, deformed, mutated, ugly, low quality, "
        "blurry, jpeg artifacts, text, watermark, signature"
    )
    return prompt, negative_prompt


# ===================== pipeline SD (IP-Adapter + LCM) =====================

def _load_pipe():
    if _pipe["obj"] is not None:
        return _pipe["obj"]

    from diffusers import AutoPipelineForText2Image, LCMScheduler

    pipe = None
    ultimo = None
    for mid in _OPCIONES:
        try:
            print(f"[generate] cargando modelo base: {mid}")
            pipe = AutoPipelineForText2Image.from_pretrained(
                mid, torch_dtype=dtype,
                safety_checker=None, requires_safety_checker=False,
            )
            _pipe["id"] = mid
            print(f"[generate] modelo base cargado: {mid}")
            break
        except Exception as e:
            ultimo = e
            print(f"[generate] fallo {mid}: {str(e)[:120]}")
    if pipe is None:
        raise RuntimeError(f"No se pudo cargar ningun modelo SD: {ultimo}")

    # IP-Adapter plus-face: condiciona con la cara real -> preserva identidad.
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus-face_sd15.bin",
    )
    # LCM-LoRA: generacion rapida (~8 pasos). No se fusiona (compat IP-Adapter).
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

    # --- memoria ---
    # NO usar enable_attention_slicing() con IP-Adapter: reemplaza los attention
    # processors por SlicedAttnProcessor y pisa los del IP-Adapter, lo que rompe con
    # "'tuple' object has no attribute 'shape'". Solo VAE slicing (no toca attention).
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    # En GPU chica (<6 GB, ej. RTX 3050 4 GB) usamos CPU offload: mueve cada modulo
    # a la GPU solo cuando se usa, asi entra en poca VRAM (mas lento pero NO da OOM).
    # PETLY_CPU_OFFLOAD: auto (default) | 1 (forzar) | 0 (todo en VRAM).
    if device == "cuda":
        try:
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            total_gb = 0.0
        modo = os.environ.get("PETLY_CPU_OFFLOAD", "auto")
        if modo == "1" or (modo == "auto" and total_gb < 6):
            pipe.enable_model_cpu_offload()
            print(f"[generate] GPU {total_gb:.1f} GB -> CPU offload (entra en poca VRAM)")
        else:
            pipe.to(device)
            print(f"[generate] GPU {total_gb:.1f} GB -> todo en VRAM")
    else:
        pipe.to(device)

    _pipe["obj"] = pipe
    print(f"[generate] pipeline listo: IP-Adapter (scale={IP_ADAPTER_SCALE}) + LCM-LoRA")
    return pipe


def generar_imagen(rasgos: dict, rasgos_geom: dict, cara_ref, seed: int,
                   steps: int = None, guidance: float = None,
                   ip_scale: float = None) -> str:
    """rasgos + cara recortada (PIL) -> imagen generada como data URL PNG (base64).

    `cara_ref` es el recorte de la cara real (lo que ve el IP-Adapter). Si es None,
    genera solo desde el texto (sin preservar identidad), como en v14.
    """
    pipe = _load_pipe()
    if ip_scale is not None:
        pipe.set_ip_adapter_scale(ip_scale)

    rgb_pelo = (rasgos_geom or {}).get("_color_pelo_rgb")
    rgb_ojos = (rasgos_geom or {}).get("_color_ojos_rgb")
    prompt, negative = construir_prompt(rasgos, rgb_pelo=rgb_pelo, rgb_ojos=rgb_ojos)

    # Generator en CPU: reproducible y compatible con CPU offload (evita device mismatch).
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps or GEN_STEPS,
        guidance_scale=guidance if guidance is not None else GEN_GUIDANCE,
        generator=generator,
        width=512, height=512,
    )
    if cara_ref is not None:
        kwargs["ip_adapter_image"] = cara_ref   # la cara real entra por el IP-Adapter

    img = pipe(**kwargs).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
