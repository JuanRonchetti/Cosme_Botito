"""
detection.py
-------------
Detección de la variable "lista_negra": normaliza el texto para
evadir ofuscación básica (leetspeak, acentos, letras repetidas) y lo
compara contra los patrones regex de config/patrones.txt para obtener
un score de densidad de lenguaje ofensivo en [0, 1].
"""

import re
import unicodedata
from pathlib import Path

def normalizar(texto):
    """Pasa a minúsculas, quita acentos, revierte leetspeak básico y colapsa letras repetidas."""
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    reemplazos = {'0':'o','1':'i','3':'e','4':'a','5':'s','@':'a','$':'s'}
    for k, v in reemplazos.items():
        texto = texto.replace(k, v)
    texto = re.sub(r'(.)\1{2,}', r'\1', texto)
    return texto

ruta_actual = Path(__file__).resolve()
ruta_patrones = ruta_actual.parent.parent / 'config' / 'patrones.txt'

def cargar_patrones(ruta=ruta_patrones):
    """
    Lee patrones.txt. Soporta peso opcional por línea:
        patron          → peso 1 (normal)
        patron:2        → peso 2 (grave, cuenta doble)
        patron:3        → peso 3 (muy grave, cuenta triple)
        # comentario    → ignorado
    """
    patrones = []
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith('#'):
                continue
            if ':' in linea:
                patron, peso = linea.rsplit(':', 1)
                try:
                    peso = int(peso)
                except ValueError:
                    patron = linea
                    peso = 1
            else:
                patron = linea
                peso = 1
            patrones.append((patron.strip(), peso))
    return patrones

_patrones_cache = None

def _get_patrones():
    global _patrones_cache
    if _patrones_cache is None:
        _patrones_cache = cargar_patrones()
    return _patrones_cache

def detectar_patrones(texto, debug=False):
    """
    Normaliza `texto` y calcula el score de densidad de lista negra:
    suma los pesos de los patrones que matchean, lo normaliza por la
    longitud del mensaje y lo escala a [0, 1].
    """
    texto_norm = normalizar(texto)
    palabras   = texto_norm.split()
    longitud   = max(len(palabras), 1)

    patrones = _get_patrones()

    matches_lista = []
    peso_total = 0
    for patron, peso in patrones:
        if re.search(patron, texto_norm):
            matches_lista.append(f'{patron}(x{peso})')
            peso_total += peso

    if debug:
        print(f"  matches: {matches_lista}")
        print(f"  longitud: {longitud}, peso_total: {peso_total}")

    # Normalizar: peso_total representa "cuántas palabras ofensivas equivalentes"
    # Divisor 0.4
    score = min(peso_total / longitud / 0.4, 1.0)

    if debug:
        print(f"  score lista_negra: {score:.2f}")

    return score