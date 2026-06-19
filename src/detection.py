from pysentimiento import create_analyzer
import re
import unicodedata
from pathlib import Path

def normalizar(texto):
    texto = texto.lower()

    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    reemplazos = {
        '0': 'o',
        '1': 'i',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '@': 'a',
        '$': 's'
    }
    for k, v in reemplazos.items():
        texto = texto.replace(k, v)

    texto = re.sub(r'(.)\1{2,}', r'\1', texto)
    return texto

# 1. Obtiene la ruta del archivo actual (src/tu_archivo.py)
ruta_actual = Path(__file__).resolve()
# 2. Sube a la raíz del proyecto (padre de src) y entra a config
ruta_patrones = ruta_actual.parent.parent / 'config' / 'patrones.txt'

# def cargar_patrones(ruta_archivo=ruta_patrones):
#     with open(ruta_archivo, "r", encoding="utf-8") as f:
#         patrones = [linea.strip() for linea in f if linea.strip() and not linea.strip().startswith('#')]
#     return patrones

def detectar_patrones(texto):
    texto = normalizar(texto)
    with open(ruta_patrones, "r", encoding="utf-8") as f:
        patrones = [linea.strip() for linea in f if linea.strip() and not linea.strip().startswith('#')]
    palabras = texto.split()
    longitud = max(len(palabras), 1)
    matches_lista = [p for p in patrones if re.search(p, texto)]
    matches = len(matches_lista)
    print(f"  matches ({matches}): {matches_lista}")
    densidad_real = matches / longitud
    print(f"  longitud: {longitud}, matches: {matches}, densidad_real: {densidad_real:.2f}, normalizada: {min(densidad_real/0.5, 1.0):.2f}")
    return min(densidad_real / 0.5, 1.0)

analyzer = create_analyzer(task="hate_speech", lang="es")

def score_hate_speech(texto):
    #print("ANALIZANDO:  [{}]".format(texto)) 
    resultado = analyzer.predict(texto)
    hateful = resultado.probas["hateful"]
    targeted = resultado.probas["targeted"]
    aggressive = resultado.probas["aggressive"]
    prob_hateful = max(hateful, targeted, aggressive)
    return prob_hateful  # número entre 0 y 1