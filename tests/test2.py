from pysentimiento import create_analyzer
import re
import re
import unicodedata

import sys
import os

# Agrega el directorio padre (la raíz del proyecto) al sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection import cargar_patrones
from src.detection import detectar_patrones

# Cargás una sola vez al iniciar el bot
PATRONES = cargar_patrones("config/patrones.txt")
analyzer = create_analyzer(task="hate_speech", lang="es")

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

def score_hate_speech(texto):
    #print("ANALIZANDO:  [{}]".format(texto)) 
    resultado = analyzer.predict(texto)
    hateful = resultado.probas["hateful"]
    targeted = resultado.probas["targeted"]
    aggressive = resultado.probas["aggressive"]
    prob_hateful = max(hateful, targeted, aggressive)
    return prob_hateful  # número entre 0 y 1

texto1 = "P3dazo de boludoooooooo"
print("ANALIZANDO:  ", texto1)
texto1_normalizado = normalizar(texto1)
print("NORMALIZADO:  ", texto1_normalizado)
print("DETECTAR PATRONES:  ", detectar_patrones(texto1_normalizado, PATRONES))
print("SCORE HATE SPEECH:  ", score_hate_speech(texto1_normalizado))

texto2 = "s3x000 gratis"
print("ANALIZANDO:  ", texto2)
texto2_normalizado = normalizar(texto2)
print("NORMALIZADO:  ", texto2_normalizado)
print("DETECTAR PATRONES:  ", detectar_patrones(texto2_normalizado, PATRONES))
print("SCORE HATE SPEECH:  ", score_hate_speech(texto2_normalizado))

texto3 = "te voy a llenar de semen"
print("ANALIZANDO:  ", texto3)
texto3_normalizado = normalizar(texto3)
print("NORMALIZADO:  ", texto3_normalizado)
print("DETECTAR PATRONES:  ", detectar_patrones(texto3_normalizado, PATRONES))
print("SCORE HATE SPEECH:  ", score_hate_speech(texto3_normalizado))