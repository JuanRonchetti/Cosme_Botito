from src.scoring import calcular_score_difuso

# TEMPORAL: pruebas unitarias para ajustar las funciones de detección y scoring antes de integrarlas al bot
from src.detection import normalizar
from src.detection import detectar_patrones
from src.detection import score_hate_speech

from src.scoring import graficar_membresias

# # Usuario nuevo, mensaje con muchas malas palabras, sin hate speech detectado
# score = calcular_score_difuso(
#     dens=0.8,        # densidad alta
#     hate_score=0.1,  # pysentimiento no detectó nada
#     hist=0.0,        # sin antecedentes
#     vel=0.1          # velocidad normal
# )
# print(f"Score: {score:.2f}")  # esperás algo entre 0.4 y 0.6

# # Reincidente con mensaje borderline
# score = calcular_score_difuso(
#     dens=0.3,
#     hate_score=0.2,
#     hist=0.9,        # reincidente
#     vel=0.2
# )
# print(f"Score: {score:.2f}")  # esperás algo alto por el historial

casos_prueba = [
    {"texto": "Feliz día de chupar pijas", "hist": 0.15, "vel": 0.0, "esperado": 0.7},
    {"texto": "Hola cómo están todos",     "hist": 0.0, "vel": 0.2, "esperado": 0.1},
    {"texto": "negro de mierda",           "hist": 0.0, "vel": 0.1, "esperado": 0.9},
    # etc
]

graficar_membresias()

for caso in casos_prueba:
    dens = detectar_patrones(normalizar(caso["texto"]))
    hate = score_hate_speech(caso["texto"])
    score = calcular_score_difuso(dens, hate, caso["hist"], caso["vel"])
    diferencia = abs(score - caso["esperado"])
    print(f"{caso['texto'][:40]:40} | obtenido: {score:.2f} | esperado: {caso['esperado']:.2f} | diff: {diferencia:.2f}")