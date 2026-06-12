# from src.scoring import calcular_score_difuso

# # TEMPORAL: pruebas unitarias para ajustar las funciones de detección y scoring antes de integrarlas al bot
# from src.detection import normalizar
# from src.detection import detectar_patrones
# from src.detection import score_hate_speech

# from src.scoring import graficar_membresias

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

# casos_prueba = [
#     {"texto": "Hola cómo están todos",     "hist": 0.0, "vel": 0.2, "esperado": 0.1},
#     {"texto": "tu comentario me parece deplorable", "hist": 0.0, "vel": 0.1, "esperado": 0.3},
#     {"texto": "vos no entendes nada", "hist": 0.0, "vel": 0.1, "esperado": 0.2},
#     {"texto": "andate a cagar", "hist": 0.0, "vel": 0.1, "esperado": 0.6},
#     {"texto": "Feliz día de chupar pijas", "hist": 0.15, "vel": 0.0, "esperado": 0.7},
#     {"texto": "negro de mierda",           "hist": 0.0, "vel": 0.1, "esperado": 0.95},
#     {"texto": "son todos putos",           "hist": 0.0, "vel": 0.1, "esperado": 0.9}
#     # etc
# ]

# graficar_membresias()

# for caso in casos_prueba:
#     texto_norm = normalizar(caso["texto"])
#     dens = detectar_patrones(texto_norm)
#     hate_score = score_hate_speech(caso["texto"])
#     score = calcular_score_difuso(dens, hate_score, caso["hist"], caso["vel"])
#     diferencia = abs(score - caso["esperado"])
#     print(f"{caso['texto'][:40]:40} | dens: {dens:.2f} | hate: {hate_score:.2f} | obtenido: {score:.2f} | esperado: {caso['esperado']:.2f} | diff: {diferencia:.2f}")


import discord
import csv
import os
import re
import unicodedata
from datetime import datetime
from collections import defaultdict
import time
from src.scoring import calcular_score_difuso
from pysentimiento import create_analyzer
from dotenv import load_dotenv

load_dotenv()

ruta_patrones = "config/patrones.txt"

def _cargar_patrones():
    with open(ruta_patrones, "r", encoding="utf-8") as f:
        return [linea.strip() for linea in f if linea.strip()]

PATRONES = _cargar_patrones()

# ============================================================
# CONFIGURACIÓN
# ============================================================

TOKEN = os.getenv("DISCORD_TOKEN")
CANAL_LOG_ID = None          # None = loguea todos los canales
ARCHIVO_LOG  = "logs/mensajes.csv"
UMBRAL_LOG   = 0.0           # loguea todo, cambiá a 0.3 para solo sospechosos

os.makedirs("logs", exist_ok=True)

analyzer = create_analyzer(task="hate_speech", lang="es")

# ============================================================
# INICIALIZAR CSV
# ============================================================

HEADERS = [
    "timestamp", "autor_id", "autor_nombre", "canal",
    "mensaje_original", "mensaje_normalizado",
    "matches_lista", "longitud_palabras",
    "densidad_real", "densidad_norm",
    "hate_hateful", "hate_targeted", "hate_aggressive", "hate_max",
    "historial", "velocidad",
    "score_obtenido", "score_esperado",
    "accion_sugerida"
]

def inicializar_csv():
    if not os.path.exists(ARCHIVO_LOG):
        with open(ARCHIVO_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()

# ============================================================
# FUNCIONES DE SCORING DETALLADO
# ============================================================

def normalizar(texto):
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

def detectar_patrones_detallado(texto_norm):
    palabras   = texto_norm.split()
    longitud   = max(len(palabras), 1)
    matches    = [p for p in PATRONES if re.search(p, texto_norm)]
    dens_real  = len(matches) / longitud
    dens_norm  = min(dens_real / 0.5, 1.0)
    return matches, longitud, dens_real, dens_norm

def hate_detallado(texto_original):
    resultado  = analyzer.predict(texto_original)
    hateful    = resultado.probas["hateful"]
    targeted   = resultado.probas["targeted"]
    aggressive = resultado.probas["aggressive"]
    return hateful, targeted, aggressive, max(hateful, targeted, aggressive)

def decidir_accion(score):
    if score < 0.30:   return "ignorar" 
    if score < 0.55:   return "alertar"
    if score < 0.75:   return "borrar_y_alertar"
    return "timeout"

# ============================================================
# PARSEAR SCORE ESPERADO DEL MENSAJE
# Formato: texto del mensaje (0.7)
# ============================================================

def parsear_mensaje(contenido):
    match = re.search(r'\((\d+(?:\.\d+)?)\)\s*$', contenido)
    if match:
        score_esperado = float(match.group(1))
        texto = contenido[:match.start()].strip()
        return texto, score_esperado
    return contenido, None

# ============================================================
# GUARDAR EN CSV
# ============================================================

def guardar_log(row):
    with open(ARCHIVO_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow(row)

# ============================================================
# BOT
# ============================================================

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = discord.Client(intents=intents)

# Historial de velocidad en memoria: {usuario_id: [timestamps]}
historial_tiempos = defaultdict(list)

def calcular_velocidad(usuario_id):
    ahora = time.time()
    historial_tiempos[usuario_id] = [
        t for t in historial_tiempos[usuario_id] if ahora - t < 30
    ]
    historial_tiempos[usuario_id].append(ahora)
    return min(len(historial_tiempos[usuario_id]) / 10, 1.0)

@bot.event
async def on_ready():
    inicializar_csv()
    print(f"Bot conectado como {bot.user}")
    print(f"Logueando en: {ARCHIVO_LOG}")

@bot.event
async def on_message(message):
    # Ignorar bots
    if message.author.bot:
        return

    # Filtrar canal si está configurado
    if CANAL_LOG_ID and message.channel.id != CANAL_LOG_ID:
        return

    contenido = message.content
    if not contenido.strip():
        return

    # Parsear score esperado si viene en el mensaje
    texto, score_esperado = parsear_mensaje(contenido)

    # Normalizar
    texto_norm = normalizar(texto)

    # Calcular señales
    matches, longitud, dens_real, dens_norm = detectar_patrones_detallado(texto_norm)
    hateful, targeted, aggressive, hate_max = hate_detallado(texto)
    velocidad = calcular_velocidad(str(message.author.id))
    historial = 0.0  # sin DB por ahora, siempre 0

    # Score difuso
    score = calcular_score_difuso(dens_norm, hate_max, historial, velocidad)
    accion = decidir_accion(score)

    # Solo loguear si supera el umbral
    if score < UMBRAL_LOG and score_esperado is None:
        return

    row = {
        "timestamp":          datetime.now().isoformat(),
        "autor_id":           str(message.author.id),
        "autor_nombre":       str(message.author.name),
        "canal":              str(message.channel.name),
        "mensaje_original":   texto,
        "mensaje_normalizado":texto_norm,
        "matches_lista":      "|".join(matches),
        "longitud_palabras":  longitud,
        "densidad_real":      round(dens_real, 4),
        "densidad_norm":      round(dens_norm, 4),
        "hate_hateful":       round(hateful, 4),
        "hate_targeted":      round(targeted, 4),
        "hate_aggressive":    round(aggressive, 4),
        "hate_max":           round(hate_max, 4),
        "historial":          historial,
        "velocidad":          round(velocidad, 4),
        "score_obtenido":     round(score, 4),
        "score_esperado":     score_esperado if score_esperado is not None else "",
        "accion_sugerida":    accion,
    }

    guardar_log(row)

    # Feedback en el mismo canal si venía con score esperado
    if score_esperado is not None:
        diff = abs(score - score_esperado)
        emoji = "✅" if diff <= 0.15 else "⚠️" if diff <= 0.30 else "❌"
        await message.reply(
            f"{emoji} **Score:** {score:.2f} | **Esperado:** {score_esperado} | **Diff:** {diff:.2f} | **Acción:** {accion}",
            mention_author=False
        )

bot.run(TOKEN)