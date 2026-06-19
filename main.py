import matplotlib
matplotlib.use('Agg')  # backend sin GUI, antes de importar pyplot o scoring

import discord
import csv
import os
import re
import unicodedata
from datetime import datetime
from collections import defaultdict
import time as _time

from src.scoring import calcular_score_difuso, graficar_membresias
from src.analisis import grafico_validacion, grafico_confusion_roc, grafico_sensibilidad, grafico_tiempos, leer_csv
from pysentimiento import create_analyzer
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURACION
# ============================================================

TOKEN        = os.getenv("DISCORD_TOKEN")
CANAL_LOG_ID = None          # None = todos los canales
ARCHIVO_LOG  = "logs/mensajes.csv"
UMBRAL_LOG   = 0.0           # 0.0 = loguear todo

os.makedirs("logs", exist_ok=True)

analyzer = create_analyzer(task="hate_speech", lang="es")

ruta_patrones = "config/patrones.txt"

def _cargar_patrones():
    with open(ruta_patrones, "r", encoding="utf-8") as f:
        return [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith('#')
        ]

PATRONES = _cargar_patrones()

# ============================================================
# CSV
# ============================================================

HEADERS = [
    "timestamp", "tipo", "autor_id", "autor_nombre", "canal",
    "mensaje_original", "mensaje_normalizado",
    "matches_lista", "longitud_palabras",
    "densidad_real", "densidad_norm",
    "hate_hateful", "hate_targeted", "hate_aggressive", "hate_max",
    "historial", "velocidad",
    "score_obtenido", "score_esperado",
    "accion_sugerida", "tiempo_ms",
]

# Headers del schema anterior (sin tiempo_ms), para migrar CSVs sin header
HEADERS_LEGACY = [
    "timestamp", "autor_id", "autor_nombre", "canal",
    "mensaje_original", "mensaje_normalizado",
    "matches_lista", "longitud_palabras",
    "densidad_real", "densidad_norm",
    "hate_hateful", "hate_targeted", "hate_aggressive", "hate_max",
    "historial", "velocidad",
    "score_obtenido", "score_esperado",
    "accion_sugerida",
]

def inicializar_csv():
    if not os.path.exists(ARCHIVO_LOG):
        with open(ARCHIVO_LOG, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=HEADERS).writeheader()
        return

    with open(ARCHIVO_LOG, "r", encoding="utf-8") as f:
        todas = list(csv.reader(f))

    if not todas:
        with open(ARCHIVO_LOG, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=HEADERS).writeheader()
        return

    # Detectar si la primera fila es un header o un dato
    primera = todas[0]
    nombres_validos = set(HEADERS) | set(HEADERS_LEGACY)
    tiene_header = all(col in nombres_validos for col in primera if col)

    headers_actuales = primera if tiene_header else HEADERS_LEGACY
    filas_datos = todas[1:] if tiene_header else todas

    headers_faltantes = [h for h in HEADERS if h not in headers_actuales]
    if not headers_faltantes:
        return

    rows = [
        {**dict(zip(headers_actuales, fila)), **{h: '' for h in headers_faltantes}}
        for fila in filas_datos if fila
    ]

    with open(ARCHIVO_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [CSV] Migrado en el lugar: {len(rows)} filas preservadas, columnas nuevas: {headers_faltantes}")

def guardar_log(row):
    with open(ARCHIVO_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow(row)


# ============================================================
# PROCESAMIENTO DE MENSAJES
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
    palabras  = texto_norm.split()
    longitud  = max(len(palabras), 1)
    matches   = [p for p in PATRONES if re.search(p, texto_norm)]
    dens_real = len(matches) / longitud
    dens_norm = min(dens_real / 0.5, 1.0)
    return matches, longitud, dens_real, dens_norm

def hate_detallado(texto_original):
    resultado  = analyzer.predict(texto_original)
    hateful    = resultado.probas["hateful"]
    targeted   = resultado.probas["targeted"]
    aggressive = resultado.probas["aggressive"]
    return hateful, targeted, aggressive, max(hateful, targeted, aggressive)

def decidir_accion(score):
    if score < 0.30:  return "ignorar"
    if score < 0.55:  return "alertar"
    if score < 0.75:  return "borrar_y_alertar"
    return "timeout"

def parsear_mensaje(contenido):
    match = re.search(r'\((\d+(?:\.\d+)?)\)\s*$', contenido)
    if match:
        return contenido[:match.start()].strip(), float(match.group(1))
    return contenido, None

# ============================================================
# ANALISIS  (delega a src/analisis.py)
# ============================================================

def analisis_inicial():
    print("=== Analisis inicial ===")
    graficar_membresias(ruta='logs/01_membresias.png', mostrar=False)
    grafico_validacion()
    print("========================\n")


def analizar_post():
    print("\n=== Analisis post-ejecucion ===")
    filas = leer_csv(ARCHIVO_LOG)
    if not filas:
        print("  Log vacio o inexistente.")
    else:
        print(f"  Leyendo {len(filas)} filas de {ARCHIVO_LOG}")
        grafico_confusion_roc(filas)
        grafico_sensibilidad()
        grafico_tiempos(filas)
    print("================================\n")

# ============================================================
# BOT
# ============================================================

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = discord.Client(intents=intents)
historial_tiempos = defaultdict(list)


def calcular_velocidad(usuario_id):
    ahora = _time.time()
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


async def _analizar_y_loguear(message, tipo):
    contenido = message.content
    if not contenido.strip():
        return

    t_inicio = _time.perf_counter()

    texto, score_esperado = parsear_mensaje(contenido)
    texto_norm = normalizar(texto)

    matches, longitud, dens_real, dens_norm = detectar_patrones_detallado(texto_norm)
    hateful, targeted, aggressive, hate_max = hate_detallado(texto)
    velocidad = calcular_velocidad(str(message.author.id))
    historial = 0.0

    score  = calcular_score_difuso(dens_norm, hate_max, historial, velocidad)
    accion = decidir_accion(score)

    tiempo_ms = (_time.perf_counter() - t_inicio) * 1000

    if score < UMBRAL_LOG and score_esperado is None:
        return

    row = {
        "timestamp":           datetime.now().isoformat(),
        "tipo":                tipo,
        "autor_id":            str(message.author.id),
        "autor_nombre":        str(message.author.name),
        "canal":               str(message.channel.name),
        "mensaje_original":    texto,
        "mensaje_normalizado": texto_norm,
        "matches_lista":       "|".join(matches),
        "longitud_palabras":   longitud,
        "densidad_real":       round(dens_real, 4),
        "densidad_norm":       round(dens_norm, 4),
        "hate_hateful":        round(hateful, 4),
        "hate_targeted":       round(targeted, 4),
        "hate_aggressive":     round(aggressive, 4),
        "hate_max":            round(hate_max, 4),
        "historial":           historial,
        "velocidad":           round(velocidad, 4),
        "score_obtenido":      round(score, 4),
        "score_esperado":      score_esperado if score_esperado is not None else "",
        "accion_sugerida":     accion,
        "tiempo_ms":           round(tiempo_ms, 2),
    }

    guardar_log(row)

    if score_esperado is not None:
        diff  = abs(score - score_esperado)
        emoji = "✅" if diff <= 0.15 else "⚠️" if diff <= 0.30 else "❌"
        prefijo = "[edicion] " if tipo == "edicion" else ""
        await message.reply(
            f"{prefijo}{emoji} **Score:** {score:.2f} | **Esperado:** {score_esperado} | "
            f"**Diff:** {diff:.2f} | **Accion:** {accion} | **t:** {tiempo_ms:.0f}ms",
            mention_author=False
        )


@bot.event
async def on_message(message):
    if message.author.bot:
        return
    if CANAL_LOG_ID and message.channel.id != CANAL_LOG_ID:
        return
    await _analizar_y_loguear(message, "nuevo")


@bot.event
async def on_message_edit(before, after):
    if after.author.bot:
        return
    if CANAL_LOG_ID and after.channel.id != CANAL_LOG_ID:
        return
    if before.content == after.content:
        return
    await _analizar_y_loguear(after, "edicion")

# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    analisis_inicial()
    try:
        bot.run(TOKEN)
    except KeyboardInterrupt:
        pass
    finally:
        analizar_post()
