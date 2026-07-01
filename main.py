import matplotlib
matplotlib.use('Agg')  # backend sin GUI, antes de importar pyplot o scoring

import discord
import csv
import os
import re
import unicodedata
from datetime import datetime
import time as _time

from src.scoring import calcular_score_difuso, graficar_membresias, etiquetar_inputs, etiquetar_output
from src.analisis import grafico_validacion, grafico_confusion_roc, grafico_sensibilidad, grafico_tiempos, leer_csv, analizar_testing
from src.modelos import score_conicet, score_detoxify
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURACION
# ============================================================

TOKEN        = os.getenv("DISCORD_TOKEN")
CANAL_LOG_ID = None          # None = todos los canales
ARCHIVO_LOG  = "logs/mensajes.csv"

os.makedirs("logs", exist_ok=True)

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
    "conicet_score", "detoxify_score",
    "historial", "score_obtenido",
    "score_esperado", "accion_sugerida", "tiempo_ms",
]

# Headers del schema anterior con hate/velocidad, para migrar CSVs de versiones previas
HEADERS_LEGACY = [
    "timestamp", "tipo", "autor_id", "autor_nombre", "canal",
    "mensaje_original", "mensaje_normalizado",
    "matches_lista", "longitud_palabras",
    "densidad_real", "densidad_norm",
    "hate_hateful", "hate_targeted", "hate_aggressive", "hate_max",
    "historial", "velocidad",
    "score_obtenido", "score_esperado",
    "accion_sugerida", "tiempo_ms",
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
        csv.DictWriter(f, fieldnames=HEADERS).writerow(row)

# ============================================================
# CSV DE TESTING (feedback de usuario via botones)
# ============================================================

ARCHIVO_TESTING = "logs/testing.csv"

HEADERS_TESTING = [
    "timestamp_mensaje", "timestamp_feedback",
    "autor_id", "autor_nombre", "canal",
    "mensaje_original",
    "lista_negra_score", "lista_negra_cat",
    "conicet_score",     "conicet_cat",
    "detoxify_score",    "detoxify_cat",
    "historial_score",   "historial_cat",
    "score_difuso",      "cat_difusa",
    "cat_esperada",
]

def inicializar_testing_csv():
    if not os.path.exists(ARCHIVO_TESTING):
        with open(ARCHIVO_TESTING, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=HEADERS_TESTING).writeheader()

def guardar_testing(row):
    with open(ARCHIVO_TESTING, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=HEADERS_TESTING).writerow(row)

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

def decidir_accion(score):
    if score < 0.30:  return "ignorar"
    if score < 0.55:  return "alertar"
    if score < 0.75:  return "borrar_y_alertar"
    return "timeout"


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
    analizar_testing()
    print("================================\n")

# ============================================================
# BOT
# ============================================================

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = discord.Client(intents=intents)


@bot.event
async def on_ready():
    inicializar_csv()
    inicializar_testing_csv()
    print(f"Bot conectado como {bot.user}")
    print(f"Logueando en: {ARCHIVO_LOG}")


_CAT_EMOJI = {"baja": "🟢", "media": "🟡", "alta": "🔴"}

# ============================================================
# UI — Botones de feedback de toxicidad
# ============================================================

class ToxicidadView(discord.ui.View):
    def __init__(self, autor_id, datos):
        super().__init__(timeout=600)
        self.autor_id = autor_id
        self.datos    = datos

    async def _registrar(self, interaction, cat_esperada):
        if interaction.user.id != self.autor_id:
            await interaction.response.send_message(
                "Solo el autor del mensaje puede calificarlo.", ephemeral=True
            )
            return
        guardar_testing({
            **self.datos,
            "timestamp_feedback": datetime.now().isoformat(),
            "cat_esperada":       cat_esperada,
        })
        self.stop()
        await interaction.response.edit_message(
            content=interaction.message.content + f"\n✅ **Registrado:** `{cat_esperada}`",
            view=None,
        )

    @discord.ui.button(label="baja",    style=discord.ButtonStyle.success)
    async def btn_baja(self, interaction, button):
        await self._registrar(interaction, "baja")

    @discord.ui.button(label="media",   style=discord.ButtonStyle.secondary)
    async def btn_media(self, interaction, button):
        await self._registrar(interaction, "media")

    @discord.ui.button(label="alta",    style=discord.ButtonStyle.danger)
    async def btn_alta(self, interaction, button):
        await self._registrar(interaction, "alta")


async def _analizar_y_loguear(message, tipo):
    contenido = message.content
    if not contenido.strip():
        return

    t_inicio = _time.perf_counter()

    texto      = contenido.strip()
    texto_norm = normalizar(texto)

    matches, longitud, dens_real, dens_norm = detectar_patrones_detallado(texto_norm)
    conicet_s = score_conicet(texto)
    detox_s   = score_detoxify(texto)
    historial = 0.0

    score  = calcular_score_difuso(dens_norm, conicet_s, detox_s, historial)
    accion = decidir_accion(score)

    cats_in         = etiquetar_inputs(dens_norm, conicet_s, detox_s, historial)
    cat_out, mu_out = etiquetar_output(score)

    tiempo_ms = (_time.perf_counter() - t_inicio) * 1000
    ts        = datetime.now().isoformat()

    ln_cat, _ = cats_in['lista_negra']
    co_cat, _ = cats_in['CONICET']
    dt_cat, _ = cats_in['detoxify']
    hu_cat, _ = cats_in['historial_usuario']

    guardar_log({
        "timestamp":           ts,
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
        "conicet_score":       round(conicet_s, 4),
        "detoxify_score":      round(detox_s, 4),
        "historial":           historial,
        "score_obtenido":      round(score, 4),
        "score_esperado":      "",
        "accion_sugerida":     accion,
        "tiempo_ms":           round(tiempo_ms, 2),
    })

    datos_testing = {
        "timestamp_mensaje":  ts,
        "autor_id":           str(message.author.id),
        "autor_nombre":       str(message.author.name),
        "canal":              str(message.channel.name),
        "mensaje_original":   texto,
        "lista_negra_score":  round(dens_norm, 4),
        "lista_negra_cat":    ln_cat,
        "conicet_score":      round(conicet_s, 4),
        "conicet_cat":        co_cat,
        "detoxify_score":     round(detox_s, 4),
        "detoxify_cat":       dt_cat,
        "historial_score":    historial,
        "historial_cat":      hu_cat,
        "score_difuso":       round(score, 4),
        "cat_difusa":         cat_out,
    }

    prefijo = "[edicion] " if tipo == "edicion" else ""
    tabla = (
        "```\n"
        f"Lista negra  │ {dens_norm:.4f} │ {ln_cat}\n"
        f"CONICET      │ {conicet_s:.4f} │ {co_cat}\n"
        f"Detoxify     │ {detox_s:.4f} │ {dt_cat}\n"
        f"Historial    │ {historial:.4f} │ {hu_cat}\n"
        "─────────────────────────────────\n"
        f"Score difuso │ {score:.4f} │ {cat_out}  µ={mu_out:.2f}\n"
        "```"
    )
    await message.reply(
        f"{prefijo}📊 **Análisis** · ⏱ `{tiempo_ms:.0f}ms`\n"
        f"{tabla}\n"
        f"{_CAT_EMOJI.get(cat_out, '❓')} Categoría: `{cat_out}`",
        view=ToxicidadView(message.author.id, datos_testing),
        mention_author=False,
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
