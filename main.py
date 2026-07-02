"""
main.py
-------
Bot de Discord de moderación. Escucha cada mensaje, lo puntúa con el
sistema difuso (src/scoring.py) y reporta los casos relevantes en un
canal configurable, además de detectar spam multicanal. Toda la
configuración por servidor y el historial de infracciones se
persisten en SQLite (src/database.py). Expone slash commands de
configuración restringidos a administradores.
"""

import os
import time as _time
from datetime import datetime, timedelta
from typing import Union

import discord
from discord import app_commands
from dotenv import load_dotenv

from src.database import (
    init_db,
    get_config,
    set_config,
    agregar_canal_excluido,
    quitar_canal_excluido,
    canal_esta_excluido,
    marcar_sospechoso,
    desmarcar_sospechoso,
    es_sospechoso,
    get_infracciones,
    sumar_infraccion,
    registrar_moderacion,
)
from src.detection import detectar_patrones
from src.modelos import score_conicet, score_detoxify
from src.scoring import calcular_score_difuso, etiquetar_output

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")

# ============================================================
# BOT
# ============================================================

intents = discord.Intents.default()
intents.message_content = True
intents.members = True


class Cosme(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)


bot = Cosme()


@bot.event
async def on_ready():
    init_db()
    try:
        await bot.tree.sync()
    except Exception as e:
        print(f"[Cosme] Error sincronizando slash commands: {e}")
    print(f"[Cosme] Conectado como {bot.user} — {len(bot.guilds)} servidor(es).")


# ============================================================
# ERRORES DE SLASH COMMANDS
# ============================================================

@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions):
        mensaje = "❌ Necesitás permisos de administrador para usar este comando."
    else:
        mensaje = "❌ Ocurrió un error al ejecutar el comando."
        print(f"[Cosme] Error en slash command: {error}")
    try:
        if interaction.response.is_done():
            await interaction.followup.send(mensaje, ephemeral=True)
        else:
            await interaction.response.send_message(mensaje, ephemeral=True)
    except Exception as e:
        print(f"[Cosme] No se pudo responder el error del comando: {e}")


# ============================================================
# SLASH COMMANDS DE CONFIGURACION (solo admins)
# ============================================================

@bot.tree.command(name="cosme_canal", description="Configura el canal o hilo donde Cosme envía sus reportes.")
@app_commands.describe(canal="Canal o hilo de destino de los reportes")
@app_commands.guild_only()
@app_commands.checks.has_permissions(administrator=True)
async def cosme_canal(interaction: discord.Interaction, canal: Union[discord.TextChannel, discord.Thread]):
    set_config(interaction.guild_id, "canal_reportes", canal.id)
    await interaction.response.send_message(f"✅ Canal de reportes configurado en {canal.mention}.", ephemeral=True)


@bot.tree.command(name="cosme_rol", description="Configura el rol que se pinguea ante mensajes de alta toxicidad o spam.")
@app_commands.describe(rol="Rol a mencionar en las alertas")
@app_commands.guild_only()
@app_commands.checks.has_permissions(administrator=True)
async def cosme_rol(interaction: discord.Interaction, rol: discord.Role):
    set_config(interaction.guild_id, "rol_alerta", rol.id)
    await interaction.response.send_message(f"✅ Rol de alerta configurado en {rol.mention}.", ephemeral=True)


@bot.tree.command(name="cosme_excluir", description="Agrega o quita un canal de la exclusión de análisis.")
@app_commands.describe(canal="Canal a excluir/incluir del análisis")
@app_commands.guild_only()
@app_commands.checks.has_permissions(administrator=True)
async def cosme_excluir(interaction: discord.Interaction, canal: discord.TextChannel):
    guild_id = interaction.guild_id
    if canal_esta_excluido(guild_id, canal.id):
        quitar_canal_excluido(guild_id, canal.id)
        await interaction.response.send_message(f"✅ {canal.mention} ya no está excluido del análisis.", ephemeral=True)
    else:
        agregar_canal_excluido(guild_id, canal.id)
        await interaction.response.send_message(f"🚫 {canal.mention} fue excluido del análisis.", ephemeral=True)


@bot.tree.command(name="cosme_activar", description="Activa o desactiva a Cosme en este servidor.")
@app_commands.guild_only()
@app_commands.checks.has_permissions(administrator=True)
async def cosme_activar(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    # Sin registro previo el bot se considera desactivado (ver gate en on_message).
    if get_config(guild_id, "activo") == "1":
        set_config(guild_id, "activo", "0")
        await interaction.response.send_message("⏸️ Bot desactivado", ephemeral=True)
    else:
        set_config(guild_id, "activo", "1")
        await interaction.response.send_message("✅ Bot activado", ephemeral=True)


@bot.tree.command(name="cosme_sospechoso", description="Marca o desmarca a un usuario como sospechoso.")
@app_commands.describe(usuario="Usuario a marcar/desmarcar")
@app_commands.guild_only()
@app_commands.checks.has_permissions(administrator=True)
async def cosme_sospechoso(interaction: discord.Interaction, usuario: discord.Member):
    guild_id = interaction.guild_id
    if es_sospechoso(guild_id, usuario.id):
        desmarcar_sospechoso(guild_id, usuario.id)
        await interaction.response.send_message(f"✅ {usuario.mention} ya no está marcado como sospechoso.", ephemeral=True)
    else:
        marcar_sospechoso(guild_id, usuario.id, str(interaction.user.id), datetime.now().isoformat())
        await interaction.response.send_message(f"🔍 {usuario.mention} fue marcado como sospechoso.", ephemeral=True)


# ============================================================
# DETECCION DE SPAM MULTICANAL
# ============================================================

SPAM_VENTANA_SEG = 10
SPAM_CANALES_MAX = 3

_actividad_canales = {}  # {user_id: {channel_id: (timestamp, message)}}


def _registrar_actividad(user_id, channel_id, message):
    """Registra el canal/timestamp/mensaje actual del usuario y descarta entradas de hace más de SPAM_VENTANA_SEG."""
    ahora = _time.time()
    canales = _actividad_canales.setdefault(user_id, {})
    canales[channel_id] = (ahora, message)
    for cid in [c for c, (ts, _m) in canales.items() if ahora - ts > SPAM_VENTANA_SEG]:
        del canales[cid]
    return canales


async def _reportar_spam(message, eventos, tiempo_ms):
    """eventos: snapshot inmutable de [(channel_id, (timestamp, message)), ...] tomado antes de awaitear nada."""
    guild = message.guild

    # Borra todos los mensajes de la ráfaga, no solo el que disparó la detección.
    for _cid, (_ts, msg) in eventos:
        try:
            await msg.delete()
        except discord.Forbidden:
            print(f"[Cosme] Sin permisos para borrar mensaje de spam de {message.author} en '{guild.name}'.")
        except discord.NotFound:
            pass
        except Exception as e:
            print(f"[Cosme] Error borrando mensaje de spam: {e}")

    if message.author.guild_permissions.administrator:
        timeout_campo = "Usuario es administrador, no se aplicó timeout"
    else:
        try:
            await message.author.timeout(
                timedelta(hours=1),
                reason="Spam multicanal detectado automáticamente",
            )
            timeout_campo = "⏳ Timeout aplicado: 1 hora"
        except discord.Forbidden:
            # Al bot le falta el permiso "Moderate Members" o su rol está por debajo del rol del usuario.
            timeout_campo = "⚠️ No se pudo aplicar timeout (revisar permisos o jerarquía de roles)"
        except Exception as e:
            print(f"[Cosme] Error al aplicar timeout: {e}")
            timeout_campo = "⚠️ No se pudo aplicar timeout (revisar permisos o jerarquía de roles)"

    registrar_moderacion(
        guild.id, message.author.id, message.content, "spam", None,
        None, None, None, "timeout_spam",
    )

    canal_id = get_config(guild.id, "canal_reportes")
    if not canal_id:
        print(f"[Cosme] Spam multicanal de {message.author} en '{guild.name}' sin canal de reportes configurado.")
        return
    # get_channel_or_thread: el canal de reportes puede ser un hilo, y get_channel nunca los resuelve.
    canal_reportes = guild.get_channel_or_thread(int(canal_id))
    if canal_reportes is None:
        print(f"[Cosme] Canal de reportes {canal_id} no encontrado en '{guild.name}'.")
        return

    nombres_canales = []
    for cid, _ in eventos:
        c = guild.get_channel_or_thread(cid)
        nombres_canales.append(c.mention if c else f"`{cid}`")

    embed = discord.Embed(
        title="🚫 Spam multicanal detectado",
        color=0x9B59B6,
        timestamp=message.created_at,
    )
    embed.add_field(name="Usuario", value=f"{message.author.mention} ({message.author.name})", inline=False)
    embed.add_field(name="Canales (últimos 10s)", value=str(len(eventos)), inline=False)
    embed.add_field(name="Lista de canales", value="\n".join(nombres_canales) or "-", inline=False)
    embed.add_field(name="Timeout", value=timeout_campo, inline=False)
    embed.set_footer(text=f"{tiempo_ms:.2f} ms")

    contenido = None
    allowed = discord.AllowedMentions.none()
    rol_id = get_config(guild.id, "rol_alerta")
    if rol_id:
        contenido = f"<@&{rol_id}>"
        allowed = discord.AllowedMentions(roles=True)

    try:
        await canal_reportes.send(content=contenido, embed=embed, allowed_mentions=allowed)
    except discord.Forbidden:
        print(f"[Cosme] Sin permisos para enviar reporte de spam en canal {canal_id}.")
    except Exception as e:
        print(f"[Cosme] Error enviando reporte de spam: {e}")


# ============================================================
# EMBED DE MODERACION
# ============================================================

_TITULOS = {
    "alta": "🚨 Mensaje de alta toxicidad",
    "media": "⚠️ Mensaje detectado",
    "sospechoso": "👁️ Usuario vigilado",
}

_COLORES = {
    "alta": 0xE74C3C,
    "media": 0xF1C40F,
    "sospechoso": 0xE67E22,
}


def _construir_embed(message, variante, sospechoso, lista_negra_score, conicet_score,
                      detoxify_score, score_difuso, cat_difusa, infracciones, tiempo_ms):
    embed = discord.Embed(
        title=_TITULOS[variante],
        color=_COLORES[variante],
        timestamp=message.created_at,
    )
    embed.add_field(name="Autor", value=f"{message.author.mention} ({message.author.name})", inline=False)
    embed.add_field(name="Canal", value=message.channel.mention, inline=False)

    contenido = message.content
    if len(contenido) > 1024:
        contenido = contenido[:1021] + "..."
    embed.add_field(name="Mensaje", value=contenido, inline=False)
    embed.add_field(name="Categoría", value=cat_difusa.upper(), inline=False)

    scores_txt = (
        f"Lista negra: {lista_negra_score:.2f}\n"
        f"CONICET:     {conicet_score:.2f}\n"
        f"Detoxify:    {detoxify_score:.2f}\n"
        f"Score difuso: {score_difuso:.2f}"
    )
    embed.add_field(name="Scores", value=f"```{scores_txt}```", inline=True)
    embed.add_field(name="Historial", value=f"{infracciones} infracciones previas", inline=False)

    if sospechoso:
        embed.add_field(name="🔍 Usuario vigilado", value="Sí", inline=False)

    embed.set_footer(text=f"{tiempo_ms:.2f} ms")
    return embed


async def _enviar_reporte(message, variante, con_ping, **datos_embed):
    guild = message.guild
    canal_id = get_config(guild.id, "canal_reportes")
    if not canal_id:
        print(f"[Cosme] Mensaje de {message.author} en '{guild.name}' no reportado: falta canal de reportes.")
        return
    canal_reportes = guild.get_channel_or_thread(int(canal_id))
    if canal_reportes is None:
        print(f"[Cosme] Canal de reportes {canal_id} no encontrado en '{guild.name}'.")
        return

    embed = _construir_embed(message, variante, **datos_embed)

    contenido = None
    allowed = discord.AllowedMentions.none()
    if con_ping:
        rol_id = get_config(guild.id, "rol_alerta")
        if rol_id:
            contenido = f"<@&{rol_id}>"
            allowed = discord.AllowedMentions(roles=True)

    try:
        await canal_reportes.send(content=contenido, embed=embed, allowed_mentions=allowed)
    except discord.Forbidden:
        print(f"[Cosme] Sin permisos para enviar reportes en canal {canal_id}.")
    except Exception as e:
        print(f"[Cosme] Error enviando reporte: {e}")


# ============================================================
# ON_MESSAGE / ON_MESSAGE_EDIT
# ============================================================

def _debe_analizar(message):
    """Gates comunes a mensajes nuevos y editados: guild válida, autor no-bot, bot activo, canal no excluido, con contenido."""
    if message.guild is None:
        return False
    if message.author.bot:
        return False
    guild_id = message.guild.id
    if get_config(guild_id, "activo") != "1":
        return False
    if canal_esta_excluido(guild_id, message.channel.id):
        return False
    if not message.content or not message.content.strip():
        return False
    return True


async def _analizar_toxicidad(message, t_inicio):
    guild_id = message.guild.id

    infracciones = get_infracciones(guild_id, message.author.id)
    historial_score = min(infracciones * 0.15, 1.0)

    lista_negra_score = detectar_patrones(message.content)
    conicet_s = score_conicet(message.content)
    detoxify_s = score_detoxify(message.content)
    score_difuso = calcular_score_difuso(lista_negra_score, conicet_s, detoxify_s, historial_score)
    cat_difusa, _mu = etiquetar_output(score_difuso)

    tiempo_ms = (_time.perf_counter() - t_inicio) * 1000

    if cat_difusa == "alta":
        sumar_infraccion(guild_id, message.author.id)

    sospechoso = es_sospechoso(guild_id, message.author.id)

    if cat_difusa == "alta":
        variante, con_ping = "alta", True
    elif cat_difusa == "media":
        variante, con_ping = "media", False
    elif sospechoso:
        variante, con_ping = "sospechoso", False
    else:
        return

    await _enviar_reporte(
        message, variante, con_ping,
        sospechoso=sospechoso,
        lista_negra_score=lista_negra_score,
        conicet_score=conicet_s,
        detoxify_score=detoxify_s,
        score_difuso=score_difuso,
        cat_difusa=cat_difusa,
        infracciones=infracciones,
        tiempo_ms=tiempo_ms,
    )

    registrar_moderacion(
        guild_id, message.author.id, message.content, cat_difusa, score_difuso,
        lista_negra_score, conicet_s, detoxify_s, variante,
    )


@bot.event
async def on_message(message):
    if not _debe_analizar(message):
        return

    t_inicio = _time.perf_counter()

    canales_recientes = _registrar_actividad(message.author.id, message.channel.id, message)
    if len(canales_recientes) > SPAM_CANALES_MAX:
        tiempo_ms = (_time.perf_counter() - t_inicio) * 1000
        # Snapshot antes de cualquier await: canales_recientes es el dict vivo de
        # _actividad_canales y un on_message concurrente del mismo usuario podría
        # mutarlo mientras _reportar_spam itera (RuntimeError: dict changed size).
        eventos = list(canales_recientes.items())
        # Corta la ráfaga ya reportada para no re-disparar spam en cada mensaje siguiente
        # mientras las entradas viejas siguen dentro de la ventana de 10s.
        _actividad_canales.pop(message.author.id, None)
        await _reportar_spam(message, eventos, tiempo_ms)
        return

    await _analizar_toxicidad(message, t_inicio)


@bot.event
async def on_message_edit(before, after):
    if before.content == after.content:
        return
    if not _debe_analizar(after):
        return
    await _analizar_toxicidad(after, _time.perf_counter())


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Falta DISCORD_TOKEN en el archivo .env")
    bot.run(TOKEN)
