"""
database.py
------------
Persistencia en SQLite (config/moderacion.db) para configuración por
servidor, canales excluidos, usuarios sospechosos, historial de
infracciones y registro de moderación. Cada función abre y cierra su
propia conexión; ninguna excepción escapa de este módulo.
"""

import os
import sqlite3
import logging

logger = logging.getLogger("database")

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_RAIZ, "config", "moderacion.db")


def _conectar():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Crea las tablas de la base de datos si todavía no existen."""
    try:
        conn = _conectar()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS config (
                guild_id TEXT,
                clave TEXT,
                valor TEXT,
                PRIMARY KEY (guild_id, clave)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS canales_excluidos (
                guild_id TEXT,
                channel_id TEXT,
                PRIMARY KEY (guild_id, channel_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS usuarios_sospechosos (
                guild_id TEXT,
                user_id TEXT,
                marcado_por TEXT,
                timestamp TEXT,
                PRIMARY KEY (guild_id, user_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS historial_usuarios (
                guild_id TEXT,
                user_id TEXT,
                infracciones INTEGER DEFAULT 0,
                ultima_infraccion TEXT,
                PRIMARY KEY (guild_id, user_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS registro_moderacion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id TEXT,
                user_id TEXT,
                mensaje TEXT,
                cat_difusa TEXT,
                score_difuso REAL,
                score_lista_negra REAL,
                score_conicet REAL,
                score_detoxify REAL,
                accion TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Error inicializando la base de datos")


# ============================================================
# CONFIG
# ============================================================

def get_config(guild_id, clave):
    """Devuelve el valor guardado para (guild_id, clave), o None si no existe."""
    try:
        conn = _conectar()
        cur = conn.cursor()
        cur.execute(
            "SELECT valor FROM config WHERE guild_id = ? AND clave = ?",
            (str(guild_id), clave),
        )
        fila = cur.fetchone()
        conn.close()
        return fila[0] if fila else None
    except Exception:
        logger.exception("Error en get_config(%s, %s)", guild_id, clave)
        return None


def set_config(guild_id, clave, valor):
    """Inserta o actualiza el valor de configuración para (guild_id, clave)."""
    try:
        conn = _conectar()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO config (guild_id, clave, valor) VALUES (?, ?, ?) "
            "ON CONFLICT(guild_id, clave) DO UPDATE SET valor = excluded.valor",
            (str(guild_id), clave, str(valor)),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Error en set_config(%s, %s, %s)", guild_id, clave, valor)


# ============================================================
# CANALES EXCLUIDOS
# ============================================================

def agregar_canal_excluido(guild_id, channel_id):
    """Marca channel_id como excluido del análisis en guild_id."""
    try:
        conn = _conectar()
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO canales_excluidos (guild_id, channel_id) VALUES (?, ?)",
            (str(guild_id), str(channel_id)),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Error en agregar_canal_excluido(%s, %s)", guild_id, channel_id)


def quitar_canal_excluido(guild_id, channel_id):
    """Quita channel_id de la lista de canales excluidos de guild_id."""
    try:
        conn = _conectar()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM canales_excluidos WHERE guild_id = ? AND channel_id = ?",
            (str(guild_id), str(channel_id)),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Error en quitar_canal_excluido(%s, %s)", guild_id, channel_id)


def canal_esta_excluido(guild_id, channel_id):
    """Devuelve True si channel_id está excluido del análisis en guild_id."""
    try:
        conn = _conectar()
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM canales_excluidos WHERE guild_id = ? AND channel_id = ?",
            (str(guild_id), str(channel_id)),
        )
        fila = cur.fetchone()
        conn.close()
        return fila is not None
    except Exception:
        logger.exception("Error en canal_esta_excluido(%s, %s)", guild_id, channel_id)
        return False


# ============================================================
# USUARIOS SOSPECHOSOS
# ============================================================

def marcar_sospechoso(guild_id, user_id, marcado_por, timestamp):
    """Marca a user_id como sospechoso en guild_id."""
    try:
        conn = _conectar()
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO usuarios_sospechosos "
            "(guild_id, user_id, marcado_por, timestamp) VALUES (?, ?, ?, ?)",
            (str(guild_id), str(user_id), str(marcado_por), timestamp),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Error en marcar_sospechoso(%s, %s)", guild_id, user_id)


def desmarcar_sospechoso(guild_id, user_id):
    """Quita la marca de sospechoso de user_id en guild_id."""
    try:
        conn = _conectar()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM usuarios_sospechosos WHERE guild_id = ? AND user_id = ?",
            (str(guild_id), str(user_id)),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Error en desmarcar_sospechoso(%s, %s)", guild_id, user_id)


def es_sospechoso(guild_id, user_id):
    """Devuelve True si user_id está marcado como sospechoso en guild_id."""
    try:
        conn = _conectar()
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM usuarios_sospechosos WHERE guild_id = ? AND user_id = ?",
            (str(guild_id), str(user_id)),
        )
        fila = cur.fetchone()
        conn.close()
        return fila is not None
    except Exception:
        logger.exception("Error en es_sospechoso(%s, %s)", guild_id, user_id)
        return False


# ============================================================
# HISTORIAL DE INFRACCIONES
# ============================================================

def get_infracciones(guild_id, user_id):
    """Devuelve la cantidad de infracciones registradas para user_id en guild_id (0 si no existe)."""
    try:
        conn = _conectar()
        cur = conn.cursor()
        cur.execute(
            "SELECT infracciones FROM historial_usuarios WHERE guild_id = ? AND user_id = ?",
            (str(guild_id), str(user_id)),
        )
        fila = cur.fetchone()
        conn.close()
        return fila[0] if fila else 0
    except Exception:
        logger.exception("Error en get_infracciones(%s, %s)", guild_id, user_id)
        return 0


def sumar_infraccion(guild_id, user_id):
    """Suma 1 al contador de infracciones de user_id en guild_id, creando el registro si no existe."""
    try:
        from datetime import datetime
        conn = _conectar()
        cur = conn.cursor()
        ts = datetime.now().isoformat()
        cur.execute(
            "INSERT INTO historial_usuarios (guild_id, user_id, infracciones, ultima_infraccion) "
            "VALUES (?, ?, 1, ?) "
            "ON CONFLICT(guild_id, user_id) DO UPDATE SET "
            "infracciones = infracciones + 1, ultima_infraccion = excluded.ultima_infraccion",
            (str(guild_id), str(user_id), ts),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Error en sumar_infraccion(%s, %s)", guild_id, user_id)


# ============================================================
# REGISTRO DE MODERACION
# ============================================================

def registrar_moderacion(guild_id, user_id, mensaje, cat_difusa, score_difuso,
                          score_lista_negra, score_conicet, score_detoxify, accion):
    """Inserta una fila en el historial de moderación de guild_id."""
    try:
        from datetime import datetime
        conn = _conectar()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO registro_moderacion "
            "(guild_id, user_id, mensaje, cat_difusa, score_difuso, score_lista_negra, "
            "score_conicet, score_detoxify, accion, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(guild_id), str(user_id), mensaje, cat_difusa, score_difuso,
                score_lista_negra, score_conicet, score_detoxify, accion,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Error en registrar_moderacion(%s, %s)", guild_id, user_id)
