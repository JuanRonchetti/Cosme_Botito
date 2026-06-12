# Cosme Botito
## Discord Fuzzy Moderation Bot

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Discord.py](https://img.shields.io/badge/discord.py-2.7.1-5865F2)
![Status](https://img.shields.io/badge/status-development-orange)
![License](https://img.shields.io/badge/license-Academic-lightgrey)

Bot de moderación asistida para Discord que combina análisis de lenguaje natural y lógica difusa para detectar mensajes potencialmente tóxicos en tiempo real.

## Objetivo

Reducir la carga operativa de los moderadores mediante la detección automática y clasificación de mensajes problemáticos, manteniendo siempre la decisión final en manos humanas.

## Características

* Análisis en tiempo real de mensajes.
* Detección de hate speech en español mediante `pysentimiento`.
* Sistema de scoring continuo de toxicidad (0-1).
* Clasificación por niveles de severidad.
* Alertas automáticas para moderadores.
* Registro histórico de infracciones y acciones.
* Menú interactivo para timeout, kick, ban y otras acciones.

## Entradas del Sistema

* Contenido del mensaje.
* Densidad de lenguaje ofensivo.
* Score de hate speech.
* Historial de infracciones.
* Frecuencia de mensajes.

## Salidas

* Score de toxicidad.
* Nivel de severidad.
* Alertas al canal de moderación.
* Registro de eventos y decisiones.

## Tecnologías

* Python
* discord.py
* scikit-fuzzy
* pysentimiento
* SQLite / JSON
