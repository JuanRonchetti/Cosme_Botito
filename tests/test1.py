"""
test1.py
--------
Prototipo mínimo de bot de Discord: solo confirma que el cliente se
conecta y loguea cada mensaje recibido por consola. No hace scoring;
es el punto de partida sobre el que se construyó main.py / tests/log_bot.py.
"""

import discord

intents = discord.Intents.default()
intents.message_content = True  # necesitás el intent habilitado en el portal

print("Iniciando...")
bot = discord.Client(intents=intents)

TOKEN = ""

@bot.event
async def on_message(message):
    if message.author.bot:  # ignorar mensajes de otros bots
        return
    
    texto = message.content
    autor = message.author
    canal = message.channel

    print(f"[{autor}] **__mandó__** [{texto}] **__en__** [{canal}]")

print("Ejecutando...")
bot.run(TOKEN)