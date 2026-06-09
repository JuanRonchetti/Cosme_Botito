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
    
    texto = message.content  # acá tenés el texto del mensaje
    autor = message.author
    canal = message.channel
    
    # acá llamás a tu función de scoring
    #score = calcular_toxicidad(texto)
    print(f"[{autor}] **__mandó__** [{texto}] **__en__** [{canal}]") #→ score: {score}")

print("Ejecutando...")
bot.run(TOKEN)