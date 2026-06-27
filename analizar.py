"""
Script standalone de análisis académico.
Lee logs/mensajes.csv y genera los cinco gráficos sin necesidad de correr el bot.

Uso:
    python analizar.py
    python analizar.py logs/mensajes.csv          # ruta alternativa
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os

# Asegura que los imports de src/ funcionen desde cualquier directorio
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.analisis import analizar_todo, analizar_testing

if __name__ == "__main__":
    archivo         = sys.argv[1] if len(sys.argv) > 1 else "logs/mensajes.csv"
    archivo_testing = sys.argv[2] if len(sys.argv) > 2 else "logs/testing.csv"
    analizar_todo(archivo=archivo)
    analizar_testing(archivo=archivo_testing)
