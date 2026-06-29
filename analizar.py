"""
Script standalone de análisis académico.

Uso:
    python analizar.py                              # análisis completo (mensajes + testing)
    python analizar.py logs/mensajes.csv            # ruta alternativa para mensajes.csv
    python analizar.py mensajes.csv testing.csv     # ambas rutas explícitas
    python analizar.py --rescore                    # re-scorea testing.csv con el sistema actual
    python analizar.py testing.csv --rescore        # ruta alternativa + rescore
    python analizar.py --directorio resultados/     # carpeta de salida personalizada
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import csv
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

# Asegura que los imports de src/ funcionen desde cualquier directorio
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.analisis import (
    analizar_todo, analizar_testing,
    grafico_confusion_testing, CATS_TOXICIDAD, leer_csv,
)

# ============================================================
# UTILIDADES DE TEXTO  (mismas que main.py)
# ============================================================

_ruta_patrones = Path('config/patrones.txt')

def _cargar_patrones():
    with open(_ruta_patrones, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f
                if line.strip() and not line.strip().startswith('#')]

def _normalizar(texto):
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    for k, v in {'0':'o','1':'i','3':'e','4':'a','5':'s','@':'a','$':'s'}.items():
        texto = texto.replace(k, v)
    return re.sub(r'(.)\1{2,}', r'\1', texto)

def _detectar(texto_norm, patrones):
    palabras  = texto_norm.split()
    longitud  = max(len(palabras), 1)
    matches   = [p for p in patrones if re.search(p, texto_norm)]
    dens_real = len(matches) / longitud
    return dens_real, min(dens_real / 0.5, 1.0)

# ============================================================
# RE-SCORING
# ============================================================

HEADERS_RESCORE = [
    'timestamp_original', 'autor_id', 'autor_nombre', 'canal',
    'mensaje_original',
    'lista_negra_score', 'lista_negra_cat',
    'conicet_score',     'conicet_cat',
    'detoxify_score',    'detoxify_cat',
    'historial_score',   'historial_cat',
    'score_difuso',      'cat_difusa',
    'cat_esperada',
]

def rescorar_y_analizar(archivo_testing='logs/testing.csv', directorio='logs'):
    from src.scoring import calcular_score_difuso, etiquetar_inputs, etiquetar_output
    from src.modelos import score_conicet, score_detoxify

    filas   = leer_csv(archivo_testing)
    validos = [
        f for f in filas
        if f.get('cat_esperada', '') in CATS_TOXICIDAD and f.get('mensaje_original', '')
    ]

    if not validos:
        print(f"  Re-scoring: ninguna muestra con cat_esperada válida en {archivo_testing}.")
        return

    patrones = _cargar_patrones()
    ts       = datetime.now().strftime('%Y-%m-%d_%H-%M')
    os.makedirs(directorio, exist_ok=True)

    print(f"\n=== Re-scoring — {len(validos)} mensajes de {archivo_testing} ===")

    resultados = []
    for i, f in enumerate(validos):
        texto      = f['mensaje_original']
        texto_norm = _normalizar(texto)
        dens_real, dens_norm = _detectar(texto_norm, patrones)
        conicet_s  = score_conicet(texto)
        detox_s    = score_detoxify(texto)
        historial  = 0.0

        score    = calcular_score_difuso(dens_norm, conicet_s, detox_s, historial)
        cats_in  = etiquetar_inputs(dens_norm, conicet_s, detox_s, historial)
        cat_out, _ = etiquetar_output(score)

        ln_cat, _ = cats_in['lista_negra']
        co_cat, _ = cats_in['CONICET']
        dt_cat, _ = cats_in['detoxify']
        hu_cat, _ = cats_in['historial_usuario']

        cat_esp = f['cat_esperada']
        ok      = '✓' if cat_out == cat_esp else '✗'
        print(f"  [{i+1:3}/{len(validos)}] {ok}  pred={cat_out:8}  esp={cat_esp:8}  {texto[:55]}")

        resultados.append({
            'timestamp_original': f.get('timestamp_mensaje', ''),
            'autor_id':           f.get('autor_id', ''),
            'autor_nombre':       f.get('autor_nombre', ''),
            'canal':              f.get('canal', ''),
            'mensaje_original':   texto,
            'lista_negra_score':  round(dens_norm, 4),
            'lista_negra_cat':    ln_cat,
            'conicet_score':      round(conicet_s, 4),
            'conicet_cat':        co_cat,
            'detoxify_score':     round(detox_s, 4),
            'detoxify_cat':       dt_cat,
            'historial_score':    historial,
            'historial_cat':      hu_cat,
            'score_difuso':       round(score, 4),
            'cat_difusa':         cat_out,
            'cat_esperada':       cat_esp,
        })

    correctos = sum(1 for r in resultados if r['cat_difusa'] == r['cat_esperada'])
    print(f"\n  Accuracy: {correctos}/{len(resultados)} ({correctos/len(resultados):.1%})")

    ruta_csv = os.path.join(directorio, f'rescoring_{ts}.csv')
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=HEADERS_RESCORE)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"  CSV guardado: {ruta_csv}")

    grafico_confusion_testing(
        resultados, directorio,
        nombre=f'rescoring_confusion_{ts}.png',
    )
    print("==========================================\n")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Análisis académico del bot de moderación.')
    parser.add_argument('mensajes',     nargs='?', default='logs/mensajes.csv',
                        help='Ruta a mensajes.csv (default: logs/mensajes.csv)')
    parser.add_argument('testing',      nargs='?', default='logs/testing.csv',
                        help='Ruta a testing.csv  (default: logs/testing.csv)')
    parser.add_argument('--rescore',    action='store_true',
                        help='Re-scorea los mensajes de testing.csv con el sistema actual')
    parser.add_argument('--directorio', default='logs',
                        help='Carpeta de salida para gráficos (default: logs)')
    args = parser.parse_args()

    analizar_todo(archivo=args.mensajes, directorio=args.directorio)
    analizar_testing(archivo=args.testing, directorio=args.directorio)

    if args.rescore:
        rescorar_y_analizar(archivo_testing=args.testing, directorio=args.directorio)
