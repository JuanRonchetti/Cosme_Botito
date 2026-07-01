"""
analizar_dataset.py
-------------------
Lee config/dataset.csv (formato: mensaje,cat_esperada sin header),
corre el pipeline completo de scoring y genera:

  logs/data/dataset_results.csv     — scores detallados por mensaje
  logs/data/dataset_confusion.png   — matriz de confusión 4×4 + métricas
  logs/data/dataset_roc.png         — curvas ROC one-vs-rest

Uso:
    python analizar_dataset.py
    python analizar_dataset.py config/mi_dataset.csv
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import csv
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.scoring import calcular_score_difuso, etiquetar_inputs, etiquetar_output
from src.modelos import score_conicet, score_detoxify

# ============================================================
# CONFIGURACIÓN
# ============================================================

CATS        = ['baja', 'media', 'alta']
DIR_SALIDA  = os.path.join("logs", "data")
RUTA_OUT    = os.path.join(DIR_SALIDA, "dataset_results.csv")

HEADERS_OUT = [
    'mensaje_original', 'cat_esperada',
    'lista_negra_score', 'lista_negra_cat',
    'conicet_score',     'conicet_cat',
    'detoxify_score',    'detoxify_cat',
    'historial_score',   'historial_cat',
    'score_difuso',      'cat_difusa',
    'correcto',
]

# ============================================================
# UTILIDADES DE TEXTO
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
    for k, v in {'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '@': 'a', '$': 's'}.items():
        texto = texto.replace(k, v)
    return re.sub(r'(.)\1{2,}', r'\1', texto)


def _detectar(texto_norm, patrones):
    palabras  = texto_norm.split()
    longitud  = max(len(palabras), 1)
    matches   = [p for p in patrones if re.search(p, texto_norm)]
    dens_real = len(matches) / longitud
    return dens_real, min(dens_real / 0.5, 1.0)


# ============================================================
# LEER DATASET
# ============================================================

def leer_dataset(ruta):
    """Lee CSV sin header formato: mensaje,cat_esperada"""
    filas = []
    with open(ruta, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            # Última columna es la categoría; el resto es el mensaje (puede contener comas)
            cat = row[-1].strip()
            msg = ','.join(row[:-1]).strip()
            if cat not in CATS or not msg:
                continue
            filas.append({'mensaje': msg, 'cat_esperada': cat})
    return filas


# ============================================================
# PIPELINE DE SCORING
# ============================================================

def procesar_dataset(filas, patrones):
    resultados = []
    n = len(filas)
    print(f"\n  Procesando {n} mensajes...")

    for i, fila in enumerate(filas):
        texto      = fila['mensaje']
        texto_norm = _normalizar(texto)
        _, dens_norm = _detectar(texto_norm, patrones)
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

        cat_esp = fila['cat_esperada']
        ok = cat_out == cat_esp

        if (i + 1) % 50 == 0 or i == 0:
            simbolo = '✓' if ok else '✗'
            print(f"    [{i+1:4}/{n}] {simbolo}  pred={cat_out:8}  esp={cat_esp:8}  {texto[:50]}")

        resultados.append({
            'mensaje_original':  texto,
            'cat_esperada':      cat_esp,
            'lista_negra_score': round(dens_norm, 4),
            'lista_negra_cat':   ln_cat,
            'conicet_score':     round(conicet_s, 4),
            'conicet_cat':       co_cat,
            'detoxify_score':    round(detox_s, 4),
            'detoxify_cat':      dt_cat,
            'historial_score':   historial,
            'historial_cat':     hu_cat,
            'score_difuso':      round(score, 4),
            'cat_difusa':        cat_out,
            'correcto':          '1' if ok else '0',
        })

    return resultados


# ============================================================
# CONSOLA — RESUMEN
# ============================================================

def imprimir_resumen(resultados):
    n         = len(resultados)
    correctos = sum(1 for r in resultados if r['correcto'] == '1')
    accuracy  = correctos / n if n > 0 else 0.0

    print(f"\n{'='*52}")
    print(f"  ACCURACY GLOBAL: {correctos}/{n}  ({accuracy:.1%})")
    print(f"{'='*52}")

    por_cat = {c: {'ok': 0, 'total': 0} for c in CATS}
    for r in resultados:
        esp = r['cat_esperada']
        if esp in por_cat:
            por_cat[esp]['total'] += 1
            if r['correcto'] == '1':
                por_cat[esp]['ok'] += 1

    print(f"\n  {'Categoría':<12} {'Correctos':>10} {'Total':>7} {'Recall':>8}")
    print(f"  {'-'*42}")
    for cat in CATS:
        ok    = por_cat[cat]['ok']
        total = por_cat[cat]['total']
        rec   = ok / total if total > 0 else 0.0
        print(f"  {cat:<12} {ok:>10} {total:>7} {rec:>8.1%}")
    print()


# ============================================================
# GRÁFICO — MATRIZ DE CONFUSIÓN + MÉTRICAS
# ============================================================

def grafico_confusion(resultados, directorio):
    nc = len(CATS)
    cm = np.zeros((nc, nc), dtype=int)
    for r in resultados:
        if r['cat_esperada'] in CATS and r['cat_difusa'] in CATS:
            i = CATS.index(r['cat_esperada'])
            j = CATS.index(r['cat_difusa'])
            cm[i, j] += 1

    n        = int(cm.sum())
    accuracy = float(np.trace(cm)) / n if n > 0 else 0.0

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f'Dataset — Categoría esperada vs predicha  ({n} mensajes)', fontsize=13)
    ax_cm  = fig.add_subplot(1, 2, 1)
    ax_met = fig.add_subplot(1, 2, 2)

    ax_cm.set_xlim(-0.5, nc - 0.5)
    ax_cm.set_ylim(nc - 0.5, -0.5)
    ax_cm.set_aspect('equal')
    for i in range(nc):
        for j in range(nc):
            color = '#88c999' if i == j else '#f08080'
            ax_cm.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                          facecolor=color, edgecolor='white', linewidth=2))
            ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center',
                       fontsize=15, color='black', fontweight='bold')
    ax_cm.set_xticks(range(nc))
    ax_cm.set_yticks(range(nc))
    ax_cm.set_xticklabels(CATS, fontsize=10)
    ax_cm.set_yticklabels(CATS, fontsize=10)
    ax_cm.set_xlabel('Predicho  (cat_difusa)')
    ax_cm.set_ylabel('Real  (cat_esperada)')
    ax_cm.set_title('Matriz de Confusión')

    ax_met.axis('off')
    lineas = [f"Accuracy total: {accuracy:.3f}   (n={n})\n\n"]
    lineas.append(f"{'Clase':<12} {'Prec':>6} {'Recall':>7} {'F1':>6} {'n':>4}\n")
    lineas.append("─" * 40 + "\n")
    for i, cat in enumerate(CATS):
        tp_c   = cm[i, i]
        fp_c   = int(cm[:, i].sum()) - tp_c
        fn_c   = int(cm[i, :].sum()) - tp_c
        prec   = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        recall = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c   = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        ni     = int(cm[i, :].sum())
        lineas.append(f"{cat:<12} {prec:>6.3f} {recall:>7.3f} {f1_c:>6.3f} {ni:>4}\n")
    ax_met.text(0.05, 0.95, "".join(lineas), ha='left', va='top', fontsize=12,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
                transform=ax_met.transAxes)

    plt.tight_layout()
    ruta = os.path.join(directorio, 'dataset_confusion.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}  (accuracy={accuracy:.3f})")


# ============================================================
# GRÁFICO — ROC ONE-VS-REST
# ============================================================

def grafico_roc(resultados, directorio):
    """
    Curvas ROC one-vs-rest para cada categoría.
    Score continuo: score_difuso, con transformación por clase:
      baja    → 1 - score_difuso  (más bajo = más probable baja)
      media   → 1 - |score_difuso - 0.37|  (pico aprox de media)
      alta    → 1 - |score_difuso - 0.60|  (pico aprox de alta)
      extrema → score_difuso
    """
    scores   = np.array([float(r['score_difuso']) for r in resultados])
    esperadas = [r['cat_esperada'] for r in resultados]

    # Score continuo por clase (proxy razonable con un solo score difuso)
    score_por_clase = {
        'baja':  1.0 - scores,
        'media': 1.0 - np.abs(scores - 0.35),
        'alta':  scores,
    }

    colores = ['steelblue', 'seagreen', 'tomato', 'goldenrod']
    thresholds = np.linspace(0, 1, 300)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title('Curvas ROC — One-vs-Rest por categoría', fontsize=12)

    for cat, color in zip(CATS, colores):
        y_true = np.array([1 if e == cat else 0 for e in esperadas])
        if y_true.sum() == 0:
            continue
        sc = score_por_clase[cat]

        tprs, fprs = [], []
        for t in thresholds:
            yp  = (sc >= t).astype(int)
            tp  = int(np.sum((y_true == 1) & (yp == 1)))
            fp  = int(np.sum((y_true == 0) & (yp == 1)))
            fn  = int(np.sum((y_true == 1) & (yp == 0)))
            tn  = int(np.sum((y_true == 0) & (yp == 0)))
            tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

        auc = float(np.trapezoid(tprs[::-1], fprs[::-1]))
        ax.plot(fprs, tprs, color=color, linewidth=2, label=f'{cat}  (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatorio')
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    ruta = os.path.join(directorio, 'dataset_roc.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analiza dataset.csv con el pipeline de scoring.')
    parser.add_argument('dataset', nargs='?', default='config/dataset.csv',
                        help='Ruta al dataset CSV (default: config/dataset.csv)')
    args = parser.parse_args()

    os.makedirs(DIR_SALIDA, exist_ok=True)

    print(f"=== Análisis de dataset: {args.dataset} ===")

    filas = leer_dataset(args.dataset)
    if not filas:
        print("ERROR: no se encontraron filas válidas en el dataset.")
        raise SystemExit(1)
    print(f"  {len(filas)} mensajes cargados")

    patrones = _cargar_patrones()

    resultados = procesar_dataset(filas, patrones)

    # CSV de resultados
    with open(RUTA_OUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS_OUT)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"\n  Resultados guardados: {RUTA_OUT}")

    imprimir_resumen(resultados)

    print("  Generando gráficos...")
    grafico_confusion(resultados, DIR_SALIDA)
    grafico_roc(resultados, DIR_SALIDA)

    print("\n=== Listo. ===\n")
