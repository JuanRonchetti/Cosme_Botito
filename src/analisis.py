import matplotlib
matplotlib.use('Agg')

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from src.scoring import calcular_score_difuso, graficar_membresias, CONFIG

ARCHIVO_LOG_DEFAULT = "logs/mensajes.csv"

# ============================================================
# CASOS DE VALIDACIÓN — ground truth manual del experto
# ============================================================

CASOS_VALIDACION = [
    {"nombre": "Saludo normal",            "dens": 0.00, "hate": 0.02, "hist": 0.00, "vel": 0.10, "esperado": 0.08},
    {"nombre": "Lenguaje suave",           "dens": 0.15, "hate": 0.05, "hist": 0.00, "vel": 0.15, "esperado": 0.18},
    {"nombre": "Insulto leve",             "dens": 0.28, "hate": 0.10, "hist": 0.00, "vel": 0.10, "esperado": 0.30},
    {"nombre": "Insulto sin hate",         "dens": 0.65, "hate": 0.05, "hist": 0.00, "vel": 0.10, "esperado": 0.48},
    {"nombre": "Hate sin insultos",        "dens": 0.05, "hate": 0.55, "hist": 0.00, "vel": 0.10, "esperado": 0.52},
    {"nombre": "Insulto + hate medio",     "dens": 0.50, "hate": 0.35, "hist": 0.00, "vel": 0.10, "esperado": 0.65},
    {"nombre": "Reincidente leve",         "dens": 0.20, "hate": 0.10, "hist": 0.75, "vel": 0.20, "esperado": 0.62},
    {"nombre": "Flood sin contenido",      "dens": 0.05, "hate": 0.02, "hist": 0.00, "vel": 0.85, "esperado": 0.22},
    {"nombre": "Flood + insultos",         "dens": 0.50, "hate": 0.20, "hist": 0.00, "vel": 0.90, "esperado": 0.70},
    {"nombre": "Hate extremo reincidente", "dens": 0.80, "hate": 0.75, "hist": 0.55, "vel": 0.30, "esperado": 0.88},
    {"nombre": "Maxima toxicidad",         "dens": 1.00, "hate": 1.00, "hist": 1.00, "vel": 1.00, "esperado": 0.95},
]

# ============================================================
# LEER CSV
# ============================================================

def leer_csv(archivo=ARCHIVO_LOG_DEFAULT):
    if not os.path.exists(archivo):
        return []
    with open(archivo, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ============================================================
# GRÁFICO 2 — Validación del sistema difuso
# ============================================================

def grafico_validacion(directorio='logs'):
    print("Validando sistema difuso con casos de referencia...")
    nombres, obtenidos, esperados, errores = [], [], [], []

    for caso in CASOS_VALIDACION:
        score = calcular_score_difuso(
            caso["dens"], caso["hate"], caso["hist"], caso["vel"], debug=False
        )
        nombres.append(caso["nombre"])
        obtenidos.append(score)
        esperados.append(caso["esperado"])
        errores.append(abs(score - caso["esperado"]))
        print(f"  {caso['nombre'][:35]:35} | obtenido: {score:.3f} | esperado: {caso['esperado']:.3f} | diff: {errores[-1]:.3f}")

    rmse = float(np.sqrt(np.mean(np.array(errores) ** 2)))
    mae  = float(np.mean(errores))
    print(f"\n  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Validacion del Sistema Difuso  --  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}', fontsize=13)

    ax = axes[0]
    colors = ['green' if e <= 0.10 else 'orange' if e <= 0.20 else 'red' for e in errores]
    ax.scatter(esperados, obtenidos, c=colors, s=80, zorder=3)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Ideal')
    ax.plot([0, 1], [0.10, 1.10], 'g--', linewidth=0.8, alpha=0.4)
    ax.plot([0, 1], [-0.10, 0.90], 'g--', linewidth=0.8, alpha=0.4, label='+-0.10')
    for i, n in enumerate(nombres):
        ax.annotate(n, (esperados[i], obtenidos[i]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel('Score esperado (ground truth)')
    ax.set_ylabel('Score obtenido (sistema)')
    ax.set_title('Esperado vs Obtenido')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'verde <=0.10  naranja <=0.20  rojo >0.20',
            transform=ax.transAxes, va='top', fontsize=7, style='italic')

    ax2 = axes[1]
    bar_colors = ['green' if e <= 0.10 else 'orange' if e <= 0.20 else 'red' for e in errores]
    ax2.barh(range(len(nombres)), errores, color=bar_colors)
    ax2.set_yticks(range(len(nombres)))
    ax2.set_yticklabels(nombres, fontsize=8)
    ax2.axvline(x=0.10, color='green',  linestyle='--', linewidth=1, label='0.10')
    ax2.axvline(x=0.20, color='orange', linestyle='--', linewidth=1, label='0.20')
    ax2.set_xlabel('Error absoluto |esperado - obtenido|')
    ax2.set_title('Error por caso de prueba')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    ruta = os.path.join(directorio, '02_validacion_sistema.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

# ============================================================
# GRÁFICO 3 — Confusión y ROC
# ============================================================

def grafico_confusion_roc(filas, directorio='logs'):
    etiquetados = [f for f in filas if f.get("score_esperado") not in ("", None)]
    if len(etiquetados) < 5:
        print(f"  Confusion/ROC: solo {len(etiquetados)} mensajes etiquetados (minimo 5).")
        print("  Etiqueta mensajes con el formato: texto (0.85)")
        return

    UMBRAL_BIN = 0.50
    y_true  = np.array([float(f["score_esperado"]) >= UMBRAL_BIN for f in etiquetados], dtype=int)
    y_score = np.array([float(f["score_obtenido"])  for f in etiquetados])

    y_pred = (y_score >= UMBRAL_BIN).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    cm = np.array([[tn, fp], [fn, tp]])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(y_true)

    thresholds = np.linspace(0, 1, 300)
    tprs, fprs = [], []
    for t in thresholds:
        yp  = (y_score >= t).astype(int)
        _tp = np.sum((y_true == 1) & (yp == 1))
        _tn = np.sum((y_true == 0) & (yp == 0))
        _fp = np.sum((y_true == 0) & (yp == 1))
        _fn = np.sum((y_true == 1) & (yp == 0))
        tprs.append(_tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0)
        fprs.append(_fp / (_fp + _tn) if (_fp + _tn) > 0 else 0.0)
    auc_val = float(np.trapezoid(tprs[::-1], fprs[::-1]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'Confusion y ROC -- {len(etiquetados)} mensajes etiquetados  (umbral: {UMBRAL_BIN})',
        fontsize=12
    )

    ax = axes[0]
    labels = ['No toxico', 'Toxico']
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusion')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > thresh else 'black')
    metricas = (
        f"Accuracy:  {accuracy:.3f}\n"
        f"Precision: {precision:.3f}\n"
        f"Recall:    {recall:.3f}\n"
        f"F1-score:  {f1:.3f}\n\n"
        f"TP={tp}  FP={fp}\n"
        f"FN={fn}  TN={tn}"
    )
    ax.text(1.08, 0.5, metricas, transform=ax.transAxes, fontsize=10,
            va='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    ax2 = axes[1]
    ax2.plot(fprs, tprs, 'b-', linewidth=2, label=f'ROC (AUC = {auc_val:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatorio')
    fpr_op = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr_op = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ax2.scatter([fpr_op], [tpr_op], color='red', s=80, zorder=5, label=f'Umbral={UMBRAL_BIN}')
    ax2.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax2.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    ax2.set_title('Curva ROC')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    ruta = os.path.join(directorio, '03_confusion_roc.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}  (AUC={auc_val:.3f}, F1={f1:.3f})")

# ============================================================
# GRÁFICO 4 — Sensibilidad analítica
# ============================================================

def grafico_sensibilidad(directorio='logs'):
    print("  Calculando sensibilidad analitica...")
    x = np.linspace(0.0, 1.0, 50)

    variables = [
        ('densidad',  0, 'steelblue'),
        ('hate',      1, 'tomato'),
        ('historial', 2, 'seagreen'),
        ('velocidad', 3, 'goldenrod'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Analisis de Sensibilidad -- Efecto de cada variable sobre toxicidad', fontsize=13)

    for idx, (nombre, var_idx, color) in enumerate(variables):
        ax = axes[idx // 2][idx % 2]

        for fijo_val, estilo, etiq in [(0.10, '-', 'otros = 0.10'), (0.50, '--', 'otros = 0.50')]:
            y = []
            for val in x:
                args = [fijo_val, fijo_val, fijo_val, fijo_val]
                args[var_idx] = float(val)
                try:
                    s = calcular_score_difuso(args[0], args[1], args[2], args[3], debug=False)
                except Exception:
                    s = 0.0
                y.append(s)
            ax.plot(x, y, linestyle=estilo, color=color, linewidth=2, label=etiq)

        for i_k, (k, cfg) in enumerate(CONFIG[nombre].items()):
            p = cfg['params']
            ax.axvspan(p[0], p[3], alpha=0.07, color=f'C{i_k}')
            mid = (p[1] + p[2]) / 2
            ax.text(mid, 0.03, k, ha='center', fontsize=7, color=f'C{i_k}')

        ax.set_xlabel(f'Valor de {nombre} [0-1]')
        ax.set_ylabel('Toxicidad [0-1]')
        ax.set_title(f'Sensibilidad: {nombre}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    ruta = os.path.join(directorio, '04_sensibilidad.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}")

# ============================================================
# GRÁFICO 5 — Tiempos de respuesta
# ============================================================

def grafico_tiempos(filas, directorio='logs'):
    tiempos = []
    for f in filas:
        t = f.get("tiempo_ms", "")
        if t not in ("", None):
            try:
                tiempos.append(float(t))
            except ValueError:
                pass

    if len(tiempos) < 2:
        print(f"  Tiempos: {len(tiempos)} medicion/es (minimo 2). Necesitas correr el bot al menos una vez con el nuevo schema.")
        return

    arr = np.array(tiempos)
    media   = float(np.mean(arr))
    mediana = float(np.median(arr))
    p95     = float(np.percentile(arr, 95))
    p99     = float(np.percentile(arr, 99))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Tiempos de procesamiento  ({len(arr)} mensajes)', fontsize=13)

    ax = axes[0]
    ax.hist(arr, bins=min(30, len(arr)), color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(media,   color='red',    linestyle='--', linewidth=1.5, label=f'Media: {media:.0f} ms')
    ax.axvline(mediana, color='orange', linestyle='--', linewidth=1.5, label=f'Mediana: {mediana:.0f} ms')
    ax.axvline(p95,     color='purple', linestyle=':',  linewidth=1.5, label=f'P95: {p95:.0f} ms')
    ax.set_xlabel('Tiempo (ms)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribucion de tiempos')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.boxplot(arr, vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
    ax2.set_ylabel('Tiempo (ms)')
    ax2.set_title('Box plot')
    ax2.set_xticklabels(['procesamiento'])
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(range(len(arr)), arr, 'o-', markersize=2, linewidth=0.8,
             color='steelblue', alpha=0.7)
    ax3.axhline(media, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax3.set_xlabel('N de mensaje')
    ax3.set_ylabel('Tiempo (ms)')
    ax3.set_title('Tiempos a lo largo de la sesion')
    ax3.grid(True, alpha=0.3)
    stats_txt = (
        f"n        = {len(arr)}\n"
        f"Media    = {media:.1f} ms\n"
        f"Mediana  = {mediana:.1f} ms\n"
        f"Desv.std = {np.std(arr):.1f} ms\n"
        f"Minimo   = {arr.min():.1f} ms\n"
        f"Maximo   = {arr.max():.1f} ms\n"
        f"P95      = {p95:.1f} ms\n"
        f"P99      = {p99:.1f} ms"
    )
    ax3.text(0.98, 0.98, stats_txt, transform=ax3.transAxes, fontsize=8,
             va='top', ha='right', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    plt.tight_layout()
    ruta = os.path.join(directorio, '05_tiempos.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {ruta}  (media={media:.0f}ms  P95={p95:.0f}ms)")

# ============================================================
# PUNTO DE ENTRADA UNIFICADO
# ============================================================

def analizar_todo(archivo=ARCHIVO_LOG_DEFAULT, directorio='logs'):
    """Genera los cinco gráficos académicos. Puede llamarse desde main.py o analizar.py."""
    os.makedirs(directorio, exist_ok=True)
    print(f"\n=== Analisis academico ({archivo}) ===")

    graficar_membresias(ruta=os.path.join(directorio, '01_membresias.png'), mostrar=False)
    grafico_validacion(directorio)

    filas = leer_csv(archivo)
    if not filas:
        print("  CSV vacio o no encontrado — saltando graficos de datos reales.")
    else:
        print(f"  Leyendo {len(filas)} filas de {archivo}")
        grafico_confusion_roc(filas, directorio)
        grafico_sensibilidad(directorio)
        grafico_tiempos(filas, directorio)

    print(f"  Graficos guardados en {directorio}/")
    print("==========================================\n")
