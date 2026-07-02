"""
optimizar_pesos_reglas.py
-------------------------
Algoritmo genético que optimiza los PESOS de las reglas difusas.

Qué optimiza:
  - 29 pesos (uno por regla), cada uno entre 0.1 y 2.0
  - Peso > 1.0 amplifica la regla
  - Peso < 1.0 silencia la regla

Qué NO optimiza:
  - Las membresías de entrada ni de salida (eso lo hace optimizar_membresias.py)
  - La estructura de las reglas

Acepta dos formatos de entrada:
  1. config/dataset.csv  (sin header: mensaje,cat_esperada)
     → pre-computa los scores con los modelos ML al inicio (una sola vez)
  2. logs/testing.csv    (con header, scores ya calculados)
     → los usa directamente

Los resultados se guardan en logs/optimizacion/ con timestamp.

Uso:
    python optimizar_pesos_reglas.py                   # usa config/dataset.csv
    python optimizar_pesos_reglas.py logs/testing.csv  # usa scores pre-computados
"""

import sys
import csv
import os
import re
import unicodedata
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# El script vive en scripts/; agregamos la raíz del proyecto a sys.path
# (para "from src..." ) y hacemos chdir ahí (para rutas relativas config/, logs/).
_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RAIZ)
os.chdir(_RAIZ)

# ============================================================
# CONFIGURACIÓN DEL ALGORITMO GENÉTICO
# ============================================================

POBLACION_SIZE = 40
GENERACIONES   = 50
TASA_MUTACION  = 0.15
TASA_CRUCE     = 0.7
ELITE_SIZE     = 4
TORNEO_SIZE    = 4
SEMILLA        = 99

PESO_MIN = 0.1
PESO_MAX = 2.0

random.seed(SEMILLA)
np.random.seed(SEMILLA)

# ============================================================
# RUTAS
# ============================================================

RUTA_CSV   = sys.argv[1] if len(sys.argv) > 1 else "config/dataset.csv"
TS         = datetime.now().strftime('%Y-%m-%d_%H-%M')
DIR_SALIDA = os.path.join("logs", "optimizacion")
os.makedirs(DIR_SALIDA, exist_ok=True)

RUTA_INCREMENTAL = os.path.join(DIR_SALIDA, "mejor_pesos.txt")

# ============================================================
# UTILIDADES DE TEXTO
# ============================================================

def _normalizar(texto):
    """Pasa a minúsculas, quita acentos, revierte leetspeak básico y colapsa letras repetidas."""
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    for k, v in {'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '@': 'a', '$': 's'}.items():
        texto = texto.replace(k, v)
    return re.sub(r'(.)\1{2,}', r'\1', texto)


def _cargar_patrones():
    """Lee config/patrones.txt y devuelve las líneas no vacías ni comentadas."""
    with open(Path('config/patrones.txt'), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]


def _detectar(texto_norm, patrones):
    """Devuelve la densidad normalizada de lista negra para texto_norm."""
    palabras  = texto_norm.split()
    longitud  = max(len(palabras), 1)
    matches   = [p for p in patrones if re.search(p, texto_norm)]
    dens_real = len(matches) / longitud
    return min(dens_real / 0.5, 1.0)


# ============================================================
# CARGAR DATASET (mismo detector de formato que optimizar_membresias.py)
# ============================================================

_CATS_VALIDAS = ('baja', 'media', 'alta')


def _es_formato_precomputado(ruta):
    """Devuelve True si el CSV tiene header con lista_negra_score."""
    with open(ruta, 'r', encoding='utf-8') as f:
        primera = f.readline()
    return 'lista_negra_score' in primera


def _cargar_precomputado(ruta):
    """Lee un CSV con header y scores ML ya calculados (columnas *_score) para cada mensaje."""
    filas = []
    with open(ruta, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                esperada = row["cat_esperada"].strip()
                if esperada not in _CATS_VALIDAS:
                    continue
                filas.append({
                    "lista_negra": float(row["lista_negra_score"]),
                    "conicet":     float(row["conicet_score"]),
                    "detoxify":    float(row["detoxify_score"]),
                    "historial":   float(row["historial_score"]),
                    "esperada":    esperada,
                })
            except (KeyError, ValueError):
                continue
    return filas


def _cargar_y_precomputar(ruta):
    """Lee mensaje,cat_esperada y calcula scores con los modelos ML (una sola vez)."""
    from src.modelos import score_conicet, score_detoxify

    mensajes = []
    with open(ruta, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            cat = row[-1].strip()
            msg = ','.join(row[:-1]).strip()
            if cat not in _CATS_VALIDAS or not msg:
                continue
            mensajes.append((msg, cat))

    if not mensajes:
        return []

    patrones = _cargar_patrones()
    filas    = []
    n        = len(mensajes)
    print(f"  Pre-computando scores ML para {n} mensajes (se hace una sola vez)...")

    for i, (texto, esperada) in enumerate(mensajes):
        dens_norm = _detectar(_normalizar(texto), patrones)
        filas.append({
            "lista_negra": dens_norm,
            "conicet":     score_conicet(texto),
            "detoxify":    score_detoxify(texto),
            "historial":   0.0,
            "esperada":    esperada,
        })
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"    {i+1}/{n} mensajes procesados")

    return filas


print(f"Cargando dataset: {RUTA_CSV}")

if _es_formato_precomputado(RUTA_CSV):
    print("  Formato detectado: scores pre-computados (CSV con header)")
    dataset = _cargar_precomputado(RUTA_CSV)
else:
    print("  Formato detectado: mensaje,cat_esperada — se correrán los modelos ML")
    dataset = _cargar_y_precomputar(RUTA_CSV)

if not dataset:
    print("ERROR: no hay muestras válidas en el dataset.")
    sys.exit(1)

print(f"  {len(dataset)} mensajes listos para optimización\n")

# ============================================================
# MEMBRESÍAS FIJAS (sincronizadas con scoring.py CONFIG)
# ============================================================

_MF_ENTRADA = {
    'lista_negra': {
        'nulo':     (0.00, 0.00, 0.15, 0.30),
        'medio':    (0.15, 0.28, 0.45, 0.58),
        'alto':     (0.45, 0.58, 0.78, 0.90),
        'muy_alto': (0.78, 0.90, 1.00, 1.00),
    },
    'CONICET': {
        'nulo':  (0.00, 0.00, 0.08, 0.15),
        'medio': (0.08, 0.15, 0.45, 0.55),
        'alto':  (0.45, 0.55, 1.00, 1.00),
    },
    'detoxify': {
        'nulo':     (0.00, 0.00, 0.15, 0.30),
        'medio':    (0.15, 0.30, 0.55, 0.70),
        'alto':     (0.55, 0.70, 0.88, 0.95),
        'muy_alto': (0.88, 0.95, 1.00, 1.00),
    },
    'historial_usuario': {
        'limpio':       (0.00, 0.00, 0.12, 0.25),
        'antecedentes': (0.12, 0.25, 0.42, 0.55),
        'reincidente':  (0.42, 0.55, 0.72, 0.85),
        'cronico':      (0.72, 0.85, 1.00, 1.00),
    },
    'toxicidad': {
        'baja':  (0.00, 0.00, 0.15, 0.28),
        'media': (0.18, 0.28, 0.42, 0.55),
        'alta':  (0.42, 0.55, 1.00, 1.00),
    },
}

CATS = ['baja', 'media', 'alta']

# ============================================================
# NOMBRES DE REGLAS (29 reglas — mismo orden que scoring.py)
# ============================================================

NOMBRES_REGLAS = [
    # ALTA (0-13)  — orden idéntico a scoring.py
    "CONICET alto & detoxify muy_alto       → alta",
    "CONICET alto & detoxify alto           → alta",
    "CONICET alto & hist cronico            → alta",
    "lista_negra muy_alto & CONICET alto    → alta",
    "CONICET alto & lista_negra nulo        → alta",
    "detoxify muy_alto & CONICET nulo       → alta",
    "detoxify alto & CONICET nulo           → alta",
    "lista_negra muy_alto & detox medio     → alta",
    "lista_negra muy_alto & CONICET medio   → alta",
    "lista_negra alto & detoxify alto       → alta",
    "lista_negra alto & detox muy_alto      → alta",
    "lista_negra alto & CONICET alto        → alta",
    "lista_negra muy_alto                   → alta",
    "detoxify alto & hist reincidente       → alta",
    # MEDIA (14-20)
    "lista_negra medio & detox medio        → media",
    "lista_negra medio & detox alto         → media",
    "lista_negra medio & CONICET/detox nulo → media",
    "CONICET medio & detox nulo             → media",
    "detoxify medio & CONICET/ln nulo       → media",
    "lista_negra medio & hist antecedentes  → media",
    "detoxify medio & hist antecedentes     → media",
    # BAJA (21)
    "todo nulo                              → baja",
]

N_REGLAS = len(NOMBRES_REGLAS)  # 29

# ============================================================
# CONSTRUCCIÓN DEL SISTEMA DIFUSO CON PESOS
# Se crea desde cero en cada evaluación para evitar que skfuzzy
# comparta estado interno entre ControlSystem distintos.
# ============================================================

def _construir_sistema_con_pesos(pesos):
    """Crea un ControlSystem con las reglas de scoring.py, aplicando `pesos` (uno por regla) sobre cada consecuente."""
    univ = np.arange(0, 1.01, 0.01)

    ln  = ctrl.Antecedent(univ, 'lista_negra')
    co  = ctrl.Antecedent(univ, 'CONICET')
    dt  = ctrl.Antecedent(univ, 'detoxify')
    hu  = ctrl.Antecedent(univ, 'historial_usuario')
    tox = ctrl.Consequent(univ, 'toxicidad')

    for cat, params in _MF_ENTRADA['lista_negra'].items():
        ln[cat]  = fuzz.trapmf(univ, list(params))
    for cat, params in _MF_ENTRADA['CONICET'].items():
        co[cat]  = fuzz.trapmf(univ, list(params))
    for cat, params in _MF_ENTRADA['detoxify'].items():
        dt[cat]  = fuzz.trapmf(univ, list(params))
    for cat, params in _MF_ENTRADA['historial_usuario'].items():
        hu[cat]  = fuzz.trapmf(univ, list(params))
    for cat, params in _MF_ENTRADA['toxicidad'].items():
        tox[cat] = fuzz.trapmf(univ, list(params))

    reglas = [
        # ALTA (0-13) — idéntico a scoring.py
        ctrl.Rule(co['alto'] & dt['muy_alto'],              tox['alta']),
        ctrl.Rule(co['alto'] & dt['alto'],                  tox['alta']),
        ctrl.Rule(co['alto'] & hu['cronico'],               tox['alta']),
        ctrl.Rule(ln['muy_alto'] & co['alto'],              tox['alta']),
        ctrl.Rule(co['alto'] & ln['nulo'],                  tox['alta']),
        ctrl.Rule(dt['muy_alto'] & co['nulo'],              tox['alta']),
        ctrl.Rule(dt['alto'] & co['nulo'],                  tox['alta']),
        ctrl.Rule(ln['muy_alto'] & dt['medio'],             tox['alta']),
        ctrl.Rule(ln['muy_alto'] & co['medio'],             tox['alta']),
        ctrl.Rule(ln['alto'] & dt['alto'],                  tox['alta']),
        ctrl.Rule(ln['alto'] & dt['muy_alto'],              tox['alta']),
        ctrl.Rule(ln['alto'] & co['alto'],                  tox['alta']),
        ctrl.Rule(ln['muy_alto'],                           tox['alta']),
        ctrl.Rule(dt['alto'] & hu['reincidente'],           tox['alta']),
        # MEDIA (14-20)
        ctrl.Rule(ln['medio'] & dt['medio'],                tox['media']),
        ctrl.Rule(ln['medio'] & dt['alto'],                 tox['media']),
        ctrl.Rule(ln['medio'] & co['nulo'] & dt['nulo'],    tox['media']),
        ctrl.Rule(co['medio'] & dt['nulo'],                 tox['media']),
        ctrl.Rule(dt['medio'] & co['nulo'] & ln['nulo'],    tox['media']),
        ctrl.Rule(ln['medio'] & hu['antecedentes'],         tox['media']),
        ctrl.Rule(dt['medio'] & hu['antecedentes'],         tox['media']),
        # BAJA (21)
        ctrl.Rule(co['nulo'] & dt['nulo'] & ln['nulo'],     tox['baja']),
    ]

    for regla, peso in zip(reglas, pesos):
        regla.consequent[0].weight = float(peso)

    return ctrl.ControlSystem(reglas), univ, tox


def _score_a_categoria(score, univ, tox_var):
    """Categoría de mayor membresía para el score dado."""
    best, best_mu = 'baja', -1.0
    for cat in CATS:
        mu = fuzz.interp_membership(univ, tox_var[cat].mf, score)
        if mu > best_mu:
            best_mu = mu
            best = cat
    return best


# ============================================================
# FITNESS
# ============================================================

def evaluar(pesos):
    """Fitness: accuracy del sistema difuso (con `pesos` aplicados a cada regla) sobre todo el dataset."""
    try:
        sistema, univ, tox_var = _construir_sistema_con_pesos(pesos)
        correctos = 0
        for muestra in dataset:
            try:
                sim = ctrl.ControlSystemSimulation(sistema)
                sim.input['lista_negra']       = muestra['lista_negra']
                sim.input['CONICET']           = muestra['conicet']
                sim.input['detoxify']          = muestra['detoxify']
                sim.input['historial_usuario'] = muestra['historial']
                sim.compute()
                cat = _score_a_categoria(sim.output['toxicidad'], univ, tox_var)
            except Exception:
                cat = 'baja'
            if cat == muestra['esperada']:
                correctos += 1
        return correctos / len(dataset)
    except Exception:
        return 0.0


# ============================================================
# GUARDAR INCREMENTALMENTE
# ============================================================

def guardar_incremental(gen, fitness, pesos):
    """Sobrescribe RUTA_INCREMENTAL con los mejores pesos encontrados hasta la generación `gen`."""
    with open(RUTA_INCREMENTAL, "w", encoding="utf-8") as f:
        f.write(f"Gen {gen} | Fitness: {fitness:.4f} ({fitness*len(dataset):.0f}/{len(dataset)})\n\n")
        f.write("PESOS_REGLAS = [\n")
        for i, (p, nombre) in enumerate(zip(pesos, NOMBRES_REGLAS)):
            f.write(f"    {p:.4f},  # {i:2d}  {nombre}\n")
        f.write("]\n")


# ============================================================
# OPERADORES GENÉTICOS
# ============================================================

def individuo_aleatorio():
    """Genera un vector de N_REGLAS pesos aleatorios entre PESO_MIN y PESO_MAX."""
    return [random.uniform(PESO_MIN, PESO_MAX) for _ in range(N_REGLAS)]


INICIO = [1.0] * N_REGLAS


def seleccion_torneo(poblacion, fitnesses):
    """Selecciona el mejor individuo entre TORNEO_SIZE candidatos aleatorios."""
    candidatos = random.sample(range(len(poblacion)), TORNEO_SIZE)
    mejor = max(candidatos, key=lambda i: fitnesses[i])
    return deepcopy(poblacion[mejor])


def cruce(p1, p2):
    """Cruce de un punto entre dos padres, con probabilidad TASA_CRUCE."""
    if random.random() > TASA_CRUCE:
        return deepcopy(p1), deepcopy(p2)
    punto = random.randint(1, N_REGLAS - 1)
    return p1[:punto] + p2[punto:], p2[:punto] + p1[punto:]


def mutar(ind):
    """Perturba cada peso con probabilidad TASA_MUTACION, dentro de los límites [PESO_MIN, PESO_MAX]."""
    nuevo = deepcopy(ind)
    for i in range(N_REGLAS):
        if random.random() < TASA_MUTACION:
            nuevo[i] = max(PESO_MIN, min(PESO_MAX, nuevo[i] + random.gauss(0, 0.15)))
    return nuevo


# ============================================================
# ALGORITMO GENÉTICO
# ============================================================

print("Iniciando AG de pesos de reglas")
print(f"  Población: {POBLACION_SIZE} | Generaciones: {GENERACIONES}")
print(f"  Reglas a ponderar: {N_REGLAS}")
print(f"  Dataset: {len(dataset)} mensajes\n")

print("Evaluando baseline (todos los pesos = 1.0)...")
baseline = evaluar(INICIO)
print(f"  Accuracy baseline: {baseline:.4f} ({baseline*len(dataset):.0f}/{len(dataset)})\n")

poblacion = [deepcopy(INICIO)]
while len(poblacion) < POBLACION_SIZE:
    poblacion.append(individuo_aleatorio())

mejor_global  = deepcopy(INICIO)
mejor_fitness = baseline

for gen in range(GENERACIONES):
    fitnesses = [evaluar(ind) for ind in poblacion]

    idx_mejor = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    if fitnesses[idx_mejor] > mejor_fitness:
        mejor_fitness = fitnesses[idx_mejor]
        mejor_global  = deepcopy(poblacion[idx_mejor])
        guardar_incremental(gen + 1, mejor_fitness, mejor_global)

    if (gen + 1) % 5 == 0 or gen == 0:
        avg = sum(fitnesses) / len(fitnesses)
        print(f"  Gen {gen+1:3d}/{GENERACIONES} | Mejor: {mejor_fitness:.4f} "
              f"({mejor_fitness*len(dataset):.0f}/{len(dataset)}) | Promedio: {avg:.4f}")

    elite_idx       = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:ELITE_SIZE]
    nueva_poblacion = [deepcopy(poblacion[i]) for i in elite_idx]

    while len(nueva_poblacion) < POBLACION_SIZE:
        h1, h2 = cruce(seleccion_torneo(poblacion, fitnesses),
                        seleccion_torneo(poblacion, fitnesses))
        h1, h2 = mutar(h1), mutar(h2)
        nueva_poblacion.append(h1)
        if len(nueva_poblacion) < POBLACION_SIZE:
            nueva_poblacion.append(h2)

    poblacion = nueva_poblacion

# ============================================================
# RESULTADO FINAL
# ============================================================

print(f"\n{'='*60}")
print("RESULTADO FINAL — PESOS DE REGLAS")
print(f"{'='*60}")
print(f"Accuracy baseline:   {baseline:.4f}")
print(f"Accuracy optimizada: {mejor_fitness:.4f}")
mejora = (mejor_fitness - baseline) * 100
signo  = '+' if mejora >= 0 else ''
print(f"Mejora: {signo}{mejora:.1f} puntos porcentuales\n")

print(f"{'Peso':>6}   Regla")
print("-" * 65)
for peso, nombre in zip(mejor_global, NOMBRES_REGLAS):
    indicador = "▲" if peso > 1.2 else ("▼" if peso < 0.5 else " ")
    print(f"  {peso:.3f} {indicador}  {nombre}")

# Guardar resultado final
guardar_incremental('FINAL', mejor_fitness, mejor_global)
print(f"\nArchivo guardado en: {RUTA_INCREMENTAL}")

# Verificación por categoría
print("\nVerificación por categoría con los pesos óptimos:")
sistema_final, univ_final, tox_final = _construir_sistema_con_pesos(mejor_global)
por_cat = {c: [0, 0] for c in CATS}

for m in dataset:
    try:
        sim = ctrl.ControlSystemSimulation(sistema_final)
        sim.input['lista_negra']       = m['lista_negra']
        sim.input['CONICET']           = m['conicet']
        sim.input['detoxify']          = m['detoxify']
        sim.input['historial_usuario'] = m['historial']
        sim.compute()
        cat = _score_a_categoria(sim.output['toxicidad'], univ_final, tox_final)
    except Exception:
        cat = 'baja'
    esp = m['esperada']
    if esp in por_cat:
        por_cat[esp][1] += 1
        if cat == esp:
            por_cat[esp][0] += 1

print(f"\n{'Categoría':<12} {'Correctos':>10} {'Total':>8} {'Recall':>8}")
print("-" * 42)
for cat in CATS:
    ok, total = por_cat[cat]
    recall = ok / total if total > 0 else 0.0
    print(f"{cat:<12} {ok:>10} {total:>8} {recall:>8.3f}")

print("\nListo.")
