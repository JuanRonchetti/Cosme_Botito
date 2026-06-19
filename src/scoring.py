import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ============================================================
# PARÁMETROS CONFIGURABLES
# Modificá estos valores para calibrar el sistema.
# Los cambios acá se reflejan tanto en los gráficos
# como en el scoring real automáticamente.
# ============================================================

CONFIG = {
    'densidad': {
        'bajo':     {'tipo': 'trapezoidal', 'params': (0.00, 0.00, 0.15, 0.30)},
        'medio':    {'tipo': 'trapezoidal', 'params': (0.15, 0.28, 0.45, 0.58)},
        'alto':     {'tipo': 'trapezoidal', 'params': (0.45, 0.58, 0.78, 0.90)},
        'muy_alto': {'tipo': 'trapezoidal', 'params': (0.78, 0.90, 1.00, 1.00)},
    },

    'hate': {
        'bajo':     {'tipo': 'trapezoidal', 'params': (0.00, 0.00, 0.05, 0.12)},
        'medio':    {'tipo': 'trapezoidal', 'params': (0.05, 0.12, 0.22, 0.32)},
        'alto':     {'tipo': 'trapezoidal', 'params': (0.22, 0.32, 0.50, 0.65)},
        'muy_alto': {'tipo': 'trapezoidal', 'params': (0.50, 0.65, 1.00, 1.00)},
    },

    'historial': {
        'limpio':       {'tipo': 'trapezoidal', 'params': (0.00, 0.00, 0.12, 0.25)},
        'antecedentes': {'tipo': 'trapezoidal', 'params': (0.12, 0.25, 0.42, 0.55)},
        'reincidente':  {'tipo': 'trapezoidal', 'params': (0.42, 0.55, 0.72, 0.85)},
        'cronico':      {'tipo': 'trapezoidal', 'params': (0.72, 0.85, 1.00, 1.00)},
    },

    'velocidad': {
        'normal': {'tipo': 'trapezoidal', 'params': (0.00, 0.00, 0.15, 0.30)},
        'rapido': {'tipo': 'trapezoidal', 'params': (0.15, 0.28, 0.45, 0.58)},
        'flood':  {'tipo': 'trapezoidal', 'params': (0.45, 0.58, 0.78, 0.90)},
        'spam':   {'tipo': 'trapezoidal', 'params': (0.78, 0.90, 1.00, 1.00)},
    },

    'toxicidad': {
        'baja':    {'tipo': 'trapezoidal', 'params': (0.00, 0.00, 0.18, 0.33)},
        'media':   {'tipo': 'trapezoidal', 'params': (0.18, 0.30, 0.48, 0.60)},
        'alta':    {'tipo': 'trapezoidal', 'params': (0.48, 0.60, 0.78, 0.88)},
        'extrema': {'tipo': 'trapezoidal', 'params': (0.78, 0.88, 1.00, 1.00)},
    },
}

# ============================================================
# UMBRALES DE ACCIÓN
# Controlás qué pasa con cada nivel de score
# ============================================================

UMBRALES = {
    'ignorar':         0.30,   # por debajo de esto, no se hace nada
    'alertar':         0.55,   # alerta a mods sin borrar
    'borrar_y_alertar': 0.75,  # borra el mensaje y alerta
    # por encima de borrar_y_alertar → timeout automático
}

# ============================================================
# CONSTRUCTOR DE FUNCIONES DE MEMBRESÍA
# No necesitás tocar esto
# ============================================================

def construir_membresia(universe, config):
    tipo = config['tipo']
    p = config['params']

    if tipo == 'triangular':
        return fuzz.trimf(universe, list(p))
    elif tipo == 'trapezoidal':
        return fuzz.trapmf(universe, list(p))
    elif tipo == 'gaussiana':
        return fuzz.gaussmf(universe, p[0], p[1])
    elif tipo == 'sigmoidea':
        return fuzz.sigmf(universe, p[1], p[0])
    elif tipo == 'sigmoidea_inv':
        return 1 - fuzz.sigmf(universe, p[1], p[0])
    else:
        raise ValueError(f"Tipo desconocido: {tipo}")

# ============================================================
# CONSTRUCCIÓN DE VARIABLES Y MEMBRESÍAS
# ============================================================

universe = np.arange(0, 1.01, 0.01)

densidad  = ctrl.Antecedent(universe, 'densidad')
hate      = ctrl.Antecedent(universe, 'hate')
historial = ctrl.Antecedent(universe, 'historial')
velocidad = ctrl.Antecedent(universe, 'velocidad')
toxicidad = ctrl.Consequent(universe, 'toxicidad')

variables = {
    'densidad':  densidad,
    'hate':      hate,
    'historial': historial,
    'velocidad': velocidad,
    'toxicidad': toxicidad,
}

for nombre_var, variable in variables.items():
    for nombre_cat, config_cat in CONFIG[nombre_var].items():
        variable[nombre_cat] = construir_membresia(universe, config_cat)

# ============================================================
# REGLAS
# ============================================================

reglas = [
    # ── EXTREMA ─────────────────────────────────────────────────
    ctrl.Rule(hate['muy_alto'] & densidad['muy_alto'],         toxicidad['extrema']),
    ctrl.Rule(hate['muy_alto'] & densidad['alto'],             toxicidad['extrema']),
    ctrl.Rule(hate['alto']     & densidad['muy_alto'],         toxicidad['extrema']),
    ctrl.Rule(hate['muy_alto'] & historial['cronico'],         toxicidad['extrema']),
    ctrl.Rule(hate['alto']     & historial['cronico'],         toxicidad['extrema']),
    ctrl.Rule(densidad['muy_alto'] & historial['cronico'],     toxicidad['extrema']),
    ctrl.Rule(hate['muy_alto'] & velocidad['spam'],            toxicidad['extrema']),
    ctrl.Rule(densidad['muy_alto'] & velocidad['spam'],        toxicidad['extrema']),

    # ── ALTA ────────────────────────────────────────────────────
    ctrl.Rule(hate['alto']     & densidad['alto'],             toxicidad['alta']),
    ctrl.Rule(hate['alto']     & densidad['medio'],            toxicidad['alta']),
    ctrl.Rule(hate['muy_alto'] & densidad['medio'],            toxicidad['alta']),
    ctrl.Rule(hate['muy_alto'] & densidad['bajo'],             toxicidad['alta']),
    ctrl.Rule(densidad['muy_alto'] & hate['medio'],            toxicidad['alta']),
    ctrl.Rule(densidad['muy_alto'] & hate['bajo'],             toxicidad['alta']),
    ctrl.Rule(densidad['alto'] & historial['reincidente'],     toxicidad['alta']),
    ctrl.Rule(densidad['alto'] & historial['cronico'],         toxicidad['alta']),
    ctrl.Rule(densidad['medio'] & historial['reincidente'],    toxicidad['alta']),
    ctrl.Rule(densidad['medio'] & historial['cronico'],        toxicidad['alta']),
    ctrl.Rule(hate['medio']    & historial['reincidente'],     toxicidad['alta']),
    ctrl.Rule(hate['medio']    & historial['cronico'],         toxicidad['alta']),
    ctrl.Rule(hate['alto']     & historial['reincidente'],     toxicidad['alta']),
    ctrl.Rule(densidad['alto'] & velocidad['flood'],           toxicidad['alta']),
    ctrl.Rule(densidad['alto'] & velocidad['spam'],            toxicidad['alta']),
    ctrl.Rule(hate['medio']    & velocidad['flood'],           toxicidad['alta']),
    ctrl.Rule(hate['medio']    & velocidad['spam'],            toxicidad['alta']),
    ctrl.Rule(densidad['medio'] & velocidad['flood'],          toxicidad['alta']),
    ctrl.Rule(densidad['medio'] & velocidad['spam'],           toxicidad['alta']),

    # ── MEDIA ───────────────────────────────────────────────────
    ctrl.Rule(densidad['alto']   & hate['bajo'],               toxicidad['media']),
    ctrl.Rule(densidad['medio']  & hate['medio'],              toxicidad['media']),
    ctrl.Rule(densidad['medio']  & hate['bajo'],               toxicidad['media']),
    ctrl.Rule(hate['alto']       & densidad['bajo'],           toxicidad['media']),
    ctrl.Rule(hate['medio']      & historial['antecedentes'],  toxicidad['media']),
    ctrl.Rule(densidad['bajo']   & historial['antecedentes'],  toxicidad['media']),
    ctrl.Rule(velocidad['flood'] & densidad['bajo'],           toxicidad['media']),
    ctrl.Rule(velocidad['rapido'] & hate['medio'],             toxicidad['media']),
    ctrl.Rule(velocidad['rapido'] & densidad['medio'],         toxicidad['media']),

    # ── BAJA ────────────────────────────────────────────────────
    ctrl.Rule(densidad['bajo']   & hate['bajo'],               toxicidad['baja']),
]

sistema_control = ctrl.ControlSystem(reglas)

# ============================================================
# FUNCIÓN PRINCIPAL DE SCORING
# ============================================================

def calcular_score_difuso(dens, hate_score, hist, vel, debug=True):
    sim = ctrl.ControlSystemSimulation(sistema_control)
    sim.input['densidad']  = dens
    sim.input['hate']      = hate_score
    sim.input['historial'] = hist
    sim.input['velocidad'] = vel

    if debug:
        print(f"    densidad  → bajo:{fuzz.interp_membership(densidad.universe, densidad['bajo'].mf, dens):.2f} medio:{fuzz.interp_membership(densidad.universe, densidad['medio'].mf, dens):.2f} alto:{fuzz.interp_membership(densidad.universe, densidad['alto'].mf, dens):.2f} muy_alto:{fuzz.interp_membership(densidad.universe, densidad['muy_alto'].mf, dens):.2f}")
        print(f"    hate      → bajo:{fuzz.interp_membership(hate.universe, hate['bajo'].mf, hate_score):.2f} medio:{fuzz.interp_membership(hate.universe, hate['medio'].mf, hate_score):.2f} alto:{fuzz.interp_membership(hate.universe, hate['alto'].mf, hate_score):.2f} muy_alto:{fuzz.interp_membership(hate.universe, hate['muy_alto'].mf, hate_score):.2f}")

    try:
        sim.compute()
        return sim.output['toxicidad']
    except KeyError:
        return 0.05

def decidir_accion(score):
    if score < UMBRALES['ignorar']:
        return 'ignorar'
    elif score < UMBRALES['alertar']:
        return 'alertar'
    elif score < UMBRALES['borrar_y_alertar']:
        return 'borrar_y_alertar'
    else:
        return 'timeout'

# ============================================================
# GRAFICADOR
# Lee CONFIG directamente, siempre sincronizado con el scoring
# ============================================================

TITULOS = {
    'densidad':  'Densidad de palabras ofensivas',
    'hate':      'Score hate speech (pysentimiento)',
    'historial': 'Historial de infracciones',
    'velocidad': 'Velocidad de mensajes',
    'toxicidad': 'Toxicidad (salida)',
}

def graficar_membresias(ruta='membresias.png', mostrar=True):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    fig.suptitle('Funciones de Membresía del Sistema Difuso', fontsize=14)

    posiciones = [(0,0), (0,1), (0,2), (1,0), (1,1)]
    nombres_vars = ['densidad', 'hate', 'historial', 'velocidad', 'toxicidad']

    for idx, nombre_var in enumerate(nombres_vars):
        ax = axes[posiciones[idx]]
        for nombre_cat, config_cat in CONFIG[nombre_var].items():
            y = construir_membresia(universe, config_cat)
            ax.plot(universe, y, label=nombre_cat)
        ax.set_title(TITULOS[nombre_var])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    # Panel de info
    ax_info = axes[1, 2]
    ax_info.axis('off')
    info = (
        f"Umbrales de acción:\n\n"
        f"< {UMBRALES['ignorar']}  → ignorar\n"
        f"< {UMBRALES['alertar']}  → alertar mods\n"
        f"< {UMBRALES['borrar_y_alertar']}  → borrar + alertar\n"
        f"≥ {UMBRALES['borrar_y_alertar']}  → timeout\n\n"
        f"Tipos usados:\n"
        + "\n".join(
            f"  {k}: {list(v.values())[0]['tipo']}"
            for k, v in CONFIG.items()
        )
    )
    ax_info.text(0.05, 0.95, info, ha='left', va='top', fontsize=10,
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                 transform=ax_info.transAxes)

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    if mostrar:
        plt.show()
    else:
        plt.close()
    print(f"Guardado como {ruta}")