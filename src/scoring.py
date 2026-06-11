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
    # --- DENSIDAD DE PALABRAS OFENSIVAS ---
    # Gaussiana: (media, desviacion_estandar)
    'densidad': {
        'bajo':  {'tipo': 'gaussiana', 'params': (0.0,  0.15)},
        'medio': {'tipo': 'gaussiana', 'params': (0.4,  0.12)},
        'alto':  {'tipo': 'gaussiana', 'params': (0.85, 0.12)},
    },

    # --- HATE SPEECH (pysentimiento) ---
    # Scores bajos son lo normal, por eso las categorías están comprimidas a la izquierda
    # Trapezoidal: (a, b, c, d) donde b-c es la zona plana en grado 1
    'hate': {
        'bajo':  {'tipo': 'trapezoidal', 'params': (0.0, 0.0,  0.1,  0.2)},
        'medio': {'tipo': 'trapezoidal', 'params': (0.1, 0.1, 0.2, 0.3)},
        'alto':  {'tipo': 'trapezoidal', 'params': (0.2, 0.25, 1.0, 1.0)},
    },

    # --- HISTORIAL DE INFRACCIONES ---
    # Trapezoidal: (a, b, c, d) donde b-c es la zona plana en grado 1
    'historial': {
        'limpio':       {'tipo': 'trapezoidal', 'params': (0.0, 0.0,  0.1,  0.25)},
        'antecedentes': {'tipo': 'trapezoidal', 'params': (0.2, 0.35, 0.55, 0.7)},
        'reincidente':  {'tipo': 'trapezoidal', 'params': (0.6, 0.8,  1.0,  1.0)},
    },

    # --- VELOCIDAD DE MENSAJES ---
    # Triangular: (inicio, pico, fin)
    'velocidad': {
        'normal': {'tipo': 'triangular', 'params': (0.0, 0.0,  0.4)},
        'rapido': {'tipo': 'triangular', 'params': (0.3, 0.55, 0.75)},
        'flood':  {'tipo': 'triangular', 'params': (0.7, 1.0,  1.0)},
    },

    # --- TOXICIDAD (salida) ---
    # Sigmoidea para los extremos, gaussiana para el medio
    'toxicidad': {
        'baja':  {'tipo': 'sigmoidea_inv', 'params': (12, 0.25)},
        'media': {'tipo': 'gaussiana',     'params': (0.5, 0.12)},
        'alta':  {'tipo': 'sigmoidea',     'params': (12, 0.75)},
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
    ctrl.Rule(densidad['alto'] & hate['alto'],              toxicidad['alta']),
    ctrl.Rule(densidad['alto'] & historial['reincidente'],  toxicidad['alta']),
    ctrl.Rule(hate['alto']     & historial['reincidente'],  toxicidad['alta']),
    ctrl.Rule(densidad['alto'] & velocidad['flood'],        toxicidad['alta']),
    ctrl.Rule(densidad['medio']& historial['reincidente'],  toxicidad['alta']),
    ctrl.Rule(hate['medio']    & historial['reincidente'],  toxicidad['alta']),
    ctrl.Rule(densidad['alto'] & hate['bajo'],              toxicidad['media']),
    ctrl.Rule(densidad['medio']& hate['medio'],             toxicidad['media']),
    ctrl.Rule(densidad['medio']& historial['antecedentes'], toxicidad['media']),
    ctrl.Rule(hate['medio']    & historial['antecedentes'], toxicidad['media']),
    ctrl.Rule(densidad['bajo'] & hate['alto'],              toxicidad['media']),
    ctrl.Rule(densidad['bajo'] & hate['bajo'] & historial['limpio'],  toxicidad['baja']),
    ctrl.Rule(densidad['bajo'] & hate['bajo'] & velocidad['normal'],  toxicidad['baja']),
]

sistema_control = ctrl.ControlSystem(reglas)

# ============================================================
# FUNCIÓN PRINCIPAL DE SCORING
# ============================================================

def calcular_score_difuso(dens, hate_score, hist, vel):
    sim = ctrl.ControlSystemSimulation(sistema_control)
    sim.input['densidad']  = dens
    sim.input['hate']      = hate_score
    sim.input['historial'] = hist
    sim.input['velocidad'] = vel
    sim.compute()
    return sim.output['toxicidad']

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

def graficar_membresias():
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
    plt.savefig('membresias.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Guardado como membresias.png")