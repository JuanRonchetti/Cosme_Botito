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
    'lista_negra': {
        'nulo':     {'tipo': 'trapezoidal', 'params': (0.00, 0.00, 0.15, 0.30)},
        'medio':    {'tipo': 'trapezoidal', 'params': (0.15, 0.28, 0.45, 0.58)},
        'alto':     {'tipo': 'trapezoidal', 'params': (0.45, 0.58, 0.78, 0.90)},
        'muy_alto': {'tipo': 'trapezoidal', 'params': (0.78, 0.90, 1.00, 1.00)},
    },

    'CONICET': {
        'nulo':  {'tipo': 'trapezoidal', 'params': (0.00, 0.00, 0.08, 0.15)},
        'medio': {'tipo': 'trapezoidal', 'params': (0.08, 0.15, 0.45, 0.55)},
        'alto':  {'tipo': 'trapezoidal', 'params': (0.45, 0.55, 1.00, 1.00)},
    },

    'detoxify': {
        'nulo':     {'tipo': 'trapezoidal', 'params': (0.00, 0.00, 0.15, 0.30)},
        'medio':    {'tipo': 'trapezoidal', 'params': (0.15, 0.30, 0.55, 0.70)},
        'alto':     {'tipo': 'trapezoidal', 'params': (0.55, 0.70, 0.88, 0.95)},
        'muy_alto': {'tipo': 'trapezoidal', 'params': (0.88, 0.95, 1.00, 1.00)},
    },

    'historial_usuario': {
        'limpio':       {'tipo': 'trapezoidal', 'params': (0.00, 0.00, 0.12, 0.25)},
        'antecedentes': {'tipo': 'trapezoidal', 'params': (0.12, 0.25, 0.42, 0.55)},
        'reincidente':  {'tipo': 'trapezoidal', 'params': (0.42, 0.55, 0.72, 0.85)},
        'cronico':      {'tipo': 'trapezoidal', 'params': (0.72, 0.85, 1.00, 1.00)},
    },

    'toxicidad': {
        'baja':    {'tipo': 'trapezoidal', 'params': (0.00, 0.00, 0.15, 0.28)},
        'media':   {'tipo': 'trapezoidal', 'params': (0.18, 0.28, 0.42, 0.55)},
        'alta':    {'tipo': 'trapezoidal', 'params': (0.42, 0.55, 0.68, 0.78)},
        'extrema': {'tipo': 'trapezoidal', 'params': (0.65, 0.75, 1.00, 1.00)},
    },
}

# ============================================================
# UMBRALES DE ACCIÓN
# ============================================================

UMBRALES = {
    'ignorar':          0.30,
    'alertar':          0.55,
    'borrar_y_alertar': 0.75,
}

# ============================================================
# CONSTRUCTOR DE FUNCIONES DE MEMBRESÍA
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

lista_negra       = ctrl.Antecedent(universe, 'lista_negra')
CONICET           = ctrl.Antecedent(universe, 'CONICET')
detoxify          = ctrl.Antecedent(universe, 'detoxify')
historial_usuario = ctrl.Antecedent(universe, 'historial_usuario')
toxicidad         = ctrl.Consequent(universe, 'toxicidad')

variables = {
    'lista_negra':       lista_negra,
    'CONICET':           CONICET,
    'detoxify':          detoxify,
    'historial_usuario': historial_usuario,
    'toxicidad':         toxicidad,
}

for nombre_var, variable in variables.items():
    for nombre_cat, config_cat in CONFIG[nombre_var].items():
        variable[nombre_cat] = construir_membresia(universe, config_cat)

# ============================================================
# REGLAS
# ============================================================

reglas = [
    # # ── EXTREMA ─────────────────────────────────────────────────
    # ctrl.Rule(CONICET['alto'] & detoxify['muy_alto'],                       toxicidad['extrema']),
    # ctrl.Rule(CONICET['alto'] & detoxify['alto'],                           toxicidad['extrema']),
    # ctrl.Rule(CONICET['alto'] & historial_usuario['cronico'],               toxicidad['extrema']),
    # ctrl.Rule(detoxify['muy_alto'] & historial_usuario['cronico'],          toxicidad['extrema']),

    # # ── ALTA ────────────────────────────────────────────────────
    # ctrl.Rule(CONICET['alto'] & detoxify['nulo'],                           toxicidad['alta']),
    # ctrl.Rule(CONICET['alto'] & detoxify['medio'],                          toxicidad['alta']),
    # ctrl.Rule(detoxify['muy_alto'] & CONICET['medio'],                      toxicidad['alta']),
    # ctrl.Rule(lista_negra['alto'] & detoxify['alto'],                       toxicidad['alta']),
    # ctrl.Rule(lista_negra['alto'] & detoxify['muy_alto'],                   toxicidad['alta']),
    # ctrl.Rule(CONICET['medio'] & historial_usuario['reincidente'],          toxicidad['alta']),
    # ctrl.Rule(detoxify['alto'] & historial_usuario['reincidente'],          toxicidad['alta']),

    # # ── MEDIA ───────────────────────────────────────────────────
    # ctrl.Rule(detoxify['muy_alto'] & CONICET['nulo'],                       toxicidad['media']),
    # ctrl.Rule(detoxify['alto'] & CONICET['nulo'],                           toxicidad['media']),
    # ctrl.Rule(lista_negra['alto'] & CONICET['nulo'] & detoxify['nulo'],     toxicidad['media']),
    # ctrl.Rule(lista_negra['medio'] & CONICET['nulo'] & detoxify['nulo'],    toxicidad['media']),
    # ctrl.Rule(detoxify['medio'] & CONICET['nulo'] & lista_negra['nulo'],    toxicidad['media']),
    # ctrl.Rule(lista_negra['medio'] & detoxify['medio'],                     toxicidad['media']),
    # ctrl.Rule(CONICET['medio'] & detoxify['nulo'],                          toxicidad['media']),
    # ctrl.Rule(lista_negra['medio'] & historial_usuario['antecedentes'],     toxicidad['media']),
    # ctrl.Rule(detoxify['medio'] & historial_usuario['antecedentes'],        toxicidad['media']),

    # # ── BAJA ────────────────────────────────────────────────────
    # ctrl.Rule(
    #     detoxify['medio'] & CONICET['nulo'] & lista_negra['nulo'] & historial_usuario['limpio'],
    #     toxicidad['baja'],
    # ),
    # ctrl.Rule(CONICET['nulo'] & detoxify['nulo'] & lista_negra['nulo'],     toxicidad['baja']),

    # ── EXTREMA ────────────────────────────────────────────────
    ctrl.Rule(CONICET['alto'] & detoxify['muy_alto'],              toxicidad['extrema']),
    ctrl.Rule(CONICET['alto'] & detoxify['alto'],                  toxicidad['extrema']),
    ctrl.Rule(CONICET['alto'] & historial_usuario['cronico'],      toxicidad['extrema']),
    ctrl.Rule(detoxify['muy_alto'] & historial_usuario['cronico'], toxicidad['extrema']),
    # lista negra muy alta confirma cualquier señal media
    ctrl.Rule(lista_negra['muy_alto'] & CONICET['alto'],           toxicidad['extrema']),
    ctrl.Rule(lista_negra['muy_alto'] & detoxify['alto'],          toxicidad['extrema']),

    # ── ALTA ───────────────────────────────────────────────────
    # CONICET es la señal principal
    ctrl.Rule(CONICET['alto'] & detoxify['medio'],                 toxicidad['alta']),
    ctrl.Rule(CONICET['alto'] & lista_negra['nulo'],               toxicidad['alta']),
    # detoxify muy alto sube aunque CONICET no detecte
    ctrl.Rule(detoxify['muy_alto'] & CONICET['nulo'],              toxicidad['alta']),
    ctrl.Rule(detoxify['muy_alto'] & CONICET['medio'],             toxicidad['alta']),
    # detoxify alto sin CONICET también sube 
    ctrl.Rule(detoxify['alto'] & CONICET['nulo'],                  toxicidad['alta']),
    # lista negra con respaldo de cualquier modelo
    ctrl.Rule(lista_negra['muy_alto'] & detoxify['medio'],         toxicidad['alta']),
    ctrl.Rule(lista_negra['muy_alto'] & CONICET['medio'],          toxicidad['alta']),
    ctrl.Rule(lista_negra['alto'] & detoxify['alto'],              toxicidad['alta']),
    ctrl.Rule(lista_negra['alto'] & detoxify['muy_alto'],          toxicidad['alta']),
    ctrl.Rule(lista_negra['alto'] & CONICET['alto'],               toxicidad['alta']),
    # lista negra sola ya merece alta si es muy alta
    ctrl.Rule(lista_negra['muy_alto'],                             toxicidad['alta']),
    # reincidentes
    ctrl.Rule(CONICET['medio'] & historial_usuario['reincidente'], toxicidad['alta']),
    ctrl.Rule(detoxify['alto'] & historial_usuario['reincidente'], toxicidad['alta']),

    # ── MEDIA ──────────────────────────────────────────────────
    ctrl.Rule(CONICET['alto'] & detoxify['nulo'], toxicidad['media']),
    # lista negra medio con algo de detoxify
    ctrl.Rule(lista_negra['medio'] & detoxify['medio'],            toxicidad['media']),
    ctrl.Rule(lista_negra['medio'] & detoxify['alto'],             toxicidad['media']),
    # lista negra sola nivel medio
    ctrl.Rule(lista_negra['medio'] & CONICET['nulo'] & detoxify['nulo'], toxicidad['media']),
    ctrl.Rule(lista_negra['alto']  & CONICET['nulo'] & detoxify['nulo'], toxicidad['media']),
    # CONICET medio sin respaldo
    ctrl.Rule(CONICET['medio'] & detoxify['nulo'],                 toxicidad['media']),
    # detoxify medio sin nada más
    ctrl.Rule(detoxify['medio'] & CONICET['nulo'] & lista_negra['nulo'], toxicidad['media']),
    # antecedentes con señal baja
    ctrl.Rule(lista_negra['medio'] & historial_usuario['antecedentes'], toxicidad['media']),
    ctrl.Rule(detoxify['medio']    & historial_usuario['antecedentes'], toxicidad['media']),

    # ── BAJA ───────────────────────────────────────────────────
    ctrl.Rule(CONICET['nulo'] & detoxify['nulo'] & lista_negra['nulo'], toxicidad['baja']),
]

sistema_control = ctrl.ControlSystem(reglas)

# ============================================================
# FUNCIÓN PRINCIPAL DE SCORING
# ============================================================

def calcular_score_difuso(dens, conicet_score, detox_score, hist, debug=False):
    sim = ctrl.ControlSystemSimulation(sistema_control)
    sim.input['lista_negra']       = dens
    sim.input['CONICET']           = conicet_score
    sim.input['detoxify']          = detox_score
    sim.input['historial_usuario'] = hist

    if debug:
        ln = lista_negra
        co = CONICET
        dt = detoxify
        hu = historial_usuario
        print(
            f"    lista_negra       → "
            f"nulo:{fuzz.interp_membership(ln.universe, ln['nulo'].mf, dens):.2f} "
            f"medio:{fuzz.interp_membership(ln.universe, ln['medio'].mf, dens):.2f} "
            f"alto:{fuzz.interp_membership(ln.universe, ln['alto'].mf, dens):.2f}"
        )
        print(
            f"    CONICET           → "
            f"nulo:{fuzz.interp_membership(co.universe, co['nulo'].mf, conicet_score):.2f} "
            f"medio:{fuzz.interp_membership(co.universe, co['medio'].mf, conicet_score):.2f} "
            f"alto:{fuzz.interp_membership(co.universe, co['alto'].mf, conicet_score):.2f}"
        )
        print(
            f"    detoxify          → "
            f"nulo:{fuzz.interp_membership(dt.universe, dt['nulo'].mf, detox_score):.2f} "
            f"medio:{fuzz.interp_membership(dt.universe, dt['medio'].mf, detox_score):.2f} "
            f"alto:{fuzz.interp_membership(dt.universe, dt['alto'].mf, detox_score):.2f} "
            f"muy_alto:{fuzz.interp_membership(dt.universe, dt['muy_alto'].mf, detox_score):.2f}"
        )
        print(
            f"    historial_usuario → "
            f"limpio:{fuzz.interp_membership(hu.universe, hu['limpio'].mf, hist):.2f} "
            f"antecedentes:{fuzz.interp_membership(hu.universe, hu['antecedentes'].mf, hist):.2f} "
            f"reincidente:{fuzz.interp_membership(hu.universe, hu['reincidente'].mf, hist):.2f} "
            f"cronico:{fuzz.interp_membership(hu.universe, hu['cronico'].mf, hist):.2f}"
        )

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
    'lista_negra':       'Lista negra (patrones)',
    'CONICET':           'Score CONICET (bert-hate-speech-es)',
    'detoxify':          'Score Detoxify (multilingual)',
    'historial_usuario': 'Historial del usuario',
    'toxicidad':         'Toxicidad (salida)',
}

def graficar_membresias(ruta='membresias.png', mostrar=True):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    fig.suptitle('Funciones de Membresía del Sistema Difuso', fontsize=14)

    posiciones   = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    nombres_vars = ['lista_negra', 'CONICET', 'detoxify', 'historial_usuario', 'toxicidad']

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

# ============================================================
# HELPERS DE INTERPRETACIÓN
# ============================================================

def _cat_dominante(var_obj, cfg_key, valor):
    best, best_mu = None, -1.0
    for cat in CONFIG[cfg_key]:
        mu = fuzz.interp_membership(var_obj.universe, var_obj[cat].mf, valor)
        if mu > best_mu:
            best_mu = mu
            best = cat
    return best, round(best_mu, 2)

def etiquetar_inputs(dens, conicet_score, detox_score, hist):
    return {
        'lista_negra':       _cat_dominante(lista_negra,       'lista_negra',       dens),
        'CONICET':           _cat_dominante(CONICET,           'CONICET',           conicet_score),
        'detoxify':          _cat_dominante(detoxify,          'detoxify',          detox_score),
        'historial_usuario': _cat_dominante(historial_usuario, 'historial_usuario', hist),
    }

def etiquetar_output(score):
    return _cat_dominante(toxicidad, 'toxicidad', score)
