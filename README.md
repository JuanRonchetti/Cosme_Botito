# Cosme Botito
## Bot de Moderación con Lógica Difusa para Discord

![Python](https://img.shields.io/badge/Python-3.12-blue)
![discord.py](https://img.shields.io/badge/discord.py-2.x-5865F2)
![scikit-fuzzy](https://img.shields.io/badge/scikit--fuzzy-0.4-green)
![Status](https://img.shields.io/badge/estado-desarrollo%20académico-orange)

Bot de moderación para Discord que detecta mensajes tóxicos en tiempo real combinando tres fuentes de evidencia (lista negra léxica, dos modelos de ML en español) con un sistema de inferencia difusa (Fuzzy Logic). Desarrollado como trabajo académico para la materia Introducción a la IA — UTN.

---

## Descripción general

El bot analiza cada mensaje del servidor, calcula un **score difuso de toxicidad** en el rango [0, 1] y lo clasifica en tres categorías (`baja`, `media`, `alta`). 

El sistema nunca toma decisiones automáticas destructivas; su rol es asistir al moderador con información cuantificada.

---

## Arquitectura del sistema de scoring

```
Mensaje de texto
      │
      ├─► Lista negra léxica   → densidad_norm   [0, 1]
      ├─► Detoxify multilingual → detoxify_score [0, 1]
      ├─► BERT CONICET hate-es → conicet_score   [0, 1]
      └─► Historial del usuario → historial       [0, 1]
                │
                ▼
      ┌────────────────────┐
      │  Sistema Difuso    │  (skfuzzy, funciones trapezoidales)
      │  22 reglas if-then │
      └────────────────────┘
                │
                ▼
         score_difuso [0, 1]
                │
         ┌──────┴──────┐
         baja   media   alta
```

### Variables de entrada

| Variable | Fuente | Categorías |
|---|---|---|
| `lista_negra` | Patrones regex en `config/patrones.txt`, normalizado por longitud | nulo / medio / alto / muy_alto |
| `CONICET` | `finiteautomata/bert-non-contextualized-hate-speech-es` (prob. clase "Hateful") | nulo / medio / alto |
| `detoxify` | `Detoxify('multilingual')` (campo `toxicity`) | nulo / medio / alto / muy_alto |
| `historial_usuario` | Acumulado de infracciones previas del usuario | limpio / antecedentes / reincidente / crónico |

### Variable de salida

| Categoría | Rango aproximado |
|---|---|
| `baja` | 0.0 – 0.3 |
| `media` | 0.2 – 0.5 |
| `alta` | 0.4 – 1.0 | 

Las membresías son trapezoidales y configurables en `src/scoring.py → CONFIG`.

---

## Estructura del proyecto

```
Cosme_Botito/
├── main.py                    # Bot Discord — evento por mensaje, UI de botones
├── analizar.py                # CLI de análisis: mensajes.csv + testing.csv + --rescore
├── analizar_dataset.py        # Evalúa config/dataset.csv (700 msgs etiquetados)
├── optimizar_membresias.py    # AG: optimiza parámetros de membresía de toxicidad
├── optimizar_pesos_reglas.py  # AG: optimiza pesos de las 22 reglas difusas
│
├── src/
│   ├── scoring.py     # Sistema difuso + CONFIG + reglas + etiquetar_inputs/output
│   ├── modelos.py     # Lazy loading de Detoxify y BERT CONICET
│   └── analisis.py    # Gráficos académicos (membresías, validación, confusion, ROC, tiempos)
│
├── config/
│   ├── patrones.txt   # Patrones regex de palabras ofensivas (uno por línea)
│   └── dataset.csv    # 700 mensajes etiquetados (mensaje,cat_esperada), sin header
│
└── documentation/
    └── ag_optimizacion.ipynb         # Notebook descriptivo del algoritmo genetico de optimizacion
```

---



## Modelos de ML

Los modelos se cargan con **lazy loading** en `src/modelos.py` (primera llamada los descarga si no están en caché):

| Modelo | Librería | Uso |
|---|---|---|
| `Detoxify('multilingual')` | `detoxify` | Score de toxicidad general |
| `finiteautomata/bert-non-contextualized-hate-speech-es` | `transformers` | Score de hate speech en español |

---

## Requisitos

```bash
pip install discord.py detoxify transformers torch scikit-fuzzy matplotlib numpy python-dotenv
```

**Variables de entorno** (`.env`):
```
DISCORD_TOKEN=tu_token_aqui
```

---

## Estado actual

- Bot funcional con análisis en tiempo real y botones de feedback
- Sistema difuso con 22 reglas y 3 categorías de salida (baja / media / alta)
- Dataset de 700 mensajes etiquetados para evaluación y optimización
- Scripts de algoritmos genéticos para optimización de membresías y pesos de reglas
- Pipeline completo de análisis académico con gráficos reproducibles
