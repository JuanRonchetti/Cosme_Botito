"""
modelos.py
----------
Carga perezosa (lazy loading) de los dos modelos de ML externos que
alimentan el sistema difuso: Detoxify (toxicidad multilingüe) y el
BERT de CONICET (hate speech en español). Cada modelo se descarga y
se instancia una única vez, en la primera llamada a su función de score.
"""

from detoxify import Detoxify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

_detoxify_model    = None
_conicet_tokenizer = None
_conicet_model     = None

_CONICET_REPO = "finiteautomata/bert-non-contextualized-hate-speech-es"


def _cargar_detoxify():
    """Instancia Detoxify('multilingual') una sola vez y la reutiliza."""
    global _detoxify_model
    if _detoxify_model is None:
        _detoxify_model = Detoxify('multilingual')
    return _detoxify_model


def _cargar_conicet():
    """Descarga e instancia el tokenizer/modelo BERT de CONICET una sola vez."""
    global _conicet_tokenizer, _conicet_model
    if _conicet_model is None:
        _conicet_tokenizer = AutoTokenizer.from_pretrained(_CONICET_REPO)
        _conicet_model = AutoModelForSequenceClassification.from_pretrained(_CONICET_REPO)
        _conicet_model.eval()
    return _conicet_tokenizer, _conicet_model


def score_detoxify(texto):
    """Devuelve el score de toxicidad de Detoxify para `texto`, en [0, 1]."""
    try:
        model = _cargar_detoxify()
        resultado = model.predict(texto)
        return float(resultado['toxicity'])
    except Exception:
        return 0.0


def score_conicet(texto):
    """Devuelve la probabilidad de la clase 'hateful' del modelo CONICET, en [0, 1]."""
    try:
        tokenizer, model = _cargar_conicet()
        inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        return float(probs[0][1])
    except Exception:
        return 0.0
