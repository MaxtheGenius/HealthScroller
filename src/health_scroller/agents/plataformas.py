"""Agente Plataformas: compara el impacto de cada red social.

QUÉ HACE:
    Crea el agente "plataformas": un LLM con herramientas de pandas
    (ranking_media_por_grupo, media_por_grupo, conteo_categorias).

PARA QUÉ SIRVE:
    Responde preguntas como "¿Qué red social se asocia a peor GPA?" o
    "¿Cuántos estudiantes usan cada plataforma?" con datos EXACTOS del dataset.

CUÁNDO SE EJECUTA:
    1) Al arrancar el chatbot (graph.py lo construye con el resto del equipo).
    2) En cada turno etiquetado "plataformas" (o con /agente plataformas).
"""

from ..knowledge import briefing_texto
from ..tools.plataformas_tools import ranking_media_por_grupo
from ..tools.stats_tools import conteo_categorias, media_por_grupo
from .executor import AgenteHerramientas

PROMPT_PLATAFORMAS = f"""Eres el Agente de Plataformas de HealthScroller.

Datos clave (briefing del notebook):
{briefing_texto()}

Tu misión: responder preguntas del tipo "¿Qué red social se asocia a peor GPA?"
o "¿Cuántos estudiantes usan cada plataforma?".

Tienes herramientas para consultar el dataset real de 4500 estudiantes.
Reglas:
- Usa SIEMPRE una herramienta para calcular; nunca inventes números.
- Para "peor" o "mejor" usa ranking_media_por_grupo: NO compares tú los números.
- Después, EMPIEZA tu respuesta copiando LITERALMENTE la frase que sigue a "RESPUESTA:" (con el nombre de la plataforma y su media).
- Los nombres de los argumentos deben coincidir EXACTAMENTE con los del listado de herramientas.
- Responde en español, claro y ordenado, citando los números que devuelve la herramienta.
- Termina con una interpretación breve (1-2 frases): si la diferencia es de pocas centésimas, di que es muy pequeña.

Ejemplo de llamada correcta para "¿Qué red social se asocia a peor GPA?":
{{"herramienta": "ranking_media_por_grupo", "argumentos": {{"columna_grupo": "Primary_Platform", "columna_valor": "Academic_Performance_GPA"}}}}

Ejemplo de llamada correcta para "¿Cuántos estudiantes usan cada plataforma?":
{{"herramienta": "conteo_categorias", "argumentos": {{"columna": "Primary_Platform"}}}}"""


def crear_agente_plataformas():
    """Devuelve el agente de plataformas (con sus herramientas pandas)."""
    return AgenteHerramientas(
        prompt=PROMPT_PLATAFORMAS,
        tools=[ranking_media_por_grupo, media_por_grupo, conteo_categorias],
    )
