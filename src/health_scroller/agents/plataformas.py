"""Agente Plataformas: compara el impacto de cada red social.

QUÉ HACE:
    Crea el agente "plataformas": un LLM con 2 herramientas de pandas
    (media_por_grupo, conteo_categorias) para comparar redes sociales.

PARA QUÉ SIRVE:
    Responde preguntas como "¿Qué red social se asocia a peor GPA?" o
    "¿Cuántos estudiantes usan cada plataforma?" con datos EXACTOS del dataset.

CUÁNDO SE EJECUTA:
    1) Al arrancar el chatbot (graph.py lo construye con el resto del equipo).
    2) En cada turno etiquetado "plataformas" (o con /agente plataformas).
"""

from ..knowledge import briefing_texto
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
- Los nombres de los argumentos deben coincidir EXACTAMENTE con los del listado de herramientas.
- Responde en español, claro y ordenado, citando los números que devuelve la herramienta.
- Termina con una interpretación breve de lo que significan esos números.

Ejemplo de llamada correcta para "¿Qué red social se asocia a peor GPA?":
{{"herramienta": "media_por_grupo", "argumentos": {{"columna_grupo": "Primary_Platform", "columna_valor": "Academic_Performance_GPA"}}}}"""


def crear_agente_plataformas():
    """Devuelve el agente de plataformas (con sus herramientas pandas)."""
    return AgenteHerramientas(
        prompt=PROMPT_PLATAFORMAS,
        tools=[media_por_grupo, conteo_categorias],
    )
