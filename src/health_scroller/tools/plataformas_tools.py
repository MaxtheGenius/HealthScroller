"""Herramientas del Agente Plataformas: rankings exactos por grupo (pandas).

QUÉ HACE:
    Expone ranking_media_por_grupo: ordena los grupos de MENOR a MAYOR media
    (3 decimales) y deja escrito cuál es el menor y el mayor y su diferencia.

PARA QUÉ SIRVE:
    gemma3:1b es muy pequeño para comparar 7 medias casi iguales; con esta
    tool la comparación la hace pandas y el modelo solo tiene que citarla.

CUÁNDO SE EJECUTA:
    Solo cuando el agente "plataformas" pide un JSON con su nombre.
"""

from langchain_core.tools import tool

from ..knowledge import cargar_dataset
from .stats_tools import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS

# En estas columnas un valor ALTO es malo (más estrés, más horas de pantalla)
MAYOR_ES_PEOR = {"Perceived_Stress_Score", "Daily_Usage_Hours", "Weekend_Extra_Hours"}
# En estas no hay "mejor" ni "peor": solo menor y mayor
NEUTRAS = {"Age"}


@tool
def ranking_media_por_grupo(columna_grupo: str, columna_valor: str) -> str:
    """Ordena los grupos de columna_grupo de MENOR a MAYOR media de columna_valor (3 decimales) e indica cuál es la PEOR y la MEJOR. Ej.: qué red social (Primary_Platform) tiene peor GPA (Academic_Performance_GPA)."""
    df = cargar_dataset()
    if columna_grupo not in COLUMNAS_CATEGORICAS:
        return f"Error: '{columna_grupo}' no es categórica. Usa una de: {', '.join(COLUMNAS_CATEGORICAS)}"
    if columna_valor not in COLUMNAS_NUMERICAS:
        return f"Error: '{columna_valor}' no es numérica. Usa una de: {', '.join(COLUMNAS_NUMERICAS)}"
    grupos = df.groupby(columna_grupo)[columna_valor]
    medias = grupos.mean().sort_values(ascending=True)
    tamanos = grupos.count()
    lineas = [
        f"{i}. {grupo}: {media:.3f} ({tamanos[grupo]} estudiantes)"
        for i, (grupo, media) in enumerate(medias.items(), start=1)
    ]
    menor, mayor = medias.index[0], medias.index[-1]
    v_menor, v_mayor = medias.iloc[0], medias.iloc[-1]
    diferencia = v_mayor - v_menor
    if columna_valor in NEUTRAS:
        frase = (f"{menor} tiene la MENOR media de {columna_valor} ({v_menor:.3f}) y {mayor} la MAYOR "
                 f"({v_mayor:.3f}); la diferencia es de {diferencia:.3f}.")
    else:
        if columna_valor in MAYOR_ES_PEOR:
            peor, v_peor, mejor, v_mejor = mayor, v_mayor, menor, v_menor
        else:
            peor, v_peor, mejor, v_mejor = menor, v_menor, mayor, v_mayor
        frase = (f"{peor} es la que se asocia al PEOR {columna_valor} (media {v_peor:.3f}) y {mejor} al MEJOR "
                 f"(media {v_mejor:.3f}); la diferencia es de solo {diferencia:.3f} puntos.")
    return (
        f"RESPUESTA: {frase}\n"
        "(Empieza tu respuesta al usuario copiando la frase RESPUESTA tal cual.)\n\n"
        f"Ranking completo de '{columna_valor}' por '{columna_grupo}' (de MENOR a MAYOR media):\n"
        + "\n".join(lineas)
    )
