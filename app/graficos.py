"""Gráficos de las páginas."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from comun import etiqueta_variable, numero
from src import config

_AZUL, _NARANJA = config.COLORES_SERIES[:2]


def _ajustar(figura: go.Figure, alto: int = 340) -> go.Figure:
    return figura.update_layout(height=alto, margin=dict(l=8, r=8, t=48, b=8), bargap=0.15)


def histograma_p(p: np.ndarray) -> go.Figure:
    figura = go.Figure(go.Histogram(x=p, xbins=dict(start=0, end=1, size=0.05), marker_color=_AZUL))
    for umbral, nombre in ((config.P_RIESGO_INTERMEDIO, "intermedio"), (config.P_RIESGO_CRITICO, "crítico")):
        figura.add_vline(x=umbral, line_dash="dash", line_width=1, annotation_text=f"riesgo {nombre}: {numero(umbral)}")
    figura.update_layout(title="Distribución de la probabilidad de fuga", xaxis_title="p calibrada", yaxis_title="Clientes")
    return _ajustar(figura.update_xaxes(range=[0, 1]))


def barras_segmento(tabla: pd.DataFrame, columna: str, titulo: str, eje: str) -> go.Figure:
    figura = px.bar(tabla, x=columna, y="segmento", orientation="h", title=titulo, labels={columna: eje, "segmento": ""})
    figura.update_traces(marker_color=_AZUL).update_yaxes(autorange="reversed")
    return _ajustar(figura)


def barras_escenarios(tabla: pd.DataFrame) -> go.Figure:
    nombres = {"inversion": "Inversión", "retorno_esperado": "Retorno esperado", "ganancia_neta": "Ganancia neta"}
    largo = tabla.melt("escenario", list(nombres), "indicador", "monto")
    figura = px.bar(
        largo.assign(indicador=largo["indicador"].map(nombres)),
        x="indicador",
        y="monto",
        color="escenario",
        barmode="group",
        title="Retención selectiva frente a retener a todos",
        labels={"indicador": "", "monto": "Dólares", "escenario": "Escenario"},
        color_discrete_sequence=[_AZUL, _NARANJA],
    )
    return _ajustar(figura)


def _barras_shap(valores, variables, colores, titulo: str, eje: str) -> go.Figure:
    figura = go.Figure(go.Bar(x=valores, y=variables, orientation="h", marker_color=colores))
    figura.update_layout(title=titulo, xaxis_title=eje).update_yaxes(autorange="reversed")
    return _ajustar(figura, 120 + 26 * len(variables))


def grafico_shap(filas: pd.DataFrame) -> go.Figure:
    colores = [_NARANJA if c > 0 else _AZUL for c in filas["contribucion"]]
    titulo = "Naranja sube la probabilidad de fuga; azul la baja"
    return _barras_shap(filas["contribucion"], filas["variable"], colores, titulo, "Contribución a p")


def grafico_shap_global(resumen: dict) -> go.Figure | None:
    """Importancia global de `shap_global.json`, o None si no tiene la forma esperada."""
    try:
        variables = [etiqueta_variable(v) for v in resumen["variables"]]
        importancia = [float(v) for v in resumen["importancia_media_abs"]]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if not variables or len(variables) != len(importancia):
        return None
    return _barras_shap(importancia, variables, _AZUL, "Variables que más mueven la probabilidad de fuga", "Media de |SHAP|")
