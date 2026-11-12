"""Portada: qué hace la app, estado del modelo y métricas de Colab."""
import streamlit as st

import comun
import graficos
import tablas
from src import inference


def _leer(ruta) -> dict:
    """JSON del modelo, o {} si falta o está dañado: la portada avisa sin detenerse."""
    try:
        return inference.leer_json(ruta)
    except inference.ErrorArtefactos:
        return {}


comun.configurar("Deep Retention")
st.write(
    "Estima con una red neuronal la probabilidad de fuga de cada cliente y solo autoriza una acción de "
    "retención si la ganancia neta esperada es positiva."
)
st.markdown(
    "- **Buscador:** decisión y explicación para un cliente existente o nuevo.\n"
    "- **Cartera:** indicadores, segmentos e intervenciones recomendadas.\n"
    "- **Simulador:** cambie costos, efectividades y márgenes, y vea el impacto en la rentabilidad."
)

if not inference.artefactos_disponibles():
    st.warning(comun.AVISO_SIN_MODELO)
    st.stop()

st.success("Modelo disponible. Elija una página en el menú lateral.")

st.subheader("Métricas en el conjunto de prueba")
metricas = tablas.tabla_metricas(_leer(inference.RUTA_METRICAS))
if metricas is None:
    st.info(f"No se pudieron leer las métricas de `{inference.RUTA_METRICAS.name}`.")
else:
    st.dataframe(metricas, hide_index=True)

st.subheader("Factores globales (SHAP)")
factores = graficos.grafico_shap_global(_leer(inference.RUTA_SHAP_GLOBAL))
if factores is None:
    st.info(f"No se pudo leer la importancia global de `{inference.RUTA_SHAP_GLOBAL.name}`.")
else:
    st.plotly_chart(factores)
