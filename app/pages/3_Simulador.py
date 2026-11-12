"""Simulador: solo recalcula las finanzas; la red no se vuelve a ejecutar."""
from dataclasses import replace

import streamlit as st

import comun
import graficos
import tablas
from src import config, finance

_MAX_DINERO = 1_000_000.0


def _dinero(zona, etiqueta: str, clave: str, valor: float) -> float:
    return zona.number_input(etiqueta, 0.0, _MAX_DINERO, valor, 10.0, key=clave)


def _catalogo() -> tuple[config.Incentivo, ...]:
    st.subheader("Incentivos")
    catalogo = []
    for zona, incentivo in zip(st.columns(len(config.CATALOGO_BASE)), config.CATALOGO_BASE):
        zona.markdown(f"**{incentivo.nombre}**")
        costo = _dinero(zona, "Costo", f"costo_{incentivo.id}", incentivo.costo)
        efectividad = zona.slider("Efectividad", 0.0, 1.0, incentivo.efectividad, 0.01, key=f"efectividad_{incentivo.id}")
        catalogo.append(replace(incentivo, costo=costo, efectividad=efectividad))
    return tuple(catalogo)


def _margen() -> config.ParametrosMargen:
    st.subheader("Margen del cliente")
    base = config.MARGEN_BASE
    zonas = st.columns(5)
    return config.ParametrosMargen(
        horizonte=zonas[0].number_input("Horizonte (años)", 0.25, 10.0, base.horizonte, 0.25, key="horizonte"),
        tasa_saldo=zonas[1].number_input("Tasa sobre saldo (%)", 0.0, 100.0, base.tasa_saldo * 100, 0.1, key="tasa_saldo") / 100,
        por_producto=_dinero(zonas[2], "Por producto", "por_producto", base.por_producto),
        por_tarjeta=_dinero(zonas[3], "Por tarjeta", "por_tarjeta", base.por_tarjeta),
        costo_fijo=_dinero(zonas[4], "Costo fijo", "costo_fijo", base.costo_fijo),
    )


comun.configurar("Simulador de parámetros")
clientes, p = comun.elegir_conjunto(*comun.requerir_cartera())
st.info(comun.NOTA_SUPUESTOS + " Solo se recalculan las finanzas, con las probabilidades ya calculadas.")

catalogo = _catalogo()
parametros = _margen()
p_min = st.slider("p_min: riesgo mínimo para actuar", 0.0, 1.0, config.P_MIN, 0.01, key="p_min")

decisiones = finance.decidir_cartera(clientes, p, parametros, catalogo, p_min)
simulado = finance.resumir_kpis(decisiones)
base = finance.resumir_kpis(finance.decidir_cartera(clientes, p))

st.subheader("Resultado frente al escenario base")
comun.mostrar_kpis(simulado, base)
st.caption(f"ROI del escenario base: {comun.porcentaje(base.roi)}. Las diferencias de cada tarjeta se miden contra ese escenario.")

st.subheader("Retener a todos frente a retención selectiva")
st.caption(
    f"Retener a todos interviene a cada cliente con p ≥ {comun.numero(config.P_RIESGO_CRITICO)}, "
    "también a los deficitarios, con el incentivo de su saldo."
)
selectiva, retener = finance.comparar_retener_a_todos(decisiones, catalogo)
escenarios = tablas.tabla_escenarios(selectiva, retener)
st.metric("Ventaja de la retención selectiva", comun.dinero(selectiva.ganancia_neta - retener.ganancia_neta))
st.plotly_chart(graficos.barras_escenarios(escenarios))
tablas.mostrar_tabla(escenarios)
