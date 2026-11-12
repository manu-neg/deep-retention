"""Tablas de las páginas: datos numéricos, mostrados con las etiquetas y formatos de las tarjetas."""
from dataclasses import asdict

import pandas as pd
import streamlit as st

from comun import ETIQUETAS, dinero, entero, etiqueta_segmento, nombre_incentivo, numero, porcentaje, probabilidad
from src import config, finance

_ORDEN_MODELOS = ["red_calibrada", "red_cruda", "linea_base"]  # primero el modelo que decide
_METRICAS_MODELO = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier", "log_loss": "Log-loss", "ece": "ECE"}
_MONEDA = ("Balance", "EstimatedSalary", "saldo", "margen", "ganancia", "costo", "inversion", "retorno_esperado", "ganancia_neta")
_FORMATOS = dict.fromkeys(_MONEDA, dinero) | {
    "clientes": entero,
    "intervenidos": entero,
    "p": probabilidad,
    "roi": porcentaje,
    "contribucion": lambda v: numero(v, 3),
}
_ETIQUETAS_COLUMNAS = ETIQUETAS | {
    config.COLUMNA_ID: "Cliente",
    "escenario": "Escenario",
    "segmento": "Segmento",
    "incentivo": "Incentivo",
    "clientes": "Clientes",
    "intervenidos": "Intervenidos",
    "p": "Probabilidad de fuga",
    "saldo": "Saldo",
    "margen": "Margen",
    "ganancia": "G*",
    "costo": "Costo",
    "inversion": "Inversión",
    "retorno_esperado": "Retorno esperado",
    "ganancia_neta": "Ganancia neta",
    "roi": "ROI",
    "variable": "Variable",
    "valor": "Valor",
    "contribucion": "Contribución a p",
}


def mostrar_tabla(tabla: pd.DataFrame) -> None:
    """El estilo solo cambia cómo se ve cada celda: las columnas siguen ordenándose por su número."""
    st.dataframe(tabla.style.format(_FORMATOS), hide_index=True, column_config=_ETIQUETAS_COLUMNAS)


def tabla_segmentos(decisiones: pd.DataFrame) -> pd.DataFrame:
    """Clientes y dinero por segmento, siempre con todos los segmentos."""
    tabla = finance.resumir_segmentos(decisiones).drop(columns="ganancia_neta").rename(columns={"capital": "saldo"})
    tabla.insert(0, "segmento", [etiqueta_segmento(s) for s in tabla.index])
    return tabla.reset_index(drop=True)


def tabla_intervenidos(clientes: pd.DataFrame, decisiones: pd.DataFrame) -> pd.DataFrame:
    """Clientes intervenidos, de mayor a menor G*."""
    d = decisiones[decisiones["actua"]].sort_values("ganancia", ascending=False, kind="stable")
    columnas = {
        config.COLUMNA_ID: clientes.loc[d.index, config.COLUMNA_ID],
        "segmento": d["segmento"],
        "incentivo": d["incentivo"].map(nombre_incentivo),
    }
    return pd.DataFrame(columnas | {c: d[c] for c in ("p", "saldo", "margen", "ganancia", "costo")})


def tabla_metricas(metricas: dict) -> pd.DataFrame | None:
    """Métricas en prueba de `metricas.json`, o None si no tienen la forma esperada."""
    try:
        tabla = pd.DataFrame(metricas["prueba"]).T.loc[_ORDEN_MODELOS, list(_METRICAS_MODELO)].astype(float)
    except (KeyError, TypeError, ValueError):
        return None
    return tabla.rename(index=config.ETIQUETAS_MODELOS, columns=_METRICAS_MODELO).rename_axis("Modelo").reset_index()


def tabla_escenarios(selectiva: finance.Escenario, retener: finance.Escenario) -> pd.DataFrame:
    filas = {"Retención selectiva": selectiva, "Retener a todos": retener}
    return pd.DataFrame([{"escenario": nombre, **asdict(e)} for nombre, e in filas.items()])
