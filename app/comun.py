"""Lógica compartida de las páginas: modelo y datos en caché, formatos y tarjetas."""
import logging
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:  # `streamlit run` solo añade app/ a la ruta
    sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
import streamlit as st

from src import config, data, explain, finance, inference

LOG = logging.getLogger(__name__)

CARTERA_COMPLETA = "Cartera completa"
CONJUNTO_PRUEBA = "Conjunto de prueba"

AVISO_SIN_MODELO = """**Aún no hay un modelo entrenado.** No se muestran probabilidades ni cifras hasta tenerlo.

El entrenamiento se hace en Google Colab, no en este equipo:

1. Ejecute `python -m scripts.empaquetar_colab` para crear `colab_bundle.zip`.
2. Abra `notebooks/entrenar_colab.ipynb` en Colab, suba el zip, ejecute todo y descargue `artefactos.zip`.
3. Ejecute `python -m scripts.importar_artefactos artefactos.zip`.
4. Recargue esta página."""

NOTA_SUPUESTOS = (
    "Todas las cifras son valores esperados según los supuestos de margen, costos y efectividad; "
    "no son resultados garantizados."
)

ETIQUETAS = {
    "CreditScore": "Puntaje crediticio",
    "Age": "Edad",
    "Tenure": "Antigüedad (años)",
    "NumOfProducts": "Número de productos",
    "Balance": "Saldo",
    "EstimatedSalary": "Salario estimado",
    "HasCrCard": "Tiene tarjeta de crédito",
    "IsActiveMember": "Es miembro activo",
    "Geography": "País",
    "Gender": "Género",
    "Complain": "Presentó una queja",
    "SatisfactionScore": "Satisfacción (1–5)",
    "PointEarned": "Puntos acumulados",
    "CardType": "Tipo de tarjeta",
}
TRADUCCIONES = {
    "France": "Francia", "Germany": "Alemania", "Spain": "España", "Female": "Mujer", "Male": "Hombre",
    "SILVER": "Plata", "GOLD": "Oro", "PLATINUM": "Platino", "DIAMOND": "Diamante",
}


# Formatos ------------------------------------------------------------------

def _es_nulo(valor) -> bool:
    return valor is None or pd.isna(valor)


def numero(valor, decimales: int = 2) -> str:
    """Número con separadores en español: 1.234,56."""
    return f"{valor:,.{decimales}f}".translate(str.maketrans(",.", ".,"))


def probabilidad(valor) -> str:
    """Con 4 decimales, para que 0,6496 (segmento V) no se lea como el umbral 0,65."""
    return numero(valor, 4)


def dinero(valor, decimales: int = 2) -> str:
    if _es_nulo(valor):
        return "n/d"
    return f"{'-' if round(valor, decimales) < 0 else ''}${numero(abs(valor), decimales)}"


def porcentaje(valor) -> str:
    return "n/d" if _es_nulo(valor) else f"{numero(valor * 100, 1)} %"


def entero(valor) -> str:
    return numero(valor, 0)


def ganancia_texto(ganancia) -> str:
    """G*, con signo; NaN significa que el cliente no tiene ningún incentivo posible."""
    if _es_nulo(ganancia):
        return "sin opción"
    return f"{'+' if ganancia > 0 else ''}{dinero(ganancia)}"


def etiqueta_segmento(segmento: str) -> str:
    return f"{segmento} · {config.SEGMENTOS[segmento]}"


def nombre_incentivo(id_incentivo, catalogo=config.CATALOGO_BASE) -> str:
    return next((i.nombre for i in catalogo if i.id == id_incentivo), "Sin intervención")


def etiqueta_variable(nombre: str) -> str:
    """Nombre de salida del preprocesador en español, p. ej. Geography_France → País: Francia."""
    base, _, categoria = nombre.partition("_")
    if base in config.CATEGORIAS:
        return f"{ETIQUETAS[base]}: {TRADUCCIONES.get(categoria, categoria)}" if categoria else ETIQUETAS[base]
    return "Saldo cero" if nombre == config.COLUMNA_SALDO_CERO else ETIQUETAS.get(nombre, nombre)


# Modelo y datos ------------------------------------------------------------

def _huella() -> int:
    """Invalida las cachés si se importa un modelo nuevo con la app abierta."""
    ruta = inference.RUTA_MODELO
    return ruta.stat().st_mtime_ns if ruta.exists() else 0


@st.cache_resource(show_spinner="Cargando el modelo…")
def _artefactos(huella: int):
    return inference.cargar_artefactos()


@st.cache_data(show_spinner="Cargando la cartera…")
def _cartera() -> pd.DataFrame:
    return data.cargar_datos()


def _predecir(huella: int, clientes: pd.DataFrame) -> np.ndarray:
    p = np.asarray(inference.predecir_fuga(_artefactos(huella), clientes), dtype=float).ravel()
    if len(p) != len(clientes) or not ((p >= 0) & (p <= 1)).all():  # un NaN tampoco pasa
        raise ValueError("Predicción inválida")
    return p


@st.cache_data(show_spinner="Calculando probabilidades de la cartera…")
def _p_cartera(huella: int) -> np.ndarray:
    """Única ejecución de la red para la cartera; el simulador reutiliza este resultado."""
    return _predecir(huella, _cartera()[config.COLUMNAS_ENTRADA])


@st.cache_data(show_spinner=False)
def _mascara_prueba() -> np.ndarray:
    clientes = _cartera()
    return clientes[config.COLUMNA_ID].isin(data.particionar(clientes).prueba[config.COLUMNA_ID]).to_numpy()


def requerir_cartera() -> tuple[pd.DataFrame, np.ndarray]:
    """Clientes y p en caché; sin modelo o con error, detiene la página con un aviso."""
    if not inference.artefactos_disponibles():
        st.warning(AVISO_SIN_MODELO)
        st.stop()
    try:
        return _cartera(), _p_cartera(_huella())
    except Exception:  # artefactos o datos dañados: el usuario ve un aviso, no una traza
        LOG.exception("No se pudo preparar la cartera")
        st.error("No se pudo preparar la cartera. Revise los artefactos de `models/` y los datos, y vuelva a importarlos.")
        st.stop()


def elegir_conjunto(clientes: pd.DataFrame, p: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """El diseño pide informar por separado la cartera completa y el conjunto de prueba."""
    opcion = st.radio("Clientes analizados", (CARTERA_COMPLETA, CONJUNTO_PRUEBA), horizontal=True, key="conjunto")
    mascara = _mascara_prueba()
    if opcion == CONJUNTO_PRUEBA:
        st.caption("Conjunto de prueba: el 15 % de los clientes, no usado para entrenar ni calibrar.")
        return clientes[mascara], p[mascara]
    roi_prueba = finance.resumir_kpis(finance.decidir_cartera(clientes[mascara], p[mascara])).roi
    st.caption(
        "Cartera completa: incluye clientes con los que entrenó la red, así que puede verse mejor que el conjunto "
        f"de prueba (ROI en prueba con los supuestos base: {porcentaje(roi_prueba)})."
    )
    return clientes, p


def predecir_cliente(cliente: pd.DataFrame) -> float:
    """p de un cliente nuevo ya validado; con error, detiene la página con un aviso."""
    try:
        return float(_predecir(_huella(), cliente)[0])
    except Exception:
        LOG.exception("No se pudo predecir el cliente")
        st.error("No se pudo calcular la probabilidad de este cliente. Revise los artefactos de `models/`.")
        st.stop()


@st.cache_data(show_spinner="Calculando la explicación SHAP…")
def _explicar(huella: int, cliente: pd.DataFrame) -> list[dict]:
    return explain.explicar_cliente(_artefactos(huella), cliente)


def explicar(cliente: pd.DataFrame) -> list[dict]:
    return _explicar(_huella(), cliente)


# Bloques de página ---------------------------------------------------------

def configurar(titulo: str) -> None:
    st.set_page_config(page_title=f"Deep Retention · {titulo}", layout="wide")
    st.title(titulo)


def mostrar_kpis(k: finance.ResumenKPI, base: finance.ResumenKPI | None = None) -> None:
    """Tarjetas de indicadores; con `base`, cada una muestra la diferencia frente a ese escenario."""
    dinero_entero = lambda v: dinero(v, 0)
    tarjetas = (
        ("Clientes analizados", "clientes", entero),
        ("Clientes intervenidos", "intervenidos", entero),
        ("Efectividad proyectada", "efectividad_proyectada", porcentaje),
        ("Capital en riesgo crítico", "capital_riesgo_critico", dinero_entero),
        ("Capital de intervenidos", "capital_intervenido", dinero_entero),
        ("Capital rescatable esperado", "capital_rescatable", dinero_entero),
        ("Inversión", "inversion", dinero),
        ("Retorno esperado", "retorno_esperado", dinero),
        ("Ganancia neta esperada", "ganancia_neta", dinero),
        ("ROI", "roi", porcentaje),
        ("Ahorro por salida permitida", "ahorro_salida_permitida", dinero),
    )
    for zona, (etiqueta, campo, formato) in zip(st.columns(4) * 3, tarjetas):
        valor = getattr(k, campo)
        delta = _delta(valor, getattr(base, campo), formato) if base else None
        zona.metric(etiqueta, formato(valor), delta, delta_color="off" if campo == "inversion" else "normal")


def _delta(valor, base, formato) -> str | None:
    if _es_nulo(valor) or _es_nulo(base):
        return None
    texto = formato(valor - base)
    if not set(texto) & set("123456789"):  # con esta precisión la diferencia no se ve: nada de «+$0»
        return None
    return f"{'+' if valor > base else ''}{texto}".replace(" %", " pp")
