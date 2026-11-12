"""Consulta individual: por cliente_id o con un formulario de cliente nuevo."""
import pandas as pd
import streamlit as st

import comun
import graficos
import tablas
from src import config, data, finance


def _campo(columna: str):
    etiqueta, clave = comun.ETIQUETAS[columna], f"form_{columna}"
    if columna in config.CATEGORIAS:
        return st.selectbox(
            etiqueta, config.CATEGORIAS[columna], index=None, format_func=comun.TRADUCCIONES.get, key=clave, placeholder="Elija"
        )
    if columna in config.COLUMNAS_BINARIAS:
        return st.selectbox(
            etiqueta, (1, 0), index=None, format_func=lambda v: "Sí" if v else "No", key=clave, placeholder="Elija"
        )
    if columna in config.RANGOS_ENTEROS:
        minimo, maximo = config.RANGOS_ENTEROS[columna]
        return st.number_input(f"{etiqueta} ({minimo}–{maximo})", value=None, step=1, key=clave)
    return st.number_input(f"{etiqueta} (≥ 0)", value=None, step=1000.0, format="%.2f", key=clave)


def _mensaje_validacion(error: Exception) -> str:
    texto = str(error).replace("Valores nulos en", "Complete estos campos:")
    for columna, etiqueta in comun.ETIQUETAS.items():
        texto = texto.replace(columna, etiqueta)
    return texto.translate(str.maketrans("", "", "[]'"))


def _formulario_cliente() -> pd.DataFrame | None:
    """Cliente nuevo; `validar_datos` es la única fuente de reglas, para no duplicar rangos."""
    with st.form("cliente_nuevo"):
        valores = {}
        for columna, zona in zip(comun.ETIQUETAS, st.columns(2) * len(comun.ETIQUETAS)):
            with zona:
                valores[columna] = _campo(columna)
        enviado = st.form_submit_button("Evaluar cliente", key="form_enviar")
    if not enviado:
        return None
    cliente = pd.DataFrame([valores])[config.COLUMNAS_ENTRADA]
    try:
        data.validar_datos(cliente, con_objetivo=False)
    except data.ErrorDatos as error:
        st.error(_mensaje_validacion(error))
        return None
    return cliente


def _motivo(decision: dict) -> str:
    """Por qué se tomó la decisión, en lenguaje de negocio."""
    if decision["actua"]:
        return "La ganancia esperada es positiva: rescatar al cliente vale más que el costo del incentivo."
    if decision["margen"] <= 0:
        return "Cliente deficitario (margen ≤ 0): se bloquea cualquier gasto de fidelización."
    if not decision["ganancia"] > 0:
        return "Ningún incentivo compensa su costo (G* ≤ 0), así que no se gasta en retención."
    return f"G* es positiva, pero el riesgo de fuga está por debajo del mínimo para actuar (p_min = {comun.numero(config.P_MIN)})."


def _mostrar_shap(cliente: pd.DataFrame) -> None:
    st.subheader("Factores que explican la probabilidad")
    try:
        filas = pd.DataFrame(comun.explicar(cliente))[["variable", "valor", "contribucion"]]
    except Exception:  # sin SHAP la decisión sigue siendo válida
        comun.LOG.exception("No se pudo calcular SHAP")
        st.warning("No se pudo calcular la explicación SHAP de este cliente. La decisión anterior sigue siendo válida.")
        return
    filas["variable"] = filas["variable"].map(comun.etiqueta_variable)
    filas["valor"] = filas["valor"].astype(str)
    st.plotly_chart(graficos.grafico_shap(filas))
    tablas.mostrar_tabla(filas)
    st.caption("SHAP indica cuánto empuja cada variable la probabilidad; describe al modelo, no una causa.")


def _mostrar_cliente(cliente: pd.DataFrame, p: float) -> None:
    """Decisión y explicación de un cliente (una fila con las columnas de entrada)."""
    fila = cliente.iloc[0]
    decision = finance.decidir_cliente(p, fila["Balance"], fila["NumOfProducts"], fila["HasCrCard"])
    st.subheader(f"Decisión: {comun.etiqueta_segmento(decision['segmento'])}")
    columnas = st.columns(4)
    columnas[0].metric("Probabilidad de fuga", comun.probabilidad(p))
    columnas[1].metric("Margen", comun.dinero(decision["margen"]))
    columnas[2].metric("G* (ganancia neta esperada)", comun.ganancia_texto(decision["ganancia"]))
    columnas[3].metric("Costo", comun.dinero(decision["costo"]))
    st.markdown(f"**Incentivo:** {comun.nombre_incentivo(decision['incentivo'])}")
    st.markdown(f"**Por qué:** {_motivo(decision)}")
    st.caption(comun.NOTA_SUPUESTOS)
    _mostrar_shap(cliente)


comun.configurar("Buscador de clientes")
clientes, p = comun.requerir_cartera()

if st.radio("Consulta", ("Cliente existente", "Cliente nuevo"), horizontal=True, key="modo") == "Cliente nuevo":
    cliente = _formulario_cliente()
    if cliente is not None:
        _mostrar_cliente(cliente, comun.predecir_cliente(cliente))
    st.stop()

identificador = st.text_input("cliente_id", placeholder="CLI-00001", max_chars=20, key="cliente_id").strip().upper()
if not identificador:
    st.info(f"Escriba un identificador, de CLI-00001 a CLI-{len(clientes):05d}.")
    st.stop()

encontrado = (clientes[config.COLUMNA_ID] == identificador).to_numpy()
if not encontrado.any():  # no se repite lo escrito: el texto del usuario no se muestra como markdown
    st.warning(f"No existe ese cliente. Use un identificador de CLI-00001 a CLI-{len(clientes):05d}.")
    st.stop()

fila = clientes.loc[encontrado, config.COLUMNAS_ENTRADA]
tablas.mostrar_tabla(fila)
_mostrar_cliente(fila, float(p[encontrado][0]))
