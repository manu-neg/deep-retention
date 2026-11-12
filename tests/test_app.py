"""Páginas de la app con AppTest. Los stubs del modelo viven solo aquí."""
import io
import json
import os
import subprocess
import sys
from dataclasses import replace

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"  # sin telemetría ni archivos en el directorio personal

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
from streamlit.testing.v1 import AppTest

from src import config, data, explain, finance, inference
from src.preprocessing import _nombres_esperados
from test_finance import EJEMPLOS

APP = config.RAIZ / "app"
sys.path.insert(0, str(APP))

import comun  # noqa: E402
import tablas  # noqa: E402

BUSCADOR, CARTERA, SIMULADOR = "pages/1_Buscador.py", "pages/2_Cartera.py", "pages/3_Simulador.py"
PAGINAS = [None, BUSCADOR, CARTERA, SIMULADOR]

P_POR_SALDO = {e[1]: e[0] for e in EJEMPLOS}
CARGAR_DATOS_REALES = data.cargar_datos


def _cartera_stub() -> pd.DataFrame:
    """Los cinco ejemplos más relleno, suficiente para la partición estratificada."""
    filas = [(e[1], e[2], e[3]) for e in EJEMPLOS] + [(30_000.0 + 3_000 * i, 1 + i % 3, i % 2) for i in range(35)]
    df = pd.DataFrame(filas, columns=["Balance", "NumOfProducts", "HasCrCard"])
    n = len(df)
    df.insert(0, config.COLUMNA_ID, [f"CLI-{i:05d}" for i in range(1, n + 1)])
    df["CreditScore"] = 650
    df["Age"] = [20 + i for i in range(n)]
    df["EstimatedSalary"] = 50_000.0
    df["Geography"] = "France"
    df["IsActiveMember"] = 1
    df["Tenure"] = 3
    df["Gender"] = "Female"
    df["Complain"] = 0
    df["SatisfactionScore"] = 3
    df["PointEarned"] = 500
    df["CardType"] = "GOLD"
    df[config.OBJETIVO] = [i % 2 for i in range(n)]
    return df


class ModeloStub:
    """Sustituye a la red y a SHAP; cuenta las ejecuciones de la red."""

    def __init__(self):
        self.predicciones = 0
        self.predecir = lambda df: np.array([P_POR_SALDO.get(s, a / 100) for s, a in zip(df["Balance"], df["Age"])])
        self.explicar = lambda df: [
            {"variable": "Age", "valor": 40, "contribucion": 0.2},
            {"variable": "Balance", "valor": 1000.5, "contribucion": 0.1},
            {"variable": "Geography", "valor": "France", "contribucion": -0.05},
        ]
        self.cargar = lambda: object()

    def predecir_fuga(self, artefactos, clientes):
        assert config.OBJETIVO not in clientes.columns  # el objetivo nunca llega a la red
        self.predicciones += 1
        return self.predecir(clientes)


@pytest.fixture(autouse=True)
def _limpiar_cache():
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def red_stub(monkeypatch):
    stub = ModeloStub()
    monkeypatch.setattr(inference, "artefactos_disponibles", lambda: True)
    monkeypatch.setattr(inference, "cargar_artefactos", lambda: stub.cargar())
    monkeypatch.setattr(inference, "predecir_fuga", stub.predecir_fuga)
    monkeypatch.setattr(explain, "explicar_cliente", lambda _, cliente: stub.explicar(cliente))
    monkeypatch.setattr(data, "cargar_datos", _cartera_stub)
    return stub


def usar_directorio_modelos(monkeypatch, directorio):
    monkeypatch.setattr(config, "DIR_MODELOS", directorio)
    for constante in ("RUTA_MODELO", "RUTA_METRICAS", "RUTA_SHAP_GLOBAL"):
        monkeypatch.setattr(inference, constante, directorio / getattr(inference, constante).name)


def abrir(pagina=None, tiempo_maximo: int = 30) -> AppTest:
    at = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=tiempo_maximo)
    if pagina:
        at.switch_page(pagina)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def correr(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def metricas(at: AppTest) -> dict:
    return {m.label: m.value for m in at.metric}


def graficos(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


def textos(at: AppTest) -> str:
    return "\n".join(e.value for e in [*at.markdown, *at.caption, *at.info, *at.warning, *at.error, *at.subheader])


# Sin artefactos --------------------------------------------------------------

@pytest.mark.parametrize("pagina", PAGINAS)
def test_sin_artefactos_avisa_y_no_inventa_cifras(monkeypatch, tmp_path, pagina):
    usar_directorio_modelos(monkeypatch, tmp_path)
    monkeypatch.setattr(data, "cargar_datos", lambda: pytest.fail("Sin modelo no deben cargarse datos"))
    at = abrir(pagina)
    aviso = at.warning[0].value
    assert "Aún no hay un modelo" in aviso
    assert "scripts.empaquetar_colab" in aviso
    assert "notebooks/entrenar_colab.ipynb" in aviso
    assert "scripts.importar_artefactos" in aviso
    assert not at.metric and not at.dataframe and not graficos(at)
    assert not at.number_input and not at.slider and not at.text_input


def test_importar_la_app_no_carga_keras_ni_shap():
    codigo = "import sys, comun; sys.exit(bool({'keras', 'tensorflow', 'shap'} & set(sys.modules)))"
    assert subprocess.run([sys.executable, "-c", codigo], cwd=APP).returncode == 0


# Inicio ----------------------------------------------------------------------

def _metricas_json() -> dict:
    fila = {"roc_auc": 0.8, "pr_auc": 0.6, "brier": 0.1, "log_loss": 0.4, "ece": 0.02}
    return {"prueba": {"red_cruda": fila, "red_calibrada": fila, "linea_base": fila}, "semilla": 42}


def test_inicio_muestra_metricas_y_factores(red_stub, monkeypatch, tmp_path):
    usar_directorio_modelos(monkeypatch, tmp_path)
    (tmp_path / "metricas.json").write_text(json.dumps(_metricas_json()), encoding="utf-8")
    resumen = {"variables": ["Age", "Geography_Germany"], "importancia_media_abs": [0.2, 0.1]}
    (tmp_path / "shap_global.json").write_text(json.dumps(resumen), encoding="utf-8")
    at = abrir()
    assert not at.info
    assert at.dataframe[0].value["Modelo"].tolist() == ["Red calibrada", "Red cruda", "Línea base"]
    assert at.dataframe[0].value["ROC-AUC"].tolist() == [0.8, 0.8, 0.8]
    assert graficos(at) == 1


@pytest.mark.parametrize(
    "metricas, resumen",
    [
        ('{"prueba": {}}', '{"variables": []}'),
        ("{dañado", "[1, 2]"),
        ('{"prueba": {"red_cruda": 1}}', '{"variables": ["Age"], "importancia_media_abs": []}'),
        ('{"prueba": NaN}', '{"variables": ["Age"], "importancia_media_abs": [NaN]}'),  # JSON estricto
    ],
)
def test_inicio_con_json_dañado_avisa(red_stub, monkeypatch, tmp_path, metricas, resumen):
    usar_directorio_modelos(monkeypatch, tmp_path)
    (tmp_path / "metricas.json").write_text(metricas, encoding="utf-8")
    (tmp_path / "shap_global.json").write_text(resumen, encoding="utf-8")
    at = abrir()
    assert "metricas.json" in at.info[0].value
    assert "shap_global.json" in at.info[1].value
    assert not at.dataframe and not graficos(at)


def test_etiquetas_de_variables_en_espanol():
    assert comun.etiqueta_variable("Geography_France") == "País: Francia"
    assert comun.etiqueta_variable("Gender_Male") == "Género: Hombre"
    assert comun.etiqueta_variable("saldo_cero") == "Saldo cero"
    assert comun.etiqueta_variable("Balance") == "Saldo"
    for nombre in _nombres_esperados():
        assert comun.etiqueta_variable(nombre) != nombre


# Buscador --------------------------------------------------------------------

MOTIVOS = ("positiva", "positiva", "deficitario", "compensa", "p_min")


@pytest.mark.parametrize("posicion", range(5))
def test_buscador_cinco_ejemplos(red_stub, posicion):
    p, _, _, _, margen, ganancia, segmento, optimo = EJEMPLOS[posicion]
    at = abrir(BUSCADOR)
    at.text_input(key="cliente_id").set_value(f"CLI-{posicion + 1:05d}")
    correr(at)
    valores = metricas(at)
    assert valores["Probabilidad de fuga"] == comun.probabilidad(p)
    assert valores["Margen"] == comun.dinero(margen)
    assert valores["G* (ganancia neta esperada)"] == comun.ganancia_texto(ganancia)
    assert at.subheader[0].value == f"Decisión: {comun.etiqueta_segmento(segmento)}"
    incentivo = comun.nombre_incentivo(optimo if segmento in "AB" else None)
    assert f"**Incentivo:** {incentivo}" in textos(at)
    motivo = next(m.value for m in at.markdown if m.value.startswith("**Por qué:**"))
    assert MOTIVOS[posicion] in motivo
    assert graficos(at) == 1  # SHAP
    assert red_stub.predicciones == 1  # la cartera, una sola vez


def test_buscador_no_redondea_p_hasta_el_umbral(red_stub):
    red_stub.predecir = lambda df: np.full(len(df), 0.6496)
    at = abrir(BUSCADOR)
    at.text_input(key="cliente_id").set_value("CLI-00004")  # sin intervención: segmento según la banda de p
    correr(at)
    assert metricas(at)["Probabilidad de fuga"] == "0,6496"
    assert at.subheader[0].value.startswith("Decisión: V")


def test_buscador_normaliza_el_identificador(red_stub):
    at = abrir(BUSCADOR)
    at.text_input(key="cliente_id").set_value("  cli-00001 ")
    correr(at)
    assert at.subheader[0].value.startswith("Decisión: A")


def test_buscador_sin_identificador_orienta(red_stub):
    at = abrir(BUSCADOR)
    assert "CLI-00001" in at.info[0].value
    assert not at.metric


def test_buscador_cliente_inexistente_no_repite_la_entrada(red_stub):
    at = abrir(BUSCADOR)
    at.text_input(key="cliente_id").set_value("![x](http://a.b)")
    correr(at)
    assert "No existe ese cliente" in at.warning[0].value
    assert "http" not in textos(at)
    assert not at.metric


def _llenar_formulario(at, **cambios):
    valores = {
        "CreditScore": 650,
        "Age": 40,
        "Tenure": 3,
        "NumOfProducts": 2,
        "Balance": 120_000.0,
        "EstimatedSalary": 50_000.0,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "Geography": "Spain",
        "Gender": "Male",
        "Complain": 0,
        "SatisfactionScore": 3,
        "PointEarned": 500,
        "CardType": "GOLD",
    } | cambios
    for columna, valor in valores.items():
        if valor is None:
            continue
        lista = at.selectbox if columna in (*config.COLUMNAS_BINARIAS, *config.COLUMNAS_CATEGORICAS) else at.number_input
        lista(key=f"form_{columna}").set_value(valor)
    at.button(key="form_enviar").click()
    return correr(at)


def _abrir_formulario(tiempo_maximo: int = 30):
    at = abrir(BUSCADOR, tiempo_maximo)
    at.radio(key="modo").set_value("Cliente nuevo")
    return correr(at)


def test_formulario_valido_decide_con_el_modelo(red_stub):
    at = _llenar_formulario(_abrir_formulario())
    assert not at.error
    assert metricas(at)["Margen"] == "$3.090,00"
    assert at.subheader[0].value == "Decisión: A · Retención prioritaria"
    assert graficos(at) == 1
    assert red_stub.predicciones == 2  # la cartera y el cliente nuevo


@pytest.mark.parametrize(
    "cambio, esperado",
    [
        ({"CreditScore": 100}, "Puntaje crediticio: se esperan enteros entre 350 y 850"),
        ({"Age": 17}, "Edad: se esperan enteros entre 18 y 92"),
        ({"Age": 93}, "Edad: se esperan enteros entre 18 y 92"),
        ({"Tenure": 11}, "Antigüedad (años): se esperan enteros entre 0 y 10"),
        ({"NumOfProducts": 5}, "Número de productos: se esperan enteros entre 1 y 4"),
        ({"Balance": -1.0}, "Saldo: no puede ser negativo"),
        ({"EstimatedSalary": -0.5}, "Salario estimado: no puede ser negativo"),
        ({"Balance": None}, "Complete estos campos: Saldo"),
        ({"Geography": None, "Gender": None}, "Complete estos campos: Género, País"),
    ],
)
def test_formulario_invalido_da_mensaje(red_stub, cambio, esperado):
    at = _llenar_formulario(_abrir_formulario(), **cambio)
    assert esperado in at.error[0].value
    assert not at.metric
    assert red_stub.predicciones == 1  # solo la cartera: el cliente inválido no llega a la red


def test_formulario_vacio_da_mensaje(red_stub):
    at = _abrir_formulario()
    at.button(key="form_enviar").click()
    correr(at)
    assert "Complete estos campos" in at.error[0].value


# Fallos del modelo: mensaje, no traza ----------------------------------------

def test_fallo_de_shap_conserva_la_decision(red_stub):
    def falla(_):
        raise RuntimeError("shap roto")

    red_stub.explicar = falla
    at = abrir(BUSCADOR)
    at.text_input(key="cliente_id").set_value("CLI-00001")
    correr(at)
    assert "explicación SHAP" in at.warning[0].value
    assert metricas(at)["Margen"] == "$3.090,00"


def test_explicacion_sin_las_claves_del_contrato_avisa(red_stub):
    red_stub.explicar = lambda _: [{"otra": 1}]
    at = abrir(BUSCADOR)
    at.text_input(key="cliente_id").set_value("CLI-00001")
    correr(at)
    assert "explicación SHAP" in at.warning[0].value


@pytest.mark.parametrize("pagina", [BUSCADOR, CARTERA, SIMULADOR])
def test_artefactos_danados_dan_mensaje(red_stub, pagina):
    def falla():
        raise OSError("modelo dañado")

    red_stub.cargar = falla
    at = abrir(pagina)
    assert "No se pudo preparar la cartera" in at.error[0].value
    assert not at.metric


@pytest.mark.parametrize("pagina", [CARTERA, SIMULADOR])
@pytest.mark.parametrize("p_mala", [float("nan"), 1.5])
def test_probabilidad_invalida_da_mensaje(red_stub, pagina, p_mala):
    red_stub.predecir = lambda df: np.full(len(df), p_mala)
    at = abrir(pagina)
    assert "No se pudo preparar la cartera" in at.error[0].value
    assert not at.metric


def test_datos_corruptos_dan_mensaje(red_stub, monkeypatch):
    def falla():
        raise data.ErrorDatos("SHA-256 incorrecto")

    monkeypatch.setattr(data, "cargar_datos", falla)
    at = abrir(CARTERA)
    assert "No se pudo preparar la cartera" in at.error[0].value


def test_prediccion_invalida_de_cliente_nuevo_da_mensaje(red_stub):
    at = _abrir_formulario()
    red_stub.predecir = lambda df: np.array([float("nan")])
    _llenar_formulario(at)
    assert "No se pudo calcular la probabilidad" in at.error[0].value
    assert not at.metric


# Cartera ---------------------------------------------------------------------

def _decisiones_stub(**argumentos):
    clientes = _cartera_stub()
    return finance.decidir_cartera(clientes, ModeloStub().predecir(clientes), **argumentos)


def test_cartera_kpis_coinciden_con_finanzas(red_stub):
    k = finance.resumir_kpis(_decisiones_stub())
    valores = metricas(abrir(CARTERA))
    assert valores["Clientes analizados"] == "40"
    assert valores["Clientes intervenidos"] == comun.entero(k.intervenidos)
    assert valores["Capital en riesgo crítico"] == comun.dinero(k.capital_riesgo_critico, 0)
    assert valores["Capital de intervenidos"] == comun.dinero(k.capital_intervenido, 0)
    assert valores["Capital rescatable esperado"] == comun.dinero(k.capital_rescatable, 0)
    assert valores["Efectividad proyectada"] == comun.porcentaje(k.efectividad_proyectada)
    assert valores["Inversión"] == comun.dinero(k.inversion)
    assert valores["Retorno esperado"] == comun.dinero(k.retorno_esperado)
    assert valores["Ganancia neta esperada"] == comun.dinero(k.ganancia_neta)
    assert valores["ROI"] == comun.porcentaje(k.roi)
    assert valores["Ahorro por salida permitida"] == comun.dinero(k.ahorro_salida_permitida)


def test_cartera_graficos_y_tablas(red_stub):
    at = abrir(CARTERA)
    assert graficos(at) == 3
    segmentos, intervenidos = at.dataframe[0].value, at.dataframe[1].value
    assert segmentos["clientes"].sum() == 40
    assert len(segmentos) == 5
    assert intervenidos["ganancia"].is_monotonic_decreasing
    assert (intervenidos["ganancia"] > 0).all()
    assert set(intervenidos["segmento"]) <= {"A", "B"}


def test_cartera_avisa_de_supuestos_y_de_clientes_de_entrenamiento(red_stub):
    texto = textos(abrir(CARTERA))
    assert comun.NOTA_SUPUESTOS in texto
    assert "clientes con los que entrenó la red" in texto
    assert "ROI en prueba" in texto


def _celdas_visibles(tabla) -> set:
    return set(convert_arrow_bytes_to_pandas_df(tabla.proto.arrow_data.styler.display_values).to_numpy().ravel())


def test_tablas_usan_etiquetas_y_formatos_de_las_tarjetas(red_stub):
    tabla = abrir(CARTERA).dataframe[1]
    etiquetas = json.loads(tabla.proto.columns)
    assert etiquetas["segmento"]["label"] == "Segmento"
    assert etiquetas[config.COLUMNA_ID]["label"] == "Cliente"
    assert etiquetas["ganancia"]["label"] == "G*"
    assert {"$3.090,00", "$601,16", "$400,00", "$120.000,00", "0,7200"} <= _celdas_visibles(tabla)
    assert tabla.value["margen"].dtype == float  # el número se conserva para ordenar


def test_tabla_de_escenarios_muestra_negativos_como_las_tarjetas(red_stub):
    at = abrir(SIMULADOR)
    at.number_input(key="tasa_saldo").set_value(0.0)
    correr(at)
    tabla = at.dataframe[0]
    assert json.loads(tabla.proto.columns)["escenario"]["label"] == "Escenario"
    assert comun.dinero(tabla.value["ganancia_neta"].iloc[1]) in _celdas_visibles(tabla)
    assert tabla.value["ganancia_neta"].iloc[1] < 0


def test_exportacion_csv_tiene_columnas_y_filas(red_stub, monkeypatch):
    exportado = {}
    boton = st.download_button

    def capturar(etiqueta, contenido, **opciones):
        exportado.update(contenido=contenido, **opciones)
        return boton(etiqueta, contenido, **opciones)

    monkeypatch.setattr(st, "download_button", capturar)
    abrir(CARTERA)
    assert exportado["file_name"] == "clientes_a_intervenir.csv"
    assert exportado["contenido"].startswith(b"\xef\xbb\xbf")  # BOM: Excel lee bien los acentos
    tabla = pd.read_csv(io.BytesIO(exportado["contenido"]), encoding="utf-8-sig")
    assert tabla.columns.tolist() == [config.COLUMNA_ID, "segmento", "incentivo", "p", "saldo", "margen", "ganancia", "costo"]

    d = _decisiones_stub()
    assert len(tabla) == d["actua"].sum()
    assert tabla["ganancia"].is_monotonic_decreasing
    assert tabla[config.COLUMNA_ID].iloc[0] == f"CLI-{d['ganancia'].where(d['actua']).idxmax() + 1:05d}"
    primero = tabla.set_index(config.COLUMNA_ID).loc["CLI-00001"]  # ejemplo 1 del diseño
    assert primero["segmento"] == "A"
    assert primero["incentivo"] == "Gestor VIP + tarifa plana"
    assert primero[["p", "saldo", "margen", "ganancia", "costo"]].tolist() == pytest.approx([0.72, 120_000, 3090, 601.16, 400])
    assert "CLI-00003" not in set(tabla[config.COLUMNA_ID])  # deficitario: nunca recibe gasto


def test_cartera_conjunto_de_prueba_se_informa_aparte(red_stub):
    at = abrir(CARTERA)
    at.radio(key="conjunto").set_value(comun.CONJUNTO_PRUEBA)
    correr(at)
    clientes = _cartera_stub()
    prueba = data.particionar(clientes).prueba
    k = finance.resumir_kpis(finance.decidir_cartera(prueba, ModeloStub().predecir(prueba)))
    valores = metricas(at)
    assert valores["Clientes analizados"] == "6"
    assert valores["Inversión"] == comun.dinero(k.inversion)
    assert red_stub.predicciones == 1


def test_cartera_sin_intervenidos_lo_dice_y_no_ofrece_csv(red_stub):
    red_stub.predecir = lambda df: np.full(len(df), 0.05)
    at = abrir(CARTERA)
    assert "ningún cliente" in at.info[0].value
    assert metricas(at)["ROI"] == "n/d"
    assert metricas(at)["Efectividad proyectada"] == "n/d"
    assert not at.get("download_button")


# Simulador -------------------------------------------------------------------

def test_simulador_valores_iniciales_igualan_el_escenario_base(red_stub):
    at = abrir(SIMULADOR)
    assert all(not m.delta for m in at.metric)
    assert "valores esperados" in at.info[0].value
    assert "clientes con los que entrenó la red" in textos(at)
    assert f"p ≥ {comun.numero(config.P_RIESGO_CRITICO)}" in textos(at)
    assert graficos(at) == 1


def _catalogo_con(id_incentivo, **cambios):
    return tuple(replace(i, **cambios) if i.id == id_incentivo else i for i in config.CATALOGO_BASE)


@pytest.mark.parametrize(
    "tipo, clave, valor, argumentos",
    [
        ("number_input", "costo_vip", 900.0, {"catalogo": _catalogo_con("vip", costo=900.0)}),
        ("number_input", "costo_bon", 20.0, {"catalogo": _catalogo_con("bon", costo=20.0)}),
        ("slider", "efectividad_exo", 0.6, {"catalogo": _catalogo_con("exo", efectividad=0.6)}),
        ("number_input", "horizonte", 2.0, {"parametros": replace(config.MARGEN_BASE, horizonte=2.0)}),
        ("number_input", "tasa_saldo", 1.0, {"parametros": replace(config.MARGEN_BASE, tasa_saldo=0.01)}),
        ("number_input", "por_producto", 10.0, {"parametros": replace(config.MARGEN_BASE, por_producto=10.0)}),
        ("number_input", "por_tarjeta", 0.0, {"parametros": replace(config.MARGEN_BASE, por_tarjeta=0.0)}),
        ("number_input", "costo_fijo", 500.0, {"parametros": replace(config.MARGEN_BASE, costo_fijo=500.0)}),
        ("slider", "p_min", 0.6, {"p_min": 0.6}),
    ],
)
def test_simulador_recalcula_igual_que_finanzas_sin_ejecutar_la_red(red_stub, tipo, clave, valor, argumentos):
    at = abrir(SIMULADOR)
    getattr(at, tipo)(key=clave).set_value(valor)
    correr(at)

    decisiones = _decisiones_stub(**argumentos)
    k = finance.resumir_kpis(decisiones)
    base = finance.resumir_kpis(_decisiones_stub())
    assert k != base  # el caso de prueba debe mover algo
    valores = metricas(at)
    assert valores["Inversión"] == comun.dinero(k.inversion)
    assert valores["Retorno esperado"] == comun.dinero(k.retorno_esperado)
    assert valores["ROI"] == comun.porcentaje(k.roi)
    assert valores["Efectividad proyectada"] == comun.porcentaje(k.efectividad_proyectada)
    assert valores["Clientes intervenidos"] == comun.entero(k.intervenidos)
    assert red_stub.predicciones == 1

    selectiva, retener = finance.comparar_retener_a_todos(decisiones, argumentos.get("catalogo", config.CATALOGO_BASE))
    tabla = at.dataframe[0].value
    assert tabla["ganancia_neta"].tolist() == pytest.approx([selectiva.ganancia_neta, retener.ganancia_neta])
    assert tabla["inversion"].tolist() == pytest.approx([selectiva.inversion, retener.inversion])
    assert valores["Ventaja de la retención selectiva"] == comun.dinero(selectiva.ganancia_neta - retener.ganancia_neta)


def test_simulador_muestra_diferencia_de_roi_frente_a_la_base(red_stub):
    at = abrir(SIMULADOR)
    at.number_input(key="costo_vip").set_value(900.0)
    correr(at)
    k = finance.resumir_kpis(_decisiones_stub(catalogo=_catalogo_con("vip", costo=900.0)))
    base = finance.resumir_kpis(_decisiones_stub())
    roi = next(m for m in at.metric if m.label == "ROI")
    assert roi.delta == comun._delta(k.roi, base.roi, comun.porcentaje)
    assert roi.delta.endswith(" pp")
    assert comun.porcentaje(base.roi) in textos(at)


@pytest.mark.parametrize(
    "valor, base, formato, esperado",
    [
        (100.4, 100.0, lambda v: comun.dinero(v, 0), None),  # no «+$0»
        (100.0, 100.4, lambda v: comun.dinero(v, 0), None),
        (101.0, 100.0, lambda v: comun.dinero(v, 0), "+$1"),
        (95.0, 100.0, comun.dinero, "-$5,00"),
        (0.50001, 0.5, comun.porcentaje, None),
        (0.6, 0.5, comun.porcentaje, "+10,0 pp"),
        (None, 0.5, comun.porcentaje, None),
    ],
)
def test_delta(valor, base, formato, esperado):
    assert comun._delta(valor, base, formato) == esperado


# Datos reales y funciones auxiliares -----------------------------------------

def test_datos_reales_con_red_simulada(red_stub, monkeypatch):
    monkeypatch.setattr(data, "cargar_datos", CARGAR_DATOS_REALES)
    red_stub.predecir = lambda df: (df["Age"].to_numpy() - 18) / 74
    assert metricas(abrir(CARTERA))["Clientes analizados"] == "10.000"
    at = abrir(BUSCADOR)
    at.text_input(key="cliente_id").set_value("CLI-10000")
    correr(at)
    assert at.subheader[0].value.startswith("Decisión:")


def test_integracion_con_inferencia_real_y_red_sin_entrenar(monkeypatch, tmp_path, artefactos_validos):
    """Contrato real con src.inference y src.explain; la red no se entrena y vive en un directorio temporal."""
    for nombre, contenido in artefactos_validos.items():
        (tmp_path / nombre).write_bytes(contenido)
    usar_directorio_modelos(monkeypatch, tmp_path)
    monkeypatch.setattr(data, "cargar_datos", _cartera_stub)

    abrir(tiempo_maximo=180)
    assert metricas(abrir(CARTERA, 180))["Clientes analizados"] == "40"
    assert "Inversión" in metricas(abrir(SIMULADOR, 180))

    at = abrir(BUSCADOR, 180)
    at.text_input(key="cliente_id").set_value("CLI-00001")
    correr(at)
    assert not at.warning  # SHAP real sin error
    assert graficos(at) == 1
    explicacion = at.dataframe[1].value
    assert len(explicacion) == 21
    assert "Saldo cero" in explicacion["variable"].tolist()

    at = _llenar_formulario(_abrir_formulario(180))
    assert not at.error and not at.warning
    assert at.subheader[0].value.startswith("Decisión:")


@pytest.mark.parametrize(
    "funcion, valor, esperado",
    [
        (comun.dinero, 1234.5, "$1.234,50"),
        (comun.dinero, -60, "-$60,00"),
        (comun.dinero, -0.001, "$0,00"),
        (comun.dinero, None, "n/d"),
        (comun.porcentaje, 0.326, "32,6 %"),
        (comun.porcentaje, None, "n/d"),
        (comun.ganancia_texto, float("nan"), "sin opción"),
        (comun.ganancia_texto, 43.6, "+$43,60"),
        (comun.ganancia_texto, -28.16, "-$28,16"),
        (comun.entero, 10_000, "10.000"),
    ],
)
def test_formatos(funcion, valor, esperado):
    assert funcion(valor) == esperado


def test_tabla_segmentos_siempre_tiene_los_cinco():
    cartera = pd.DataFrame({"Balance": [120_000.0], "NumOfProducts": [2], "HasCrCard": [1]})
    tabla = tablas.tabla_segmentos(finance.decidir_cartera(cartera, [0.72]))
    assert tabla["clientes"].tolist() == [1, 0, 0, 0, 0]
    assert tabla["inversion"].tolist() == [400, 0, 0, 0, 0]
    assert tabla["retorno_esperado"].iloc[0] == pytest.approx(1001.16)
    assert tabla["saldo"].tolist() == [120_000, 0, 0, 0, 0]
    assert tabla.columns.tolist() == ["segmento", "clientes", "saldo", "inversion", "retorno_esperado"]
