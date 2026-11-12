"""Preprocesamiento: paridad JSON/sklearn, orden fijo y ajuste solo con entrenamiento."""
import ast
import inspect
import json

import keras
import numpy as np
import pandas as pd
import pytest

from src import config, evaluate, explain, train
from src.data import cargar_datos
from src.preprocessing import (
    N_COLUMNAS,
    agregar_saldo_cero,
    ajustar_preprocesador,
    exportar_parametros,
    transformar,
    validar_parametros,
)

NOMBRES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "SatisfactionScore", "PointEarned",
    "HasCrCard", "IsActiveMember", "Complain", "saldo_cero",
    "Geography_France", "Geography_Germany", "Geography_Spain", "Gender_Female", "Gender_Male",
    "CardType_SILVER", "CardType_GOLD", "CardType_PLATINUM", "CardType_DIAMOND",
]  # fmt: skip


@pytest.fixture()
def ajuste(clientes):
    transformador = ajustar_preprocesador(clientes())
    return transformador, exportar_parametros(transformador)


def test_ida_y_vuelta_json_coincide_con_sklearn(ajuste, clientes):
    transformador, parametros = ajuste
    nuevos = clientes(25, semilla=1)
    leidos = json.loads(json.dumps(parametros, allow_nan=False))
    salida_sklearn = transformador.transform(agregar_saldo_cero(nuevos))
    assert np.array_equal(transformar(leidos, nuevos), salida_sklearn)


def test_veintiuna_columnas_en_orden_fijo(ajuste, clientes):
    transformador, parametros = ajuste
    assert N_COLUMNAS == 21
    assert parametros["nombres_salida"] == NOMBRES
    assert list(transformador.get_feature_names_out()) == NOMBRES
    assert transformar(parametros, clientes(5)).shape == (5, 21)


def test_el_orden_de_entrada_no_cambia_la_salida(ajuste, clientes):
    _, parametros = ajuste
    df = clientes(5)
    assert np.array_equal(transformar(parametros, df), transformar(parametros, df[df.columns[::-1]]))


def test_valores_a_mano(clientes):
    parametros = exportar_parametros(ajustar_preprocesador(clientes()))
    parametros["media"] = [0.0, 0, 0, 0, 0, 0, 0, 0]
    parametros["escala"] = [1.0, 1, 1, 1, 1, 2, 1, 10]
    fila = pd.DataFrame(
        [{"CreditScore": 600, "Age": 40, "Tenure": 3, "Balance": 0.0, "NumOfProducts": 2, "EstimatedSalary": 100.0,
          "SatisfactionScore": 4, "PointEarned": 500, "HasCrCard": 1, "IsActiveMember": 0, "Complain": 1, "Geography": "Germany",
          "Gender": "Male", "CardType": "GOLD"}]
    )  # fmt: skip
    assert transformar(parametros, fila).tolist() == [[600, 40, 3, 0, 2, 50, 4, 50, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]]


def test_transforma_los_datos_reales_sin_ajustar_con_ellos(ajuste):
    """Los parámetros salen de datos sintéticos; solo se comprueban tipos y forma con el CSV real."""
    _, parametros = ajuste
    cartera = cargar_datos()
    matriz = transformar(parametros, cartera)
    assert matriz.shape == (10_000, 21)
    assert np.array_equal(matriz[:, 11], (cartera["Balance"] == 0).to_numpy(dtype=float))
    assert all((matriz[:, a:b].sum(axis=1) == 1).all() for a, b in ((12, 15), (15, 17), (17, 21)))


def test_saldo_cero_y_sin_mutar_la_entrada():
    df = pd.DataFrame({"Balance": [0.0, 0.01]})
    assert agregar_saldo_cero(df)[config.COLUMNA_SALDO_CERO].tolist() == [1.0, 0.0]
    assert config.COLUMNA_SALDO_CERO not in df


def test_los_parametros_dependen_solo_de_lo_ajustado(clientes):
    entrenamiento = clientes(40)
    medias = exportar_parametros(ajustar_preprocesador(entrenamiento))["media"]
    assert medias == pytest.approx(entrenamiento[config.COLUMNAS_ESCALADAS].mean().tolist())


@pytest.mark.parametrize(
    "columna, valor", [("Geography", "Italy"), ("Gender", "Otro"), ("CardType", "BRONZE"), ("Age", np.nan), ("Balance", np.inf)]
)
def test_rechaza_categorias_desconocidas_y_no_finitos(ajuste, clientes, columna, valor):
    _, parametros = ajuste
    df = clientes(3)
    df[columna] = df[columna].astype(object if columna in config.COLUMNAS_CATEGORICAS else float)
    df.loc[0, columna] = valor
    with pytest.raises(ValueError):
        transformar(parametros, df)


@pytest.mark.parametrize(
    "cambio",
    [{"media": [0.0]}, {"escala": [0.0] * 8}, {"nombres_salida": ["x"]}, {"categorias": {}}, {"media": ["a"] * 8}],
)
def test_validar_parametros_rechaza_formas_invalidas(ajuste, cambio):
    _, parametros = ajuste
    validar_parametros(parametros)
    with pytest.raises(ValueError):
        validar_parametros({**parametros, **cambio})


def _llamadas(modulo, nombre: str) -> list[ast.Call]:
    arbol = ast.parse(inspect.getsource(modulo))
    return [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and (getattr(nodo.func, "attr", None) == nombre or getattr(nodo.func, "id", None) == nombre)
    ]


def _argumentos(llamada: ast.Call) -> list[str]:
    return [ast.unparse(a) for a in llamada.args]


def test_el_codigo_ajusta_solo_con_entrenamiento():
    """El camino de entrenamiento no corre aquí: se comprueba leyendo su AST."""
    assert list(inspect.signature(ajustar_preprocesador).parameters) == ["entrenamiento"]
    [llamada] = _llamadas(train, "ajustar_preprocesador")
    assert _argumentos(llamada) == ["particion.entrenamiento"]
    assert not _llamadas(evaluate, "ajustar_preprocesador")
    assert not _llamadas(explain, "ajustar_preprocesador")

    [red] = [c for c in _llamadas(train, "fit") if ast.unparse(c.func) == "modelo.fit"]
    assert _argumentos(red) == ["crear_lotes(x_entrenamiento, y_entrenamiento)"]
    [platt] = _llamadas(train, "ajustar_platt")
    assert _argumentos(platt) == ["p_validacion", "y_validacion"]
    [base] = _llamadas(evaluate, "fit")
    assert _argumentos(base) == ["x_entrenamiento", "y_entrenamiento"]
    [prediccion] = _llamadas(evaluate, "predict_proba")
    assert _argumentos(prediccion) == ["x_prueba"]
    [fondo_global, fondo_vivo] = _llamadas(explain, "kmeans")
    assert ast.unparse(fondo_global.args[0]) == ast.unparse(fondo_vivo.args[0]) == "x_entrenamiento"


def test_fit_recibe_lotes_y_validacion_sin_class_weight():
    [red] = [c for c in _llamadas(train, "fit") if ast.unparse(c.func) == "modelo.fit"]
    nombrados = {k.arg: ast.unparse(k.value) for k in red.keywords}
    assert nombrados["validation_data"] == "(x_validacion, y_validacion)"
    assert nombrados["validation_batch_size"] == "TAM_LOTE"
    assert not {"class_weight", "batch_size", "shuffle"} & nombrados.keys()  # con un dataset, los fija crear_lotes
    assert nombrados.keys() <= inspect.signature(keras.Model.fit).parameters.keys()
