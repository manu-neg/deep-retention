"""Inferencia y funciones puras de entrenamiento, evaluación y SHAP, sin entrenar nada."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import config, evaluate, explain, inference, train
from src.finance import decidir_cartera, resumir_kpis
from src.inference import (
    ARCHIVOS_ARTEFACTOS,
    Artefactos,
    ErrorArtefactos,
    aplicar_platt,
    artefactos_disponibles,
    cargar_artefactos,
    cargar_nucleo,
    guardar_json,
    leer_json,
    logit,
    predecir_cruda,
    predecir_fuga,
    validar_calibrador,
)
from src.preprocessing import transformar


@pytest.fixture(scope="module")
def directorio(tmp_path_factory, artefactos_validos):
    """Artefactos completos con una red SIN entrenar."""
    ruta = tmp_path_factory.mktemp("artefactos")
    for nombre, contenido in artefactos_validos.items():
        (ruta / nombre).write_bytes(contenido)
    return ruta


# --- Arquitectura ---


def test_arquitectura_y_parametros(modelo):
    tipos = [capa.__class__.__name__ for capa in modelo.layers]
    assert tipos == ["Dense", "BatchNormalization", "Dropout", "Dense", "Dropout", "Dense"]
    assert [c.units for c in modelo.layers if c.__class__.__name__ == "Dense"] == [128, 64, 1]
    assert [c.rate for c in modelo.layers if c.__class__.__name__ == "Dropout"] == [0.3, 0.2]
    assert modelo.layers[0].activation.__name__ == modelo.layers[3].activation.__name__ == "relu"
    assert modelo.layers[-1].activation.__name__ == "sigmoid"
    assert modelo.input_shape == (None, 21)
    assert sum(int(np.prod(p.shape)) for p in modelo.trainable_weights) == 11_393


def test_compilacion(modelo):
    assert modelo.optimizer.__class__.__name__ == "Adam"
    assert float(modelo.optimizer.learning_rate) == pytest.approx(1e-3)
    assert modelo.loss == "binary_crossentropy"


# --- Carga e inferencia ---


def test_ida_y_vuelta_keras_con_safe_mode(directorio, modelo, parametros, clientes):
    cargado = cargar_nucleo(directorio).modelo
    x = transformar(parametros, clientes(8)).astype("float32")
    assert np.allclose(cargado.predict_on_batch(x), modelo.predict_on_batch(x))


def test_keras_con_lambda_no_carga_con_safe_mode(tmp_path, parametros):
    import keras

    modelo_lambda = keras.Sequential([keras.layers.Input((21,)), keras.layers.Lambda(lambda x: x, output_shape=(21,))])
    modelo_lambda.save(tmp_path / "modelo.keras")
    guardar_json(tmp_path / "preprocesador.json", parametros)
    guardar_json(tmp_path / "calibrador.json", {"a": 1.0, "b": 0.0})
    with pytest.raises(ErrorArtefactos):
        cargar_nucleo(tmp_path)


def test_disponibilidad_exige_todos_los_archivos(tmp_path, directorio):
    assert artefactos_disponibles(directorio)
    assert not artefactos_disponibles(tmp_path)
    for nombre in ARCHIVOS_ARTEFACTOS[:-1]:
        (tmp_path / nombre).write_bytes(b"x")
    assert not artefactos_disponibles(tmp_path)


def test_predecir_fuga_aplica_platt_sobre_la_red(directorio, clientes):
    artefactos = cargar_artefactos(directorio)
    df = clientes(6)
    p = predecir_fuga(artefactos, df)
    assert isinstance(p, np.ndarray) and p.shape == (6,)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.allclose(p, aplicar_platt({"a": 1.5, "b": -0.5}, predecir_cruda(artefactos, df)))


def test_cargar_rechaza_json_invalido(tmp_path, directorio):
    for nombre in ARCHIVOS_ARTEFACTOS:
        (tmp_path / nombre).write_bytes((directorio / nombre).read_bytes())
    (tmp_path / "calibrador.json").write_text('{"a": NaN, "b": 0}', encoding="utf-8")
    with pytest.raises(ErrorArtefactos):
        cargar_artefactos(tmp_path)


def test_leer_json_es_estricto(tmp_path):
    ruta = tmp_path / "x.json"
    ruta.write_text('{"a": 1}', encoding="utf-8")
    assert leer_json(ruta) == {"a": 1}
    ruta.write_text('{"a": Infinity}', encoding="utf-8")
    with pytest.raises(ErrorArtefactos, match="x.json"):
        leer_json(ruta)


@pytest.mark.parametrize("a", [0, 0.0, -1.5])
def test_calibrador_exige_a_positivo(a):
    with pytest.raises(ValueError, match="a > 0"):
        validar_calibrador({"a": a, "b": 0.0})


def test_platt_con_coeficientes_conocidos():
    assert aplicar_platt({"a": 1.0, "b": 0.0}, [0.2, 0.5, 0.9]) == pytest.approx([0.2, 0.5, 0.9])
    # p=0,5 → logit 0 → sigmoide(1) = 0,731058...
    assert aplicar_platt({"a": 2.0, "b": 1.0}, 0.5) == pytest.approx(0.7310585786)
    # p=0,8 → logit ln 4 → sigmoide(2·ln 4 − 1) = 16/(16 + e)
    assert aplicar_platt({"a": 2.0, "b": -1.0}, 0.8) == pytest.approx(16 / (16 + np.e))


def test_logit_finito_en_los_extremos():
    assert np.isfinite(logit([0.0, 1.0])).all()


def test_inferencia_no_importa_entrenamiento_ni_shap():
    codigo = (
        "import sys, src.inference;"
        "prohibidos = {'src.train', 'src.evaluate', 'src.explain', 'shap', 'tensorflow', 'keras', 'sklearn'};"
        "print(sorted(prohibidos & set(sys.modules)))"
    )
    salida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, check=True, cwd=config.RAIZ)
    assert salida.stdout.strip() == "[]"


def test_explain_no_carga_shap_al_importarse():
    codigo = "import sys, src.explain; print('shap' in sys.modules)"
    salida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, check=True, cwd=config.RAIZ)
    assert salida.stdout.strip() == "False"


# --- Entrenamiento: piezas comprobables sin ajustar ---


def test_pesos_clase_a_mano():
    # N=10, N0=8, N1=2 → 10/16 y 10/4
    assert train.pesos_clase([0] * 8 + [1] * 2) == {0: 0.625, 1: 2.5}
    with pytest.raises(ValueError):
        train.pesos_clase([0, 0])


def test_callbacks_del_diseno():
    parada, tasa = train.crear_callbacks()
    assert (parada.monitor, parada.patience, parada.restore_best_weights) == ("val_loss", 15, True)
    assert (tasa.monitor, tasa.factor, tasa.patience) == ("val_loss", 0.5, 5)
    assert (train.TAM_LOTE, train.MAX_EPOCAS) == (256, 200)


def _epoca(lotes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.concatenate(columna) for columna in zip(*lotes.as_numpy_iterator()))


def test_lotes_se_rebarajan_en_cada_epoca_y_llevan_pesos_de_clase():
    n = 600  # más de dos lotes de 256
    x = np.arange(n, dtype=float).reshape(-1, 1)
    y = (np.arange(n) % 5 == 0).astype("float32")
    lotes = train.crear_lotes(x, y)
    assert [len(b[0]) for b in lotes.as_numpy_iterator()] == [256, 256, 88]

    (x1, y1, w1), (x2, _, _) = _epoca(lotes), _epoca(lotes)
    assert not np.array_equal(x1, x2)  # otro orden…
    assert np.array_equal(np.sort(x1, axis=0), x) and np.array_equal(np.sort(x2, axis=0), x)  # …con las mismas muestras

    # Cada muestra conserva su etiqueta y el peso N/(2·N_k): 600/240 = 2,5 y 600/960 = 0,625.
    indices = x1.ravel().astype(int)
    assert np.array_equal(y1, y[indices])
    assert np.array_equal(w1, np.where(y1 == 1, 2.5, 0.625))


# --- Evaluación con predicciones sintéticas ---


def test_ece_a_mano():
    # Bin [0,1; 0,2): conf 0,15, tasa 0,5, n=2. Bin [0,8; 0,9): conf 0,85, tasa 1, n=2.
    y, p = [0, 1, 1, 1], [0.12, 0.18, 0.82, 0.88]
    assert evaluate.calcular_ece(y, p) == pytest.approx((2 * 0.35 + 2 * 0.15) / 4)


def test_ece_p_uno_cae_en_el_ultimo_bin_y_calibracion_perfecta_es_cero():
    assert evaluate.calcular_ece([1, 1], [1.0, 1.0]) == pytest.approx(0)
    assert [f["bin"] for f in evaluate.tabla_fiabilidad([1], [1.0])] == [9]


def test_metricas_probabilidad_a_mano():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    m = evaluate.metricas_probabilidad(y, p)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx((0.01 + 0.16 + 0.4225 + 0.04) / 4)
    assert set(m) == {"roc_auc", "pr_auc", "brier", "log_loss", "ece"}


def test_comparar_modelos_no_mide_la_calibracion_de_la_linea_base():
    fila = {"roc_auc": 0.8, "pr_auc": 0.6, "brier": 0.1, "log_loss": 0.3, "ece": 0.02}
    comparacion = evaluate.comparar_modelos({"red_cruda": fila, "red_calibrada": fila, "linea_base": fila})
    assert set(comparacion["auc"]) == {"linea_base", "red_calibrada"}
    assert set(comparacion["calibracion"]) == {"red_cruda", "red_calibrada"}


def test_auc_por_grupo():
    df = pd.DataFrame(
        {
            "Gender": ["Female", "Female", "Male", "Male"],
            "Geography": ["France", "France", "Spain", "Spain"],
            config.OBJETIVO: [0, 1, 1, 1],
        }
    )
    resultado = evaluate.auc_por_grupo(df, [0.1, 0.9, 0.5, 0.6])
    assert resultado["Gender"]["Female"]["auc"] == 1.0
    assert resultado["Gender"]["Male"]["auc"] is None  # solo una clase
    assert resultado["Geography"]["Germany"] == {"n": 0, "tasa_fuga": None, "auc": None}
    json.dumps(resultado, allow_nan=False)


def test_kpis_incluyen_ganancia_neta_y_retener_a_todos(clientes):
    df = clientes(10)
    p = np.linspace(0.05, 0.95, 10)
    kpis = evaluate.kpis_financieros(df, p)
    resumen = resumir_kpis(decidir_cartera(df, p))
    assert kpis["clientes"] == 10
    assert kpis["inversion"] == resumen.inversion
    assert kpis["ganancia_neta"] == pytest.approx(resumen.retorno_esperado - resumen.inversion)
    comparacion = kpis["comparacion"]
    assert comparacion["selectiva"]["ganancia_neta"] == pytest.approx(kpis["ganancia_neta"])
    assert comparacion["retener_a_todos"]["intervenidos"] == int((p >= config.P_RIESGO_CRITICO).sum())
    assert list(kpis["por_segmento"]) == list(config.SEGMENTOS)
    assert set(kpis["por_segmento"]["A"]) == {"clientes", "capital", "inversion", "retorno_esperado", "ganancia_neta"}
    assert sum(s["clientes"] for s in kpis["por_segmento"].values()) == 10
    assert sum(s["ganancia_neta"] for s in kpis["por_segmento"].values()) == pytest.approx(kpis["ganancia_neta"])
    json.dumps(kpis, allow_nan=False)


@pytest.mark.parametrize(
    "val_loss, esperada",
    [
        ([0.7, 0.5, 0.6, 0.65], 2),  # no cuenta las épocas de paciencia
        ([0.4, 0.5], 1),
        ([0.7, 0.5, 0.5], 2),  # en un empate, EarlyStopping se queda con la primera
        ([0.7, float("nan"), 0.6], 3),
    ],
)
def test_mejor_epoca(val_loss, esperada):
    assert evaluate.mejor_epoca({"val_loss": val_loss}) == esperada


def test_figuras_se_escriben_como_png(tmp_path):
    y = np.array([0, 1] * 20)
    p = np.linspace(0, 1, 40)
    evaluate.figura_fiabilidad(y, {"Red cruda": p, "Red calibrada": 1 - p}, tmp_path / "f.png")
    evaluate.figura_historial({"loss": [0.7, 0.6], "val_loss": [0.71, 0.65]}, tmp_path / "h.png")
    explain.figura_beeswarm(np.random.default_rng(0).normal(size=(10, 3)), np.ones((10, 3)), list("abc"), tmp_path / "b.png")
    for nombre in ("f.png", "h.png", "b.png"):
        assert (tmp_path / nombre).read_bytes().startswith(b"\x89PNG")


class _LineaBaseSimulada:
    """Sustituye a la regresión logística: aquí no se ajusta ningún modelo."""

    def __init__(self, **_):
        pass

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        p = np.linspace(0.05, 0.95, len(x))
        return np.column_stack([1 - p, p])


def test_evaluate_escribe_metricas_completas(monkeypatch, tmp_path, directorio, clientes):
    """Recorre main con una red sin entrenar, datos sintéticos y la línea base simulada."""
    monkeypatch.setattr(config, "DIR_MODELOS", tmp_path / "models")
    monkeypatch.setattr(config, "DIR_FIGURAS", tmp_path / "figuras")
    monkeypatch.setattr(config, "RUTA_HISTORIAL", tmp_path / "historial.json")
    monkeypatch.setattr(evaluate, "cargar_nucleo", lambda: cargar_nucleo(directorio))
    monkeypatch.setattr(evaluate, "cargar_datos", lambda: clientes(200))
    monkeypatch.setattr(evaluate, "LogisticRegression", _LineaBaseSimulada)
    guardar_json(config.RUTA_HISTORIAL, {"loss": [0.7, 0.6], "val_loss": [0.71, 0.65]})

    evaluate.main()

    metricas = leer_json(tmp_path / "models" / "metricas.json")
    assert set(metricas["prueba"]) == set(config.ETIQUETAS_MODELOS)
    assert "validación" in metricas["nota"] and "cartera completa" in metricas["nota"]
    assert set(metricas["comparacion_modelos"]) == {"auc", "calibracion"}
    for conjunto, n in (("cartera_completa", 200), ("prueba", 30)):
        kpi = metricas["kpi"][conjunto]
        assert kpi["clientes"] == n
        assert kpi["ganancia_neta"] == pytest.approx(kpi["retorno_esperado"] - kpi["inversion"])
        assert set(kpi["comparacion"]) == {"selectiva", "retener_a_todos"}
        assert sum(s["clientes"] for s in kpi["por_segmento"].values()) == n
    assert metricas["epocas_entrenadas"] == 2
    assert metricas["mejor_epoca"] == 2
    assert sorted(p.name for p in (tmp_path / "figuras").iterdir()) == [evaluate.FIGURA_FIABILIDAD, evaluate.FIGURA_HISTORIAL]


# --- SHAP con red sin entrenar ---


def test_explicar_cliente_contrato(directorio, clientes):
    artefactos = cargar_artefactos(directorio)
    cliente = clientes(1, semilla=7)
    filas = explain.explicar_cliente(artefactos, cliente)
    assert len(filas) == 21
    assert all(set(f) == {"variable", "valor", "contribucion"} for f in filas)
    assert {f["variable"] for f in filas} == set(artefactos.preprocesador["nombres_salida"])
    magnitudes = [abs(f["contribucion"]) for f in filas]
    assert magnitudes == sorted(magnitudes, reverse=True)
    json.dumps(filas, allow_nan=False)
    por_variable = {f["variable"]: f["valor"] for f in filas}
    assert por_variable["Age"] == float(cliente["Age"].iloc[0])  # unidades originales, no z
    # Propiedad de eficiencia de SHAP: la suma explica p − E[p] sobre el fondo ponderado.
    fondo = np.asarray(artefactos.fondo_shap["datos"])
    pesos = np.asarray(artefactos.fondo_shap["pesos"])
    esperada = np.average(inference.predecir_matriz(artefactos, fondo), weights=pesos)
    diferencia = predecir_fuga(artefactos, cliente)[0] - esperada
    assert sum(f["contribucion"] for f in filas) == pytest.approx(diferencia, abs=1e-4)


def test_explicar_cliente_exige_una_fila(directorio, clientes):
    with pytest.raises(ValueError):
        explain.explicar_cliente(cargar_artefactos(directorio), clientes(2))


def test_valores_shap_globales_con_fondo_matriz(directorio):
    artefactos = cargar_artefactos(directorio)
    fondo = np.asarray(artefactos.fondo_shap["datos"])
    valores = explain.valores_shap(artefactos, fondo, fondo[:2], 100)
    assert valores.shape == (2, 21)


def test_resumen_global_a_mano():
    valores = np.array([[1.0, -3.0, 0.0, 0.5], [-1.0, 1.0, 0.0, 0.5]])
    resumen = explain.resumen_global(valores, list("abcd"))
    assert resumen["variables"] == ["b", "a", "d", "c"]
    assert resumen["importancia_media_abs"] == [2.0, 1.0, 0.5, 0.0]
    assert resumen["factores_principales"] == ["b", "a", "d"]


def test_resumen_global_agrupa_one_hot_con_signo_por_cliente():
    nombres = ["Age", "Geography_France", "Geography_Germany", "Gender_Female", "Gender_Male"]
    valores = np.array([[0.1, 0.3, -0.3, 0.2, 0.1], [-0.1, 0.5, 0.1, 0.0, -0.4]])
    # Por cliente: Geography 0,0 y 0,6 → 0,3 (por separado serían 0,4 + 0,2). Gender 0,3 y −0,4 → 0,35. Age → 0,1.
    resumen = explain.resumen_global(valores, nombres)
    assert resumen["variables"] == ["Gender", "Geography", "Age"]
    assert resumen["importancia_media_abs"] == pytest.approx([0.35, 0.3, 0.1])
    assert resumen["factores_principales"] == ["Gender", "Geography", "Age"]
