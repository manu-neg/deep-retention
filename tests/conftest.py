"""Clientes sintéticos y artefactos con una red sin entrenar, compartidos por los tests."""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # antes de que algún test importe matplotlib: sin pantalla

import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocessing import ajustar_preprocesador, exportar_parametros, transformar


def _clientes(n: int = 60, semilla: int = 0) -> pd.DataFrame:
    g = np.random.default_rng(semilla)
    return pd.DataFrame(
        {
            "CreditScore": g.integers(350, 851, n),
            "Age": g.integers(18, 93, n),
            "Tenure": g.integers(0, 11, n),
            "Balance": g.choice([0.0, 1234.5, 80_000.25, 150_000.0], n),
            "NumOfProducts": g.integers(1, 5, n),
            "EstimatedSalary": g.uniform(0, 200_000, n),
            "HasCrCard": g.integers(0, 2, n),
            "IsActiveMember": g.integers(0, 2, n),
            "Complain": g.integers(0, 2, n),
            "Geography": g.choice(config.CATEGORIAS["Geography"], n),
            "Gender": g.choice(config.CATEGORIAS["Gender"], n),
            "SatisfactionScore": g.integers(1, 6, n),
            "PointEarned": g.integers(100, 1001, n),
            "CardType": g.choice(config.CATEGORIAS["CardType"], n),
            config.OBJETIVO: g.integers(0, 2, n),
        }
    )


@pytest.fixture(scope="session")
def clientes():
    """Fábrica: clientes(n, semilla) devuelve un DataFrame sintético nuevo."""
    return _clientes


@pytest.fixture(scope="session")
def parametros() -> dict:
    return exportar_parametros(ajustar_preprocesador(_clientes()))


@pytest.fixture(scope="session")
def modelo():
    from src.model import construir_mlp  # diferido: los tests de finanzas no cargan TensorFlow

    return construir_mlp()


@pytest.fixture(scope="session")
def artefactos_validos(tmp_path_factory, parametros, modelo) -> dict[str, bytes]:
    """Contenido de cada artefacto, con una red SIN entrenar."""
    ruta = tmp_path_factory.mktemp("modelo") / "modelo.keras"
    modelo.save(ruta)
    fondo = transformar(parametros, _clientes(5, semilla=3))
    return {
        "modelo.keras": ruta.read_bytes(),
        "preprocesador.json": json.dumps(parametros).encode(),
        "calibrador.json": b'{"a": 1.5, "b": -0.5}',
        "metricas.json": b'{"prueba": {}, "roi": null}',
        "shap_global.json": b'{"variables": []}',
        "fondo_shap.json": json.dumps({"datos": fondo.tolist(), "pesos": [1.0, 2.0, 1.0, 3.0, 1.0]}).encode(),
    }
