"""Etiquetas de variables en SHAP, con y sin categoría."""
import pytest

from app.comun import etiqueta_variable


@pytest.mark.parametrize(
    "nombre, esperado",
    [("Geography", "País"), ("Gender", "Género"), ("Geography_France", "País: Francia"), ("saldo_cero", "Saldo cero")],
)
def test_etiqueta_variable(nombre, esperado):
    assert etiqueta_variable(nombre) == esperado
