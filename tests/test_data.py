import hashlib

import pandas as pd
import pytest

from src import config, data
from src.data import ErrorDatos, cargar_datos, particionar, validar_datos

FILA_VALIDA = {
    "RowNumber": "1", "CustomerId": "15634602", "Surname": "Prueba", "CreditScore": "600", "Geography": "France",
    "Gender": "Male", "Age": "40", "Tenure": "10", "Balance": "1000.5", "NumOfProducts": "2", "HasCrCard": "0",
    "IsActiveMember": "1", "EstimatedSalary": "2000.0", "Exited": "1", "Complain": "1", "Satisfaction Score": "3",
    "Card Type": "GOLD", "Point Earned": "450",
}  # fmt: skip


def _escribir(tmp_path, texto):
    """Escribe un CSV temporal y devuelve su ruta y su SHA-256."""
    ruta = tmp_path / "datos.csv"
    ruta.write_bytes(texto.encode("utf-8"))
    return ruta, hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _csv(*cambios_por_fila):
    filas = [{**FILA_VALIDA, "RowNumber": str(i), **cambios} for i, cambios in enumerate(cambios_por_fila, 1)]
    cabecera = ",".join(filas[0])
    return cabecera + "\n" + "\n".join(",".join(f.values()) for f in filas) + "\n"


@pytest.fixture(scope="module")
def datos():
    return cargar_datos()


def test_suma_incorrecta_se_rechaza_antes_de_interpretar(tmp_path, monkeypatch):
    def no_debe_llamarse(_):
        raise AssertionError("se interpretó antes de verificar")

    monkeypatch.setattr(data, "_interpretar_csv", no_debe_llamarse)
    ruta, _ = _escribir(tmp_path, _csv({}))
    with pytest.raises(ErrorDatos, match="SHA-256"):
        cargar_datos(ruta, sha256="0" * 64)


def test_archivo_real_con_suma_incorrecta_se_rechaza():
    with pytest.raises(ErrorDatos, match="SHA-256"):
        cargar_datos(sha256="0" * 64)


def test_csv_valido_se_carga_y_convierte(tmp_path):
    df = cargar_datos(*_escribir(tmp_path, _csv({}, {"Geography": "Spain", "Exited": "0"})))
    assert list(df[config.COLUMNA_ID]) == ["CLI-00001", "CLI-00002"]
    assert set(df.columns) == {config.COLUMNA_ID, *config.COLUMNAS_ENTRADA, config.OBJETIVO}
    assert list(df["Exited"]) == [1, 0]
    assert list(df["Geography"]) == ["France", "Spain"]
    for columna in config.RANGOS_ENTEROS:
        assert pd.api.types.is_integer_dtype(df[columna])
    assert df["Tenure"].iloc[0] == 10 and df["CardType"].iloc[0] == "GOLD"


def test_descarta_identificadores(tmp_path):
    df = cargar_datos(*_escribir(tmp_path, _csv({})))
    assert not {"CustomerId", "Surname", "RowNumber"} & set(df.columns)
    assert df["Complain"].iloc[0] == 1


@pytest.mark.parametrize(
    "cambio, columna",
    [
        ({"CreditScore": "349"}, "CreditScore"),
        ({"CreditScore": "851"}, "CreditScore"),
        ({"CreditScore": "600.5"}, "CreditScore"),
        ({"Age": "17"}, "Age"),
        ({"Age": "93"}, "Age"),
        ({"Balance": "-0.01"}, "Balance"),
        ({"EstimatedSalary": "-1"}, "EstimatedSalary"),
        ({"Tenure": "11"}, "Tenure"),
        ({"NumOfProducts": "5"}, "NumOfProducts"),
        ({"IsActiveMember": "2"}, "IsActiveMember"),
        ({"Complain": "2"}, "Complain"),
        ({"Satisfaction Score": "0"}, "SatisfactionScore"),
        ({"Point Earned": "1001"}, "PointEarned"),
        ({"Exited": "2"}, "Exited"),
        ({"Geography": "Mars"}, "Geography"),
        ({"Gender": "Otro"}, "Gender"),
        ({"Card Type": "BRONZE"}, "CardType"),
        ({"Balance": ""}, "Balance"),
        ({"Geography": ""}, "Geography"),
        ({"HasCrCard": "x"}, "HasCrCard"),
        ({"Balance": "inf"}, "Balance"),
    ],
)
def test_valores_invalidos_se_rechazan(tmp_path, cambio, columna):
    with pytest.raises(ErrorDatos, match=columna):
        cargar_datos(*_escribir(tmp_path, _csv({}, cambio)))


def test_limites_validos_se_aceptan(tmp_path):
    bajo = {"CreditScore": "350", "Age": "18", "Balance": "0", "EstimatedSalary": "0", "Tenure": "0", "NumOfProducts": "1"}
    alto = {"CreditScore": "850", "Age": "92", "Tenure": "10", "NumOfProducts": "4", "Point Earned": "1000"}
    assert len(cargar_datos(*_escribir(tmp_path, _csv(bajo, alto)))) == 2


def test_columna_faltante_se_rechaza(tmp_path):
    texto = _csv({}).replace(",Age", "", 1).replace("Male,40,", "Male,", 1)
    with pytest.raises(ErrorDatos, match="Age"):
        cargar_datos(*_escribir(tmp_path, texto))


def test_numero_de_fila_repetido_se_rechaza(tmp_path):
    with pytest.raises(ErrorDatos, match="RowNumber"):
        cargar_datos(*_escribir(tmp_path, _csv({"RowNumber": "1"}, {"RowNumber": "1"})))


def test_csv_mal_formado_se_rechaza(tmp_path):
    with pytest.raises(ErrorDatos):
        cargar_datos(*_escribir(tmp_path, "esto no es un csv"))


def test_csv_sin_filas_se_rechaza(tmp_path):
    with pytest.raises(ErrorDatos, match="sin filas"):
        cargar_datos(*_escribir(tmp_path, ",".join(FILA_VALIDA) + "\n"))


def test_validar_cliente_sin_objetivo(datos):
    validar_datos(datos[config.COLUMNAS_ENTRADA].head(1), con_objetivo=False)
    with pytest.raises(ErrorDatos, match="Exited"):
        validar_datos(datos[config.COLUMNAS_ENTRADA].head(1))


def test_archivo_real(datos):
    assert len(datos) == 10_000
    assert datos["Exited"].mean() == pytest.approx(0.2038)
    assert datos[config.COLUMNA_ID].iloc[0] == "CLI-00001"
    assert datos[config.COLUMNA_ID].iloc[-1] == "CLI-10000"
    assert datos[config.COLUMNA_ID].is_unique
    # Primera fila del archivo
    fila = datos.iloc[0]
    assert (fila["CreditScore"], fila["Age"], fila["Geography"], fila["Tenure"]) == (619, 42, "France", 2)
    assert (fila["SatisfactionScore"], fila["CardType"], fila["PointEarned"]) == (2, "DIAMOND", 464)
    assert fila["Balance"] == 0


def test_particion_tamanos_y_estratificacion(datos):
    p = particionar(datos)
    assert (len(p.entrenamiento), len(p.validacion), len(p.prueba)) == (7_000, 1_500, 1_500)
    for parte in p:
        assert parte["Exited"].mean() == pytest.approx(0.2038, abs=0.001)
    ids = pd.concat(p)[config.COLUMNA_ID]
    assert ids.is_unique
    assert set(ids) == set(datos[config.COLUMNA_ID])


def test_particion_reproducible(datos):
    a, b = particionar(datos), particionar(datos)
    for parte_a, parte_b in zip(a, b):
        assert list(parte_a[config.COLUMNA_ID]) == list(parte_b[config.COLUMNA_ID])
    assert list(a.prueba[config.COLUMNA_ID]) != list(particionar(datos, semilla=7).prueba[config.COLUMNA_ID])
