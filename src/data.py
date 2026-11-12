"""Carga verificada, validación y partición del conjunto de datos."""
import hashlib
import io
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


class ErrorDatos(ValueError):
    """El archivo o los datos no cumplen lo exigido por el diseño."""


class Particion(NamedTuple):
    entrenamiento: pd.DataFrame
    validacion: pd.DataFrame
    prueba: pd.DataFrame


def cargar_datos(ruta: Path = config.RUTA_DATOS, sha256: str = config.SHA256_DATOS) -> pd.DataFrame:
    """Verifica el SHA-256, interpreta el CSV, lo valida y añade `cliente_id` (CLI-<RowNumber>)."""
    # Se interpretan exactamente los bytes verificados, sin releer el archivo.
    contenido = Path(ruta).read_bytes()
    obtenido = hashlib.sha256(contenido).hexdigest()
    if obtenido != sha256:
        raise ErrorDatos(f"SHA-256 incorrecto en {Path(ruta).name}: se esperaba {sha256}, se obtuvo {obtenido}")

    df = _normalizar(_interpretar_csv(contenido))
    filas = df.pop(config.COLUMNA_FILA)
    validar_datos(df)
    df = df.astype({columna: "int64" for columna in config.RANGOS_ENTEROS})
    df.insert(0, config.COLUMNA_ID, [f"CLI-{int(i):05d}" for i in filas])
    return df


def _interpretar_csv(contenido: bytes) -> pd.DataFrame:
    try:
        # Todo como texto: los tipos se deciden después, columna por columna.
        df = pd.read_csv(io.BytesIO(contenido), dtype=str, keep_default_na=False, encoding="utf-8")
    except Exception as error:
        raise ErrorDatos(f"CSV no interpretable: {error!r}") from error
    if df.empty:
        raise ErrorDatos("CSV no interpretable: sin filas")
    return df


def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra, descarta columnas excluidas, marca vacíos como nulos y convierte los numéricos."""
    df = df.rename(columns=config.RENOMBRES_CSV).drop(columns=config.COLUMNAS_DESCARTADAS, errors="ignore")
    if config.COLUMNA_FILA not in df:
        raise ErrorDatos(f"Falta la columna {config.COLUMNA_FILA}")
    for columna in df.columns:
        df[columna] = df[columna].str.strip().replace("", None)
        if columna not in config.COLUMNAS_CATEGORICAS:
            try:
                df[columna] = pd.to_numeric(df[columna])
            except (ValueError, TypeError) as error:
                raise ErrorDatos(f"{columna}: valor no numérico") from error
    filas = df[config.COLUMNA_FILA]
    if filas.isna().any() or not filas.is_unique or (filas != filas.round()).any() or (filas < 1).any():
        raise ErrorDatos(f"{config.COLUMNA_FILA}: se esperan enteros positivos únicos")
    return df


def validar_datos(df: pd.DataFrame, con_objetivo: bool = True) -> None:
    """Exige columnas, nulos, rangos y categorías del diseño; lanza ErrorDatos si no se cumplen."""
    esperadas = set(config.COLUMNAS_ENTRADA) | ({config.OBJETIVO} if con_objetivo else set())
    presentes = set(df.columns) - {config.COLUMNA_ID}
    if presentes != esperadas:
        raise ErrorDatos(f"Columnas incorrectas: faltan {esperadas - presentes}, sobran {presentes - esperadas}")

    nulas = [columna for columna in esperadas if df[columna].isna().any()]
    if nulas:
        raise ErrorDatos(f"Valores nulos en {sorted(nulas)}")

    for columna, (minimo, maximo) in config.RANGOS_ENTEROS.items():
        if columna in esperadas:
            valores = _numericos(df, columna)
            if (valores != np.floor(valores)).any() or (valores < minimo).any() or (valores > maximo).any():
                raise ErrorDatos(f"{columna}: se esperan enteros entre {minimo} y {maximo}")

    for columna in config.COLUMNAS_NO_NEGATIVAS:
        if (_numericos(df, columna) < 0).any():
            raise ErrorDatos(f"{columna}: no puede ser negativo")

    for columna, categorias in config.CATEGORIAS.items():
        invalidas = set(df[columna]) - set(categorias)
        if invalidas:
            raise ErrorDatos(f"{columna}: categorías inválidas {sorted(map(str, invalidas))}")


def _numericos(df: pd.DataFrame, columna: str) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(df[columna]):
        raise ErrorDatos(f"{columna}: debe ser numérica")
    valores = df[columna].to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise ErrorDatos(f"{columna}: valores no finitos")
    return valores


def particionar(df: pd.DataFrame, semilla: int = config.SEMILLA) -> Particion:
    """Partición estratificada entrenamiento/validación/prueba (70/15/15)."""
    n_validacion = round(len(df) * config.FRACCION_VALIDACION)
    n_prueba = round(len(df) * config.FRACCION_PRUEBA)
    entrenamiento, resto = train_test_split(
        df, test_size=n_validacion + n_prueba, random_state=semilla, stratify=df[config.OBJETIVO]
    )
    validacion, prueba = train_test_split(
        resto, test_size=n_prueba, random_state=semilla, stratify=resto[config.OBJETIVO]
    )
    return Particion(*(parte.reset_index(drop=True) for parte in (entrenamiento, validacion, prueba)))
