"""Preprocesamiento: ajuste con sklearn solo en entrenamiento; aplicación desde JSON, sin sklearn."""
import numpy as np
import pandas as pd

from src import config

COLUMNAS_DIRECTAS = [*config.COLUMNAS_BINARIAS, config.COLUMNA_SALDO_CERO]
N_COLUMNAS = 21


def agregar_saldo_cero(df: pd.DataFrame) -> pd.DataFrame:
    """Copia de `df` con el indicador de saldo exactamente cero."""
    return df.assign(**{config.COLUMNA_SALDO_CERO: (df["Balance"] == 0).astype(float)})


def ajustar_preprocesador(entrenamiento: pd.DataFrame):
    """Ajusta el ColumnTransformer; recibe únicamente la partición de entrenamiento."""
    from sklearn.compose import ColumnTransformer  # diferido: la inferencia no necesita sklearn
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorias = [list(config.CATEGORIAS[columna]) for columna in config.COLUMNAS_CATEGORICAS]
    transformador = ColumnTransformer(
        [
            ("escala", StandardScaler(), config.COLUMNAS_ESCALADAS),
            ("directas", "passthrough", COLUMNAS_DIRECTAS),
            ("categorias", OneHotEncoder(categories=categorias, sparse_output=False), config.COLUMNAS_CATEGORICAS),
        ],
        verbose_feature_names_out=False,
    )
    return transformador.fit(agregar_saldo_cero(entrenamiento))


def exportar_parametros(transformador) -> dict:
    """Parámetros ajustados como datos JSON puros."""
    escala = transformador.named_transformers_["escala"]
    codificador = transformador.named_transformers_["categorias"]
    return {
        "columnas_escaladas": list(config.COLUMNAS_ESCALADAS),
        "media": escala.mean_.tolist(),
        "escala": escala.scale_.tolist(),
        "columnas_directas": list(COLUMNAS_DIRECTAS),
        "categorias": {
            columna: valores.tolist() for columna, valores in zip(config.COLUMNAS_CATEGORICAS, codificador.categories_)
        },
        "nombres_salida": transformador.get_feature_names_out().tolist(),
    }


def validar_parametros(parametros: dict) -> None:
    """Exige la forma esperada, porque el JSON llega desde fuera de esta máquina."""
    try:
        media = np.asarray(parametros["media"], dtype=float)
        escala = np.asarray(parametros["escala"], dtype=float)
        esperados = (
            parametros["columnas_escaladas"] == config.COLUMNAS_ESCALADAS
            and parametros["columnas_directas"] == COLUMNAS_DIRECTAS
            and parametros["categorias"] == {c: list(v) for c, v in config.CATEGORIAS.items()}
            and parametros["nombres_salida"] == _nombres_esperados()
            and media.shape == escala.shape == (len(config.COLUMNAS_ESCALADAS),)
            and bool(np.isfinite(media).all() and np.isfinite(escala).all() and (escala > 0).all())
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Parámetros de preprocesamiento inválidos: {error!r}") from error
    if not esperados:
        raise ValueError("Parámetros de preprocesamiento inválidos")


def transformar(parametros: dict, df: pd.DataFrame) -> np.ndarray:
    """Matriz de N×21 en el orden de `nombres_salida`, idéntica a la salida de sklearn."""
    df = agregar_saldo_cero(df)
    escaladas = df[parametros["columnas_escaladas"]].to_numpy(dtype=float)
    partes = [
        (escaladas - np.asarray(parametros["media"])) / np.asarray(parametros["escala"]),
        df[parametros["columnas_directas"]].to_numpy(dtype=float),
    ]
    for columna, categorias in parametros["categorias"].items():
        desconocidas = set(df[columna]) - set(categorias)
        if desconocidas:
            raise ValueError(f"{columna}: categorías desconocidas {sorted(map(str, desconocidas))}")
        partes.append(np.column_stack([(df[columna] == categoria).to_numpy(dtype=float) for categoria in categorias]))
    matriz = np.hstack(partes)
    if not np.isfinite(matriz).all():
        raise ValueError("Valores nulos o no finitos en los datos de entrada")
    return matriz


def _nombres_esperados() -> list[str]:
    codificadas = [f"{c}_{v}" for c in config.COLUMNAS_CATEGORICAS for v in config.CATEGORIAS[c]]
    return [*config.COLUMNAS_ESCALADAS, *COLUMNAS_DIRECTAS, *codificadas]
