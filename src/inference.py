"""Canal de inferencia: carga artefactos y predice. Nunca importa código de entrenamiento ni shap."""
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import N_COLUMNAS, transformar, validar_parametros

ARCHIVO_MODELO = "modelo.keras"
ARCHIVO_PREPROCESADOR = "preprocesador.json"
ARCHIVO_CALIBRADOR = "calibrador.json"
ARCHIVO_METRICAS = "metricas.json"
ARCHIVO_SHAP_GLOBAL = "shap_global.json"
ARCHIVO_FONDO_SHAP = "fondo_shap.json"
ARCHIVOS_ARTEFACTOS = (
    ARCHIVO_MODELO,
    ARCHIVO_PREPROCESADOR,
    ARCHIVO_CALIBRADOR,
    ARCHIVO_METRICAS,
    ARCHIVO_SHAP_GLOBAL,
    ARCHIVO_FONDO_SHAP,
)
RUTA_MODELO = config.DIR_MODELOS / ARCHIVO_MODELO
RUTA_METRICAS = config.DIR_MODELOS / ARCHIVO_METRICAS
RUTA_SHAP_GLOBAL = config.DIR_MODELOS / ARCHIVO_SHAP_GLOBAL
EPSILON_LOGIT = 1e-7  # acota p para que el logit sea finito con la salida float32 de la red


class ErrorArtefactos(ValueError):
    """Un artefacto falta o no tiene la forma esperada."""


@dataclass(frozen=True)
class Artefactos:
    modelo: object
    preprocesador: dict
    calibrador: dict
    fondo_shap: dict = field(default_factory=dict)


def artefactos_disponibles(directorio: Path | None = None) -> bool:
    """Indica si están todos los artefactos, sin cargar TensorFlow."""
    directorio = Path(directorio or config.DIR_MODELOS)
    return all((directorio / nombre).is_file() for nombre in ARCHIVOS_ARTEFACTOS)


def cargar_artefactos(directorio: Path | None = None) -> Artefactos:
    """Carga y valida lo necesario para predecir y explicar; lanza ErrorArtefactos si algo falla."""
    directorio = Path(directorio or config.DIR_MODELOS)
    base = cargar_nucleo(directorio)
    fondo = leer_json(directorio / ARCHIVO_FONDO_SHAP)
    try:
        validar_fondo(fondo)
    except ValueError as error:
        raise ErrorArtefactos(str(error)) from error
    return replace(base, fondo_shap=fondo)


def cargar_nucleo(directorio: Path | None = None) -> Artefactos:
    """Solo lo necesario para predecir; lo usan también evaluate y explain, antes de que exista el resto."""
    import keras  # diferido: la app arranca sin TensorFlow mientras no haya artefactos

    directorio = Path(directorio or config.DIR_MODELOS)
    preprocesador = leer_json(directorio / ARCHIVO_PREPROCESADOR)
    calibrador = leer_json(directorio / ARCHIVO_CALIBRADOR)
    try:
        validar_parametros(preprocesador)
        validar_calibrador(calibrador)
        modelo = keras.models.load_model(directorio / ARCHIVO_MODELO, compile=False, safe_mode=True)
    except (OSError, ValueError, TypeError) as error:
        raise ErrorArtefactos(f"Artefacto inválido: {error}") from error
    return Artefactos(modelo=modelo, preprocesador=preprocesador, calibrador=calibrador)


def predecir_fuga(artefactos: Artefactos, clientes_df: pd.DataFrame) -> np.ndarray:
    """Probabilidad de fuga calibrada, una por cliente."""
    return predecir_matriz(artefactos, transformar(artefactos.preprocesador, clientes_df))


def predecir_cruda(artefactos: Artefactos, clientes_df: pd.DataFrame) -> np.ndarray:
    """Salida de la red sin calibrar; solo para evaluar."""
    return _salida_red(artefactos, transformar(artefactos.preprocesador, clientes_df))


def predecir_matriz(artefactos: Artefactos, matriz: np.ndarray) -> np.ndarray:
    """Probabilidad calibrada desde la matriz ya transformada (la que perturba SHAP)."""
    return aplicar_platt(artefactos.calibrador, _salida_red(artefactos, matriz))


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPSILON_LOGIT, 1 - EPSILON_LOGIT)
    return np.log(p) - np.log1p(-p)


def aplicar_platt(calibrador: dict, p) -> np.ndarray:
    """Platt sobre el logit: sigmoide(a·logit(p) + b)."""
    return 1.0 / (1.0 + np.exp(-(calibrador["a"] * logit(p) + calibrador["b"])))


def validar_calibrador(calibrador: dict) -> None:
    try:
        coeficientes = [calibrador["a"], calibrador["b"]]
        validos = all(type(c) in (int, float) and np.isfinite(c) for c in coeficientes) and calibrador["a"] > 0
    except (KeyError, TypeError) as error:
        raise ValueError(f"Calibrador inválido: {error!r}") from error
    if not validos:  # a ≤ 0 anularía o invertiría el orden de riesgo de la red
        raise ValueError("Calibrador inválido: los coeficientes deben ser números finitos y a > 0")


def validar_fondo(fondo: dict) -> None:
    """Fondo SHAP: matriz K×14 con un peso positivo por fila."""
    try:
        datos = np.asarray(fondo["datos"], dtype=float)
        pesos = np.asarray(fondo["pesos"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Fondo SHAP inválido: {error!r}") from error
    forma_valida = datos.ndim == 2 and len(datos) > 0 and datos.shape[1] == N_COLUMNAS and pesos.shape == (len(datos),)
    if not (forma_valida and np.isfinite(datos).all() and np.isfinite(pesos).all() and (pesos > 0).all()):
        raise ValueError("Fondo SHAP inválido: se espera una matriz K×14 con pesos positivos")


def leer_json(ruta: Path) -> dict:
    """Lee un objeto JSON estricto: rechaza NaN e Infinity."""
    try:
        return interpretar_json(Path(ruta).read_bytes())
    except (OSError, ValueError) as error:
        raise ErrorArtefactos(f"{Path(ruta).name}: {error}") from error


def interpretar_json(contenido: bytes) -> dict:
    def _rechazar(constante: str):
        raise ValueError(f"constante no permitida: {constante}")

    try:
        objeto = json.loads(contenido.decode("utf-8"), parse_constant=_rechazar)
    except RecursionError as error:
        raise ValueError("JSON demasiado anidado") from error
    if not isinstance(objeto, dict):
        raise ValueError("se esperaba un objeto JSON")
    return objeto


def guardar_json(ruta: Path, objeto: dict) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(objeto, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _salida_red(artefactos: Artefactos, matriz: np.ndarray) -> np.ndarray:
    salida = artefactos.modelo.predict_on_batch(np.asarray(matriz, dtype="float32"))
    return np.asarray(salida, dtype=float).ravel()
