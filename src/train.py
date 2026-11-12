"""Entrena la red y escribe sus artefactos. Se ejecuta solo en Colab: python -m src.train."""
import keras
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from src import config
from src.data import Particion, cargar_datos, particionar
from src.inference import ARCHIVO_CALIBRADOR, ARCHIVO_MODELO, ARCHIVO_PREPROCESADOR, guardar_json, logit
from src.model import construir_mlp
from src.preprocessing import ajustar_preprocesador, exportar_parametros, transformar

TAM_LOTE = 256
MAX_EPOCAS = 200
PACIENCIA_PARADA = 15
PACIENCIA_TASA = 5
FACTOR_TASA = 0.5


def pesos_clase(y) -> dict[int, float]:
    """N/(2·N_k), calculados con la partición que se reciba (entrenamiento)."""
    y = np.asarray(y)
    conteos = {k: int((y == k).sum()) for k in (0, 1)}
    if min(conteos.values()) == 0:
        raise ValueError("Se necesitan ambas clases para calcular los pesos")
    return {k: len(y) / (2 * n) for k, n in conteos.items()}


def crear_lotes(x, y) -> tf.data.Dataset:
    """Lotes (x, y, peso de clase) que se rebarajan en cada época."""
    y = np.asarray(y, dtype="float32")
    pesos = pesos_clase(y)
    w = np.where(y == 1, pesos[1], pesos[0]).astype("float32")
    return (
        tf.data.Dataset.from_tensor_slices((np.asarray(x, dtype="float32"), y, w))
        .shuffle(len(y), seed=config.SEMILLA, reshuffle_each_iteration=True)
        .batch(TAM_LOTE)
    )


def crear_callbacks() -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA_PARADA, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=FACTOR_TASA, patience=PACIENCIA_TASA),
    ]


def ajustar_platt(p_validacion, y_validacion) -> dict[str, float]:
    """Coeficientes de Platt sobre el logit, ajustados con la partición de validación."""
    # C infinito: sin penalización, Platt puro.
    calibrador = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(p_validacion).reshape(-1, 1), np.asarray(y_validacion))
    return {"a": float(calibrador.coef_[0, 0]), "b": float(calibrador.intercept_[0])}


def entrenar(particion: Particion) -> None:
    preprocesador = exportar_parametros(ajustar_preprocesador(particion.entrenamiento))
    x_entrenamiento = transformar(preprocesador, particion.entrenamiento)
    x_validacion = transformar(preprocesador, particion.validacion)
    y_entrenamiento = particion.entrenamiento[config.OBJETIVO].to_numpy(dtype="float32")
    y_validacion = particion.validacion[config.OBJETIVO].to_numpy(dtype="float32")

    modelo = construir_mlp()
    historia = modelo.fit(
        crear_lotes(x_entrenamiento, y_entrenamiento),
        validation_data=(x_validacion, y_validacion),
        validation_batch_size=TAM_LOTE,
        epochs=MAX_EPOCAS,
        callbacks=crear_callbacks(),
        verbose=2,
    )
    p_validacion = modelo.predict(x_validacion, batch_size=TAM_LOTE, verbose=0).ravel()
    calibrador = ajustar_platt(p_validacion, y_validacion)

    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    modelo.save(config.DIR_MODELOS / ARCHIVO_MODELO)
    guardar_json(config.DIR_MODELOS / ARCHIVO_PREPROCESADOR, preprocesador)
    guardar_json(config.DIR_MODELOS / ARCHIVO_CALIBRADOR, calibrador)
    guardar_json(config.RUTA_HISTORIAL, {clave: [float(v) for v in valores] for clave, valores in historia.history.items()})
    print(f"Entrenamiento terminado en {len(historia.history['loss'])} épocas. Calibrador: {calibrador}")


def main() -> None:
    keras.utils.set_random_seed(config.SEMILLA)  # Python, NumPy y TensorFlow
    tf.config.experimental.enable_op_determinism()
    entrenar(particionar(cargar_datos()))


if __name__ == "__main__":
    main()
