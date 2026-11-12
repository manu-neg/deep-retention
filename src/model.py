"""Arquitectura de la red. Ver docs/DESIGN.md."""
import keras

from src.preprocessing import N_COLUMNAS

TASA_APRENDIZAJE = 1e-3


def construir_mlp(n_entradas: int = N_COLUMNAS) -> keras.Model:
    """MLP compilado, sin entrenar."""
    modelo = keras.Sequential(
        [
            keras.layers.Input(shape=(n_entradas,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="deep_retention",
    )
    modelo.compile(optimizer=keras.optimizers.Adam(TASA_APRENDIZAJE), loss="binary_crossentropy")
    return modelo
