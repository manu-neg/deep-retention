"""SHAP global (solo en Colab: python -m src.explain) y explicación individual en vivo para la app."""
import numpy as np
import pandas as pd

from src import config
from src.inference import ARCHIVO_FONDO_SHAP, ARCHIVO_SHAP_GLOBAL, Artefactos, guardar_json, predecir_matriz
from src.preprocessing import agregar_saldo_cero, transformar

K_FONDO_GLOBAL = 100
K_FONDO_VIVO = 25  # el SHAP en vivo usa un fondo reducido para responder en segundos
N_CLIENTES_BEESWARM = 500
MUESTRAS_GLOBAL = 1000
MUESTRAS_VIVO = 500
N_FACTORES = 3
FIGURA_BEESWARM = "shap_beeswarm.png"


def valores_shap(artefactos: Artefactos, fondo, matriz: np.ndarray, muestras: int) -> np.ndarray:
    """SHAP de la predicción calibrada, N×14. `fondo` es una matriz o un DenseData de shap."""
    import shap  # diferido: la app no carga shap al arrancar

    explicador = shap.KernelExplainer(lambda x: predecir_matriz(artefactos, x), fondo)
    # l1_reg=False: usa todas las variables en vez de preseleccionarlas con LARS
    valores = explicador.shap_values(matriz, nsamples=muestras, l1_reg=False, silent=True)
    return np.asarray(valores, dtype=float).reshape(len(matriz), -1)


def explicar_cliente(artefactos: Artefactos, cliente_df: pd.DataFrame) -> list[dict]:
    """Contribuciones a p de un cliente, de mayor a menor |contribución|; `valor` en unidades originales."""
    from shap.utils._legacy import DenseData  # conserva los pesos del k-means guardado

    if len(cliente_df) != 1:
        raise ValueError("Se explica un cliente por vez")
    nombres = artefactos.preprocesador["nombres_salida"]
    matriz = transformar(artefactos.preprocesador, cliente_df)
    fondo = DenseData(
        np.asarray(artefactos.fondo_shap["datos"], dtype=float),
        nombres,
        None,
        np.asarray(artefactos.fondo_shap["pesos"], dtype=float),
    )
    contribuciones = valores_shap(artefactos, fondo, matriz, MUESTRAS_VIVO)[0]
    originales = agregar_saldo_cero(cliente_df)
    filas = [
        {
            "variable": nombre,
            "valor": float(originales[nombre].iloc[0]) if nombre in originales else float(matriz[0, i]),
            "contribucion": float(contribuciones[i]),
        }
        for i, nombre in enumerate(nombres)
    ]
    return sorted(filas, key=lambda fila: abs(fila["contribucion"]), reverse=True)


def _variable(nombre: str) -> str:
    """Variable original de una columna: Geography_France → Geography."""
    return next((c for c in config.COLUMNAS_CATEGORICAS if nombre.startswith(f"{c}_")), nombre)


def resumen_global(valores: np.ndarray, nombres: list[str]) -> dict:
    """Importancia global por variable, ordenada: suma con signo de sus columnas por cliente y luego media de |SHAP|."""
    originales = [_variable(nombre) for nombre in nombres]
    variables = list(dict.fromkeys(originales))
    por_variable = np.column_stack([valores[:, [o == v for o in originales]].sum(axis=1) for v in variables])
    importancia = np.abs(por_variable).mean(axis=0)
    orden = np.argsort(-importancia)
    return {
        "variables": [variables[i] for i in orden],
        "importancia_media_abs": [float(importancia[i]) for i in orden],
        "factores_principales": [variables[i] for i in orden[:N_FACTORES]],
        "n_clientes": int(len(valores)),
    }


def figura_beeswarm(valores: np.ndarray, matriz: np.ndarray, nombres: list[str], ruta) -> None:
    import matplotlib.pyplot as plt
    import shap

    shap.summary_plot(valores, matriz, feature_names=nombres, show=False)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close("all")


def main() -> None:
    import shap

    from src.data import cargar_datos, particionar
    from src.inference import cargar_nucleo

    np.random.seed(config.SEMILLA)  # KernelExplainer muestrea con el generador global
    artefactos = cargar_nucleo()
    nombres = artefactos.preprocesador["nombres_salida"]
    particion = particionar(cargar_datos())
    x_entrenamiento = transformar(artefactos.preprocesador, particion.entrenamiento)
    x_prueba = transformar(artefactos.preprocesador, particion.prueba)

    # Los fondos resumen entrenamiento; los clientes explicados salen de prueba.
    fondo_global = shap.kmeans(x_entrenamiento, K_FONDO_GLOBAL)
    fondo_vivo = shap.kmeans(x_entrenamiento, K_FONDO_VIVO)
    generador = np.random.default_rng(config.SEMILLA)
    indices = generador.choice(len(x_prueba), size=min(N_CLIENTES_BEESWARM, len(x_prueba)), replace=False)
    muestra = x_prueba[indices]
    valores = valores_shap(artefactos, fondo_global, muestra, MUESTRAS_GLOBAL)

    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_beeswarm(valores, muestra, nombres, config.DIR_FIGURAS / FIGURA_BEESWARM)
    guardar_json(config.DIR_MODELOS / ARCHIVO_SHAP_GLOBAL, {**resumen_global(valores, nombres), "fondo_k": K_FONDO_GLOBAL})
    guardar_json(
        config.DIR_MODELOS / ARCHIVO_FONDO_SHAP,
        {"datos": np.asarray(fondo_vivo.data).tolist(), "pesos": np.asarray(fondo_vivo.weights).tolist()},
    )
    print("SHAP terminado.")


if __name__ == "__main__":
    main()
