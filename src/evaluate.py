"""Evalúa en prueba y escribe metricas.json y figuras. Se ejecuta solo en Colab: python -m src.evaluate."""
import platform
from dataclasses import asdict
from importlib import metadata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src import config
from src.data import cargar_datos, particionar
from src.finance import comparar_retener_a_todos, decidir_cartera, resumir_kpis, resumir_segmentos
from src.inference import ARCHIVO_METRICAS, cargar_nucleo, guardar_json, leer_json, predecir_cruda, predecir_fuga
from src.preprocessing import transformar

N_BINS = 10
FIGURA_FIABILIDAD = "fiabilidad.png"
FIGURA_HISTORIAL = "historial_entrenamiento.png"
LIBRERIAS = ("tensorflow", "keras", "numpy", "pandas", "scikit-learn", "scipy", "shap", "matplotlib")
NOTA_METRICAS = (
    "La validación se reutilizó para early stopping y calibración. "
    "La cartera completa incluye clientes de entrenamiento."
)
NOTA_HISTORIAL = "La pérdida de entrenamiento lleva pesos de clase y dropout; la de validación, no."
MARCADORES = ("o", "s")  # cada serie lleva además su color y su leyenda
TINTA = "#52514e"
SUPERFICIE = "#fcfcfb"


def tabla_fiabilidad(y, p) -> list[dict]:
    """Bins uniformes no vacíos: [0, 0,1), …, [0,9, 1]."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    indices = np.clip((p * N_BINS).astype(int), 0, N_BINS - 1)
    return [
        {
            "bin": int(b),
            "n": int((indices == b).sum()),
            "confianza": float(p[indices == b].mean()),
            "tasa_observada": float(y[indices == b].mean()),
        }
        for b in np.unique(indices)
    ]


def calcular_ece(y, p) -> float:
    """Error de calibración esperado: Σ (n_b/N)·|tasa_b − confianza_b|."""
    return float(
        sum(fila["n"] * abs(fila["tasa_observada"] - fila["confianza"]) for fila in tabla_fiabilidad(y, p))
        / len(np.asarray(y))
    )


def metricas_probabilidad(y, p) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": calcular_ece(y, p),
    }


def comparar_modelos(prueba: dict) -> dict:
    """Línea base frente a la red calibrada en AUC; la calibración solo se compara con la red cruda."""
    return {
        "auc": {m: {k: prueba[m][k] for k in ("roc_auc", "pr_auc")} for m in ("linea_base", "red_calibrada")},
        "calibracion": {m: {k: prueba[m][k] for k in ("brier", "log_loss", "ece")} for m in ("red_cruda", "red_calibrada")},
    }


def auc_por_grupo(df: pd.DataFrame, p) -> dict:
    """AUC por categoría; None si el grupo no tiene ambas clases."""
    y, p = df[config.OBJETIVO].to_numpy(), np.asarray(p)
    resultado = {}
    for columna in ("Gender", "Geography"):
        resultado[columna] = {}
        for categoria in config.CATEGORIAS[columna]:
            grupo = (df[columna] == categoria).to_numpy()
            ambas = len(np.unique(y[grupo])) == 2
            resultado[columna][categoria] = {
                "n": int(grupo.sum()),
                "tasa_fuga": float(y[grupo].mean()) if grupo.any() else None,
                "auc": float(roc_auc_score(y[grupo], p[grupo])) if ambas else None,
            }
    return resultado


def kpis_financieros(df: pd.DataFrame, p) -> dict:
    decisiones = decidir_cartera(df, p)
    kpi = resumir_kpis(decisiones)
    selectiva, retener = comparar_retener_a_todos(decisiones)
    return {
        **asdict(kpi),
        "por_segmento": resumir_segmentos(decisiones).to_dict("index"),
        "comparacion": {"selectiva": asdict(selectiva), "retener_a_todos": asdict(retener)},
    }


def mejor_epoca(historial: dict[str, list[float]]) -> int:
    """Época (desde 1) de menor val_loss: la de los pesos que restaura EarlyStopping."""
    return int(np.nanargmin(historial["val_loss"])) + 1


def _ejes():
    figura, ejes = plt.subplots(figsize=(6.5, 4.5), facecolor=SUPERFICIE, layout="constrained")
    ejes.set_facecolor(SUPERFICIE)
    ejes.spines[["top", "right"]].set_visible(False)
    ejes.spines[["left", "bottom"]].set_color(TINTA)
    ejes.tick_params(colors=TINTA)
    ejes.grid(color="#e8e7e3", linewidth=0.8)
    ejes.set_axisbelow(True)
    return figura, ejes


def _guardar(figura, ejes, titulo: str, ruta) -> None:
    ejes.set_title(titulo, loc="left", color=TINTA)
    ejes.legend(frameon=False)
    figura.savefig(ruta, dpi=150)
    plt.close(figura)


def figura_fiabilidad(y, probabilidades: dict[str, np.ndarray], ruta) -> None:
    figura, ejes = _ejes()
    ejes.plot([0, 1], [0, 1], color=TINTA, linewidth=1, linestyle="--", label="Calibración perfecta")
    for (nombre, p), color, marcador in zip(probabilidades.items(), config.COLORES_SERIES, MARCADORES):
        filas = tabla_fiabilidad(y, p)
        ejes.plot(
            [f["confianza"] for f in filas],
            [f["tasa_observada"] for f in filas],
            color=color,
            marker=marcador,
            markersize=6,
            linewidth=2,
            label=f"{nombre} (ECE {calcular_ece(y, p):.3f})",
        )
    ejes.set(xlim=(0, 1), ylim=(0, 1), xlabel="Probabilidad predicha", ylabel="Tasa de fuga observada")
    _guardar(figura, ejes, "Fiabilidad en el conjunto de prueba", ruta)


def figura_historial(historial: dict[str, list[float]], ruta) -> None:
    figura, ejes = _ejes()
    series = (("loss", "Entrenamiento"), ("val_loss", "Validación"))
    for (clave, nombre), color in zip(series, config.COLORES_SERIES):
        ejes.plot(range(1, len(historial[clave]) + 1), historial[clave], color=color, linewidth=2, label=nombre)
    ejes.set(xlabel="Época", ylabel="Entropía cruzada binaria")
    figura.supxlabel(NOTA_HISTORIAL, fontsize=8, color=TINTA)
    _guardar(figura, ejes, "Historial de entrenamiento", ruta)


def main() -> None:
    artefactos = cargar_nucleo()
    cartera = cargar_datos()
    particion = particionar(cartera)
    prueba = particion.prueba
    y_prueba = prueba[config.OBJETIVO].to_numpy()

    x_entrenamiento = transformar(artefactos.preprocesador, particion.entrenamiento)
    x_prueba = transformar(artefactos.preprocesador, prueba)
    y_entrenamiento = particion.entrenamiento[config.OBJETIVO].to_numpy()
    linea_base = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.SEMILLA)
    probabilidades = {
        "red_cruda": predecir_cruda(artefactos, prueba),
        "red_calibrada": predecir_fuga(artefactos, prueba),
        # Solo para comparar: se ajusta con entrenamiento y no deja artefacto.
        "linea_base": linea_base.fit(x_entrenamiento, y_entrenamiento).predict_proba(x_prueba)[:, 1],
    }
    p_prueba = probabilidades["red_calibrada"]
    historial = leer_json(config.RUTA_HISTORIAL)

    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_fiabilidad(
        y_prueba,
        {config.ETIQUETAS_MODELOS[m]: probabilidades[m] for m in ("red_cruda", "red_calibrada")},
        config.DIR_FIGURAS / FIGURA_FIABILIDAD,
    )
    figura_historial(historial, config.DIR_FIGURAS / FIGURA_HISTORIAL)

    metricas_prueba = {m: metricas_probabilidad(y_prueba, p) for m, p in probabilidades.items()}
    guardar_json(
        config.DIR_MODELOS / ARCHIVO_METRICAS,
        {
            "nota": NOTA_METRICAS,
            "prueba": metricas_prueba,
            "comparacion_modelos": comparar_modelos(metricas_prueba),
            "fiabilidad_red_calibrada": tabla_fiabilidad(y_prueba, p_prueba),
            "auc_por_grupo": auc_por_grupo(prueba, p_prueba),
            "kpi": {
                "cartera_completa": kpis_financieros(cartera, predecir_fuga(artefactos, cartera)),
                "prueba": kpis_financieros(prueba, p_prueba),
            },
            "calibrador": artefactos.calibrador,
            "particion": {nombre: len(parte) for nombre, parte in particion._asdict().items()},
            "epocas_entrenadas": len(historial["loss"]),
            "mejor_epoca": mejor_epoca(historial),
            "historial": historial,
            "semilla": config.SEMILLA,
            "versiones": {"python": platform.python_version(), **{n: metadata.version(n) for n in LIBRERIAS}},
        },
    )
    print("Evaluación terminada.")


if __name__ == "__main__":
    main()
