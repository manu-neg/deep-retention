"""Constantes y supuestos editables del proyecto. Ver docs/DESIGN.md."""
from dataclasses import dataclass
from math import inf
from pathlib import Path

# Rutas (relativas a la raíz del proyecto)
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "Customer-Churn-Records.csv"
DIR_MODELOS = RAIZ / "models"
DIR_REPORTES = RAIZ / "reports"
DIR_FIGURAS = DIR_REPORTES / "figuras"
RUTA_HISTORIAL = DIR_REPORTES / "historial_entrenamiento.json"  # intermedio para evaluate; no es artefacto

SHA256_DATOS = "2abdc051d9170540777bf35652ccb02a6ce9c9645892b078762cb729039e44d2"

# Presentación
ETIQUETAS_MODELOS = {"red_cruda": "Red cruda", "red_calibrada": "Red calibrada", "linea_base": "Línea base"}
COLORES_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # paleta categórica validada (CVD ≥ 8)

# Partición
SEMILLA = 42
FRACCION_VALIDACION = 0.15
FRACCION_PRUEBA = 0.15

# Columnas
COLUMNA_ID = "cliente_id"
OBJETIVO = "Exited"
COLUMNA_FILA = "RowNumber"  # da el cliente_id
# Nombres del CSV con espacios, renombrados al cargar
RENOMBRES_CSV = {"Satisfaction Score": "SatisfactionScore", "Card Type": "CardType", "Point Earned": "PointEarned"}
# Descartadas al cargar: identificadores personales. `Complain` se usa por decisión del dueño aunque coincide
# con Exited en el 99,86 % de las filas; el informe lo declara como limitación. Ver docs/DESIGN.md.
COLUMNAS_DESCARTADAS = ["CustomerId", "Surname"]
COLUMNA_SALDO_CERO = "saldo_cero"  # derivada; la crea el preprocesamiento
COLUMNAS_ESCALADAS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "SatisfactionScore", "PointEarned",
]
COLUMNAS_BINARIAS = ["HasCrCard", "IsActiveMember", "Complain"]
COLUMNAS_CATEGORICAS = ["Geography", "Gender", "CardType"]
COLUMNAS_ENTRADA = COLUMNAS_ESCALADAS + COLUMNAS_BINARIAS + COLUMNAS_CATEGORICAS

# Validación de datos
RANGOS_ENTEROS = {
    "CreditScore": (350, 850),
    "Age": (18, 92),
    "Tenure": (0, 10),
    "NumOfProducts": (1, 4),
    "HasCrCard": (0, 1),
    "IsActiveMember": (0, 1),
    "Complain": (0, 1),
    "SatisfactionScore": (1, 5),
    "PointEarned": (0, 1000),
    OBJETIVO: (0, 1),
}
COLUMNAS_NO_NEGATIVAS = ["Balance", "EstimatedSalary"]
CATEGORIAS = {
    "Geography": ("France", "Germany", "Spain"),
    "Gender": ("Female", "Male"),
    "CardType": ("SILVER", "GOLD", "PLATINUM", "DIAMOND"),
}

# Bandas de riesgo (fijas) y umbral de actuación (editable)
P_RIESGO_INTERMEDIO = 0.30
P_RIESGO_CRITICO = 0.65
P_MIN = 0.30

# Segmentos
SEGMENTOS = {
    "A": "Retención prioritaria",
    "B": "Retención automatizada",
    "C": "Salida Permitida",
    "V": "Vigilancia",
    "D": "Estable",
}
ID_INCENTIVO_PRIORITARIO = "vip"  # actuar con este incentivo es segmento A


@dataclass(frozen=True)
class ParametrosMargen:
    """M = horizonte × (tasa_saldo·saldo + por_producto·productos + por_tarjeta·tarjeta − costo_fijo)."""

    horizonte: float = 1.0
    tasa_saldo: float = 0.025
    por_producto: float = 60.0
    por_tarjeta: float = 90.0
    costo_fijo: float = 120.0


@dataclass(frozen=True)
class Incentivo:
    """Elegible si saldo_min ≤ saldo < saldo_max."""

    id: str
    nombre: str
    costo: float
    efectividad: float
    saldo_min: float = 0.0
    saldo_max: float = inf

    def __post_init__(self):
        if self.costo < 0 or not 0 <= self.efectividad <= 1 or self.saldo_min >= self.saldo_max:
            raise ValueError(f"Incentivo inválido: {self}")


MARGEN_BASE = ParametrosMargen()
CATALOGO_BASE = (
    Incentivo("vip", "Gestor VIP + tarifa plana", 400.0, 0.45, saldo_min=100_000.0),
    Incentivo("exo", "Exoneración de comisiones (12 meses)", 150.0, 0.28, saldo_min=25_000.0, saldo_max=100_000.0),
    Incentivo("bon", "Bonificación transaccional", 50.0, 0.20),
)
