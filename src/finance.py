"""Margen, decisión de retención e indicadores. Solo finanzas: sin dependencia de la red."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src import config
from src.config import Incentivo, ParametrosMargen


@dataclass(frozen=True)
class Escenario:
    """Valor esperado de una estrategia de retención; `roi` es None si la inversión es 0."""

    intervenidos: int
    inversion: float
    retorno_esperado: float
    ganancia_neta: float
    roi: float | None


@dataclass(frozen=True)
class ResumenKPI(Escenario):
    """Escenario de la decisión selectiva con los demás indicadores del conjunto de clientes."""

    clientes: int
    capital_riesgo_critico: float
    capital_intervenido: float
    capital_rescatable: float
    ahorro_salida_permitida: float
    efectividad_proyectada: float | None  # Σ p·e / Σ p de los intervenidos; None si no hay


def calcular_margen(saldo, productos, tarjeta, parametros: ParametrosMargen = config.MARGEN_BASE):
    """Margen esperado del cliente; acepta escalares o arreglos."""
    saldo, productos, tarjeta = (np.asarray(x, dtype=float) for x in (saldo, productos, tarjeta))
    return parametros.horizonte * (
        parametros.tasa_saldo * saldo
        + parametros.por_producto * productos
        + parametros.por_tarjeta * tarjeta
        - parametros.costo_fijo
    )


def decidir_cartera(
    clientes: pd.DataFrame,
    p,
    parametros: ParametrosMargen = config.MARGEN_BASE,
    catalogo: tuple[Incentivo, ...] = config.CATALOGO_BASE,
    p_min: float = config.P_MIN,
) -> pd.DataFrame:
    """Una fila por cliente. `ganancia` (G*) es NaN sin opción e `incentivo_optimo` es el de G*
    aunque no se actúe; `incentivo`, `costo`, `efectividad` y `retorno` solo cuentan si se actúa."""
    if not catalogo:
        raise ValueError("El catálogo está vacío")
    if not 0 <= p_min <= 1:
        raise ValueError("p_min debe estar entre 0 y 1")
    p = np.asarray(p, dtype=float)
    saldo = clientes["Balance"].to_numpy(dtype=float)
    productos = clientes["NumOfProducts"].to_numpy(dtype=float)
    tarjeta = clientes["HasCrCard"].to_numpy(dtype=float)
    if p.shape != saldo.shape:
        raise ValueError("p debe tener un valor por cliente")
    if not ((p >= 0) & (p <= 1)).all():  # un NaN no debe acabar como «Estable»
        raise ValueError("p debe estar entre 0 y 1, sin nulos")
    if not (np.isfinite(saldo) & np.isfinite(productos) & np.isfinite(tarjeta)).all():
        raise ValueError("Saldo, productos y tarjeta no admiten nulos")

    margen = calcular_margen(saldo, productos, tarjeta, parametros)
    costos = np.array([i.costo for i in catalogo])
    efectividades = np.array([i.efectividad for i in catalogo])
    elegibles = _elegibilidad(saldo, catalogo) & (margen > 0)[:, None]  # deficitario: sin opción

    ganancias = np.where(elegibles, (p * margen)[:, None] * efectividades - costos, -np.inf)
    mejor = ganancias.argmax(axis=1)
    filas = np.arange(len(p))
    con_opcion = elegibles.any(axis=1)
    ganancia = np.where(con_opcion, ganancias[filas, mejor], np.nan)
    actua = con_opcion & (ganancia > 0) & (p >= p_min)
    efectividad = np.where(actua, efectividades[mejor], 0.0)

    ids =np.array([i.id for i in catalogo], dtype=object)
    es_prioritario = ids[mejor] == config.ID_INCENTIVO_PRIORITARIO
    segmento = np.select(
        [actua & es_prioritario, actua, p >= config.P_RIESGO_CRITICO, p >= config.P_RIESGO_INTERMEDIO],
        ["A", "B", "C", "V"],
        default="D",
    )
    return pd.DataFrame(
        {
            "p": p,
            "saldo": saldo,
            "margen": margen,
            "ganancia": ganancia,
            "incentivo_optimo": _ids_donde(ids[mejor], con_opcion),
            "actua": actua,
            "segmento": segmento.astype(object),
            "incentivo": _ids_donde(ids[mejor], actua),
            "costo": np.where(actua, costos[mejor], 0.0),
            "efectividad": efectividad,
            "retorno": p * efectividad * margen,
            "costo_por_saldo": _por_saldo(saldo, catalogo, "costo"),
        },
        index=clientes.index,
    )


def decidir_cliente(
    p: float,
    saldo: float,
    productos: int,
    tarjeta: int,
    parametros: ParametrosMargen = config.MARGEN_BASE,
    catalogo: tuple[Incentivo, ...] = config.CATALOGO_BASE,
    p_min: float = config.P_MIN,
) -> dict:
    """Decisión de un cliente, con la misma lógica y claves que `decidir_cartera`."""
    cliente = pd.DataFrame({"Balance": [saldo], "NumOfProducts": [productos], "HasCrCard": [tarjeta]})
    return decidir_cartera(cliente, [p], parametros, catalogo, p_min).to_dict("records")[0]


def resumir_kpis(decisiones: pd.DataFrame) -> ResumenKPI:
    """Indicadores sobre la salida de `decidir_cartera`; llamar por separado para cartera y prueba."""
    p, saldo, efectividad, actua = (decisiones[c] for c in ("p", "saldo", "efectividad", "actua"))
    critico = p >= config.P_RIESGO_CRITICO
    fuga_intervenidos = float(p[actua].sum())
    escenario = _escenario(int(actua.sum()), float(decisiones["costo"].sum()), float(decisiones["retorno"].sum()))
    return ResumenKPI(
        **asdict(escenario),
        clientes=len(decisiones),
        capital_riesgo_critico=float(saldo[critico].sum()),
        capital_intervenido=float(saldo[actua].sum()),
        capital_rescatable=float((saldo * p * efectividad).sum()),
        ahorro_salida_permitida=float(decisiones.loc[critico & ~actua, "costo_por_saldo"].sum()),
        efectividad_proyectada=float((p * efectividad).sum()) / fuga_intervenidos if fuga_intervenidos > 0 else None,
    )


def resumir_segmentos(decisiones: pd.DataFrame) -> pd.DataFrame:
    """Una fila por segmento, siempre los cinco; solo A y B pueden tener inversión y retorno."""
    grupos = decisiones.groupby("segmento")
    tabla = pd.DataFrame(
        {
            "clientes": grupos.size(),
            "capital": grupos["saldo"].sum(),
            "inversion": grupos["costo"].sum(),
            "retorno_esperado": grupos["retorno"].sum(),
        }
    ).reindex(list(config.SEGMENTOS), fill_value=0)
    tabla["clientes"] = tabla["clientes"].astype(int)
    tabla["ganancia_neta"] = tabla["retorno_esperado"] - tabla["inversion"]
    return tabla


def comparar_retener_a_todos(
    decisiones: pd.DataFrame, catalogo: tuple[Incentivo, ...] = config.CATALOGO_BASE
) -> tuple[Escenario, Escenario]:
    """(selectiva, retener a todos) sobre la salida de `decidir_cartera` con el mismo catálogo.

    Retener a todos interviene a cada cliente con p ≥ `config.P_RIESGO_CRITICO`, también
    deficitarios, con el incentivo de su saldo; un margen negativo resta del retorno.
    """
    k = resumir_kpis(decisiones)
    p = decisiones["p"].to_numpy()
    saldo = decisiones["saldo"].to_numpy()
    todos = (p >= config.P_RIESGO_CRITICO) & _elegibilidad(saldo, catalogo).any(axis=1)
    retorno = p * _por_saldo(saldo, catalogo, "efectividad") * decisiones["margen"].to_numpy()
    return (
        _escenario(k.intervenidos, k.inversion, k.retorno_esperado),
        _escenario(int(todos.sum()), float(_por_saldo(saldo, catalogo, "costo")[todos].sum()), float(retorno[todos].sum())),
    )


def _escenario(intervenidos: int, inversion: float, retorno: float) -> Escenario:
    return Escenario(intervenidos, inversion, retorno, retorno - inversion, _roi(retorno, inversion))


def _roi(retorno: float, inversion: float) -> float | None:
    return (retorno - inversion) / inversion if inversion > 0 else None


def _elegibilidad(saldo: np.ndarray, catalogo: tuple[Incentivo, ...]) -> np.ndarray:
    return np.column_stack([(saldo >= i.saldo_min) & (saldo < i.saldo_max) for i in catalogo])


def _por_saldo(saldo: np.ndarray, catalogo: tuple[Incentivo, ...], atributo: str) -> np.ndarray:
    """Atributo del incentivo que corresponde al saldo: el elegible de mayor saldo_min (el más específico)."""
    valores = np.zeros_like(saldo)
    for incentivo in sorted(catalogo, key=lambda i: i.saldo_min):
        valores[(saldo >= incentivo.saldo_min) & (saldo < incentivo.saldo_max)] = getattr(incentivo, atributo)
    return valores


def _ids_donde(ids: np.ndarray, condicion: np.ndarray) -> np.ndarray:
    resultado = np.full(len(ids), None, dtype=object)
    resultado[condicion] = ids[condicion]
    return resultado
