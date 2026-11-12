import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src import config
from src.config import Incentivo, ParametrosMargen
from src.finance import (
    calcular_margen,
    comparar_retener_a_todos,
    decidir_cartera,
    decidir_cliente,
    resumir_kpis,
    resumir_segmentos,
)

# Los cinco ejemplos de docs/DESIGN.md; test_app.py también los usa.
EJEMPLOS = [
    # p, saldo, productos, tarjeta, margen, ganancia, segmento, incentivo óptimo
    (0.72, 120_000.0, 2, 1, 3090, 601.16, "A", "vip"),
    (0.70, 60_000.0, 1, 1, 1530, 164.20, "B", "bon"),  # bonificación, no exoneración
    (0.81, 0.0, 1, 0, -60, None, "C", None),
    (0.78, 8_000.0, 1, 0, 140, -28.16, "C", "bon"),
    (0.12, 150_000.0, 3, 1, 3900, 43.60, "D", "bon"),
]


def _cartera(*filas):
    """Filas (saldo, productos, tarjeta) como DataFrame."""
    return pd.DataFrame(filas, columns=["Balance", "NumOfProducts", "HasCrCard"])


def _cinco_ejemplos():
    return decidir_cartera(_cartera(*[e[1:4] for e in EJEMPLOS]), [e[0] for e in EJEMPLOS])


@pytest.mark.parametrize("p, saldo, productos, tarjeta, margen, ganancia, segmento, optimo", EJEMPLOS)
def test_ejemplos_del_diseno(p, saldo, productos, tarjeta, margen, ganancia, segmento, optimo):
    r = decidir_cliente(p, saldo, productos, tarjeta)
    assert r["margen"] == pytest.approx(margen)
    if ganancia is None:
        assert math.isnan(r["ganancia"])
    else:
        assert r["ganancia"] == pytest.approx(ganancia)
    assert r["segmento"] == segmento
    assert r["incentivo_optimo"] == optimo
    assert r["actua"] is (segmento in "AB")
    if segmento not in "AB":
        assert r["incentivo"] is None
        assert r["costo"] == 0


def test_margen_escalar_y_vectorial():
    assert calcular_margen(120_000, 2, 1) == pytest.approx(3090)
    assert calcular_margen(np.array([120_000, 0]), np.array([2, 1]), np.array([1, 0])) == pytest.approx([3090, -60])


def test_margen_respeta_horizonte():
    assert calcular_margen(120_000, 2, 1, ParametrosMargen(horizonte=2)) == pytest.approx(6180)


def test_margen_cero_es_deficitario():
    # 2,5 % de 2.400 + 60 − 120 = 0
    r = decidir_cliente(0.90, 2_400, 1, 0)
    assert r["margen"] == pytest.approx(0)
    assert not r["actua"]
    assert math.isnan(r["ganancia"])
    assert r["segmento"] == "C"


def test_ganancia_cero_no_actua():
    # Un solo incentivo sin costo y p·e·M = 0 por efectividad nula: G = 0 exacto.
    catalogo = (Incentivo("bon", "Bonificación", 0.0, 0.0),)
    r = decidir_cliente(0.70, 60_000, 1, 1, catalogo=catalogo)
    assert r["ganancia"] == 0
    assert not r["actua"]
    assert r["segmento"] == "C"


def test_ganancia_apenas_positiva_actua():
    catalogo = (Incentivo("bon", "Bonificación", 0.0, 0.01),)
    r = decidir_cliente(0.70, 60_000, 1, 1, catalogo=catalogo)
    assert r["ganancia"] > 0
    assert r["actua"]
    assert r["segmento"] == "B"


@pytest.mark.parametrize(
    "p, segmento",
    [(0.2999, "D"), (0.30, "V"), (0.6499, "V"), (0.65, "C"), (1.0, "C"), (0.0, "D")],
)
def test_bandas_de_riesgo_sin_intervencion(p, segmento):
    # Saldo 8.000: M = 140 y la bonificación nunca es rentable (G máx. = 28 − 50).
    r = decidir_cliente(p, 8_000, 1, 0)
    assert not r["actua"]
    assert r["segmento"] == segmento


@pytest.mark.parametrize("p, actua", [(0.2999, False), (0.30, True)])
def test_p_min_es_inclusivo(p, actua):
    # Saldo 150.000 con 3 productos y tarjeta: G bonificación > 0 con p ≥ 0,30.
    assert decidir_cliente(p, 150_000, 3, 1)["actua"] is actua


@pytest.mark.parametrize(
    "saldo, optimo_elegibles",
    [
        (24_999.99, {"bon"}),
        (25_000, {"exo", "bon"}),
        (99_999.99, {"exo", "bon"}),
        (100_000, {"vip", "bon"}),
    ],
)
def test_elegibilidad_en_los_limites_de_saldo(saldo, optimo_elegibles):
    # Catálogos con un solo incentivo: solo hay opción si ese incentivo es elegible.
    for incentivo in config.CATALOGO_BASE:
        r = decidir_cliente(0.9, saldo, 2, 1, catalogo=(incentivo,))
        assert (r["incentivo_optimo"] == incentivo.id) == (incentivo.id in optimo_elegibles)


def test_saldo_100000_y_25000_eligen_el_incentivo_del_tramo():
    catalogo_sin_bon = config.CATALOGO_BASE[:2]
    assert decidir_cliente(0.9, 100_000, 2, 1, catalogo=catalogo_sin_bon)["incentivo"] == "vip"
    assert decidir_cliente(0.9, 25_000, 2, 1, catalogo=catalogo_sin_bon)["incentivo"] == "exo"
    assert decidir_cliente(0.9, 24_999.99, 2, 1, catalogo=catalogo_sin_bon)["incentivo_optimo"] is None


@pytest.mark.parametrize("p, segmento", [(0.0, "D"), (0.10, "D"), (0.29, "D"), (0.30, "V"), (0.64, "V"), (0.65, "C"), (0.99, "C")])
@pytest.mark.parametrize("p_min", [0.0, config.P_MIN])
def test_deficitario_nunca_recibe_gasto(p, segmento, p_min):
    # Incentivo gratis y muy efectivo: ni así se gasta en un deficitario.
    catalogo = (Incentivo("bon", "Bonificación", 0.0, 1.0),)
    r = decidir_cliente(p, 0, 1, 0, catalogo=catalogo, p_min=p_min)
    assert r["margen"] < 0
    assert not r["actua"]
    assert r["costo"] == 0
    assert r["incentivo"] is None
    assert r["segmento"] == segmento


def test_p_min_cero_permite_riesgo_bajo_y_no_p_nula():
    assert decidir_cliente(0.12, 150_000, 3, 1, p_min=0.0)["segmento"] == "B"
    r = decidir_cliente(0.0, 150_000, 3, 1, p_min=0.0)  # G = −costo
    assert not r["actua"]
    assert r["segmento"] == "D"


def test_p_min_alto_no_cambia_las_bandas():
    assert decidir_cliente(0.70, 60_000, 1, 1, p_min=0.80)["segmento"] == "C"
    assert decidir_cliente(0.50, 60_000, 1, 1, p_min=0.80)["segmento"] == "V"


def test_el_catalogo_es_un_argumento():
    catalogo = (Incentivo("bon", "Bonificación", 5_000.0, 0.20),)
    assert not decidir_cliente(0.70, 60_000, 1, 1, catalogo=catalogo)["actua"]


def test_cartera_coincide_con_cliente_y_conserva_indice():
    cartera = _cartera(*[e[1:4] for e in EJEMPLOS])
    cartera.index = list("abcde")
    decisiones = decidir_cartera(cartera, [e[0] for e in EJEMPLOS])
    assert list(decisiones.index) == list("abcde")
    for (etiqueta, fila), ejemplo in zip(decisiones.iterrows(), EJEMPLOS):
        unico = decidir_cliente(*ejemplo[:4])
        assert fila["segmento"] == unico["segmento"]
        assert fila["actua"] == unico["actua"]
        assert fila["margen"] == pytest.approx(unico["margen"])


@pytest.mark.parametrize("p", [[-0.1], [1.1], [float("nan")]])
def test_rechaza_p_invalida(p):
    with pytest.raises(ValueError):
        decidir_cartera(_cartera((1_000, 1, 0)), p)


def test_rechaza_longitud_distinta_y_catalogo_vacio():
    with pytest.raises(ValueError):
        decidir_cartera(_cartera((1_000, 1, 0)), [0.1, 0.2])
    with pytest.raises(ValueError):
        decidir_cartera(_cartera((1_000, 1, 0)), [0.1], catalogo=())


def test_kpis_caso_a_mano():
    k = resumir_kpis(_cinco_ejemplos())
    assert k.clientes == 5
    assert k.intervenidos == 2
    # p ≥ 0,65: 120.000 + 60.000 + 0 + 8.000
    assert k.capital_riesgo_critico == pytest.approx(188_000)
    # Solo intervenidos (ejemplos 1 y 2): 120.000 + 60.000
    assert k.capital_intervenido == pytest.approx(180_000)
    # 120.000·0,72·0,45 + 60.000·0,70·0,20
    assert k.capital_rescatable == pytest.approx(38_880 + 8_400)
    assert k.inversion == pytest.approx(450)
    # 0,72·0,45·3.090 + 0,70·0,20·1.530
    assert k.retorno_esperado == pytest.approx(1001.16 + 214.20)
    assert k.roi == pytest.approx((1215.36 - 450) / 450)
    # Ejemplos 3 y 4, sin intervención con saldo bajo: 50 + 50
    assert k.ahorro_salida_permitida == pytest.approx(100)


def test_segmentos_caso_a_mano():
    # Los cinco ejemplos más un deficitario de riesgo intermedio, que cae en V.
    filas = [e[1:4] for e in EJEMPLOS] + [(0.0, 1, 0)]
    decisiones = decidir_cartera(_cartera(*filas), [e[0] for e in EJEMPLOS] + [0.50])
    tabla = resumir_segmentos(decisiones)
    assert tabla.index.tolist() == ["A", "B", "C", "V", "D"]
    assert tabla["clientes"].tolist() == [1, 1, 2, 1, 1]
    assert tabla["capital"].tolist() == pytest.approx([120_000, 60_000, 8_000, 0, 150_000])
    assert tabla["inversion"].tolist() == pytest.approx([400, 50, 0, 0, 0])
    # 0,72·0,45·3.090 y 0,70·0,20·1.530; su neto es el G* de los ejemplos 1 y 2
    assert tabla["retorno_esperado"].tolist() == pytest.approx([1001.16, 214.20, 0, 0, 0])
    assert tabla["ganancia_neta"].tolist() == pytest.approx([601.16, 164.20, 0, 0, 0])
    k = resumir_kpis(decisiones)
    a_b = tabla.loc[["A", "B"]].sum()
    assert a_b["clientes"] == k.intervenidos
    assert a_b["capital"] == pytest.approx(k.capital_intervenido)
    assert a_b["inversion"] == pytest.approx(k.inversion)
    assert a_b["retorno_esperado"] == pytest.approx(k.retorno_esperado)
    assert a_b["ganancia_neta"] == pytest.approx(k.ganancia_neta)
    assert tabla["clientes"].sum() == k.clientes


def test_segmentos_cartera_vacia():
    tabla = resumir_segmentos(decidir_cartera(_cartera(), []))
    assert tabla.index.tolist() == ["A", "B", "C", "V", "D"]
    assert (tabla == 0).all(axis=None)


def test_ahorro_usa_el_incentivo_por_saldo_no_el_optimo():
    # p = 0,65; saldos en cada tramo; margen anulado para que nadie sea intervenido.
    parametros = ParametrosMargen(tasa_saldo=0, por_producto=0, por_tarjeta=0)
    cartera = _cartera((150_000, 1, 0), (30_000, 1, 0), (1_000, 1, 0))
    k = resumir_kpis(decidir_cartera(cartera, [0.65, 0.65, 0.65], parametros))
    assert k.ahorro_salida_permitida == pytest.approx(400 + 150 + 50)


def test_kpis_inversion_cero():
    k = resumir_kpis(decidir_cartera(_cartera((0, 1, 0), (8_000, 1, 0)), [0.81, 0.20]))
    assert k.intervenidos == 0
    assert k.inversion == 0
    assert k.retorno_esperado == 0
    assert k.capital_rescatable == 0
    assert k.roi is None


def test_kpis_cartera_vacia():
    k = resumir_kpis(decidir_cartera(_cartera(), []))
    assert k.clientes == 0
    assert k.roi is None


def test_retorno_por_cliente_solo_si_actua():
    # 0,72·0,45·3.090; 0,70·0,20·1.530; el resto no recibe gasto ni retorno
    assert _cinco_ejemplos()["retorno"].tolist() == pytest.approx([1001.16, 214.20, 0, 0, 0])


def test_efectividad_proyectada_a_mano():
    # (0,72·0,45 + 0,70·0,20) / (0,72 + 0,70) = 0,464 / 1,42; p de no intervenidos no cuenta
    assert resumir_kpis(_cinco_ejemplos()).efectividad_proyectada == pytest.approx(0.464 / 1.42)


def test_efectividad_proyectada_nula_sin_intervenidos():
    assert resumir_kpis(decidir_cartera(_cartera((0, 1, 0)), [0.81])).efectividad_proyectada is None
    assert resumir_kpis(decidir_cartera(_cartera(), [])).efectividad_proyectada is None


def test_ganancia_neta_del_resumen():
    assert resumir_kpis(_cinco_ejemplos()).ganancia_neta == pytest.approx(1215.36 - 450)


def test_retener_a_todos_a_mano():
    selectiva, retener = comparar_retener_a_todos(_cinco_ejemplos())
    assert selectiva.intervenidos == 2
    assert selectiva.inversion == pytest.approx(450)
    assert selectiva.retorno_esperado == pytest.approx(1215.36)
    assert selectiva.ganancia_neta == pytest.approx(765.36)
    assert selectiva.roi == pytest.approx(765.36 / 450)
    # p ≥ 0,65: vip 400, exo 150, bon 50, bon 50 (el ejemplo 5, p = 0,12, queda fuera)
    assert retener.intervenidos == 4
    assert retener.inversion == pytest.approx(650)
    # 0,72·0,45·3.090 + 0,70·0,28·1.530 + 0,81·0,20·(−60) + 0,78·0,20·140
    assert retener.retorno_esperado == pytest.approx(1001.16 + 299.88 - 9.72 + 21.84)
    assert retener.ganancia_neta == pytest.approx(1313.16 - 650)
    assert retener.roi == pytest.approx(663.16 / 650)


def test_retener_a_todos_usa_incentivo_por_saldo_no_el_optimo():
    # Ejemplo 2: la selectiva elige bonificación; retener a todos aplica exoneración por saldo.
    _, retener = comparar_retener_a_todos(decidir_cartera(_cartera((60_000, 1, 1)), [0.70]))
    assert retener.inversion == pytest.approx(150)
    assert retener.retorno_esperado == pytest.approx(0.70 * 0.28 * 1530)


def test_retener_a_todos_incluye_deficitario_con_retorno_negativo():
    _, retener = comparar_retener_a_todos(decidir_cartera(_cartera((0, 1, 0)), [0.81]))
    assert retener.intervenidos == 1
    assert retener.retorno_esperado == pytest.approx(-9.72)
    assert retener.ganancia_neta == pytest.approx(-59.72)
    assert retener.roi == pytest.approx(-59.72 / 50)


def test_retener_a_todos_respeta_el_catalogo_editado():
    catalogo = tuple(replace(i, costo=i.costo * 2, efectividad=i.efectividad / 2) for i in config.CATALOGO_BASE)
    decisiones = decidir_cartera(_cartera((120_000, 2, 1)), [0.72], catalogo=catalogo)
    _, retener = comparar_retener_a_todos(decisiones, catalogo)
    assert retener.inversion == pytest.approx(800)
    assert retener.retorno_esperado == pytest.approx(0.72 * 0.225 * 3090)


def test_retener_a_todos_sin_elegible_no_cuenta_como_intervenido():
    catalogo = config.CATALOGO_BASE[:1]  # solo VIP: saldo 1.000 no tiene incentivo
    decisiones = decidir_cartera(_cartera((1_000, 1, 0)), [0.90], catalogo=catalogo)
    _, retener = comparar_retener_a_todos(decisiones, catalogo)
    assert retener.intervenidos == 0
    assert retener.roi is None


def test_retener_a_todos_sin_criticos_y_cartera_vacia():
    for decisiones in (decidir_cartera(_cartera((120_000, 2, 1)), [0.40]), decidir_cartera(_cartera(), [])):
        _, retener = comparar_retener_a_todos(decisiones)
        assert retener.intervenidos == 0
        assert retener.inversion == 0
        assert retener.roi is None


def test_no_importa_tensorflow():
    codigo = "import sys, src.finance; sys.exit('tensorflow' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", codigo], cwd=config.RAIZ).returncode == 0
