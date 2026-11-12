"""El importador rechaza cada zip malicioso sin copiar nada."""
import io
import json
import stat
import zipfile
from pathlib import Path

import keras
import pytest

from scripts import importar_artefactos as imp
from src.inference import ARCHIVOS_ARTEFACTOS

PNG = imp.MAGIA_PNG + b"resto"


@pytest.fixture(scope="module")
def validos(artefactos_validos) -> dict[str, bytes]:
    return {**artefactos_validos, "figuras/fiabilidad.png": PNG}


def crear_zip(ruta, entradas: dict[str, bytes], extra=()) -> None:
    """`extra`: ZipInfo ya preparados con su contenido, para los casos manipulados."""
    with zipfile.ZipFile(ruta, "w", zipfile.ZIP_DEFLATED) as archivo:
        for nombre, contenido in entradas.items():
            archivo.writestr(nombre, contenido)
        for info, contenido in extra:
            archivo.writestr(info, contenido)


def keras_manipulado(validos, configuracion=None, extra_interno=None) -> bytes:
    """Reescribe el .keras real cambiando su config.json o añadiéndole una entrada."""
    salida = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(validos["modelo.keras"])) as origen, zipfile.ZipFile(salida, "w") as destino:
        for nombre in origen.namelist():
            contenido = origen.read(nombre)
            if nombre == "config.json" and configuracion is not None:
                contenido = json.dumps(configuracion(json.loads(contenido))).encode()
            destino.writestr(nombre, contenido)
        if extra_interno:
            destino.writestr(extra_interno, b"x")
    return salida.getvalue()


def afirmar_rechazo(tmp_path, entradas, extra=(), mensaje=None):
    ruta = tmp_path / "artefactos.zip"
    crear_zip(ruta, entradas, extra)
    modelos, figuras = tmp_path / "models", tmp_path / "figuras"
    with pytest.raises(imp.ArtefactoRechazado, match=mensaje):
        imp.importar(ruta, modelos, figuras)
    assert not modelos.exists() and not figuras.exists(), "no debe copiarse nada"


def test_zip_valido_se_copia_a_su_destino(tmp_path, validos):
    ruta = tmp_path / "artefactos.zip"
    crear_zip(ruta, validos)
    importados = imp.importar(ruta, tmp_path / "models", tmp_path / "figs")
    assert set(importados) == set(validos)
    assert (tmp_path / "models" / "metricas.json").read_bytes() == validos["metricas.json"]
    assert (tmp_path / "figs" / "fiabilidad.png").read_bytes() == PNG
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == sorted(ARCHIVOS_ARTEFACTOS)


def test_fallo_de_escritura_no_mezcla_artefactos(tmp_path, validos, monkeypatch):
    ruta = tmp_path / "artefactos.zip"
    crear_zip(ruta, validos)
    modelos, figuras = tmp_path / "models", tmp_path / "figs"
    imp.importar(ruta, modelos, figuras)

    crear_zip(ruta, {**validos, "metricas.json": b'{"nueva": true}'})
    escribir = Path.write_bytes

    def fallar_en_la_ultima(destino, contenido):
        if destino.name == "fiabilidad.png.tmp":
            raise OSError("disco lleno")
        return escribir(destino, contenido)

    monkeypatch.setattr(Path, "write_bytes", fallar_en_la_ultima)
    with pytest.raises(OSError, match="disco lleno"):
        imp.importar(ruta, modelos, figuras)
    assert (modelos / "metricas.json").read_bytes() == validos["metricas.json"]
    assert not [*modelos.glob("*.tmp"), *figuras.glob("*.tmp")]


@pytest.mark.parametrize(
    "nombre",
    [
        "../evil.json",
        "figuras/../../evil.png",
        "models/../../evil.json",
        "/etc/evil.json",
        "/abs/metricas.json",
        "C:/evil.json",
        "..\\evil.json",
        "\\evil.json",
    ],
)
def test_rutas_inseguras(tmp_path, validos, nombre):
    afirmar_rechazo(tmp_path, {**validos, nombre: b"{}"}, mensaje="no permitido")


@pytest.mark.parametrize(
    "nombre",
    [
        "otro.json",
        "src/train.py",
        "modelo.pkl",
        "modelo.keras.bak",
        "figuras/sub/a.png",
        "figuras/a.jpg",
        "figuras/a.png.exe",
        "METRICAS.JSON",
    ],
)
def test_nombres_fuera_de_la_lista(tmp_path, validos, nombre):
    afirmar_rechazo(tmp_path, {**validos, nombre: PNG}, mensaje="no permitido")


def test_enlace_simbolico(tmp_path, validos):
    resto = {k: v for k, v in validos.items() if k != "metricas.json"}
    info = zipfile.ZipInfo("metricas.json")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    afirmar_rechazo(tmp_path, resto, [(info, b"/etc/passwd")], mensaje="archivo normal")


def test_entrada_de_directorio(tmp_path, validos):
    afirmar_rechazo(tmp_path, {**validos, "figuras/": b""}, mensaje="archivo normal")


def test_entrada_cifrada(tmp_path, validos):
    ruta = tmp_path / "artefactos.zip"
    crear_zip(ruta, validos)
    datos = bytearray(ruta.read_bytes())
    datos[datos.find(b"PK\x01\x02") + 8] |= 0x1  # zipfile no escribe este bit: se marca en el registro central
    ruta.write_bytes(datos)
    with pytest.raises(imp.ArtefactoRechazado, match="encrypted"):
        imp.importar(ruta, tmp_path / "models", tmp_path / "figuras")
    assert not (tmp_path / "models").exists()


def test_entrada_duplicada(tmp_path, validos):
    info = zipfile.ZipInfo("metricas.json")
    with pytest.warns(UserWarning):
        afirmar_rechazo(tmp_path, validos, [(info, b"{}")], mensaje="duplicada")


@pytest.mark.parametrize(
    "nombre, limite", [("metricas.json", imp.LIMITE_JSON), ("figuras/f.png", imp.LIMITE_PNG), ("modelo.keras", imp.LIMITE_MODELO)]
)
def test_entradas_demasiado_grandes(tmp_path, validos, nombre, limite):
    contenido = imp.MAGIA_PNG + b"0" * limite
    afirmar_rechazo(tmp_path, {**validos, nombre: contenido}, mensaje="tamaño máximo")


def test_tamano_declarado_falso_no_engana(tmp_path, validos):
    """Un encabezado que declara menos de lo que hay termina en rechazo, nunca en lectura ilimitada."""
    ruta = tmp_path / "artefactos.zip"
    crear_zip(ruta, validos)
    datos = bytearray(ruta.read_bytes())
    pos = datos.find(b"PK\x01\x02")  # primer registro central: tamaño sin comprimir en el desplazamiento 24
    datos[pos + 24 : pos + 28] = (1).to_bytes(4, "little")
    ruta.write_bytes(datos)
    with pytest.raises(imp.ArtefactoRechazado, match="Zip corrupto"):
        imp.importar(ruta, tmp_path / "models", tmp_path / "figuras")
    assert not (tmp_path / "models").exists()


def test_zip_enorme_en_disco(tmp_path, monkeypatch, validos):
    monkeypatch.setattr(imp, "LIMITE_ZIP", 10)
    afirmar_rechazo(tmp_path, validos, mensaje="tamaño máximo")


def test_demasiadas_entradas(tmp_path, validos):
    figuras = {f"figuras/f{i}.png": PNG for i in range(imp.LIMITE_ENTRADAS)}
    afirmar_rechazo(tmp_path, {**validos, **figuras}, mensaje="Demasiadas")


@pytest.mark.parametrize("nombre", ARCHIVOS_ARTEFACTOS)
def test_artefacto_faltante(tmp_path, validos, nombre):
    afirmar_rechazo(tmp_path, {k: v for k, v in validos.items() if k != nombre}, mensaje="Faltan")


def test_no_es_un_zip(tmp_path):
    ruta = tmp_path / "artefactos.zip"
    ruta.write_bytes(b"esto no es un zip")
    with pytest.raises(imp.ArtefactoRechazado, match="corrupto"):
        imp.importar(ruta, tmp_path / "models", tmp_path / "figuras")


@pytest.mark.parametrize("contenido", [b"GIF89a....", b"", b"\x89PNG"])
def test_png_falso(tmp_path, validos, contenido):
    afirmar_rechazo(tmp_path, {**validos, "figuras/f.png": contenido}, mensaje="no es un PNG")


@pytest.mark.parametrize(
    "contenido",
    [b"{no json", b"[1, 2]", b'{"a": NaN}', b'{"a": Infinity}', b"\xff\xfe", b"", b'{"a":' + b"[" * 200_000],
)
def test_json_invalido(tmp_path, validos, contenido):
    afirmar_rechazo(tmp_path, {**validos, "metricas.json": contenido}, mensaje="metricas.json")


@pytest.mark.parametrize(
    "nombre, contenido",
    [
        ("calibrador.json", b'{"a": 1}'),
        ("calibrador.json", b'{"a": "x", "b": 0}'),
        ("calibrador.json", b'{"a": 0, "b": 0}'),
        ("calibrador.json", b'{"a": -1.5, "b": 0}'),
        ("preprocesador.json", b'{"media": []}'),
        ("fondo_shap.json", b'{"datos": [[1, 2]], "pesos": [1]}'),
    ],
)
def test_json_con_forma_incorrecta(tmp_path, validos, nombre, contenido):
    afirmar_rechazo(tmp_path, {**validos, nombre: contenido}, mensaje=nombre)


def test_keras_con_lambda_real(tmp_path, validos):
    modelo = keras.Sequential([keras.layers.Input((21,)), keras.layers.Lambda(lambda x: x, output_shape=(21,))])
    ruta = tmp_path / "lambda.keras"
    modelo.save(ruta)
    afirmar_rechazo(tmp_path, {**validos, "modelo.keras": ruta.read_bytes()}, mensaje="clase no permitida")


def test_keras_con_lambda_anidada(tmp_path, validos):
    lambda_ = {"module": "keras.layers", "class_name": "Lambda", "config": {}, "registered_name": None}
    interno = {"module": "keras", "class_name": "Sequential", "config": {"layers": [lambda_]}, "registered_name": None}

    def anidar(configuracion):
        configuracion["config"]["layers"].append(interno)
        return configuracion

    modelo = keras_manipulado(validos, anidar)
    afirmar_rechazo(tmp_path, {**validos, "modelo.keras": modelo}, mensaje=r"keras\.layers, Lambda")


@pytest.mark.parametrize(
    "nodo",
    [
        {"module": "keras.config", "class_name": "enable_unsafe_deserialization", "config": {}, "registered_name": None},
        {"module": "keras.utils", "class_name": "get_file", "config": {"origin": "http://x"}, "registered_name": None},
        {"module": "keras.layers", "class_name": "TFSMLayer", "config": {"filepath": "/x"}, "registered_name": None},
        {"module": "keras.layers", "class_name": "StringLookup", "config": {"vocabulary": "/etc/passwd"}, "registered_name": None},
        {"module": None, "class_name": "Lambda", "config": {}, "registered_name": None},
        {"module": None, "class_name": "Dense", "config": {}, "registered_name": None},
        {"module": "keras.layers", "class_name": "Dense", "config": {}, "registered_name": "propio>Dense"},
        {"module": "keras.initializers", "class_name": "Dense", "config": {}, "registered_name": None},
        {"class_name": "Dense", "config": {}},
        {"module": "keras.layers", "config": {}},
    ],
)
def test_keras_rechaza_lo_que_no_esta_en_la_lista(tmp_path, validos, nodo):
    def inyectar(configuracion):  # dentro de un parámetro de una capa permitida, no como capa
        configuracion["config"]["layers"][1]["config"]["kernel_initializer"] = nodo
        return configuracion

    modelo = keras_manipulado(validos, inyectar)
    afirmar_rechazo(tmp_path, {**validos, "modelo.keras": modelo}, mensaje="clase no permitida")


def test_lista_de_clases_coincide_con_un_keras_real(validos):
    def clases(nodo):
        if isinstance(nodo, dict):
            propia = {(nodo["class_name"], nodo["module"])} if "class_name" in nodo else set()
            return propia.union(*map(clases, nodo.values()))
        return set().union(*map(clases, nodo)) if isinstance(nodo, list) else set()

    with zipfile.ZipFile(io.BytesIO(validos["modelo.keras"])) as modelo:
        assert clases(json.loads(modelo.read("config.json"))) == set(imp.CLASES_KERAS.items())


def test_keras_con_configuracion_muy_anidada(tmp_path, validos):
    salida = io.BytesIO()
    with zipfile.ZipFile(salida, "w") as modelo:
        for nombre in imp.INTERNOS_KERAS - {"config.json"}:
            modelo.writestr(nombre, b"x")
        modelo.writestr("config.json", b'{"a":' + b"[" * 500 + b"]" * 500 + b"}")
    afirmar_rechazo(tmp_path, {**validos, "modelo.keras": salida.getvalue()}, mensaje="archivo inválido")


def test_keras_con_clase_ajena(tmp_path, validos):
    def ajena(configuracion):
        configuracion["config"]["layers"][1]["module"] = "os.path"
        return configuracion

    afirmar_rechazo(tmp_path, {**validos, "modelo.keras": keras_manipulado(validos, ajena)}, mensaje="os.path")


def test_keras_con_entrada_interna_extra(tmp_path, validos):
    modelo = keras_manipulado(validos, extra_interno="../evil.py")
    afirmar_rechazo(tmp_path, {**validos, "modelo.keras": modelo}, mensaje="contenido inesperado")


def test_keras_que_no_es_zip(tmp_path, validos):
    afirmar_rechazo(tmp_path, {**validos, "modelo.keras": b"pickle disfrazado"}, mensaje="archivo inválido")


def test_keras_real_pasa_la_inspeccion(validos):
    imp._validar_keras(validos["modelo.keras"])


def test_cli_devuelve_error_sin_copiar(tmp_path, capsys):
    ruta = tmp_path / "nada.zip"
    assert imp.main([str(ruta)]) == 1
    assert "Rechazado" in capsys.readouterr().err
    assert imp.main([]) == 2
