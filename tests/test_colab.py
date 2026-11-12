"""Paquete de Colab y cuaderno: contenido exacto, sin salidas y etapas en orden."""
import json
import signal
import stat
import subprocess
import sys
import types
import zipfile

import pytest

from scripts.empaquetar_colab import empaquetar
from src import config
from src.inference import ARCHIVOS_ARTEFACTOS

RUTA_CUADERNO = config.RAIZ / "notebooks" / "entrenar_colab.ipynb"


@pytest.fixture(scope="module")
def cuaderno() -> dict:
    return json.loads(RUTA_CUADERNO.read_text(encoding="utf-8"))


def fuentes(cuaderno, tipo="code") -> list[str]:
    return ["".join(c["source"]) for c in cuaderno["cells"] if c["cell_type"] == tipo]


def test_paquete_solo_contiene_lo_permitido(tmp_path):
    ruta = empaquetar(tmp_path / "bundle.zip")
    with zipfile.ZipFile(ruta) as bundle:
        nombres = set(bundle.namelist())
    fuente = {f"src/{p.name}" for p in (config.RAIZ / "src").glob("*.py")}
    assert nombres == fuente | {"data/Customer-Churn-Records.csv", "requirements.txt"}
    assert "src/config.py" in nombres and "src/train.py" in nombres


def test_paquete_rechaza_csv_alterado(tmp_path, monkeypatch):
    alterado = tmp_path / "datos.csv"
    alterado.write_bytes(config.RUTA_DATOS.read_bytes() + b"\n")
    monkeypatch.setattr(config, "RUTA_DATOS", alterado)
    with pytest.raises(ValueError, match="SHA-256"):
        empaquetar(tmp_path / "bundle.zip")
    assert not (tmp_path / "bundle.zip").exists()


def test_cuaderno_valido_sin_salidas(cuaderno):
    assert cuaderno["nbformat"] == 4
    for celda in cuaderno["cells"]:
        assert celda["cell_type"] in ("code", "markdown")
        if celda["cell_type"] == "code":
            assert celda["outputs"] == []
            assert celda["execution_count"] is None


def test_cuaderno_pasos_en_orden(cuaderno):
    texto = "\n".join(fuentes(cuaderno))
    pasos = [
        "files.upload()",
        "extraer_paquete(\"colab_bundle.zip\"",
        "hashlib.sha256",
        "--require-hashes",
        "src.train",
        "src.evaluate",
        "src.explain",
        "files.download(\"artefactos.zip\")",
    ]
    posiciones = [texto.index(paso) for paso in pasos]
    assert posiciones == sorted(posiciones)


def test_cuaderno_empaqueta_los_nombres_que_acepta_el_importador(cuaderno):
    texto = "\n".join(fuentes(cuaderno))
    assert all(f'"{nombre}"' in texto for nombre in ARCHIVOS_ARTEFACTOS)


def _subir(cuaderno, monkeypatch, subidos: dict[str, bytes]) -> None:
    """Ejecuta la celda de subida con un files.upload simulado."""
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: subidos)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    [celda] = [f for f in fuentes(cuaderno) if "files.upload()" in f]
    exec(celda, {})


def test_subida_repetida_reemplaza_el_paquete_con_el_nombre_fijo(cuaderno, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "colab_bundle.zip").write_bytes(b"anterior")
    _subir(cuaderno, monkeypatch, {"colab_bundle (1).zip": b"nuevo"})
    assert (tmp_path / "colab_bundle.zip").read_bytes() == b"nuevo"


def test_subida_rechaza_varios_archivos(cuaderno, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="único"):
        _subir(cuaderno, monkeypatch, {"a.zip": b"a", "b.zip": b"b"})
    assert not (tmp_path / "colab_bundle.zip").exists()


def _ejecutor(cuaderno, proyecto):
    """Ejecuta solo la definición de `ejecutar`, sin instalar."""
    [celda] = [f for f in fuentes(cuaderno) if "def ejecutar" in f]
    espacio = {"PROYECTO": proyecto}
    exec(celda[: celda.index('ejecutar("pip"')], espacio)
    return espacio


def test_primera_celda_comprueba_python(cuaderno, monkeypatch, capsys):
    primera = fuentes(cuaderno)[0]
    for version in [(3, 12, 1), (3, 13, 18)]:
        texto = ".".join(map(str, version))
        monkeypatch.setattr(sys, "version_info", version)
        monkeypatch.setattr(sys, "version", f"{texto} (main)")
        exec(primera, {})
        assert capsys.readouterr().out == f"Python {texto}\n"
    for version in [(3, 11, 9), (3, 14, 0)]:
        texto = ".".join(map(str, version))
        monkeypatch.setattr(sys, "version_info", version)
        monkeypatch.setattr(sys, "version", f"{texto} (main)")
        with pytest.raises(SystemExit, match=f"admite Python 3.12 y 3.13.*usa {texto}"):
            exec(primera, {})


def test_etapa_tolera_bytes_invalidos_y_usa_utf8(cuaderno, tmp_path, capsys):
    codigo = "import os, sys\nsys.stdout.buffer.write(b'caf\\xc3\\xa9 \\xff ' + os.environ['PYTHONUTF8'].encode() + b'\\n')\n"
    (tmp_path / "etapa_prueba.py").write_text(codigo, encoding="utf-8")
    _ejecutor(cuaderno, tmp_path)["ejecutar"]("etapa_prueba")
    assert capsys.readouterr().out == "café \ufffd 1\n"


def test_interrumpir_mata_la_etapa(cuaderno, tmp_path, monkeypatch):
    (tmp_path / "etapa_prueba.py").write_text("import time\nprint('x', flush=True)\ntime.sleep(60)\n", encoding="utf-8")
    procesos = []
    popen = subprocess.Popen

    def registrar(*args, **kwargs):
        procesos.append(popen(*args, **kwargs))
        return procesos[-1]

    def interrumpir(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", registrar)
    espacio = _ejecutor(cuaderno, tmp_path)
    espacio["print"] = interrumpir  # llega durante la lectura, como al detener la celda
    with pytest.raises(KeyboardInterrupt):
        espacio["ejecutar"]("etapa_prueba")
    assert procesos[0].wait(timeout=10) == -signal.SIGKILL
    procesos[0].stdout.close()


def test_etapa_muestra_salida_y_error_con_su_entorno(cuaderno, tmp_path, capsys):
    codigo = (
        "import os, sys\n"
        "print('salida', os.environ['KERAS_BACKEND'], os.environ['PYTHONUNBUFFERED'])\n"
        "print('aviso', file=sys.stderr)\n"
    )
    (tmp_path / "etapa_prueba.py").write_text(codigo, encoding="utf-8")
    _ejecutor(cuaderno, tmp_path)["ejecutar"]("etapa_prueba")
    assert capsys.readouterr().out.splitlines() == ["salida tensorflow 1", "aviso"]


def test_etapa_que_falla_detiene_el_cuaderno(cuaderno, tmp_path, capsys):
    (tmp_path / "etapa_prueba.py").write_text("print('antes'); raise SystemExit(3)\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="etapa_prueba .código 3"):
        _ejecutor(cuaderno, tmp_path)["ejecutar"]("etapa_prueba")
    assert "antes" in capsys.readouterr().out


def _extractor(cuaderno):
    """Ejecuta solo la definición de la celda de descompresión, sin llamar a Colab."""
    [celda] = [f for f in fuentes(cuaderno) if "def extraer_paquete" in f]
    definicion = celda[: celda.index('extraer_paquete("colab_bundle.zip"')]
    espacio = {}
    exec(definicion, espacio)  # código propio del proyecto, bajo control de versiones
    return espacio["extraer_paquete"]


def test_descompresion_acepta_el_paquete_real(cuaderno, tmp_path):
    ruta = empaquetar(tmp_path / "bundle.zip")
    _extractor(cuaderno)(ruta, tmp_path / "proyecto")
    assert (tmp_path / "proyecto" / "src" / "train.py").is_file()
    assert (tmp_path / "proyecto" / "data" / "Customer-Churn-Records.csv").is_file()


@pytest.mark.parametrize("nombre", ["../evil.py", "/tmp/evil.py", "src/../../evil.py", "otro.txt", "src/sub/a.py"])
def test_descompresion_rechaza_rutas(cuaderno, tmp_path, nombre):
    ruta = tmp_path / "bundle.zip"
    with zipfile.ZipFile(ruta, "w") as bundle:
        bundle.writestr("requirements.txt", b"x")
        bundle.writestr(nombre, b"x")
    with pytest.raises(ValueError):
        _extractor(cuaderno)(ruta, tmp_path / "proyecto")
    assert not (tmp_path / "proyecto").exists()


def test_descompresion_rechaza_enlaces(cuaderno, tmp_path):
    ruta = tmp_path / "bundle.zip"
    info = zipfile.ZipInfo("requirements.txt")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(ruta, "w") as bundle:
        bundle.writestr(info, b"/etc/passwd")
    with pytest.raises(ValueError):
        _extractor(cuaderno)(ruta, tmp_path / "proyecto")
    assert not (tmp_path / "proyecto").exists()
