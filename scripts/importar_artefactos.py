"""Importa artefactos.zip, que se trata como no confiable. Nada se copia si algo falla."""
import io
import os
import re
import stat
import sys
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from src import config
from src.inference import (
    ARCHIVO_CALIBRADOR,
    ARCHIVO_FONDO_SHAP,
    ARCHIVO_MODELO,
    ARCHIVO_PREPROCESADOR,
    ARCHIVOS_ARTEFACTOS,
    interpretar_json,
    validar_calibrador,
    validar_fondo,
)
from src.preprocessing import validar_parametros

MB = 1024 * 1024
LIMITE_ZIP = 100 * MB
LIMITE_ENTRADAS = 30
LIMITE_MODELO = 20 * MB
LIMITE_JSON = 5 * MB
LIMITE_PNG = 10 * MB
LIMITE_INTERNO_KERAS = 50 * MB
PROFUNDIDAD_MAXIMA_KERAS = 64
PATRON_FIGURA = re.compile(r"figuras/[A-Za-z0-9_-]+\.png")
MAGIA_PNG = b"\x89PNG\r\n\x1a\n"
INTERNOS_KERAS = {"config.json", "metadata.json", "model.weights.h5"}
# Clase → módulo exactos de la configuración de construir_mlp; comprobado contra un .keras real en los tests.
CLASES_KERAS = {
    "Sequential": "keras",
    "DTypePolicy": "keras",
    "InputLayer": "keras.layers",
    "Dense": "keras.layers",
    "BatchNormalization": "keras.layers",
    "Dropout": "keras.layers",
    "GlorotUniform": "keras.initializers",
    "Zeros": "keras.initializers",
    "Ones": "keras.initializers",
    "Adam": "keras.optimizers",
}
VALIDADORES_JSON = {
    ARCHIVO_PREPROCESADOR: validar_parametros,
    ARCHIVO_CALIBRADOR: validar_calibrador,
    ARCHIVO_FONDO_SHAP: validar_fondo,
}


class ArtefactoRechazado(ValueError):
    """El zip no pasó alguna comprobación de seguridad."""


def importar(ruta_zip: Path, dir_modelos: Path | None = None, dir_figuras: Path | None = None) -> list[str]:
    """Valida todo en memoria y, solo si todo pasa, copia. Devuelve los nombres importados."""
    contenidos = leer_validado(ruta_zip)
    destinos = {
        nombre: Path(dir_figuras or config.DIR_FIGURAS) / PurePosixPath(nombre).name
        if nombre.startswith("figuras/")
        else Path(dir_modelos or config.DIR_MODELOS) / nombre
        for nombre in contenidos
    }
    temporales = {nombre: destino.with_name(destino.name + ".tmp") for nombre, destino in destinos.items()}
    try:
        # Primero todos los temporales: un fallo de escritura no deja artefactos mezclados.
        for nombre, temporal in temporales.items():
            temporal.parent.mkdir(parents=True, exist_ok=True)
            temporal.write_bytes(contenidos[nombre])
        for nombre, temporal in temporales.items():
            os.replace(temporal, destinos[nombre])  # reemplaza la entrada de directorio, sin seguir enlaces
    finally:
        for temporal in temporales.values():
            temporal.unlink(missing_ok=True)
    return sorted(contenidos)


def leer_validado(ruta_zip: Path) -> dict[str, bytes]:
    ruta_zip = Path(ruta_zip)
    if not ruta_zip.is_file() or ruta_zip.stat().st_size > LIMITE_ZIP:
        raise ArtefactoRechazado(f"{ruta_zip.name}: no existe o supera el tamaño máximo")
    try:
        with zipfile.ZipFile(ruta_zip) as archivo:
            contenidos = _leer_entradas(archivo)
    except (zipfile.BadZipFile, EOFError, zlib.error, NotImplementedError, RuntimeError, OSError) as error:
        raise ArtefactoRechazado(f"Zip corrupto: {error}") from error

    faltantes = set(ARCHIVOS_ARTEFACTOS) - set(contenidos)
    if faltantes:
        raise ArtefactoRechazado(f"Faltan artefactos: {sorted(faltantes)}")
    for nombre, contenido in contenidos.items():
        _validar_contenido(nombre, contenido)
    return contenidos


def _leer_entradas(archivo: zipfile.ZipFile) -> dict[str, bytes]:
    entradas = archivo.infolist()
    if len(entradas) > LIMITE_ENTRADAS:
        raise ArtefactoRechazado(f"Demasiadas entradas: {len(entradas)}")
    contenidos = {}
    for entrada in entradas:
        limite = _comprobar_entrada(entrada)
        if entrada.filename in contenidos:
            raise ArtefactoRechazado(f"Entrada duplicada: {entrada.filename!r}")
        with archivo.open(entrada) as origen:
            contenido = origen.read(limite + 1)  # no se confía en el tamaño declarado
        if len(contenido) > limite:
            raise ArtefactoRechazado(f"{entrada.filename}: supera el tamaño máximo")
        contenidos[entrada.filename] = contenido
    return contenidos


def _comprobar_entrada(entrada: zipfile.ZipInfo) -> int:
    """Rechaza nombres y tipos no permitidos; devuelve el límite de la entrada."""
    nombre = entrada.filename
    tipo = stat.S_IFMT(entrada.external_attr >> 16)
    if tipo not in (0, stat.S_IFREG):
        raise ArtefactoRechazado(f"Entrada que no es un archivo normal: {nombre!r}")
    if nombre == ARCHIVO_MODELO:
        return LIMITE_MODELO
    if nombre in ARCHIVOS_ARTEFACTOS:
        return LIMITE_JSON
    if PATRON_FIGURA.fullmatch(nombre):
        return LIMITE_PNG
    raise ArtefactoRechazado(f"Nombre no permitido: {nombre!r}")


def _validar_contenido(nombre: str, contenido: bytes) -> None:
    if nombre == ARCHIVO_MODELO:
        _validar_keras(contenido)
    elif nombre.endswith(".png"):
        if not contenido.startswith(MAGIA_PNG):
            raise ArtefactoRechazado(f"{nombre}: no es un PNG")
    else:
        try:
            objeto = interpretar_json(contenido)
            VALIDADORES_JSON.get(nombre, lambda _: None)(objeto)
        except ValueError as error:  # incluye UnicodeDecodeError y JSONDecodeError
            raise ArtefactoRechazado(f"{nombre}: {error}") from error


def _validar_keras(contenido: bytes) -> None:
    """Inspecciona el .keras sin cargarlo: solo los archivos y las clases esperados."""
    try:
        with zipfile.ZipFile(io.BytesIO(contenido)) as modelo:
            entradas = modelo.infolist()
            if {e.filename for e in entradas} != INTERNOS_KERAS or len(entradas) != len(INTERNOS_KERAS):
                raise ArtefactoRechazado(f"{ARCHIVO_MODELO}: contenido inesperado")
            if sum(e.file_size for e in entradas) > LIMITE_INTERNO_KERAS:
                raise ArtefactoRechazado(f"{ARCHIVO_MODELO}: supera el tamaño máximo")
            with modelo.open("config.json") as origen:
                _revisar_configuracion(interpretar_json(origen.read(LIMITE_JSON + 1)))
    except ArtefactoRechazado:
        raise
    # RuntimeError incluye RecursionError: una configuración excesivamente anidada se rechaza.
    except (zipfile.BadZipFile, EOFError, zlib.error, NotImplementedError, RuntimeError, ValueError) as error:
        raise ArtefactoRechazado(f"{ARCHIVO_MODELO}: archivo inválido ({error})") from error


def _revisar_configuracion(nodo, profundidad: int = 0) -> None:
    """Recorre toda la configuración, porque una clase puede ir anidada en cualquier parámetro."""
    if profundidad > PROFUNDIDAD_MAXIMA_KERAS:
        raise ValueError("configuración demasiado anidada")
    if isinstance(nodo, dict):
        if {"class_name", "module", "registered_name"} & nodo.keys():
            clase, modulo = nodo.get("class_name"), nodo.get("module")
            permitida = isinstance(clase, str) and clase in CLASES_KERAS and CLASES_KERAS[clase] == modulo
            if not permitida or nodo.get("registered_name") is not None:
                raise ArtefactoRechazado(f"{ARCHIVO_MODELO}: clase no permitida ({modulo}, {clase})")
        for valor in nodo.values():
            _revisar_configuracion(valor, profundidad + 1)
    elif isinstance(nodo, list):
        for valor in nodo:
            _revisar_configuracion(valor, profundidad + 1)


def main(argumentos: list[str]) -> int:
    if len(argumentos) != 1:
        print("Uso: python -m scripts.importar_artefactos artefactos.zip", file=sys.stderr)
        return 2
    try:
        importados = importar(Path(argumentos[0]))
    except ArtefactoRechazado as error:
        print(f"Rechazado, no se copió nada: {error}", file=sys.stderr)
        return 1
    print("Importados:", ", ".join(importados))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
