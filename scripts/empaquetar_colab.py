"""Crea colab_bundle.zip con src/, el CSV y requirements.txt, y nada más."""
import hashlib
import zipfile
from pathlib import Path

from src import config

RUTA_BUNDLE = config.RAIZ / "colab_bundle.zip"


def empaquetar(destino: Path = RUTA_BUNDLE) -> Path:
    if hashlib.sha256(config.RUTA_DATOS.read_bytes()).hexdigest() != config.SHA256_DATOS:
        raise ValueError("El CSV no coincide con el SHA-256 fijado")
    archivos = [*sorted((config.RAIZ / "src").rglob("*.py")), config.RUTA_DATOS, config.RAIZ / "requirements.txt"]
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as bundle:
        for archivo in archivos:
            bundle.write(archivo, archivo.relative_to(config.RAIZ).as_posix())
    return destino


if __name__ == "__main__":
    print(f"Creado {empaquetar().name}")
