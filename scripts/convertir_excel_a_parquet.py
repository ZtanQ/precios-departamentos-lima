"""Convierte el Excel del BCRP (65 MB) a Parquet (pocos MB).

No cambia ningún dato: solo el formato, para que el repositorio sea liviano
y la carga tome segundos. Solo hace falta correrlo si el BCRP publica una
versión nueva del Excel.

Uso, desde la raíz del repositorio:
    python scripts/convertir_excel_a_parquet.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.datos import leer_excel_original  # noqa: E402


def main() -> None:
    print(f"Leyendo {config.ARCHIVO_EXCEL.name} ...")
    df = leer_excel_original()
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(config.ARCHIVO_PARQUET, index=False)
    print(
        f"Listo: {config.ARCHIVO_PARQUET.name} "
        f"({len(df):,} filas, {df.shape[1]} columnas)"
    )


if __name__ == "__main__":
    main()
