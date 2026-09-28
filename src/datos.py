"""Carga única del dataset para todo el equipo.

Todos los notebooks usan `cargar_datos()` en lugar de leer el archivo por
su cuenta. Así todos parten de los mismos datos y los mismos nombres de
columnas.

Esta función NO limpia los datos (duplicados, distritos mal escritos,
valores extremos). Eso se hace después, con las reglas acordadas.
"""
import pandas as pd

from src import config


def _renombrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip())
    faltantes = set(config.NOMBRES_COLUMNAS) - set(df.columns)
    if faltantes:
        raise ValueError(
            f"El archivo no tiene las columnas esperadas: {sorted(faltantes)}. "
            "Revisa que sea la base desagregada de venta del BCRP."
        )
    return df.rename(columns=config.NOMBRES_COLUMNAS)


def leer_excel_original() -> pd.DataFrame:
    """Lee el Excel del BCRP tal como viene (tarda cerca de un minuto)."""
    if not config.ARCHIVO_EXCEL.exists():
        raise FileNotFoundError(
            f"No se encontró {config.ARCHIVO_EXCEL.name} en data/raw/. "
            "Colócalo ahí (ver README)."
        )
    return pd.read_excel(config.ARCHIVO_EXCEL, sheet_name="Venta")


def cargar_datos() -> pd.DataFrame:
    """Devuelve el dataset con nombres de columnas cortos, sin limpiar.

    Usa la copia en Parquet incluida en el repositorio (carga en segundos).
    Si no existe, lee el Excel original.
    """
    if config.ARCHIVO_PARQUET.exists():
        df = pd.read_parquet(config.ARCHIVO_PARQUET)
    else:
        df = leer_excel_original()
    return _renombrar_columnas(df)
