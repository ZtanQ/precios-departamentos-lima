"""Rutas y constantes compartidas por todo el proyecto.

Si una decisión del equipo cambia (por ejemplo, el año de corte del
entrenamiento), se cambia aquí una sola vez y todos los notebooks la toman.
"""
from pathlib import Path

# Rutas
RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"
FIGURES = RAIZ / "reports" / "figures"

ARCHIVO_EXCEL = DATA_RAW / "precios-inmobiliarios-bd-desagregada-venta-2026-1.xlsx"
ARCHIVO_PARQUET = DATA_RAW / "precios_venta_bcrp_2026_1.parquet"

# Semilla para que los resultados se repitan igual en todas las computadoras
SEMILLA = 42

# Separación temporal (propuesta; confirmar en equipo)
ANIO_FIN_TRAIN = 2023      # entrenamiento: anuncios hasta 2023
ANIO_INICIO_TEST = 2024    # evaluación: anuncios de 2024 en adelante

# Variable objetivo
OBJETIVO = "precio_usd"

# Nombres originales del Excel -> nombres cortos para el código.
# Antes de aplicar el mapa se quitan los espacios sobrantes
# (en el Excel, "Superficie " trae un espacio al final).
NOMBRES_COLUMNAS = {
    "Unnamed: 0": "periodo",
    "Año": "anio",
    "Trimestre": "trimestre",
    "Precio en dólares corrientes": "precio_usd",
    "Tipo de cambio": "tipo_cambio",
    "IPC": "ipc",
    "Precio en soles corrientes": "precio_soles",
    "Precio en soles constantes de 2009": "precio_soles_2009",
    "Distrito": "distrito",
    "Superficie": "superficie_m2",
    "Número de habitaciones": "habitaciones",
    "Número de baños": "banos",
    "Número de garajes": "garajes",
    "Piso de ubicación": "piso",
    "Vista al exterior": "vista_exterior",
    "Años de antigüedad": "antiguedad_anios",
}

# Columnas que nunca deben entrar al modelo:
# - fuga de datos: son el precio objetivo convertido a soles
# - redundantes: solo repiten la información del período
COLUMNAS_FUGA = ["precio_soles", "precio_soles_2009"]
COLUMNAS_REDUNDANTES = ["periodo", "tipo_cambio", "ipc"]
