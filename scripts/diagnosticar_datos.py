"""Diagnóstico de solo lectura. Ejecutar: python scripts/diagnosticar_datos.py.

No entrena modelos ni cambia reglas o datos. Guarda y muestra el mismo reporte.
"""
import argparse
import hashlib
import io
import platform
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd

from src import config
from src.datos import cargar_datos


def generar_reporte(df):
    salida = io.StringIO()

    def mostrar(titulo, contenido):
        salida.write(f"\n{'=' * 78}\n{titulo}\n{'=' * 78}\n")
        if isinstance(contenido, pd.DataFrame):
            contenido = contenido.to_string(max_rows=None, max_cols=None, float_format=lambda x: f"{x:,.4f}")
        elif isinstance(contenido, pd.Series):
            contenido = contenido.to_string(max_rows=None, float_format=lambda x: f"{x:,.4f}")
        salida.write(str(contenido) + "\n")

    mostrar("CONTEXTO Y LECTURA DEL REPORTE", (
        "Objetivo declarado: precio base de oferta para posterior ajuste profesional.\n"
        "Una fila representa un anuncio; no se puede identificar un inmueble único sin un ID fiable.\n"
        "Diagnóstico descriptivo: no demuestra causalidad ni importancia predictiva del modelo.\n"
        "No se limpian, imputan ni eliminan observaciones. Solo se crean columnas auxiliares en memoria.\n"
        "Las alertas requieren revisión: un extremo no equivale automáticamente a un error.\n"
        f"Python: {platform.python_version()} | pandas: {pd.__version__} | numpy: {np.__version__}"
    ))
    archivo = config.ARCHIVO_PARQUET if config.ARCHIVO_PARQUET.exists() else config.ARCHIVO_EXCEL
    mostrar("FUENTE LOCAL", f"Archivo: {archivo.name}\nSHA256: {hashlib.sha256(archivo.read_bytes()).hexdigest()}\nDimensiones originales: {df.shape}")
    mostrar("ESQUEMA ORIGINAL", pd.DataFrame({
        "tipo": df.dtypes.astype(str), "nulos": df.isna().sum(),
        "porcentaje_nulos": df.isna().mean() * 100, "valores_distintos": df.nunique(),
    }))
    mostrar("COBERTURA HISTÓRICA (ANTES DEL CORTE)", df.groupby("anio").agg(anuncios=("precio_usd", "size"), precio_mediano=("precio_usd", "median")))

    d = df.loc[df.anio.between(2022, 2026)].copy()
    if d.empty:
        raise ValueError("No hay registros entre 2022 y 2026.")
    d["distrito_clave"] = d.distrito.astype("string").str.normalize("NFC").str.strip().str.replace(r"\s+", " ", regex=True).str.casefold()
    area_valida = d.superficie_m2.where(np.isfinite(d.superficie_m2) & d.superficie_m2.gt(0))
    d["precio_m2_diagnostico"] = d.precio_usd / area_valida
    mostrar("MUESTRA DE TRABAJO 2022–2026 SIN LIMPIEZA", f"Filas: {len(d):,}\nDistritos originales: {d.distrito.nunique()}\nClaves distritales normalizadas: {d.distrito_clave.nunique()}\nEl año 2026 puede tener cobertura parcial; consultar trimestres abajo.")
    mostrar("ANUNCIOS POR AÑO Y TRIMESTRE", pd.crosstab(d.anio, d.trimestre, margins=True))
    mostrar("ESCRITURAS ORIGINALES DE DISTRITOS", d.groupby(["distrito_clave", "distrito"], dropna=False).size().rename("anuncios"))
    mostrar("COBERTURA DISTRITAL POR AÑO", pd.crosstab(d.distrito_clave, d.anio))
    mostrar("FALTANTES POR AÑO (%)", d[list(df.columns)].groupby("anio").agg(lambda s: s.isna().mean() * 100))
    numericas = d[list(df.columns)].select_dtypes(include="number")
    mostrar("DISTRIBUCIONES NUMÉRICAS", numericas.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T)
    mostrar("VALORES NO FINITOS (NO INCLUYE NULOS)", (numericas.notna() & ~np.isfinite(numericas)).sum())
    for col in ["habitaciones", "banos", "garajes", "piso", "vista_exterior", "antiguedad_anios"]:
        mostrar(f"FRECUENCIAS COMPLETAS: {col}", d[col].value_counts(dropna=False).sort_index())

    alertas = {
        "precio ausente, no finito o <= 0": ~np.isfinite(d.precio_usd) | d.precio_usd.le(0),
        "superficie ausente, no finita o <= 0": ~np.isfinite(d.superficie_m2) | d.superficie_m2.le(0),
        "habitaciones <= 0 o fraccionarias": d.habitaciones.le(0) | d.habitaciones.mod(1).ne(0) & d.habitaciones.notna(),
        "baños <= 0 o distintos de múltiplos de 0.5": d.banos.le(0) | (d.banos.mul(2).mod(1).ne(0) & d.banos.notna()),
        "garajes negativos o fraccionarios": d.garajes.lt(0) | (d.garajes.mod(1).ne(0) & d.garajes.notna()),
        "antigüedad negativa o > 150": d.antiguedad_anios.lt(0) | d.antiguedad_anios.gt(150),
        "vista fuera de 0/1 (sin nulos)": ~d.vista_exterior.isin([0, 1]) & d.vista_exterior.notna(),
    }
    mostrar("ALERTAS LÓGICAS: NO SON EXCLUSIONES AUTOMÁTICAS", pd.Series({k: int(v.sum()) for k, v in alertas.items()}))
    originales = list(df.columns)
    seleccion = ["anio", "trimestre", "precio_usd", "distrito_clave", "superficie_m2", "habitaciones", "banos", "garajes", "antiguedad_anios"]
    mostrar("DUPLICADOS: CANTIDAD DE FILAS ADICIONALES", pd.Series({
        "exactos en columnas originales": d.duplicated(subset=originales).sum(),
        "en columnas seleccionadas y distrito normalizado": d.duplicated(subset=seleccion).sum(),
    }))
    mostrar("EJEMPLOS DE COINCIDENCIAS EN COLUMNAS SELECCIONADAS (MÁXIMO 20)", d.loc[d.duplicated(subset=seleccion, keep=False), seleccion].head(20))
    fisicas = ["distrito_clave", "superficie_m2", "habitaciones", "banos", "garajes", "antiguedad_anios"]
    coincidencias = d.groupby(fisicas, dropna=False).agg(anuncios=("anio", "size"), anios=("anio", "nunique"))
    mostrar("PERFILES FÍSICOS COINCIDENTES ENTRE AÑOS", f"Grupos presentes en varios años: {int(coincidencias.anios.gt(1).sum())}\nNo prueba que sean el mismo inmueble; no eliminar automáticamente.")
    mostrar("PRECIO Y SUPERFICIE POR DISTRITO", d.groupby("distrito_clave").agg(
        anuncios=("precio_usd", "size"), precio_mediano=("precio_usd", "median"),
        precio_q25=("precio_usd", lambda s: s.quantile(.25)), precio_q75=("precio_usd", lambda s: s.quantile(.75)),
        superficie_mediana=("superficie_m2", "median"), precio_m2_mediano=("precio_m2_diagnostico", "median"),
    ).sort_values("anuncios", ascending=False))
    mostrar("EVOLUCIÓN TRIMESTRAL", d.groupby(["anio", "trimestre"]).agg(anuncios=("precio_usd", "size"), precio_mediano=("precio_usd", "median"), precio_m2_mediano=("precio_m2_diagnostico", "median")))
    correlaciones = d.select_dtypes(include="number").corr(method="spearman")["precio_usd"].drop("precio_usd").sort_values(ascending=False)
    mostrar("ASOCIACIÓN SPEARMAN CON PRECIO (NO IMPORTANCIA DEL MODELO)", correlaciones)
    mostrar("ADVERTENCIA SOBRE ASOCIACIONES", "Los precios en soles derivan del objetivo y no son predictores válidos.\nDistrito es categórico: esta correlación no permite compararlo con las variables numéricas.\nLas asociaciones y medianas por distrito pueden reflejar diferencias de composición, superficie y período.")
    for col in ["habitaciones", "banos", "garajes"]:
        mostrar(f"PRECIO SEGÚN {col} (SIN CONTROLAR OTRAS VARIABLES)", d.groupby(col, dropna=False).agg(anuncios=("precio_usd", "size"), precio_mediano=("precio_usd", "median"), superficie_mediana=("superficie_m2", "median")))
    columnas_ejemplo = seleccion + ["precio_m2_diagnostico"]
    mostrar("10 PRECIOS POR M² MÁS BAJOS (REVISIÓN, NO BORRADO)", d.nsmallest(10, "precio_m2_diagnostico")[columnas_ejemplo])
    mostrar("10 PRECIOS POR M² MÁS ALTOS (REVISIÓN, NO BORRADO)", d.nlargest(10, "precio_m2_diagnostico")[columnas_ejemplo])
    mostrar("SEPARACIÓN CONFIGURADA ACTUALMENTE: SOLO DESCRIPCIÓN", f"Fin de entrenamiento: {config.ANIO_FIN_TRAIN}\nInicio de prueba: {config.ANIO_INICIO_TEST}\nNo se propone ni aplica una nueva separación en este script.")
    mostrar("COBERTURA POR DISTRITO EN EL CORTE ACTUAL (DATOS SIN LIMPIAR)", pd.crosstab(d.distrito_clave, np.where(d.anio <= config.ANIO_FIN_TRAIN, "hasta_fin_train", "posterior_fin_train")))
    mostrar("ATRIBUTOS DISPONIBLES Y LÍMITES", (
        f"Columnas originales: {', '.join(df.columns)}\n"
        "No se observan columnas explícitas de acabados, decoración, estado de conservación, dirección/coordenadas, distancias a servicios ni precio final de transacción.\n"
        "Sin identificador de inmueble no puede determinarse con certeza la republicación.\n"
        "Este diagnóstico no mide precisión predictiva ni confirma qué variable es la más importante.\n"
        "Los resultados describen anuncios de la fuente, no garantizan representatividad de toda Lima."
    ))
    return salida.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=RAIZ / "reports" / "diagnostico_datos.txt", help="Ruta del reporte UTF-8.")
    args = parser.parse_args()
    reporte = generar_reporte(cargar_datos())
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(reporte, encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(reporte)
    print(f"\nReporte guardado en: {args.salida.resolve()}")


if __name__ == "__main__":
    main()
