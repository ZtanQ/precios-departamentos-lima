r"""Auditoría de solo lectura del período 2022–2026-T1.

Ejecutar desde el proyecto:
    .\.venv\Scripts\python.exe scripts/auditar_preparacion.py

Genera un resumen y CSV detallados en reports/auditoria_preparacion/.
No guarda datos preparados ni cambia el criterio de limpieza del TP1.
"""
import argparse
import hashlib
import io
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd

from src import config, limpieza
from src.datos import cargar_datos


def agrupar_coincidencias(df, columnas):
    """Incluye todos los miembros de cada grupo, no solo filas adicionales."""
    grupos = df.loc[df.duplicated(subset=columnas, keep=False)].copy()
    if grupos.empty:
        grupos.insert(0, "grupo_auditoria", pd.Series(dtype="int64"))
        return grupos
    grupos.insert(0, "grupo_auditoria", grupos.groupby(
        columnas, dropna=False, sort=False
    ).ngroup() + 1)
    return grupos.sort_values(["grupo_auditoria", "fila_origen"])


def auditar(df):
    """Devuelve tablas y texto; la entrada no se modifica.

    fila_origen es la posición base cero en cargar_datos(), no un ID inmobiliario
    ni el número de fila del Excel. Toda deduplicación es una vista en memoria.
    """
    originales = list(df.columns)
    base = df.copy()
    base.insert(0, "fila_origen", np.arange(len(df)))
    periodo = base.anio.between(2022, 2025) | (base.anio.eq(2026) & base.trimestre.eq(1))
    bruto = base.loc[periodo].copy()
    if bruto.empty:
        raise ValueError("No hay anuncios en 2022–2026-T1.")
    tablas = {"duplicados_exactos": agrupar_coincidencias(bruto, originales)}

    # Reproduce en memoria el orden del notebook: exactos, normalización,
    # correcciones lógicas, selección de columnas y soporte distrital global.
    # La columna de trazabilidad nunca participa en la detección de duplicados.
    sin_exactos = base.drop_duplicates(subset=originales).copy()
    normalizado = limpieza.normalizar_distritos(sin_exactos)
    corregido = limpieza.corregir_valores_imposibles(normalizado)
    descartadas = config.COLUMNAS_FUGA + config.COLUMNAS_REDUNDANTES + ["piso", "vista_exterior"]
    seleccionadas = [c for c in originales if c not in descartadas]
    soporte = corregido.distrito.value_counts()
    vista = corregido.loc[
        corregido.fila_origen.isin(bruto.fila_origen)
        & corregido.distrito.isin(soporte[soporte >= 10].index)
    ].copy()
    residuales = agrupar_coincidencias(vista, seleccionadas)
    # Adjunta originales para identificar exactamente qué se perdió o corrigió.
    residuales = residuales.merge(
        base, on="fila_origen", how="left", suffixes=("", "_original"), validate="one_to_one"
    )
    tablas["coincidencias_residuales"] = residuales
    diferencias = []
    for grupo, filas in residuales.groupby("grupo_auditoria"):
        fuente = base.loc[base.fila_origen.isin(filas.fila_origen)]
        distintas = [c for c in originales if fuente[c].nunique(dropna=False) > 1]
        diferencias.append({
            "grupo_auditoria": grupo, "miembros": len(filas),
            "filas_adicionales": len(filas) - 1,
            "filas_origen": ", ".join(map(str, filas.fila_origen)),
            "columnas_originales_diferentes": ", ".join(distintas) or "ninguna",
            "estado": "pendiente de revisión; no se elimina",
        })
    tablas["diferencias_residuales"] = pd.DataFrame(diferencias, columns=[
        "grupo_auditoria", "miembros", "filas_adicionales", "filas_origen",
        "columnas_originales_diferentes", "estado",
    ])

    motivos = pd.Series("", index=bruto.index)

    def marcar(mask, motivo):
        mask = mask.fillna(False)
        motivos.loc[mask] += motivo + "; "

    for col in ["precio_usd", "superficie_m2"]:
        marcar(~np.isfinite(bruto[col]) | bruto[col].le(0), f"{col}: ausente, no finito o no positivo")
    for col in ["habitaciones", "banos", "garajes", "antiguedad_anios"]:
        marcar(bruto[col].isna(), f"{col}: faltante original")
        marcar(bruto[col].notna() & ~np.isfinite(bruto[col]), f"{col}: no finito")
    marcar(bruto.habitaciones.le(0) | (bruto.habitaciones.notna() & bruto.habitaciones.mod(1).ne(0)), "habitaciones: no positivo o fraccionario")
    marcar(bruto.banos.le(0) | (bruto.banos.notna() & bruto.banos.mul(2).mod(1).ne(0)), "baños: no positivo o no múltiplo de 0.5")
    marcar(bruto.garajes.lt(0) | (bruto.garajes.notna() & bruto.garajes.mod(1).ne(0)), "garajes: negativo o fraccionario")
    marcar(bruto.antiguedad_anios.lt(0) | bruto.antiguedad_anios.gt(150), "antigüedad: negativa o superior a 150")
    marcar(bruto.distrito.isna() | bruto.distrito.astype("string").str.strip().eq(""), "distrito: vacío")
    logicas = bruto.loc[motivos.ne("")].copy()
    logicas.insert(0, "motivo_revision", motivos.loc[logicas.index].str.rstrip("; "))
    tablas["inconsistencias_logicas"] = logicas

    # Referencia descriptiva en muestra sin exactos. No es un filtro de modelado:
    # utiliza todo el período, no debe trasladarse al pipeline ni a la validación.
    referencia = normalizado.loc[normalizado.fila_origen.isin(bruto.fila_origen)].copy()
    referencia["precio_m2"] = referencia.precio_usd / referencia.superficie_m2
    validos = np.isfinite(referencia.precio_m2) & referencia.precio_m2.gt(0) & referencia.superficie_m2.gt(0)
    referencia = referencia.loc[validos]
    ref = referencia.groupby(["distrito", "anio"]).precio_m2.agg(["size", "median"])
    ref.columns = ["n_referencia_distrito_anio", "mediana_m2_distrito_anio"]
    precios = bruto.copy()
    precios["distrito_referencia"] = precios.fila_origen.map(normalizado.set_index("fila_origen").distrito).fillna(precios.distrito)
    precios["precio_m2"] = precios.precio_usd / precios.superficie_m2.where(precios.superficie_m2.gt(0))
    precios = precios.merge(ref, left_on=["distrito_referencia", "anio"], right_index=True, how="left", validate="many_to_one")
    precios["ratio_mediana_m2"] = precios.precio_m2 / precios.mediana_m2_distrito_anio
    precios["alerta_ratio"] = precios.ratio_mediana_m2.lt(.25) | precios.ratio_mediana_m2.gt(4)
    precios["entre_10_menores_m2"] = precios.fila_origen.isin(precios.nsmallest(10, "precio_m2").fila_origen)
    precios["entre_10_mayores_m2"] = precios.fila_origen.isin(precios.nlargest(10, "precio_m2").fila_origen)
    tablas["precios_para_revision"] = precios.loc[
        precios.alerta_ratio | precios.entre_10_menores_m2 | precios.entre_10_mayores_m2
    ].sort_values("precio_m2")

    texto = io.StringIO()

    def escribir(titulo, valor):
        texto.write(f"\n{'=' * 78}\n{titulo}\n{'=' * 78}\n")
        if isinstance(valor, pd.DataFrame):
            valor = valor.to_string(index=False, max_rows=None, max_cols=None)
        texto.write(str(valor) + "\n")

    escribir("ALCANCE", "2022-T1 a 2026-T1. Auditoría; no aplica decisiones nuevas.\n"
             "fila_origen: posición base cero en la carga original, no ID de inmueble.\n"
             "Las vistas de preparación son copias en memoria; la fuente no se modifica.\n"
             "Coincidir en características no demuestra identidad del inmueble.\n"
             "Las estadísticas de precio son descriptivas del período completo, NO reglas para entrenar o evaluar modelos.")
    escribir("CONTEOS", pd.DataFrame([
        {"etapa": "original histórico", "filas": len(base)},
        {"etapa": "2022–2026-T1 bruto", "filas": len(bruto)},
        {"etapa": "exactos adicionales en el período", "filas": int(bruto.duplicated(originales).sum())},
        {"etapa": "período sin exactos", "filas": int(sin_exactos.fila_origen.isin(bruto.fila_origen).sum())},
        {"etapa": "vista TP1 previa a IQR (soporte global >=10)", "filas": len(vista)},
        {"etapa": "coincidencias adicionales en vista TP1", "filas": int(vista.duplicated(seleccionadas).sum())},
        {"etapa": "filas con alertas lógicas o faltantes", "filas": len(logicas)},
        {"etapa": "precios para revisión (incluye top/bottom 10)", "filas": len(tablas["precios_para_revision"])},
    ]))
    escribir("CLAVE DE COINCIDENCIAS RESIDUALES", ", ".join(seleccionadas))
    escribir("DIFERENCIAS ENTRE COINCIDENCIAS RESIDUALES", tablas["diferencias_residuales"])
    escribir("FILAS COMPLETAS DE COINCIDENCIAS RESIDUALES", residuales)
    escribir("INCONSISTENCIAS LÓGICAS Y FALTANTES ORIGINALES", logicas)
    escribir("PRECIOS PARA REVISIÓN", tablas["precios_para_revision"])
    escribir("CRITERIOS DE LECTURA Y PENDIENTES", (
        "Ratio <0.25 o >4 de la mediana de distrito/año: señal exploratoria, no error confirmado.\n"
        "Los 10 menores/mayores US$/m² se incluyen aunque no superen esos límites.\n"
        "Referencias con pocos casos son débiles; consultar n_referencia_distrito_anio.\n"
        "No se ajusta ni aplica el filtro IQR: esto no reproduce su muestra final.\n"
        "Antigüedad 1996 podría ser un año de construcción; no se convierte sin evidencia.\n"
        "No se reemplaza US$190 por US$190000 ni se corrigen monedas por inferencia.\n"
        "La política actual convierte antigüedades >150 en faltantes solo en la vista TP1.\n"
        "Los cuatro residuales se buscan después de las transformaciones actuales; pueden diferir en columnas descartadas.\n"
        "Las repeticiones exactas son repeticiones de filas, no identificación confirmada de inmuebles.\n"
        "Próximo paso: revisar estas tablas y acordar conservar, corregir con evidencia o excluir con motivo documentado."
    ))
    return tablas, texto.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=RAIZ / "reports" / "auditoria_preparacion")
    args = parser.parse_args()
    df = cargar_datos()
    tablas, reporte = auditar(df)
    fuente = config.ARCHIVO_PARQUET if config.ARCHIVO_PARQUET.exists() else config.ARCHIVO_EXCEL
    reporte = (f"Fuente: {fuente.name}\nSHA256: {hashlib.sha256(fuente.read_bytes()).hexdigest()}\n"
               f"pandas: {pd.__version__} | numpy: {np.__version__}\n" + reporte)
    args.salida.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in tablas.items():
        tabla.to_csv(args.salida / f"{nombre}.csv", index=False, encoding="utf-8-sig")
    destino = args.salida / "resumen_auditoria.txt"
    destino.write_text(reporte, encoding="utf-8")
    print(f"Auditoría terminada. No se modificaron datos ni reglas.\nResumen: {destino.resolve()}")
    print(f"Se generaron {len(tablas)} CSV con el detalle en {args.salida.resolve()}")


if __name__ == "__main__":
    main()
