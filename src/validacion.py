"""Particiones por trimestres completos y máscaras IQR sin entrenamiento de modelos."""
import numpy as np
import pandas as pd

from src import config
from src.limpieza import FiltroIQRDistrito

ENTRADAS = ["anio", "trimestre", "distrito", "superficie_m2", "habitaciones",
            "banos", "garajes", "antiguedad_anios"]


def dividir_temporal(base):
    """Base preparada con fila_origen; no se eliminan filas por precio aquí."""
    d = base.copy()
    if d.fila_origen.isna().any() or d.fila_origen.duplicated().any():
        raise ValueError("Cada fila debe tener un identificador de origen único.")
    if not d.trimestre.isin([1, 2, 3, 4]).all() or d.anio.mod(1).ne(0).any():
        raise ValueError("Año/trimestre inválido.")
    periodo = d.anio * 4 + d.trimestre - 1
    if not periodo.between(config.TF1_INICIO, config.TF1_FIN).all():
        raise ValueError("La base debe limitarse a 2022-T1–2026-T1.")
    d["particion"] = np.select([
        periodo.le(config.TF1_FIN_AJUSTE),
        periodo.le(config.TF1_FIN_VALIDACION),
    ], ["entrenamiento", "validacion"], default="prueba")
    d = d.sort_values(["anio", "trimestre", "fila_origen"]).reset_index(drop=True)
    partes = {nombre: d.loc[d.particion.eq(nombre)].copy()
              for nombre in ["entrenamiento", "validacion", "prueba"]}
    if any(parte.empty for parte in partes.values()):
        raise ValueError("Hay una partición vacía.")
    anteriores = None
    for parte in partes.values():
        p = parte.anio * 4 + parte.trimestre - 1
        if anteriores is not None and anteriores >= p.min():
            raise ValueError("Las particiones no respetan el orden temporal.")
        anteriores = p.max()
    return partes


def mascara_iqr(filtro, df):
    """Reproduce la selección del filtro existente sin perder identificadores."""
    temporal = df.copy()
    temporal["_posicion_auditoria"] = np.arange(len(df))
    seleccion = filtro.transform(temporal)
    return np.isin(np.arange(len(df)), seleccion._posicion_auditoria.to_numpy())


def construir_entrega(base):
    partes = dividir_temporal(base)
    ajuste = partes["entrenamiento"]
    # La referencia de selección nunca ve validación ni prueba.
    filtro_seleccion = FiltroIQRDistrito(factor_iqr=3.0).fit(ajuste)
    anterior = pd.concat([partes["entrenamiento"], partes["validacion"]], ignore_index=True)
    # Solo se utiliza después de fijar la configuración del modelo.
    filtro_final = FiltroIQRDistrito(factor_iqr=3.0).fit(anterior)
    entrega = pd.concat(partes.values(), ignore_index=True)
    entrega["incluido_iqr_seleccion"] = pd.Series(pd.NA, index=entrega.index, dtype="boolean")
    seleccion = entrega.particion.ne("prueba")
    entrega.loc[seleccion, "incluido_iqr_seleccion"] = mascara_iqr(filtro_seleccion, entrega.loc[seleccion])
    entrega["incluido_iqr_final"] = mascara_iqr(filtro_final, entrega)
    # Soporte calculado en los subconjuntos de entrenamiento, no en el test.
    for etiqueta, datos in [("seleccion", ajuste), ("final", anterior)]:
        conteo = datos.distrito.value_counts()
        entrega[f"n_distrito_train_{etiqueta}"] = entrega.distrito.map(conteo).fillna(0).astype(int)
    return entrega


def separar_xy(datos):
    """Lista explícita: IDs, máscaras de auditoría y objetivo no son predictores.

    Mantiene año/trimestre disponibles. La selección de variables del modelo
    corresponde al compañero 2; el pipeline actual los descarta.
    """
    return datos[ENTRADAS].copy(), datos[config.OBJETIVO].copy()
