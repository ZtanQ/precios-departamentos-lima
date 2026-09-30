"""Reglas de limpieza propuestas en el notebook 01.

Cada función hace una sola cosa y se puede usar desde cualquier notebook.
Las reglas que dependen de estadísticas (medianas) se ajustan SOLO con
los datos de entrenamiento, para no filtrar información del test.
"""
import numpy as np
import pandas as pd


def normalizar_distritos(df: pd.DataFrame) -> pd.DataFrame:
    """Regla 2: unifica mayúsculas y espacios ("surco", "SURCO" -> "Surco").

    Para cada distrito elige la escritura más frecuente del dataset.
    """
    df = df.copy()
    clave = df["distrito"].str.strip().str.lower()
    escritura_mas_comun = (
        df.assign(_clave=clave)
        .groupby("_clave")["distrito"]
        .agg(lambda s: s.str.strip().value_counts().idxmax())
    )
    df["distrito"] = clave.map(escritura_mas_comun)
    return df


def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Regla 1: elimina filas idénticas. Debe aplicarse ANTES de separar."""
    return df.drop_duplicates().reset_index(drop=True)


def corregir_valores_imposibles(df: pd.DataFrame) -> pd.DataFrame:
    """Reglas 6, 7 y 8: convierte en faltante lo que no puede ser cierto."""
    df = df.copy()
    # df.loc[~df["vista_exterior"].isin([0, 1]), "vista_exterior"] = np.nan
    baños_validos = (df["banos"] * 2) % 1 == 0          # enteros o medios baños
    df.loc[~baños_validos, "banos"] = np.nan
    df.loc[df["antiguedad_anios"] > 150, "antiguedad_anios"] = np.nan
    return df


class FiltroPrecioM2Extremo:
    """Regla 9: descarta precios por m² absurdos para su distrito y año.

    Se ajusta con el entrenamiento (medianas por distrito y año). Un anuncio
    se considera error si su US$/m² es menor que `minimo` veces o mayor que
    `maximo` veces la mediana de su grupo.
    """

    def __init__(self, minimo: float = 0.25, maximo: float = 4.0):
        self.minimo = minimo
        self.maximo = maximo

    @staticmethod
    def _precio_m2(df):
        return df["precio_usd"] / df["superficie_m2"]

    def fit(self, df_train: pd.DataFrame):
        self.medianas_ = self._precio_m2(df_train).groupby(
            [df_train["distrito"], df_train["anio"]]
        ).median()
        # Para años que no están en el entrenamiento: mediana del último año
        ultimo = df_train["anio"].max()
        self.medianas_ultimo_anio_ = self._precio_m2(
            df_train[df_train["anio"] == ultimo]
        ).groupby(df_train["distrito"]).median()
        return self

    def es_extremo(self, df: pd.DataFrame) -> pd.Series:
        claves = pd.MultiIndex.from_arrays([df["distrito"], df["anio"]])
        ref = pd.Series(self.medianas_.reindex(claves).to_numpy(), index=df.index)
        ref = ref.fillna(df["distrito"].map(self.medianas_ultimo_anio_))
        ratio = self._precio_m2(df) / ref
        return (ratio < self.minimo) | (ratio > self.maximo)
