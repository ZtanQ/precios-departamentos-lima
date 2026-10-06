"""Reglas de limpieza propuestas en el notebook 01.

Cada función hace una sola cosa y se puede usar desde cualquier notebook.
Las reglas que dependen de estadísticas (medianas) se ajustan SOLO con
los datos de entrenamiento, para no filtrar información del test.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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

def descartar_distritos_baja_frecuencia(
    df: pd.DataFrame, 
    min_anuncios: int = 10, 
    col_distrito: str = "distrito"
) -> pd.DataFrame:
    """Elimina anuncios de distritos con un conteo menor a `min_anuncios`.
    
    Evita incluir distritos con tamaño muestral marginal (ej. N=1 en San Martín 
    de Porres o Callao) que distorsionan el diagnóstico y generan categorías 
    inéditas en particiones temporales.
    """
    df = df.copy()
    conteo = df[col_distrito].value_counts()
    distritos_validos = conteo[conteo >= min_anuncios].index
    return df[df[col_distrito].isin(distritos_validos)].reset_index(drop=True)


class FiltroIQRDistrito(BaseEstimator, TransformerMixin):
    """Filtra valores atípicos (outliers) basándose en el precio por metro cuadrado,
    calculando los límites del Rango Intercuartílico (IQR) por cada distrito.
    
    Diseñado para evitar el Data Leakage: aprende los límites y distritos con soporte
    mínimo en fit() y los aplica en transform(). Se utiliza un factor IQR ajustable
    (por defecto 3.0 para preservar propiedades de lujo legítimas).
    """
    
    def __init__(
        self, 
        factor_iqr: float = 3.0, 
        col_precio: str = "precio_usd", 
        col_area: str = "superficie_m2", 
        col_distrito: str = "distrito",
        min_muestras_distrito: int = 10
    ):
        self.factor_iqr = factor_iqr
        self.col_precio = col_precio
        self.col_area = col_area
        self.col_distrito = col_distrito
        self.min_muestras_distrito = min_muestras_distrito
        self.limites_ = {}
        self.distritos_validos_ = set()
        self.limite_global_ = (0, np.inf)

    def fit(self, X, y=None):
        # Crear copia temporal para calcular los ratios
        df_temp = X[[self.col_precio, self.col_area, self.col_distrito]].copy()
        df_temp["precio_m2"] = df_temp[self.col_precio] / df_temp[self.col_area]

        # Validar soporte mínimo por distrito en entrenamiento
        conteo = df_temp[self.col_distrito].value_counts()
        if self.min_muestras_distrito > 0:
            self.distritos_validos_ = set(conteo[conteo >= self.min_muestras_distrito].index)
            # Solo estimar límites distritales para aquellos con suficiente masa
            df_temp_stats = df_temp[df_temp[self.col_distrito].isin(self.distritos_validos_)]
        else:
            self.distritos_validos_ = set(conteo.index)
            df_temp_stats = df_temp

        # Calcular Q1, Q3 e IQR por distrito usando métodos vectorizados
        stats = df_temp_stats.groupby(self.col_distrito)["precio_m2"].agg(
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75)
        )
        stats["iqr"] = stats["q3"] - stats["q1"]
        
        # Calcular límites con el factor seleccionado
        stats["lim_inf"] = np.maximum(0, stats["q1"] - (self.factor_iqr * stats["iqr"]))
        stats["lim_sup"] = stats["q3"] + (self.factor_iqr * stats["iqr"])

        # Guardar en memoria como diccionario {distrito: (lim_inf, lim_sup)}
        self.limites_ = stats[["lim_inf", "lim_sup"]].apply(tuple, axis=1).to_dict()

        # Calcular un límite global de respaldo
        q1_g = df_temp["precio_m2"].quantile(0.25)
        q3_g = df_temp["precio_m2"].quantile(0.75)
        iqr_g = q3_g - q1_g
        self.limite_global_ = (
            max(0, q1_g - (self.factor_iqr * iqr_g)),
            q3_g + (self.factor_iqr * iqr_g)
        )

        return self

    def transform(self, X, y=None):
        # Operar sobre una copia para no alterar el DataFrame original
        X_out = X.copy()
        precio_m2 = X_out[self.col_precio] / X_out[self.col_area]

        # Mapear los límites correspondientes al distrito de cada fila
        limites = X_out[self.col_distrito].map(self.limites_)

        # Asignar los límites. Si el distrito no se vio en Train, usa el límite global
        lim_inf = limites.apply(lambda x: x[0] if isinstance(x, tuple) else self.limite_global_[0])
        lim_sup = limites.apply(lambda x: x[1] if isinstance(x, tuple) else self.limite_global_[1])

        # Crear máscara binaria de retención y aplicar filtro
        mask_validos = (precio_m2 >= lim_inf) & (precio_m2 <= lim_sup)

        # Filtro de soporte distrital (excluye distritos no vistos o con soporte insuficiente en Train)
        if self.min_muestras_distrito > 0 and len(self.distritos_validos_) > 0:
            mask_distrito = X_out[self.col_distrito].isin(self.distritos_validos_)
            mask_validos = mask_validos & mask_distrito

        return X_out[mask_validos].reset_index(drop=True)