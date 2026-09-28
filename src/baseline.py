"""Baseline de dominio: la regla que usa el mercado para fijar precios.

Precio estimado = mediana de US$/m² del distrito × superficie del departamento.

Está escrita como un estimador de scikit-learn (fit / predict) para que se
evalúe exactamente igual que los modelos del notebook 03.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class ReglaMercadoM2(BaseEstimator, RegressorMixin):
    """Mediana de US$/m² por distrito × superficie.

    Parámetros
    ----------
    anios_referencia : int
        Cuántos años finales del entrenamiento se usan para calcular la
        mediana. Con 1 se usa solo el último año, que es lo que haría un
        corredor: mirar los precios más recientes del distrito.
    """

    def __init__(self, anios_referencia: int = 1):
        self.anios_referencia = anios_referencia

    def fit(self, X: pd.DataFrame, y):
        precio_m2 = np.asarray(y, dtype=float) / X["superficie_m2"].to_numpy()
        ultimo_anio = X["anio"].max()
        recientes = X["anio"].to_numpy() > ultimo_anio - self.anios_referencia
        ref = pd.Series(precio_m2[recientes], index=X["distrito"].to_numpy()[recientes])
        self.mediana_por_distrito_ = ref.groupby(level=0).median()
        # Distritos que no aparecen en los años de referencia: mediana general
        self.mediana_general_ = float(np.median(precio_m2[recientes]))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        m2 = X["distrito"].map(self.mediana_por_distrito_).fillna(self.mediana_general_)
        return (m2 * X["superficie_m2"]).to_numpy()
