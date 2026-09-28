"""Métricas comunes para comparar todos los modelos con la misma vara.

- MAE (US$): cuántos dólares se equivoca en promedio. Es lo que entiende
  un comprador.
- Error porcentual mediano: cuánto se equivoca un anuncio típico, sin que
  unos pocos casos extremos lo distorsionen.
- Error porcentual medio (MAPE): sensible a precios mal ingresados.
- % de anuncios con error <= 10%: qué tan seguido la estimación sirve
  para negociar.
"""
import numpy as np
import pandas as pd


def errores_porcentuales(y_real, y_pred) -> np.ndarray:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.abs(y_real - y_pred) / y_real


def evaluar(y_real, y_pred) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = errores_porcentuales(y_real, y_pred)
    return {
        "MAE (US$)": float(np.mean(np.abs(y_real - y_pred))),
        "RMSE (US$)": float(np.sqrt(np.mean((y_real - y_pred) ** 2))),
        "Error % mediano": float(np.median(ape)),
        "Error % medio (MAPE)": float(np.mean(ape)),
        "Anuncios con error <= 10%": float(np.mean(ape <= 0.10)),
    }


def tabla_comparativa(resultados: dict) -> pd.DataFrame:
    """Convierte {nombre_modelo: evaluar(...)} en una tabla legible."""
    tabla = pd.DataFrame(resultados).T
    formato = {
        "MAE (US$)": "{:,.0f}",
        "RMSE (US$)": "{:,.0f}",
        "Error % mediano": "{:.1%}",
        "Error % medio (MAPE)": "{:.1%}",
        "Anuncios con error <= 10%": "{:.1%}",
    }
    return tabla.style.format(formato)
