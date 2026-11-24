"""Definición de modelos y ensambles inmobiliarios para TP1 y TF1."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb


class EnsamblePonderadoPrecioM2(BaseEstimator, RegressorMixin):
    """Ensamble heterogéneo (CatBoost, RF, Ridge, LightGBM, XGBoost) que predice log(precio_m2)
       y reconstruye el precio en dólares: precio_predicho_usd = expm1(pred_log_m2) * superficie_m2
    """
    def __init__(self, pipe_cb=None, pipe_rf=None, pipe_rd=None, pipe_lgb=None, pipe_xgb=None,
                 w_cb=0.2, w_rf=0.2, w_rd=0.1, w_lgb=0.25, w_xgb=0.25):
        self.pipe_cb = pipe_cb
        self.pipe_rf = pipe_rf
        self.pipe_rd = pipe_rd
        self.pipe_lgb = pipe_lgb
        self.pipe_xgb = pipe_xgb
        self.w_cb = w_cb
        self.w_rf = w_rf
        self.w_rd = w_rd
        self.w_lgb = w_lgb
        self.w_xgb = w_xgb

    def fit(self, X, y_m2):
        y_log_m2 = np.log1p(y_m2)
        if self.pipe_cb is not None:
            self.pipe_cb.fit(X, y_log_m2)
        if self.pipe_rf is not None:
            self.pipe_rf.fit(X, y_log_m2)
        if self.pipe_rd is not None:
            self.pipe_rd.fit(X, y_log_m2)
        if self.pipe_lgb is not None:
            self.pipe_lgb.fit(X, y_log_m2)
        if self.pipe_xgb is not None:
            self.pipe_xgb.fit(X, y_log_m2)
        return self

    def predict_m2(self, X):
        preds = []
        pesos = []
        if self.pipe_cb is not None and self.w_cb > 0:
            preds.append(self.w_cb * self.pipe_cb.predict(X))
            pesos.append(self.w_cb)
        if self.pipe_rf is not None and self.w_rf > 0:
            preds.append(self.w_rf * self.pipe_rf.predict(X))
            pesos.append(self.w_rf)
        if self.pipe_rd is not None and self.w_rd > 0:
            preds.append(self.w_rd * self.pipe_rd.predict(X))
            pesos.append(self.w_rd)
        if self.pipe_lgb is not None and self.w_lgb > 0:
            preds.append(self.w_lgb * self.pipe_lgb.predict(X))
            pesos.append(self.w_lgb)
        if self.pipe_xgb is not None and self.w_xgb > 0:
            preds.append(self.w_xgb * self.pipe_xgb.predict(X))
            pesos.append(self.w_xgb)
            
        peso_total = sum(pesos) if sum(pesos) > 0 else 1.0
        pred_log_m2 = sum(preds) / peso_total
        return np.expm1(pred_log_m2)

    def predict(self, X):
        pred_m2 = self.predict_m2(X)
        superficie = np.asarray(X["superficie_m2"], dtype=float)
        return pred_m2 * superficie


def obtener_modelos_inmobiliarios(preprocesador):
    """Retorna un diccionario con los pipelines de los modelos base de TP1."""
    algoritmos = {
        "Ridge Regressor": Ridge(alpha=1.0, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=200, depth=8, random_state=42, verbose=False)
    }
    
    modelos = {}
    for nombre, estimador in algoritmos.items():
        pipe = Pipeline([
            ('prep', preprocesador),
            ('modelo', estimador)
        ])
        modelo_transformado = TransformedTargetRegressor(
            regressor=pipe,
            func=np.log1p,       
            inverse_func=np.expm1 
        )
        modelos[nombre] = modelo_transformado
        
    return modelos
