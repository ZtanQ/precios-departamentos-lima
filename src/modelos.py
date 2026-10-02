import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

def obtener_modelos_inmobiliarios(preprocesador):
    """
    Retorna un diccionario con los pipelines de los modelos a evaluar.
    Todos los modelos incluyen el preprocesador y una transformación 
    logarítmica natural sobre la variable objetivo (precio_usd).
    """
    
    # 1. Definición de algoritmos base
    algoritmos = {
        "Ridge Regressor": Ridge(alpha=1.0, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=200, depth=8, random_state=42, verbose=False)
    }
    
    modelos = {}
    
    # 2. Ensamblaje: Preprocesador + Algoritmo + Log(y)
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