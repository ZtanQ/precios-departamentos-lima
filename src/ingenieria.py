"""Módulo de ingeniería de características y clustering inmobiliario para el TF1.

Implementa:
1. Selección y justificación técnica del número óptimo de clusters (k) mediante:
   - Método del Codo (Inercia intra-cluster / WCSS).
   - Coeficiente de Silueta (Silhouette Score).
   - Índice de Davies-Bouldin.
2. Estimador scikit-learn compatible (ClusteringSubmercado) que:
   - Aprende exclusivamente con el conjunto de entrenamiento (Cero Data Leakage).
   - Puede evaluar automáticamente el k óptimo o recibirlo validado.
   - Segmenta los distritos de Lima en macro-submercados económicos homogéneos.
3. Estimador scikit-learn compatible (GeneradorCaracteristicasInmobiliarias):
   - Ratios de habitabilidad (baños/hab, m2/hab, ratio_sup_distrital, tipología).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


class ClusteringSubmercado(BaseEstimator, TransformerMixin):
    """Segmentación no supervisada de distritos en submercados de valor inmobiliario.
    
    Aprende en fit() el perfil inmobiliario de cada distrito a partir de datos de entrenamiento
    (mediana de US$/m2, mediana de superficie y mediana de antigüedad), estandariza estas
    métricas y aplica K-Means.
    
    Si n_clusters='auto', ejecuta una búsqueda sistemática evaluando el Coeficiente de Silueta
    y el Índice de Davies-Bouldin sobre k in [2, 7] y selecciona la partición que maximiza la
    cohesión y separación de clusters con sentido de negocio.
    """
    
    def __init__(self, n_clusters=4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.mapa_distrito_cluster_ = {}
        self.cluster_defecto_ = "Cluster_0"
        self.perfil_clusters_ = None
        self.evaluacion_k_ = {}
        self.k_seleccionado_ = None

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        if y is not None:
            df["_precio"] = np.asarray(y, dtype=float)
        elif "precio_usd" in df.columns:
            df["_precio"] = df["precio_usd"].astype(float)
        else:
            raise ValueError("Se requiere 'precio_usd' (o vector y) para estimar los clusters distritales.")

        df["_precio_m2"] = df["_precio"] / df["superficie_m2"]

        # Agregación por distrito (solo distritos de entrenamiento)
        stats = df.groupby("distrito").agg(
            mediana_m2=("_precio_m2", "median"),
            mediana_superficie=("superficie_m2", "median"),
            mediana_antiguedad=("antiguedad_anios", "median")
        ).fillna(0)

        # Escalamiento de las características distritales
        X_scaled = self.scaler.fit_transform(stats)
        n_muestras = len(stats)
        max_k_posible = min(7, n_muestras - 1)

        # Búsqueda / Validación de k
        if self.n_clusters == "auto":
            mejor_k = 4 if max_k_posible >= 4 else 2
            mejor_sil = -1.0
            for k in range(2, max_k_posible + 1):
                km_eval = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
                labels_eval = km_eval.fit_predict(X_scaled)
                sil = silhouette_score(X_scaled, labels_eval)
                db = davies_bouldin_score(X_scaled, labels_eval)
                self.evaluacion_k_[k] = {
                    "inercia": km_eval.inertia_,
                    "silhouette": sil,
                    "davies_bouldin": db
                }
                if sil > mejor_sil and k >= 3:
                    mejor_sil = sil
                    mejor_k = k
            self.k_seleccionado_ = mejor_k
        else:
            self.k_seleccionado_ = min(int(self.n_clusters), max_k_posible)
            for k in range(2, max_k_posible + 1):
                km_eval = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
                labels_eval = km_eval.fit_predict(X_scaled)
                self.evaluacion_k_[k] = {
                    "inercia": km_eval.inertia_,
                    "silhouette": silhouette_score(X_scaled, labels_eval),
                    "davies_bouldin": davies_bouldin_score(X_scaled, labels_eval)
                }

        self.kmeans = KMeans(n_clusters=self.k_seleccionado_, random_state=self.random_state, n_init=20)
        self.kmeans.fit(X_scaled)

        # Asignación de etiquetas descriptivas según valor de m2 promedio del cluster
        stats["cluster_raw"] = self.kmeans.labels_
        orden_clusters = stats.groupby("cluster_raw")["mediana_m2"].mean().sort_values(ascending=False).index.tolist()
        
        # Nombres interpretativos según perfil socioeconómico y valor de suelo
        nombres_semanticos = ["Lima_Top", "Lima_Moderna", "Lima_Centro_Tradicional", "Lima_Periferia_Norte"]
        if len(orden_clusters) == 4:
            mapa_orden = {raw: nombres_semanticos[i] for i, raw in enumerate(orden_clusters)}
        else:
            mapa_orden = {raw: f"Submercado_{i+1}" for i, raw in enumerate(orden_clusters)}
        stats["cluster"] = stats["cluster_raw"].map(mapa_orden)
        
        self.mapa_distrito_cluster_ = stats["cluster"].to_dict()
        self.perfil_clusters_ = stats.groupby("cluster")[["mediana_m2", "mediana_superficie", "mediana_antiguedad"]].mean().sort_values(by="mediana_m2", ascending=False)
        self.cluster_defecto_ = stats["cluster"].value_counts().idxmax()

        # Perfil inmobiliario enriquecido para interpretación exhaustiva de submercados
        df["_submercado"] = df["distrito"].map(self.mapa_distrito_cluster_)
        agg_enriquecido = {
            "n_distritos": ("distrito", "nunique"),
            "n_anuncios": ("_precio", "count"),
            "pct_mercado": ("_precio", lambda x: round(len(x) / len(df) * 100, 1)),
            "mediana_precio_usd": ("_precio", "median"),
            "mediana_m2": ("_precio_m2", "median"),
            "iqr_m2": ("_precio_m2", lambda x: round(x.quantile(0.75) - x.quantile(0.25), 1)),
            "mediana_superficie": ("superficie_m2", "median"),
            "mediana_antiguedad": ("antiguedad_anios", "median")
        }
        if "habitaciones" in df.columns:
            agg_enriquecido["mediana_habitaciones"] = ("habitaciones", "median")
        if "banos" in df.columns:
            agg_enriquecido["mediana_banos"] = ("banos", "median")
        if "garajes" in df.columns:
            agg_enriquecido["mediana_garajes"] = ("garajes", "median")
        agg_enriquecido["distritos"] = ("distrito", lambda x: ", ".join(sorted(x.unique())))

        self.perfil_clusters_enriquecido_ = df.groupby("_submercado").agg(**agg_enriquecido).sort_values(by="mediana_m2", ascending=False)
        self.perfil_clusters_enriquecido_.index.name = "submercado"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out["submercado_cluster"] = X_out["distrito"].map(
            self.mapa_distrito_cluster_
        ).fillna(self.cluster_defecto_)

        for col in ["_precio", "_precio_m2"]:
            if col in X_out.columns:
                X_out = X_out.drop(columns=[col])

        return X_out


class GeneradorCaracteristicasInmobiliarias(BaseEstimator, TransformerMixin):
    """Generador de características de ingeniería inmobiliaria con CERO data leakage."""
    
    def __init__(self):
        self.medianas_sup_distrito_ = {}
        self.mediana_sup_global_ = 85.0

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        self.mediana_sup_global_ = float(df["superficie_m2"].median())
        self.medianas_sup_distrito_ = df.groupby("distrito")["superficie_m2"].median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_out = X.copy()
        
        habs = np.maximum(df_out["habitaciones"].fillna(1), 1)
        banos = df_out["banos"].fillna(1)
        superficie = np.maximum(df_out["superficie_m2"].fillna(self.mediana_sup_global_), 10)
        
        df_out["ratio_banos_habitacion"] = banos / habs
        df_out["m2_por_habitacion"] = superficie / habs
        
        condiciones = [
            (superficie <= 45) & (habs <= 1),
            (superficie <= 110) & (habs <= 3),
            (superficie <= 180),
            (superficie > 180)
        ]
        etiquetas = [
            'Estudio_Compacto',
            'Familiar_Estandar',
            'Familiar_Amplio',
            'Premium_Penthouse'
        ]
        df_out["tipologia"] = np.select(condiciones, etiquetas, default='Familiar_Estandar')
        
        med_dist = df_out["distrito"].map(self.medianas_sup_distrito_).fillna(self.mediana_sup_global_)
        df_out["ratio_sup_distrital"] = superficie / med_dist
        
        return df_out


def generar_features_ingenieria(df: pd.DataFrame, generador: GeneradorCaracteristicasInmobiliarias = None) -> pd.DataFrame:
    if generador is None:
        gen = GeneradorCaracteristicasInmobiliarias().fit(df)
        return gen.transform(df)
    return generador.transform(df)
