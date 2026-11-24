"""Plataforma Interactiva de Valuación y Tasación Inmobiliaria en Lima Metropolitana.

Desarrollada para el Trabajo Final (TF1) - CC209 Data Mining Tools.
Arquitectura: Ensamble Ponderado Calibrado sobre US$/m² (CatBoost + RF + Ridge + LightGBM + XGBoost).
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Configuración de página con diseño profesional tipo Dashboard
st.set_page_config(
    page_title="Tasador Inteligente | Lima Metropolitana",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS sutiles y limpios para estética de producto financiero/proptech
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 12px;
    }
    .metric-card-top {
        border-left-color: #1f77b4;
    }
    .metric-card-moderna {
        border-left-color: #2ca02c;
    }
    .metric-card-centro {
        border-left-color: #ff7f0e;
    }
    .metric-card-norte {
        border-left-color: #d62728;
    }
    .tag-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        background-color: #e9ecef;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)


# 2. Carga optimizada de artefactos en caché
@st.cache_resource(show_spinner="Iniciando motor de tasación...")
def cargar_motor():
    ruta_artefacto = RAIZ / "models" / "modelo_inmobiliario_tf1.joblib"
    if not ruta_artefacto.exists():
        st.error(f"No se encontró el artefacto del modelo en {ruta_artefacto}")
        st.stop()
    return joblib.load(ruta_artefacto)

motor = cargar_motor()
ensamble = motor["ensamble"]
clusterer = motor["clusterer"]
gen_features = motor["generador_features"]
preprocesador = motor["preprocesador"]
catboost_model = motor["catboost_model"]
pesos = motor["pesos"]

DISTRITOS = sorted(list(clusterer.mapa_distrito_cluster_.keys()))

# 3. Sidebar: Interfaz de Configuración del Inmueble
st.sidebar.image("https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?auto=format&fit=crop&w=600&q=80", use_container_width=True)
st.sidebar.markdown("### 📍 Ubicación y Tipología")

distrito_sel = st.sidebar.selectbox("Distrito", DISTRITOS, index=DISTRITOS.index("Miraflores") if "Miraflores" in DISTRITOS else 0)
superficie_sel = st.sidebar.slider("Superficie Total (m²)", min_value=25, max_value=350, value=90, step=5)

st.sidebar.markdown("### 🛋️ Distribución y Ambientes")
col_hab, col_ban = st.sidebar.columns(2)
with col_hab:
    habitaciones_sel = st.number_input("Dormitorios", min_value=1, max_value=6, value=3, step=1)
with col_ban:
    banos_sel = st.number_input("Baños", min_value=1.0, max_value=6.0, value=2.0, step=0.5)

col_gar, col_ant = st.sidebar.columns(2)
with col_gar:
    garajes_sel = st.selectbox("Cocheras", options=[0, 1, 2, 3, 4], index=1)
with col_ant:
    antiguedad_sel = st.number_input("Años de Antigüedad", min_value=0, max_value=60, value=5, step=1)

# Construcción de DataFrame de entrada para inferencia
df_inmueble = pd.DataFrame([{
    "distrito": distrito_sel,
    "superficie_m2": float(superficie_sel),
    "habitaciones": float(habitaciones_sel),
    "banos": float(banos_sel),
    "garajes": float(garajes_sel),
    "antiguedad_anios": float(antiguedad_sel)
}])

# Pipeline dinámico de transformación sin data leakage
df_inmueble_eng = gen_features.transform(clusterer.transform(df_inmueble))

# Inferencia del ensamble en tiempo real
pred_usd = float(ensamble.predict(df_inmueble_eng)[0])
pred_m2 = float(ensamble.predict_m2(df_inmueble_eng)[0])

submercado = df_inmueble_eng["submercado_cluster"].iloc[0]
tipologia = df_inmueble_eng["tipologia"].iloc[0]
ratio_ban_hab = df_inmueble_eng["ratio_banos_habitacion"].iloc[0]
m2_hab = df_inmueble_eng["m2_por_habitacion"].iloc[0]
ratio_sup_dist = df_inmueble_eng["ratio_sup_distrital"].iloc[0]

# Intervalo comercial empírico validado en Test Ciego (MedAPE = 11.11%)
error_medape = 0.1111
usd_min = pred_usd * (1 - error_medape)
usd_max = pred_usd * (1 + error_medape)
m2_min = pred_m2 * (1 - error_medape)
m2_max = pred_m2 * (1 + error_medape)

# 4. Vista Principal: Panel de Tasación y Cotización
st.title("🏙️ Sistema de Tasación Inmobiliaria Inteligente")
st.markdown("Valuación hedónica contemporánea de departamentos en Lima Metropolitana basada en un Ensamble Ponderado Calibrado con Optuna.")

# Fila 1: Indicadores clave de Tasación
col_main_1, col_main_2, col_main_3 = st.columns([1.4, 1.3, 1.3])

with col_main_1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%); padding: 22px; border-radius: 12px; color: white;">
        <span style="font-size: 13px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">Valor de Tasación Estimado</span>
        <h1 style="margin: 6px 0 2px 0; font-size: 38px; color: white; font-weight: 700;">US$ {:,.0f}</h1>
        <span style="font-size: 13px; opacity: 0.95;">Banda comercial (±11.1%): <b>${:,.0f} - ${:,.0f}</b></span>
    </div>
    """.format(pred_usd, usd_min, usd_max), unsafe_allow_html=True)

with col_main_2:
    st.markdown("""
    <div style="background: white; border: 1px solid #e0e0e0; padding: 22px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
        <span style="font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">Valor Unitario por Metro Cuadrado</span>
        <h2 style="margin: 6px 0 2px 0; font-size: 32px; color: #2c3e50; font-weight: 700;">US$ {:,.0f} / m²</h2>
        <span style="font-size: 13px; color: #555;">Rango esperado: <b>${:,.0f} - ${:,.0f} / m²</b></span>
    </div>
    """.format(pred_m2, m2_min, m2_max), unsafe_allow_html=True)

with col_main_3:
    color_submercado = {
        "Lima_Top": "#1f77b4",
        "Lima_Moderna": "#2ca02c",
        "Lima_Centro_Tradicional": "#ff7f0e",
        "Lima_Periferia_Norte": "#d62728"
    }.get(submercado, "#333")

    st.markdown(f"""
    <div style="background: white; border: 1px solid #e0e0e0; padding: 22px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
        <span style="font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">Segmentación Urbana y Producto</span>
        <div style="margin: 8px 0;">
            <span style="background-color: {color_submercado}20; color: {color_submercado}; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 14px;">
                {submercado.replace('_', ' ')}
            </span>
        </div>
        <span style="font-size: 13px; color: #555;">Tipología: <b>{tipologia.replace('_', ' ')}</b></span><br>
        <span style="font-size: 12px; color: #777;">Amplitud: <b>{m2_hab:.1f} m²/hab</b> · Ratio suite: <b>{ratio_ban_hab:.2f} bñ/hab</b></span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 5. Pestañas de Análisis Especializado
tab_shap, tab_comparador, tab_mercado, tab_tecnica = st.tabs([
    "🔍 Descomposición Hedónica (SHAP)",
    "⚖️ Análisis de Sensibilidad y Simulador",
    "🏙️ Radiografía del Submercado",
    "📐 Arquitectura y Benchmarking"
])

# PESTAÑA 1: Explicabilidad SHAP en tiempo real
with tab_shap:
    st.subheader("Auditoría Explicable: ¿Cómo se forma el precio?")
    st.markdown(
        "Utilizando teoría de juegos cooperativos (**TreeSHAP**), descomponemos el valor unitario por $m^2$ "
        "revelando con precisión cuánta prima o penalización aporta cada atributo individual respecto al valor base del mercado."
    )
    
    # Transformación del inmueble para SHAP
    X_input_trans = preprocesador.transform(df_inmueble_eng)
    feature_names = preprocesador.get_feature_names_out()
    clean_feature_names = [
        name.replace("num__", "").replace("cat__", "")
            .replace("distrito_", "Distrito: ")
            .replace("submercado_cluster_", "Submercado: ")
            .replace("tipologia_", "Tipología: ")
        for name in feature_names
    ]
    
    explainer = shap.TreeExplainer(catboost_model)
    shap_vals = explainer(X_input_trans)
    shap_vals.feature_names = clean_feature_names
    
    col_plot, col_interp = st.columns([1.6, 1.0])
    
    with col_plot:
        fig_shap, ax = plt.subplots(figsize=(9, 4.8))
        shap.plots.waterfall(shap_vals[0], max_display=8, show=False)
        plt.title(f"Aportes Hedónicos Locales en {distrito_sel} ({superficie_sel} m²)", fontsize=11, fontweight="bold", pad=8)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()
        
    with col_interp:
        st.markdown("#### Factores Más Determinantes")
        st.markdown(f"""
        - **Localización ({distrito_sel}):** Se ubica en el macro-submercado **{submercado.replace('_', ' ')}**, aportando un efecto geográfico dominante sobre el precio de suelo.
        - **Amplitud Habitable:** Con **{m2_hab:.1f} m² por habitación**, clasifica como *{tipologia.replace('_', ' ')}*.
        - **Antigüedad:** Inmueble con **{antiguedad_sel} años**, aplicando una depreciación física estándar sobre el valor unitario.
        - **Confort Sanitario:** Proporción de **{ratio_ban_hab:.1f} baños por habitación**, capturando el valor agregado de baños incorporados.
        """)

# PESTAÑA 2: Simulador de Sensibilidad
with tab_comparador:
    st.subheader("Simulador de Sensibilidad: ¿Cuánto cambia el valor si varía el inmueble?")
    st.markdown("Proyección instantánea del precio estimado alterando características clave sobre este mismo departamento:")
    
    col_sim_1, col_sim_2, col_sim_3 = st.columns(3)
    
    # Simulación 1: +1 Dormitorio
    df_sim_hab = df_inmueble.copy()
    df_sim_hab["habitaciones"] = float(habitaciones_sel + 1)
    df_sim_hab_eng = gen_features.transform(clusterer.transform(df_sim_hab))
    p_sim_hab = float(ensamble.predict(df_sim_hab_eng)[0])
    diff_hab = p_sim_hab - pred_usd
    
    # Simulación 2: +1 Cochera
    df_sim_gar = df_inmueble.copy()
    df_sim_gar["garajes"] = float(garajes_sel + 1)
    df_sim_gar_eng = gen_features.transform(clusterer.transform(df_sim_gar))
    p_sim_gar = float(ensamble.predict(df_sim_gar_eng)[0])
    diff_gar = p_sim_gar - pred_usd
    
    # Simulación 3: Departamento Estreno (0 años)
    df_sim_estreno = df_inmueble.copy()
    df_sim_estreno["antiguedad_anios"] = 0.0
    df_sim_estreno_eng = gen_features.transform(clusterer.transform(df_sim_estreno))
    p_sim_estreno = float(ensamble.predict(df_sim_estreno_eng)[0])
    diff_estreno = p_sim_estreno - pred_usd
    
    with col_sim_1:
        st.metric(
            label="Si tuviera +1 Dormitorio",
            value=f"US$ {p_sim_hab:,.0f}",
            delta=f"{'+' if diff_hab >= 0 else ''}${diff_hab:,.0f}"
        )
        st.caption(f"De {habitaciones_sel} a {habitaciones_sel + 1} dorms ({m2_hab:.1f} → {superficie_sel/(habitaciones_sel+1):.1f} m²/hab)")
        
    with col_sim_2:
        st.metric(
            label="Si tuviera +1 Cochera",
            value=f"US$ {p_sim_gar:,.0f}",
            delta=f"{'+' if diff_gar >= 0 else ''}${diff_gar:,.0f}"
        )
        st.caption(f"De {garajes_sel} a {garajes_sel + 1} estacionamiento(s) privado(s)")
        
    with col_sim_3:
        st.metric(
            label="Si fuera de Estreno (0 años)",
            value=f"US$ {p_sim_estreno:,.0f}",
            delta=f"{'+' if diff_estreno >= 0 else ''}${diff_estreno:,.0f}"
        )
        st.caption(f"Eliminando la depreciación de {antiguedad_sel} años")

# PESTAÑA 3: Radiografía del Submercado
with tab_mercado:
    st.subheader(f"Macro-Submercado: {submercado.replace('_', ' ')}")
    st.markdown("Perfil estructural del submercado determinado de forma no supervisada por $K$-Means ($k=4$) sobre los datos del BCRP:")
    
    perfil_df = clusterer.perfil_clusters_enriquecido_.copy()
    display_perfil = perfil_df.rename(columns={
        "n_distritos": "N° Distritos",
        "n_anuncios": "Volumen Anuncios",
        "pct_mercado": "% Oferta de Mercado",
        "mediana_precio_usd": "Precio Mediano (USD)",
        "mediana_m2": "Mediana US$/m²",
        "iqr_m2": "Dispersión IQR (US$/m²)",
        "mediana_superficie": "Superficie Mediana",
        "mediana_antiguedad": "Antigüedad Mediana",
        "distritos": "Distritos Integrantes"
    })
    
    st.dataframe(display_perfil, use_container_width=True)
    
    # Resaltar distritos del mismo submercado
    distritos_grupo = [d.strip() for d in perfil_df.loc[submercado, "distritos"].split(",")]
    st.info(f"**Distritos que comparten este mismo perfil socio-espacial:** {', '.join(distritos_grupo)}")

# PESTAÑA 4: Benchmarking Técnico y Arquitectura
with tab_tecnica:
    st.subheader("Benchmark de Modelos y Calibración del Ensamble")
    st.markdown("Evaluación comparativa rigurosa en el **Conjunto de Prueba Ciego Temporal (2024–2026, N=25,042 anuncios)**:")
    
    tabla_bench = pd.DataFrame([
        {"Modelo": "Dummy Baseline (Mediana)", "MAE (US$)": "$54,905", "RMSE (US$)": "$73,539", "MedAPE": "28.89%", "Acierto <= 10%": "18.6%"},
        {"Modelo": "Regla Empírica de Mercado (Tasador)", "MAE (US$)": "$28,585", "RMSE (US$)": "$38,322", "MedAPE": "14.90%", "Acierto <= 10%": "34.5%"},
        {"Modelo": "Ridge Regressor (sobre m²)", "MAE (US$)": "$24,142", "RMSE (US$)": "$33,531", "MedAPE": "12.31%", "Acierto <= 10%": "41.6%"},
        {"Modelo": "Random Forest (sobre m²)", "MAE (US$)": "$23,092", "RMSE (US$)": "$32,675", "MedAPE": "11.28%", "Acierto <= 10%": "45.5%"},
        {"Modelo": "LightGBM (sobre m²)", "MAE (US$)": "$22,992", "RMSE (US$)": "$32,518", "MedAPE": "11.33%", "Acierto <= 10%": "45.0%"},
        {"Modelo": "XGBoost (sobre m²)", "MAE (US$)": "$22,946", "RMSE (US$)": "$32,465", "MedAPE": "11.27%", "Acierto <= 10%": "45.0%"},
        {"Modelo": "CatBoost Optimizado (sobre m²)", "MAE (US$)": "$22,722", "RMSE (US$)": "$32,212", "MedAPE": "11.21%", "Acierto <= 10%": "45.6%"},
        {"Modelo": "Ensamble Heterogéneo TF1 (5 Modelos)", "MAE (US$)": "$22,650", "RMSE (US$)": "$32,063", "MedAPE": "11.11%", "Acierto <= 10%": "45.77%"}
    ])
    st.dataframe(tabla_bench, use_container_width=True, hide_index=True)
    
    st.markdown("#### Ponderaciones del Ensamble Calibradas con Optuna")
    col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)
    col_w1.metric("CatBoost (Simétrico)", f"{pesos.get('w_cb', 0)*100:.1f}%")
    col_w2.metric("XGBoost (Hessian)", f"{pesos.get('w_xgb', 0)*100:.1f}%")
    col_w3.metric("LightGBM (Leaf-wise)", f"{pesos.get('w_lgb', 0)*100:.1f}%")
    col_w4.metric("Random Forest (Bagging)", f"{pesos.get('w_rf', 0)*100:.1f}%")
    col_w5.metric("Ridge (L2 Lineal)", f"{pesos.get('w_rd', 0)*100:.1f}%")

st.divider()
st.caption("Sistema de Estimación Inmobiliaria para Lima Metropolitana · Modelo TF1 entrenado sobre registros BCRP (2022–2026).")
