# Estimación y Tasación Inteligente de Departamentos en Lima Metropolitana

Proyecto Integrador de Data Science – **Trabajo Final (TF1)**  
Curso: **Data Mining Tools (CC209)**, UPC, 2026.  
Docente: **Carlos Fernando Montoya Cubas**

---

## 1. Resumen Ejecutivo y Problema

En Lima Metropolitana, los departamentos se anuncian en dólares y su precio de oferta se fija tradicionalmente mediante reglas heurísticas del mercado inmobiliario (mediana de US$/m² del distrito × metraje).

Este proyecto desarrolla una **solución de Data Science de extremo a extremo** para estimar el precio de lista de un departamento residencial a partir de sus atributos físicos y su localización espacial.

### Evolución Metodológica del TP1 al TF1
Frente al Trabajo Parcial (TP1), el Trabajo Final (TF1) implementa cuatro saltos metodológicos sustanciales:
1. **Modelado del Precio Unitario:** En lugar de predecir directamente el precio total (`precio_usd`), los modelos estiman el **precio unitario por metro cuadrado** (`precio_m2 = precio_usd / superficie_m2`) y reconstruyen el precio total en dólares. Esto desacopla el efecto del tamaño y estabiliza la varianza residual a lo largo de todas las tipologías.
2. **Técnica de la 2da Mitad del Curso:** Incorporación de **Clustering Espacial de Submercados** ($K$-Means, $k=4$) e ingeniería de habitabilidad (`m2_por_habitacion`, `ratio_banos_habitacion`, `tipologia`, `ratio_sup_distrital`) con CERO data leakage.
3. **Experimentación Sistemática con Optuna:** Partición triple *out-of-time* (Train Selección 2022-T1–2023-T2, Val 2023-T3/T4, Test Ciego 2024-T1–2026-T1) y optimización bayesiana de un **Ensamble Ponderado** (CatBoost 87.9%, Random Forest 10.4%, Ridge 1.7%).
4. **Despliegue Funcional con Interpretabilidad:** Aplicación web interactiva en **Streamlit** (`app.py`) conectada al pipeline serializado en `.joblib`, con visualización de rangos comerciales de tolerancia ($\pm 11.1\%$) y explicaciones locales **TreeSHAP (Waterfall plot)** por departamento cotizado.

---

## 2. Resultados Experimentales en Test Ciego (2024–2026, $N=25,042$)

Evaluación imparcial en horizonte temporal futuro no visto:

| Modelo / Enfoque | MAE (US$) | RMSE (US$) | MedAPE | % Error $\le 10\%$ |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy (Mediana TP1)** | \$54,905 | \$73,539 | 28.89% | 18.63% |
| **Regla de Mercado (Tasador)** | \$28,585 | \$38,322 | 14.90% | 34.49% |
| **Ridge Regressor (sobre m²)** | \$24,142 | \$33,531 | 12.31% | 41.63% |
| **Random Forest (sobre m²)** | \$23,099 | \$32,679 | 11.25% | 45.39% |
| **CatBoost Optimizado (sobre m²)** | \$22,701 | \$32,089 | 11.16% | 45.52% |
| **Ensamble Ponderado TF1 (Ganador)** | **\$22,647** | **\$32,010** | **11.13%** | **45.63%** |

*El Ensamble Ponderado bate con holgura a la regla tradicional del mercado inmobiliario (reducción de 3.77 puntos en MedAPE) y logra que el 45.63% de las tasaciones presenten un error $\le 10\%$.*

---

## 3. Matriz de Decisiones de Herramientas

| Necesidad | Herramienta Elegida | Alternativa Considerada | Justificación Técnica de la Elección |
| :--- | :--- | :--- | :--- |
| **Pipeline & Preprocesamiento** | `scikit-learn` (`Pipeline`, `ColumnTransformer`) | Limpieza manual con pandas | Asegura que imputación, escalado robusto y codificación se ajusten exclusivamente con el conjunto de entrenamiento, previniendo fuga de datos. |
| **Técnica 2da Mitad** | `K-Means Clustering` ($k=4$) | Redes Neuronales Profundas (MLP) | Segmenta distritos en clusters socio-inmobiliarios homogéneos e interpretables (Lima Top, Moderna, Centro, Periferia); un MLP aumentaría la complejidad y opacidad innecesariamente. |
| **Optimización de Hiperparámetros** | `Optuna` | `GridSearchCV` / `RandomizedSearchCV` | Optimización bayesiana con TPE; converge más rápido a hiperparámetros óptimos y permite calibrar los pesos del ensamble sobre validación temporal. |
| **Modelos del Ensamble** | CatBoost + Random Forest + Ridge | Modelos individuales aislados | La combinación de gradiente boosted, árboles no lineales y regularización lineal $L_2$ reduce el sesgo y la varianza, batiendo a cualquier estimador individual. |
| **Interpretabilidad** | `TreeSHAP` (`TreeExplainer`) | Permutation Feature Importance | Fundamentada en valores Shapley de teoría de juegos; provee aditividad en escala de dólares y genera explicaciones locales (*waterfall plots*) para cada departamento. |
| **Despliegue Funcional** | `Streamlit` | FastAPI + React | Permite desarrollar una interfaz interactiva y visual en Python para la demostración en vivo de 12 minutos requerida en la sustentación oral. |

---

## 4. Estructura del Repositorio

```
precios-departamentos-lima/
├── data/
│   ├── raw/precios_venta_bcrp_2026_1.parquet   # Datos oficiales BCRP (119,980 anuncios)
│   └── processed/base_modelado.parquet         # Base 2022-2026 (generada localmente)
├── models/
│   ├── modelo_inmobiliario_tf1.joblib          # Ensamble ponderado + preprocesador serializado
│   └── clusterer_tf1.joblib                    # Estimador K-Means para submercados
├── notebooks/                                  # Análisis exploratorio y prototipos
│   ├── 01_problema_y_datos.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preparacion_y_modelos.ipynb
│   ├── valoracion_inmobiliaria.ipynb           # Notebook Consolidado TP1
│   └── valoracion_inmobiliaria_tf1.ipynb       # NOTEBOOK CONSOLIDADO FINAL DEL TF1 (Ejecutado)
├── src/
│   ├── config.py                               # Rutas, constantes y particiones
│   ├── datos.py                                # Carga única de datos
│   ├── limpieza.py                             # Deduplicación y tratamiento IQR
│   ├── ingenieria.py                           # ClusteringSubmercado y ratios de habitabilidad
│   ├── validacion.py                           # Protocolo de partición temporal
│   ├── baseline.py                             # Estimador ReglaMercadoM2
│   ├── metricas.py                             # Cálculo de MAE, RMSE, MedAPE, MAPE
│   └── modelos.py                              # Definición de EnsamblePonderadoPrecioM2
├── scripts/
│   ├── preparar_validacion_temporal.py         # Genera base_modelado.parquet con máscaras
│   ├── ejecutar_experimentacion.py             # Script formal de Optuna y entrenamiento final
│   └── analizar_interpretabilidad_errores.py   # Script de TreeSHAP y análisis de 50 residuos
├── reports/
│   ├── report.tex                              # Informe escrito original del TP1
│   ├── report_tf1.tex                          # INFORME ESCRITO FINAL DEL TF1
│   ├── interpretabilidad_errores/figuras/      # Gráficos SHAP y diagnóstico residual
│   └── experimentos/registro_experimentos.csv  # Registro tabular de todas las corridas
├── app.py                                      # APLICACIÓN INTERACTIVA EN STREAMLIT
├── requirements.txt                            # Dependencias fijadas del proyecto
├── PLAN_TF1.md                                 # Plan técnico y tabla de cambios TP1 vs TF1
├── ESTADO_PROYECTO.md                          # Bitácora viva de control y verificación
└── README.md
```

---

## 5. Instalación y Ejecución Rápida

### Requisitos Previos
* Python 3.10 a 3.12.

```bash
# 1. Clonar el repositorio
git clone <URL-del-repositorio>
cd precios-departamentos-lima

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate       # En Linux/Mac
# .venv\Scripts\activate        # En Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Reproducción Completa del Pipeline
```bash
# Paso 1: Ejecutar pruebas unitarias
python -m unittest discover -s tests

# Paso 2: Generar datos y particiones out-of-time
python scripts/preparar_validacion_temporal.py

# Paso 3: Ejecutar optimización bayesiana y entrenar el ensamble final
python scripts/ejecutar_experimentacion.py

# Paso 4: Generar análisis SHAP y diagnóstico de residuos
python scripts/analizar_interpretabilidad_errores.py
```

### Iniciar la Aplicación Web (Streamlit)
```bash
streamlit run app.py
```
*Se abrirá la aplicación interactiva en `http://localhost:8501` para tasar cualquier inmueble en vivo con explicaciones locales SHAP.*

---

## 6. Equipo y Roles

| Integrante | Rol en el Trabajo Final (TF1) |
| :--- | :--- |
| **Gabriel Alonzo Reyna Alvarado** | Formulación de técnica adicional (K-Means Clustering), matriz de decisiones y evaluación de submercados. |
| **Guido Yair Abel Jeri Saldaña** | Interpretabilidad con TreeSHAP (global/local) y diagnóstico sistemático de casos extremos de error. |
| **José Guillermo Melgar Puertas** | Protocolo de partición temporal out-of-time, optimización con Optuna y calibración del Ensamble Ponderado. |
| **Piero Marcos Contreras Albornoz** | Desarrollo de la aplicación interactiva en Streamlit, serialización del pipeline y documentación final en LaTeX. |
