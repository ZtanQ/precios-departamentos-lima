# Estimación del precio de departamentos en Lima Metropolitana

Proyecto Integrador de Data Science del curso **Data Mining Tools (CC209)**, UPC, 2026.
Docente: Carlos Fernando Montoya Cubas.

## Problema

En Lima, los departamentos se anuncian en dólares y su precio se fija mirando otros anuncios del mismo distrito. Este proyecto busca estimar el **precio de oferta en dólares** de un departamento a partir de su distrito, metraje, dormitorios, baños, cocheras, antigüedad y trimestre de publicación.

- **Tipo de problema:** regresión supervisada.
- **Unidad de análisis:** un anuncio de venta de un departamento en un trimestre.
- **Referencia a superar:** la regla de mercado "mediana US$/m² del distrito × metraje", además de un DummyRegressor.

## Datos

| Campo | Valor |
| --- | --- |
| Fuente | Banco Central de Reserva del Perú (BCRP), base desagregada de precios de venta de departamentos |
| Archivo original | `precios-inmobiliarios-bd-desagregada-venta-2026-1.xlsx`, hoja "Venta" |
| Tamaño | 119,980 anuncios, 16 columnas |
| Período | 1998-T1 a 2026-T1, trimestral |
| Tipo de precio | Precio de oferta (anuncio), no precio final de venta |

- **Enlace de descarga oficial:** [BCRP - Indicador de Precios de Venta de Departamentos](https://www.bcrp.gob.pe/estadisticas/indicador-de-precios-de-venta-de-departamentos.html)
- **Condiciones de uso y procedencia:** Portal de series estadísticas abiertas del Banco Central de Reserva del Perú (BCRPData). Los datos son de libre acceso con fines académicos, de estudio e investigación económica, citando la atribución institucional correspondiente. Metodología oficial disponible en [BCRP, Documento de Trabajo 006-2018](https://www.bcrp.gob.pe/docs/Publicaciones/Documentos-de-Trabajo/2018/documento-de-trabajo-006-2018.pdf) y [BCRP, Nota de Estudios 65-2025](https://www.bcrp.gob.pe/docs/Publicaciones/Notas-Estudios/2025/nota-de-estudios-65-2025.pdf).

### Sobre los archivos de datos

- `data/raw/precios_venta_bcrp_2026_1.parquet` está incluido en el repositorio. Es el Excel original convertido a Parquet, **sin ningún cambio en los datos** (1.9 MB en lugar de 65 MB).
- El Excel original no se sube a GitHub por su tamaño. Si el BCRP publica una versión nueva, colócala en `data/raw/` y ejecuta `python scripts/convertir_excel_a_parquet.py`.

## Estructura del repositorio

```
precios-departamentos-lima/
├── data/
│   ├── raw/          # datos originales (Parquet incluido; Excel fuera de Git)
│   └── processed/    # datos limpios generados por los notebooks (fuera de Git)
├── notebooks/
│   ├── 01_problema_y_datos.ipynb       # problema, dataset, reglas de limpieza, baseline de mercado
│   ├── 02_eda.ipynb                    # análisis exploratorio detallado
│   ├── 03_preparacion_y_modelos.ipynb  # limpieza, pipeline, modelos y evaluación
│   └── valoracion_inmobiliaria.ipynb   # NOTEBOOK CONSOLIDADO FINAL con todo el flujo integrado
├── src/
│   ├── config.py     # rutas y decisiones compartidas (año de corte, semilla, columnas)
│   ├── datos.py      # cargar_datos(): carga única para todo el equipo
│   ├── limpieza.py   # reglas de limpieza y transformadores (FiltroIQRDistrito, normalización)
│   ├── baseline.py   # regla de mercado (distrito × m²) como estimador de scikit-learn
│   ├── metricas.py   # métricas comunes para comparar todos los modelos (MAE, RMSE, MedAPE, MAPE)
│   └── modelos.py    # definición y pipelines de modelos (Ridge, ElasticNet, RF, CatBoost)
├── scripts/
│   └── convertir_excel_a_parquet.py
├── models/           # modelos entrenados (fuera de Git)
├── reports/
│   ├── figures/      # gráficos exportados para el informe y la presentación
│   └── informe_tp1.tex # informe formal del TP1 en LaTeX
├── requirements.txt
└── README.md
```

## Instalación

Requiere Python 3.10 a 3.13.

```bash
git clone <URL-del-repositorio>
cd precios-departamentos-lima

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecución

```bash
jupyter notebook
```

Abrir los notebooks de la carpeta `notebooks/` en orden (01, 02, 03) o directamente el notebook unificado `valoracion_inmobiliaria.ipynb`. Cada uno carga los datos con:

```python
from src.datos import cargar_datos
df = cargar_datos()
```

## Equipo

| Integrante | Rol / Bloque |
| --- | --- |
| Gabriel Alonzo Reyna Alvarado | Problema, dataset, reglas de limpieza, baseline de mercado y matriz de herramientas |
| Guido Yair Abel Jeri Saldaña | Análisis exploratorio de datos (EDA) e interpretaciones |
| José Guillermo Melgar Puertas | Limpieza, separación de datos out-of-time, pipeline, modelos y evaluación preliminar |
| Piero Marcos Contreras Albornoz | Integración del repositorio, notebook final, plan hacia el TF1 y presentación |

## Cómo trabajamos con Git

1. Antes de empezar a trabajar: `git pull`.
2. Cada uno trabaja en **su** notebook. Si dos personas editan el mismo notebook a la vez, Git no puede unir los cambios.
3. Antes de subir: en Jupyter, *Kernel > Restart & Run All*, para que el notebook quede ejecutado de principio a fin.
4. Subir los cambios:
   ```bash
   git add .
   git commit -m "Descripción corta del cambio"
   git push
   ```
5. Las decisiones compartidas (año de corte, columnas descartadas) se cambian solo en `src/config.py`, y se avisa al grupo.

## Estado del TP1

- [x] Estructura del repositorio y carga de datos
- [x] Definición del problema, ficha técnica, reglas de limpieza propuestas y baseline de mercado (notebook 01)
- [x] EDA exhaustivo con interpretaciones diferenciadas (notebook 02 y consolidado)
- [x] Reglas de limpieza acordadas e implementadas en `src/limpieza.py` (deduplicación, normalización distrital, filtro IQR)
- [x] Pipeline reproducible con `ColumnTransformer`, baselines y modelos en `src/modelos.py` (notebook 03)
- [x] Notebook final consolidado (`notebooks/valoracion_inmobiliaria.ipynb`)
- [x] Informe escrito formal en LaTeX (`reports/informe_tp1.tex`)
- [x] Plan de trabajo hacia el TF1 documentado
- [ ] Presentación (PDF o PPTX)

## Decisiones de herramientas

Cada herramienta se eligió por un rasgo concreto de nuestros datos. Se actualiza hacia el TF1.

| Necesidad | Etapa | Herramienta elegida | Alternativa considerada | Justificación |
| --- | --- | --- | --- | --- |
| Carga y manipulación | TP1 | pandas | Excel / Power Query | Son 120 mil filas con reglas reproducibles. En Excel las correcciones son manuales y no quedan versionadas |
| EDA | TP1 | Matplotlib + Seaborn | ydata-profiling (reporte automático) | El EDA gira alrededor del tiempo y del distrito. Un reporte automático no detecta que piso = 0 depende del año |
| Preparación | TP1 | scikit-learn: Pipeline + ColumnTransformer + SimpleImputer + RobustScaler + OneHotEncoder | Limpieza manual en pandas antes de separar | El pipeline ajusta imputación, escalado y codificación solo con el entrenamiento, evitando fuga de datos |
| Transformación del objetivo | TP1 | TransformedTargetRegressor (logaritmo natural del precio) | Precio sin transformar | El precio tiene fuerte asimetría positiva. En escala logarítmica el modelo optimiza errores porcentuales/relativos |
| Baselines | TP1 | DummyRegressor (media y mediana) + regla de mercado (`src/baseline.py`) | Solo DummyRegressor | La regla de mercado (14.9% error mediano) es la referencia empírica real de los tasadores a superar |
| Modelamiento | TP1 | Ridge, ElasticNet, Random Forest y CatBoost Regressor (`src/modelos.py`) | Redes neuronales / AutoML | CatBoost lideró con 11.3% de error mediano, capturando no linealidades espaciales complejas sin sobreajuste |
| Experimentación | TF1 | Optuna + validación temporal (TimeSeriesSplit) + MLflow | GridSearchCV con validación aleatoria | Optuna explora eficientemente hiperparámetros respetando el orden cronológico estricto |
| Interpretabilidad | TF1 | SHAP (TreeSHAP) | Importancia nativa de variables MDI | SHAP explica cada predicción localmente en dólares reales, aportando transparencia al usuario |
| Despliegue | TF1 | Streamlit | FastAPI | Permite que un usuario final ingrese los datos del departamento e interactúe con la tasación e intervalos en tiempo real |
| Reproducibilidad | TP1 y TF1 | Git + requirements.txt + LaTeX | Carpeta compartida en Drive | Permite versionado colaborativo y redacción académica profesional reproducible |
