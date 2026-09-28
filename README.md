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

El BCRP recopila estos precios desde 1998, primero de anuncios en periódicos y hoy del portal Urbania. Metodología: [BCRP, Documento de Trabajo 006-2018](https://www.bcrp.gob.pe/docs/Publicaciones/Documentos-de-Trabajo/2018/documento-de-trabajo-006-2018.pdf) y [BCRP, Nota de Estudios 65-2025](https://www.bcrp.gob.pe/docs/Publicaciones/Notas-Estudios/2025/nota-de-estudios-65-2025.pdf).

**Pendiente:** agregar el enlace exacto de descarga del Excel y las condiciones de uso de BCRPData.

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
│   ├── 02_eda.ipynb                    # análisis exploratorio
│   └── 03_preparacion_y_modelos.ipynb  # limpieza, pipeline, modelos y evaluación
├── src/
│   ├── config.py     # rutas y decisiones compartidas (año de corte, semilla, columnas)
│   └── datos.py      # cargar_datos(): carga única para todo el equipo
├── scripts/
│   └── convertir_excel_a_parquet.py
├── models/           # modelos entrenados (fuera de Git)
├── reports/figures/  # gráficos exportados para la presentación
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

Abrir los notebooks de la carpeta `notebooks/` en orden (01, 02, 03). Cada uno carga los datos con:

```python
from src.datos import cargar_datos
df = cargar_datos()
```

## Equipo

| Integrante | Bloque |
| --- | --- |
| Gabriel Reyna | Problema, dataset, reglas de limpieza, baseline de mercado y matriz de herramientas |
| Yair | Análisis exploratorio (EDA) e interpretaciones |
| José | Limpieza, separación de datos, pipeline, modelos y evaluación |
| Marcos | Integración del repositorio, notebook final, plan hacia el TF1 y presentación |

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
- [ ] Definición del problema y ficha técnica (notebook 01)
- [ ] EDA con interpretaciones (notebook 02)
- [ ] Reglas de limpieza acordadas por el equipo
- [ ] Pipeline, baselines y modelos (notebook 03)
- [ ] Notebook final consolidado
- [ ] Plan hacia el TF1
- [ ] Presentación (PDF o PPTX)

## Uso de herramientas de IA generativa

Según la sección 10 de la guía del curso. **Pendiente:** completar al cierre del TP1 indicando para qué se usó (por ejemplo, revisión del dataset, estructura del repositorio o redacción de documentación).
