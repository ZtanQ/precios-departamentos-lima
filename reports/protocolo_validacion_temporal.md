# Protocolo de entrega: compañero 1

## Propósito y estado

Precio base de oferta para posterior ajuste profesional. Período de datos:
2022-T1–2026-T1, seleccionado por el equipo por pandemia y transición del mercado;
no implica que se haya probado estabilidad desde 2022.

Este documento define el procedimiento y recoge los conteos de la ejecución local
verificada en reports/validacion_temporal/. No se ha entrenado un modelo con este
nuevo protocolo; estos resultados verifican preparación y cobertura, no precisión.

La comparación ya ejecutada del consolidado conserva 2022–2023 como entrenamiento
y 2024–2026-T1 como prueba. Tras IQR usa 15156/25042 filas. Su tabla actualizada,
redondeada como la salida guardada, está en
`auditoria_preparacion/metricas_actualizadas.csv`. No confundir esa comparación
con resultados de la validación nueva.

## Ventanas estipuladas

| Función | Período | Uso permitido |
| --- | --- | --- |
| Entrenamiento de selección | 2022-T1–2023-T2 | Ajustar filtros, transformadores y modelo |
| Validación | 2023-T3 y 2023-T4 | Elegir variables, algoritmo e hiperparámetros |
| Reentrenamiento posterior | Todo 2022–2023 | Ajustar desde cero la configuración ya elegida |
| Prueba posterior | 2024-T1–2026-T1 | Medir la configuración final; no dirigir su selección |

La división usa trimestres completos y comprueba orden estricto, conjuntos no
vacíos e identificadores únicos. La cobertura distrital se genera para revisar
si estas ventanas tienen suficiente soporte antes de iniciar experimentos.

## Una base de trabajo

Ejecutar desde la raíz:

```powershell
.\.venv\Scripts\python.exe scripts/preparar_validacion_temporal.py
```

Reconstruye desde la fuente y sobrescribe la única base local
`data/processed/base_modelado.parquet`. No crea copias de datos train/val/test.
Incluye `fila_origen`, `particion` y máscaras de auditoría. Los identificadores
son posiciones en la fuente, no IDs de inmueble. Los cuatro pares que difieren
en piso se conservan aunque sus entradas de modelo coincidan.

`scripts/preparar_datos_actuales.py` genera la versión previa a estas etiquetas;
si se vuelve a ejecutar, ejecutar después el script de validación para recuperarlas.

## IQR y transformaciones aprendidas

Se preserva el IQR distrital factor 3 y soporte mínimo 10 de la comparación actual.
No se ejecuta sobre la base completa antes de dividir:

- `incluido_iqr_seleccion`: límites aprendidos solo en 2022-T1–2023-T2;
  aplicada a entrenamiento y validación. Es ausente en prueba intencionalmente.
- `incluido_iqr_final`: límites aprendidos en todo 2022–2023. Solo se usa para
  reentrenamiento y prueba después de fijar la configuración. No utilizarla para
  elegir hiperparámetros o filtrar el entrenamiento de selección.
- `n_distrito_train_seleccion/final`: soporte aprendido en cada entrenamiento.

Estas máscaras dependen del precio observado y delimitan una muestra condicionada.
No son transformaciones de entrada para inferencia. Reportar junto con las
métricas qué máscara y población se utilizaron; se permite analizar la muestra
completa sin IQR como sensibilidad sin duplicar la base.

Imputación de baños/antigüedad, escalado y OneHotEncoder se ajustan en el
`Pipeline` con entrenamiento. Tras seleccionar se clona/reconstruye el pipeline
y se ajusta de nuevo con 2022–2023. No reutilizar un pipeline ya ajustado al test.
No se imputa el precio objetivo ni se usa precio/m² calculado con el objetivo
como predictor. La lista de entrada excluye IDs, máscaras y columnas de soporte.

## Uso para compañero 2

```python
import pandas as pd
from src import config
from src.validacion import separar_xy

base = pd.read_parquet(config.DATA_PROCESSED / "base_modelado.parquet")
train = base.loc[base.particion.eq("entrenamiento") & base.incluido_iqr_seleccion.fillna(False)]
val = base.loc[base.particion.eq("validacion") & base.incluido_iqr_seleccion.fillna(False)]
X_train, y_train = separar_xy(train)
X_val, y_val = separar_xy(val)
# Entrenar y elegir configuraciones usando únicamente train/val.
# Después, reconstruir el pipeline para reentrenar:
final_train = base.loc[base.particion.ne("prueba") & base.incluido_iqr_final]
test = base.loc[base.particion.eq("prueba") & base.incluido_iqr_final]
X_final, y_final = separar_xy(final_train)
X_test, y_test = separar_xy(test)
```

`separar_xy` deja año/trimestre disponibles. El preprocesador actual no los usa;
incorporarlos es una alternativa a comparar con validación, no un cambio ya evaluado.
El compañero 3 recibe la misma base: técnicas que aprendan parámetros se ajustan
con entrenamiento, y las explicaciones/casos de prueba no deben guiar selección.

## Evidencia generada y límites

`reports/validacion_temporal/` recibirá resumen de particiones, cobertura distrital
y trimestral, casos de soporte insuficiente, resumen de texto y hashes de fuente
y decisiones. El reporte final se redacta tras revisar esos archivos.

No puede afirmarse ausencia absoluta de fuga: se mantienen la normalización y
el soporte global de TP1 por continuidad; las exclusiones manuales y el test
2024–2026 ya se exploraron. El informe debe declarar ambas circunstancias.
No hay ID fiable para descartar republicaciones entre períodos. Los cuatro
distritos con un anuncio quedaron fuera de la base actual. La superficie observada
es 28–150 m², y faltan acabados, conservación, coordenadas y entorno.
No extrapolar métricas a toda Lima, a la población excluida o al precio de cierre.

## Resultados verificados de preparación y validación

| Etapa | Período | Antes de IQR | Después del IQR correspondiente | Distritos antes de IQR |
| --- | --- | ---: | ---: | ---: |
| Entrenamiento para selección | 2022-T1–2023-T2 | 10061 | 10048 | 22 |
| Validación para selección | 2023-T3/T4 | 5132 | 5101 | 22 |
| Reentrenamiento final | 2022-T1–2023-T4 | 15193 | 15156 | 22 |
| Prueba posterior | 2024-T1–2026-T1 | 25154 | 25042 | 22 |

El IQR de selección retira 13 anuncios del entrenamiento y 31 de validación.
El IQR final, ajustado con todo 2022–2023, retira 37 de ese conjunto y 112 de
prueba. No sumar máscaras de referencias distintas: para selección se usan
10048/5101, y solo tras fijar la configuración se usan 15156/25042.

Los 22 distritos están presentes tanto en entrenamiento como en validación y
prueba antes de IQR. El soporte mínimo de entrenamiento es 145 en Carabayllo;
el mínimo de validación es 30, también en Carabayllo. No se detectan distritos
con menos de 10 anuncios de entrenamiento: soporte_insuficiente.csv solo contiene
la cabecera. La presencia no garantiza igual precisión ni representatividad.

Surco pasa de 749/10061 anuncios de entrenamiento (7.4%) a 4302/25154 de prueba
(17.1%); Miraflores pasa de 976/10061 (9.7%) a 4174/25154 (16.6%). Esto muestra
un cambio de composición que exige reportar errores por distrito junto con sus
tamaños; no demuestra por sí solo cambios de precio o causas económicas.

El baño faltante está en entrenamiento y la antigüedad inválida convertida en
faltante está en validación. La imputación de validación debe utilizar la mediana
aprendida en entrenamiento, sin calcular una mediana con validación. La prueba
no presenta faltantes en esas dos columnas antes del IQR.

La base única contiene 40347 filas previas a IQR y abarca 6 trimestres de
entrenamiento, 2 de validación y 9 de prueba. La fuente y el manifiesto se
identifican con SHA256 en ejecucion.json; se registró Python 3.12.10.

Estos resultados permiten entregar la preparación a los compañeros 2 y 3.
El siguiente trabajo del compañero 2 es entrenar y comparar configuraciones
con entrenamiento/validación. Las métricas actuales del consolidado corresponden
a la comparación anterior con todo 2022–2023 como entrenamiento.
