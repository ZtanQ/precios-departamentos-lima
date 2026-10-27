# Decisiones de limpieza tras la auditoría

## Alcance

Se actualiza la base de comparación de modelos del proyecto de precio base de
oferta para revisión profesional. Período: 2022-T1 a 2026-T1. La fuente BCRP
permanece intacta. No se crean etiquetas de precio imputadas ni una segunda
variante de entrenamiento. El Parquet de trabajo es `data/processed/base_modelado.parquet`.
Se sobrescribe al ejecutar `python scripts/preparar_datos_actuales.py`.
Los archivos procesados permanecen fuera de Git según `.gitignore` y cada
integrante los regenera. El manifiesto y los scripts sí se versionan.

## Precios

El manifiesto `decisiones_precios.csv` conserva posición original base cero,
valores del anuncio, decisión, motivo y firma de todas las columnas originales.
La posición no es un identificador de inmueble. Si los registros o el esquema
cambian, el código exige revisar la auditoría en vez de excluir otras filas.

- Conservar sin cambios: 89031, 97233, 81729 y 112895.
- Excluir precios bajos sospechosos (filas completas): 88630, 96182, 98617,
  88774, 87980 y 102498.
- Excluir precios altos sospechosos (filas completas): 103689, 108814, 119886,
  115896, 113059, 113193, 118518, 88185, 113427, 116150 y 117018.

Son 17 exclusiones específicas acordadas por el equipo tras revisión descriptiva
y consulta exploratoria de ofertas. No son errores confirmados ni una regla
general de que todo precio alto sea incorrecto. No se atribuye el precio a
motivaciones de los propietarios. No se reemplaza el objetivo por promedios,
conversiones de moneda ni valores supuestos.

## Duplicados y atributos

Se conserva una fila por repetición exacta en todas las columnas originales.
En el período hay 1155 filas adicionales exactas. No se deduplica por las
columnas del modelo: los cuatro pares de 2024-T4 que difieren en piso permanecen
(104708/104776, 105020/105044, 105244/105270 y 106243/106303).
No se eliminan perfiles iguales de distintos períodos por suponer republicación.

La antigüedad 1996 de la fila 90819 se convierte en faltante mediante la regla
existente >150. Si fuera año de construcción, la edad en el anuncio de 2023
sería aproximadamente 27, no 30; no hay confirmación para esa reinterpretación.
La fila 85440 conserva baños faltantes. Ambas filas permanecen: la imputación
por mediana se ajusta dentro del pipeline con entrenamiento, no aquí.

## Conteos de la base actual

| Etapa en 2022–2026-T1 | Filas |
| --- | ---: |
| Fuente sin limpiar | 41523 |
| Tras 17 exclusiones manuales | 41506 |
| Tras retirar 1155 duplicados exactos | 40351 |
| Tras criterio existente de soporte distrital (cuatro registros) | 40347 |

La base previa a IQR tiene 22 distritos: 6583 anuncios en 2022, 8610 en 2023,
12985 en 2024, 10008 en 2025 y 2161 en 2026-T1.
Contiene cuatro coincidencias adicionales en las columnas seleccionadas que se
conservan deliberadamente. No son un pendiente de eliminación automática.

## Continuidad y limitaciones

El corte entrenamiento 2022–2023 / prueba 2024–2026-T1, la normalización, el
soporte global >=10 y el IQR existente en los notebooks no se rediseñan en esta
etapa. La exportación es previa a IQR; no contiene imputación, escalado ni
codificación. Los notebooks 03 y consolidado ejecutan las mismas exclusiones
antes de su preparación existente. Los notebooks 01 y 02 siguen siendo análisis
exploratorios de la fuente y no son la nueva base de comparación.

La auditoría previa permanece como evidencia de lo observado antes de decidir.
Los conteos actuales se generan en `conteos_base_actual.csv`.
El consolidado se ejecutó después de esta limpieza: 15156 anuncios de entrenamiento
y 25042 de prueba tras IQR. Sus interpretaciones y la tabla del informe se actualizaron
con las salidas guardadas (métricas redondeadas en metricas_actualizadas.csv).
El notebook 03 conserva salidas vacías hasta su ejecución independiente.
Esta comparación no utiliza todavía la validación nueva; consultar
../protocolo_validacion_temporal.md.

Excluir precios extremos puede reducir los errores al retirar casos difíciles.
La evaluación resultante solo representa la población depurada. La revisión usó
también anuncios del test del TP1: no se debe presentar ese conjunto como nunca
explorado. Definir la validación y revisar el tratamiento estadístico del test
son los siguientes pasos; esta limpieza no los resuelve.
