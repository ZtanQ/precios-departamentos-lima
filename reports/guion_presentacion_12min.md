# GUION TÉCNICO Y ESTRUCTURA DE PRESENTACIÓN (12 MINUTOS)
## Trabajo Final (TF1) – Data Mining Tools (CC209)
**Docente:** Carlos Fernando Montoya Cubas  
**Tiempo límite estricto:** 12 minutos de exposición (incluye demo funcional) + 5 minutos de preguntas.  
**Integrantes:** 4 expositores (todos deben participar).

---

### ESTRUCTURA DE TIEMPOS Y ROLES

```
[00:00 - 02:30] Bloque 1: Problema, Evolución TP1 -> TF1 y Técnica Adicional (Gabriel Reyna)
[02:30 - 05:30] Bloque 2: Preprocesamiento, Partición Temporal y Experimentación (José Melgar)
[05:30 - 08:30] Bloque 3: Demostración Funcional en Vivo en Streamlit (Piero Contreras)
[08:30 - 11:00] Bloque 4: Interpretabilidad SHAP y Diagnóstico de Errores (Guido Jeri)
[11:00 - 12:00] Bloque 5: Conclusiones, Limitaciones y Cierre (Todos)
```

---

### DETALLE DEL GUION POR INTEGRANTE

#### BLOQUE 1: Problema, Evolución TP1 $\rightarrow$ TF1 y Técnica Adicional (2:30 min)
* **Expositor:** Gabriel Alonzo Reyna Alvarado
* **Diapositivas:**
  * *Slide 1: Portada y Problema de Negocio.*
    * "El mercado inmobiliario en Lima fija tradicionalmente precios de oferta mediante la regla empírica del tasador: mediana de US$/m² del distrito multiplicada por la superficie. En el TP1 demostramos que esta regla tiene un 14.9% de error mediano."
  * *Slide 2: Tabla de Evolución TP1 vs. TF1.*
    * "No venimos a presentar el TP1 con más datos. Hemos introducido tres saltos metodológicos: (1) cambio de variable objetivo a precio unitario US$/m² con reconstrucción matemática en dólares; (2) clustering de submercados de la segunda mitad del curso; y (3) ensamble ponderado calibrado con Optuna."
  * *Slide 3: Técnica Adicional (Clustering Espacial de Submercados).*
    * "Implementamos K-Means ($k=4$) sobre el precio/m², superficie y antigüedad aprendidos estrictamente en Train. Esto segmenta Lima en 4 clusters socio-inmobiliarios reales (Lima Top, Lima Moderna, Lima Centro y Periferia), aportando micro-zonas sin dispersar la dimensionalidad."

---

#### BLOQUE 2: Validación Temporal, Modelado y Experimentación (3:00 min)
* **Expositor:** José Guillermo Melgar Puertas
* **Diapositivas:**
  * *Slide 4: Protocolo de Partición Out-of-Time Triple.*
    * "Para garantizar CERO data leakage, no usamos K-Fold aleatorio. Separamos cronológicamente: Train de Selección (2022-T1 a 2023-T2), Validación (2023-T3 a 2023-T4) para Optuna, y congelamos una Prueba Ciega de dos años completos (2024 a 2026-T1 con 25,042 anuncios)."
  * *Slide 5: Ensamble Ponderado Calibrado sobre US$/m².*
    * "En lugar de predecir dólares totales, entrenamos sobre $\log(1 + \text{US\$/m}^2)$. Combinamos tres algoritmos con Optuna: CatBoost (87.9%), Random Forest (10.4%) y Ridge (1.7%)."
  * *Slide 6: Resultados Oficiales en Test Ciego.*
    * "En la prueba ciega de 25 mil inmuebles, el Ensamble Ponderado alcanzó un MedAPE de **11.13%**, un MAE de **\$22,647** y un RMSE de **\$32,010**, clasificando al **45.63%** de los departamentos dentro del margen estricto del $\pm 10\%$ de error comercial."

---

#### BLOQUE 3: Demostración Funcional en Vivo en Streamlit (3:00 min)
* **Expositor:** Piero Marcos Contreras Albornoz
* **Dinámica en Vivo:**
  * "Paso a compartir la aplicación web que materializa el pipeline fuera del notebook."
  * **Caso 1 (Departamento Estándar en Miraflores):**
    * Ingrese: Miraflores, 85 m², 3 dormitorios, 2 baños, 1 cochera, 5 años de antigüedad.
    * Muestre el resultado en pantalla: Precio central ~$196,000, Valor unitario ~$2,180/m², Cluster Asignado: Lima Top.
    * Destaque el intervalo de negociación ($\pm 11.1\%$).
  * **Caso 2 (Interpretabilidad en Vivo):**
    * Cambie a la pestaña de "Explicabilidad SHAP".
    * Muestre el Waterfall Plot: *"Aquí el tasador o cliente puede ver cómo la ubicación en Miraflores y los 2 baños empujan el precio al alza, mientras que la antigüedad descuenta valor."*
  * **Explicar la Arquitectura:**
    * Resaltar que `app.py` consume el archivo `.joblib` empaquetado, ejecutando imputación, clustering y ensamble en menos de 200 milisegundos.

---

#### BLOQUE 4: Interpretabilidad TreeSHAP y Análisis de Errores (2:30 min)
* **Expositor:** Guido Yair Abel Jeri Saldaña
* **Diapositivas:**
  * *Slide 7: Explicabilidad Global (Beeswarm Plot).*
    * "TreeSHAP demuestra que el valor unitario en Lima está gobernado por: (1) el cluster de submercado, (2) la superficie relativa distrital, y (3) la depreciación no lineal por antigüedad."
  * *Slide 8: Análisis Sistemático de Colas de Error.*
    * "Auditamos los 50 peores casos del conjunto de prueba:
      - **Subestimaciones:** Ocurren en departamentos de San Isidro y Miraflores con acabados de ultralujo o vista al Golf, factores no registrados por el BCRP.
      - **Sobreestimaciones:** Departamentos con precios anómalamente bajos en San Borja o Surco, correspondientes a remates judiciales o venta de derechos y acciones."
  * *Slide 9: Desempeño Espacial.*
    * "El modelo es altamente preciso en Lima Moderna (Surquillo 8.5% error, Jesús María 8.9%), pero presenta mayor dispersión en la periferia (Carabayllo 16.3%) debido a la heterogeneidad urbana."

---

#### BLOQUE 5: Conclusiones, Limitaciones y Cierre (1:00 min)
* **Expositor:** Gabriel Alonzo Reyna Alvarado / Todos
* **Diapositiva:**
  * *Slide 10: Conclusiones y Defensa.*
    * "Conclusiones: Predecir el valor unitario ($m^2$) superó conceptualmente al modelado de precios totales. El ensamble bate al mercado y reduce la incertidumbre pericial."
    * "Limitación: Los datos corresponden a precios de oferta (anuncio), no a transacciones notariales finales."
    * "Quedamos a disposición para las preguntas del docente."

---

## BANCO DE PREGUNTAS TÍPICAS DEL DOCENTE (CARLOS MONTOYA)

### P1: ¿Por qué decidieron predecir el precio por m² en lugar del precio total en dólares?
* **Respuesta del estudiante:**  
  *"En el TP1 predecíamos el precio total en dólares. Sin embargo, la superficie tiene una correlación de casi 0.85 con el precio, lo que hacía que el modelo sufriera de heterocedasticidad: los errores en departamentos grandes de 200 m² distorsionaban el aprendizaje frente a departamentos pequeños. Al modelar el logaritmo del precio por metro cuadrado $\log(\text{US\$/m}^2)$, desacoplamos el efecto de escala. El modelo se concentra en aprender el valor de la ubicación, calidad y antigüedad, y luego simplemente reescalamos por la superficie. Empíricamente esto redujo el MAE en \$281 y el RMSE en \$501 en el test ciego."*

### P2: ¿Cómo evitaron el Data Leakage al incorporar el Clustering de distritos?
* **Respuesta del estudiante:**  
  *"El estimador `ClusteringSubmercado` implementa la interfaz `BaseEstimator` y `TransformerMixin` de Scikit-Learn. En la fase de `fit()`, calcula las medianas de precio/m², superficie y antigüedad ÚNICAMENTE con los datos de entrenamiento (2022 a 2023-T2). En validación y prueba, el método `transform()` solo asigna el cluster ya aprendido por mapeo distrital. Si entrara un distrito nuevo no visto en train, cuenta con un fallback al cluster más frecuente para no fallar en producción."*

### P3: ¿Por qué usaron un Ensamble Ponderado y no se quedaron solo con CatBoost?
* **Respuesta del estudiante:**  
  *"CatBoost fue nuestro estimador individual más fuerte (MedAPE 11.16%). Sin embargo, los árboles de gradiente tienden a ser sensibles en zonas de baja densidad. Al ensamblarlo con Random Forest (10.4%) y Ridge (1.7%) optimizando los pesos en un split de validación temporal mediante Optuna, logramos regularizar las predicciones extremas, reduciendo el RMSE en \$321 adicionales en el conjunto de prueba ciego."*

### P4: ¿Por qué hay departamentos en San Isidro donde el modelo se equivoca por más de \$100,000?
* **Respuesta del estudiante:**  
  *"Analizamos esos casos específicos en el archivo `casos_extremos_error.csv`. Esos departamentos son inmuebles de lujo con vista al Lima Golf Club o acabados italianos. La base del BCRP solo tiene variables estructurales básicas (área, cuartos, baños, antigüedad). Nuestra conclusión técnica es que el modelo estima con alta fidelidad el 'precio base de mercado', pero no puede capturar primas cualitativas de ultralujo no presentes en los datos."*

