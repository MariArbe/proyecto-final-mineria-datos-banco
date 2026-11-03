# Proyecto Final: Minería de Datos

Proyecto desarrollado con la metodología **CRISP-DM** para predecir si un cliente de un banco contratará un depósito a plazo, a partir de la información disponible **antes de llamarlo**. El objetivo es ayudar al banco a priorizar a qué clientes llamar en una campaña de marketing directo.

**Datos:** [Bank Marketing — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing), con 45,211 registros y 17 variables.

## Integrantes
* Mariana Arbeláez Sánchez
* María Lucía Cadavid
* Juan Manuel Arias

---

## Entregables

| Archivo | Contenido |
|---|---|
| `METODOLOGIA - CRISP DM.docx` | Informe con todas las fases de CRISP-DM documentadas. |
| `01_Preparacion_Datos_Bank_Marketing.ipynb` | Comprensión de los datos, análisis exploratorio y limpieza. Incluye el pandas profiling. |
| `02_Modelado_Evaluacion_Bank_Marketing.ipynb` | Ingeniería de características, balanceo con SMOTE, validación cruzada, 4 métodos supervisados y 3 de ensamble, medidas de calidad, ANOVA y Tukey, hiperparametrización (Grid Search y Bayes Search) y pipeline final. |
| `03_Despliegue_Bank_Marketing.ipynb` | Carga del pipeline, predicción de datos futuros y ejecución de la interfaz con Streamlit. |
| `app.py` | Interfaz gráfica en Streamlit. |

### Archivos de apoyo

| Archivo | Contenido |
|---|---|
| `bank_marketing_profiling.html` | Reporte de pandas profiling generado en el notebook 01. |
| `bank_marketing_limpio_sin_balancear.csv` | Dataset limpio que genera el notebook 01 y usa el notebook 02. |
| `pipeline_modelo.pkl` | Pipeline final (preprocesamiento + XGBoost optimizado) que genera el notebook 02. |
| `bank_marketing_datos_futuros.csv` | 12 clientes de ejemplo tomados del dataset, sin la variable objetivo, para probar el despliegue. |
| `requirements.txt` y `.streamlit/config.toml` | Librerías y colores de la interfaz. |

---

## Resultados principales

- **Modelos comparados:** KNN, Árbol de Decisión, Red Neuronal (MLP) y SVM lineal, más tres ensambles: Random Forest, XGBoost y Stacking. Todos se evaluaron con validación cruzada estratificada de 10 particiones sobre el 80% de entrenamiento, aplicando SMOTE solo dentro de cada partición.
- **Selección:** con ANOVA y Tukey sobre el ROC AUC de la validación cruzada, Stacking, XGBoost y Random Forest forman el grupo superior. Stacking se excluyó por costo computacional: es equivalente a XGBoost, pero 58 veces más lento. Los modelos seleccionados fueron **XGBoost, Random Forest y SVM lineal**.
- **Modelo final:** XGBoost optimizado con Bayes Search. Su ROC AUC es 0.795 en validación cruzada y 0.804 en prueba, y su PR AUC en prueba es 0.464. Con el umbral 0.5, **de cada 100 clientes que recomienda llamar, unos 62 contratan el depósito**, frente a unos 12 si se llamara sin priorizar.

---

## Cómo ejecutarlo

### Notebooks (Google Colab o Jupyter)
Ejecutarlos en orden:
1. `01_Preparacion_Datos_Bank_Marketing.ipynb`: descarga los datos desde UCI (requiere conexión a internet) y genera el dataset limpio.
2. `02_Modelado_Evaluacion_Bank_Marketing.ipynb`: necesita el CSV limpio y genera `pipeline_modelo.pkl`. La ejecución completa toma alrededor de una hora, sobre todo por la validación cruzada del Stacking.
3. `03_Despliegue_Bank_Marketing.ipynb`: necesita `pipeline_modelo.pkl` y `bank_marketing_datos_futuros.csv`.

### Interfaz con Streamlit (en el computador)
```bash
pip install -r requirements.txt
streamlit run app.py
```
La interfaz se abre en http://localhost:8501 y tiene dos modos:
- **Un cliente:** formulario con los datos del cliente. Muestra la probabilidad de que contrate y si conviene llamarlo.
- **Lista de clientes:** recibe un CSV y devuelve la lista ordenada de mayor a menor probabilidad.

Para ejecutarla en Google Colab, seguir la sección 4 del notebook 03, que usa *localtunnel*.

> **Versiones:** `pipeline_modelo.pkl` se guardó con scikit-learn 1.9.1 y XGBoost 3.4.1, y `requirements.txt` fija esas versiones. Si se vuelve a generar el pipeline con el notebook 02 en otro entorno, hay que actualizar esas dos versiones en `requirements.txt`.
