"""Despliegue del modelo Bank Marketing con Streamlit.

Carga el pipeline final (preprocesamiento + XGBoost optimizado) guardado en el notebook
02_Modelado_Evaluacion_Bank_Marketing.ipynb y predice si un cliente contratará el depósito a plazo.

Ejecutar con:  streamlit run app.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Umbral de decisión y referencias del modelo (notebook 02)
UMBRAL = 0.5
TASA_PROMEDIO = 0.117        # proporción de clientes que contrataron en el conjunto de prueba
PRECISION_TEST = 0.62        # de cada 100 clientes recomendados, ~62 contratan
ROC_AUC_TEST = 0.804

COLUMNAS = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
            "contact", "day_of_week", "month", "campaign", "pdays", "previous", "poutcome"]

# Etiquetas en español -> valores originales del dataset (None = valor no registrado)
OCUPACIONES = {
    "Administrativo": "admin.", "Directivo o gerente": "management", "Técnico": "technician",
    "Obrero u operario": "blue-collar", "Servicios": "services", "Emprendedor": "entrepreneur",
    "Independiente": "self-employed", "Servicio doméstico": "housemaid", "Jubilado": "retired",
    "Estudiante": "student", "Desempleado": "unemployed", "No registrado": None,
}
ESTADO_CIVIL = {"Casado(a)": "married", "Soltero(a)": "single", "Divorciado(a) o viudo(a)": "divorced"}
EDUCACION = {"Primaria": "primary", "Secundaria": "secondary", "Universitaria": "tertiary", "No registrado": None}
MEDIO_CONTACTO = {"Celular": "cellular", "Teléfono fijo": "telephone", "No registrado": None}
MESES = {"Enero": "jan", "Febrero": "feb", "Marzo": "mar", "Abril": "apr", "Mayo": "may", "Junio": "jun",
         "Julio": "jul", "Agosto": "aug", "Septiembre": "sep", "Octubre": "oct", "Noviembre": "nov",
         "Diciembre": "dec"}
RESULTADO_ANTERIOR = {"Contrató": "success", "No contrató": "failure", "Otro resultado": "other",
                      "No registrado": None}

ESTILOS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap');

:root {
  --papel: #F3F6F4;
  --tinta: #16243F;
  --verde: #23704F;
  --pizarra: #56636B;
  --linea: #D3DBD6;
  --blanco: #FFFFFF;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
.stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stRadio, .stToggle, button, input, label {
  font-family: 'Archivo', system-ui, sans-serif;
}

[data-testid="stAppViewContainer"] { background: var(--papel); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.5rem; max-width: 1180px; }

h1.titulo {
  font-family: 'Archivo', sans-serif;
  font-stretch: 112%;
  font-weight: 750;
  font-size: clamp(1.9rem, 3.4vw, 2.7rem);
  line-height: 1.1;
  letter-spacing: -0.015em;
  color: var(--tinta);
  margin: 0 0 0.5rem 0;
  padding: 0;
  max-width: 30ch;
}
p.bajada { color: var(--pizarra); font-size: 1.05rem; line-height: 1.55; max-width: 62ch; margin: 0 0 1.5rem 0; }

h3.grupo {
  font-family: 'Archivo', sans-serif;
  font-weight: 650;
  font-size: 1.05rem;
  color: var(--tinta);
  margin: 1.4rem 0 0.25rem 0;
  padding: 0 0 0.4rem 0;
  border-bottom: 1px solid var(--linea);
}

/* El comprobante acompaña al formulario mientras se recorre */
[data-testid="stColumn"]:has(.comprobante) { position: sticky; top: 1.5rem; align-self: flex-start; }

/* Comprobante de decisión: la pieza central de la interfaz */
.comprobante {
  background: var(--blanco);
  border: 1px solid var(--linea);
  border-top: 2px dashed var(--linea);
  border-radius: 2px 2px 10px 10px;
  padding: 1.6rem 1.6rem 1.4rem;
  color: var(--tinta);
}
.comprobante .etiqueta { color: var(--pizarra); font-size: 0.95rem; margin: 0; }
.comprobante .cifra {
  font-stretch: 125%;
  font-weight: 800;
  font-size: clamp(3.6rem, 7vw, 5rem);
  line-height: 1;
  letter-spacing: -0.03em;
  margin: 0.35rem 0 1.4rem 0;
  font-variant-numeric: tabular-nums;
}
.comprobante .cifra span { font-size: 0.45em; font-weight: 600; margin-left: 0.08em; }

.regla { position: relative; height: 10px; background: #E7EDE9; border-radius: 5px; margin-bottom: 2.6rem; }
.regla .relleno { position: absolute; inset: 0 auto 0 0; border-radius: 5px; background: var(--tinta);
                  transition: width 300ms ease-out; }
.regla.llamar .relleno { background: var(--verde); }
.regla .marca { position: absolute; top: -5px; width: 2px; height: 20px; background: var(--pizarra); }
.regla .marca small { position: absolute; top: 24px; transform: translateX(-50%); white-space: nowrap;
                      font-size: 0.78rem; color: var(--pizarra); line-height: 1.25; text-align: center; }

.veredicto { display: inline-block; font-weight: 700; font-size: 1.1rem; padding: 0.55rem 1rem;
             border-radius: 6px; margin: 0 0 0.8rem 0; }
.veredicto.llamar { background: var(--verde); color: var(--blanco); }
.veredicto.esperar { border: 1.5px solid var(--pizarra); color: var(--tinta); }
.comprobante .nota { color: var(--pizarra); font-size: 0.95rem; line-height: 1.55; margin: 0.35rem 0 0 0; max-width: 46ch; }

.resumen-lista { font-size: 1.15rem; color: var(--tinta); margin: 1rem 0 0.75rem 0; }
.resumen-lista strong { font-stretch: 112%; }

.pie { color: var(--pizarra); font-size: 0.85rem; line-height: 1.5; border-top: 1px solid var(--linea);
       padding-top: 1rem; margin-top: 2.5rem; }

/* En pantallas angostas las columnas se apilan: el comprobante va primero */
@media (max-width: 640px) {
  [data-testid="stHorizontalBlock"]:has(.comprobante) { flex-direction: column-reverse; }
  [data-testid="stColumn"]:has(.comprobante) { position: static; }
}

@media (prefers-reduced-motion: reduce) { .regla .relleno { transition: none; } }
</style>
"""


@st.cache_resource
def cargar_pipeline():
    """Carga el pipeline final guardado con pickle en el notebook de modelado."""
    carpeta = Path(__file__).parent
    for ruta in (carpeta / "pipeline_modelo.pkl", carpeta / "entregables_modelado_bank" / "pipeline_modelo.pkl"):
        if ruta.exists():
            with open(ruta, "rb") as archivo:
                return pickle.load(archivo)
    return None


def preparar(datos):
    """Deja los datos en el formato del dataset limpio: mismas columnas y valores no registrados como NaN."""
    datos = datos[COLUMNAS].copy()
    for columna in ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]:
        valores = datos[columna].astype("object").replace({"unknown": np.nan, "": np.nan})
        datos[columna] = valores.where(valores.notna(), np.nan)
    return datos


def comprobante(probabilidad):
    """HTML del comprobante de decisión para un cliente."""
    llamar = probabilidad > UMBRAL
    veces = probabilidad / TASA_PROMEDIO
    if veces >= 1:
        comparacion = f"Es {veces:.1f} veces la tasa promedio de la campaña ({TASA_PROMEDIO:.1%})."
    else:
        comparacion = f"Está por debajo de la tasa promedio de la campaña ({TASA_PROMEDIO:.1%})."
    if llamar:
        veredicto = '<div class="veredicto llamar">Llamar a este cliente</div>'
        nota = (f"La probabilidad supera el {UMBRAL:.0%}. En el conjunto de prueba, de cada 100 clientes con esta "
                f"recomendación unos {PRECISION_TEST * 100:.0f} contrataron el depósito.")
    else:
        veredicto = '<div class="veredicto esperar">No priorizar por ahora</div>'
        nota = (f"La probabilidad no alcanza el {UMBRAL:.0%}. Puede quedar para una ronda posterior si hay "
                "capacidad de llamadas disponible.")
    return f"""
<div class="comprobante" role="status" aria-live="polite">
  <p class="etiqueta">Probabilidad de que contrate el depósito</p>
  <p class="cifra">{probabilidad * 100:.0f}<span>%</span></p>
  <div class="regla {'llamar' if llamar else ''}" aria-hidden="true">
    <div class="relleno" style="width:{probabilidad * 100:.1f}%"></div>
    <div class="marca" style="left:{TASA_PROMEDIO * 100:.1f}%"><small>Promedio<br>{TASA_PROMEDIO:.1%}</small></div>
    <div class="marca" style="left:{UMBRAL * 100:.0f}%"><small>Umbral para llamar<br>{UMBRAL:.0%}</small></div>
  </div>
  {veredicto}
  <p class="nota">{comparacion}</p>
  <p class="nota">{nota}</p>
</div>
"""


def formulario_cliente():
    """Formulario de un cliente; devuelve un DataFrame de una fila con las variables del modelo."""
    st.markdown('<h3 class="grupo">Perfil del cliente</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    edad = c1.number_input("Edad", min_value=18, max_value=100, value=39, step=1)
    ocupacion = c2.selectbox("Ocupación", list(OCUPACIONES), index=1)
    estado_civil = c1.selectbox("Estado civil", list(ESTADO_CIVIL))
    educacion = c2.selectbox("Nivel educativo", list(EDUCACION), index=2)

    st.markdown('<h3 class="grupo">Situación financiera</h3>', unsafe_allow_html=True)
    c1, _ = st.columns(2)
    saldo = c1.number_input("Saldo promedio anual (€)", min_value=-10000, max_value=150000, value=450, step=50,
                            help="Puede ser negativo si el cliente tiene sobregiros.")
    c1, c2, c3 = st.columns(3)
    hipoteca = c1.toggle("Crédito hipotecario", value=True)
    consumo = c2.toggle("Crédito de consumo", value=False)
    mora = c3.toggle("Créditos en mora", value=False)

    st.markdown('<h3 class="grupo">Llamada en esta campaña</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    medio = c1.selectbox("Medio de contacto", list(MEDIO_CONTACTO))
    llamadas = c2.number_input("Llamadas en esta campaña", min_value=1, max_value=60, value=1, step=1,
                               help="Incluye la llamada que se va a hacer.")
    mes = c1.selectbox("Mes de la llamada", list(MESES), index=4)
    dia = c2.number_input("Día del mes de la llamada", min_value=1, max_value=31, value=15, step=1)

    st.markdown('<h3 class="grupo">Campañas anteriores</h3>', unsafe_allow_html=True)
    contactado_antes = st.toggle("Fue contactado en una campaña anterior", value=False)
    if contactado_antes:
        c1, c2, c3 = st.columns(3)
        pdays = c1.number_input("Días desde el último contacto", min_value=1, max_value=900, value=180, step=1)
        previous = c2.number_input("Contactos anteriores", min_value=1, max_value=300, value=1, step=1)
        resultado = c3.selectbox("Resultado de esa campaña", list(RESULTADO_ANTERIOR))
    else:
        # En el dataset, pdays = -1 y previous = 0 significan que no hubo contacto previo
        pdays, previous, resultado = -1, 0, "No registrado"

    return pd.DataFrame([{
        "age": edad, "job": OCUPACIONES[ocupacion], "marital": ESTADO_CIVIL[estado_civil],
        "education": EDUCACION[educacion], "default": "yes" if mora else "no", "balance": saldo,
        "housing": "yes" if hipoteca else "no", "loan": "yes" if consumo else "no",
        "contact": MEDIO_CONTACTO[medio], "day_of_week": dia, "month": MESES[mes], "campaign": llamadas,
        "pdays": pdays, "previous": previous, "poutcome": RESULTADO_ANTERIOR[resultado],
    }])


def lista_clientes(pipeline):
    """Predicción para un archivo CSV con varios clientes y descarga de la lista priorizada."""
    st.markdown('<p class="bajada">Sube un archivo CSV con las mismas columnas del dataset limpio. '
                'Las columnas <code>y</code> y <code>duration</code> se ignoran si vienen en el archivo.</p>',
                unsafe_allow_html=True)
    plantilla = pd.DataFrame(columns=COLUMNAS).to_csv(index=False).encode("utf-8")
    st.download_button("Descargar plantilla CSV", plantilla, "plantilla_clientes.csv", "text/csv")

    archivo = st.file_uploader("Archivo de clientes (CSV)", type="csv")
    if archivo is None:
        st.caption("Cuando subas el archivo, aquí aparecerá la lista de clientes ordenada de mayor a menor probabilidad.")
        return

    try:
        datos = pd.read_csv(archivo)
    except Exception:
        st.error("No se pudo leer el archivo. Revisa que sea un CSV separado por comas.")
        return
    faltantes = [c for c in COLUMNAS if c not in datos.columns]
    if faltantes:
        st.error(f"Al archivo le faltan estas columnas: {', '.join(faltantes)}. Usa la plantilla como referencia.")
        return
    if datos.empty:
        st.error("El archivo no tiene clientes. Agrega al menos una fila.")
        return

    X = preparar(datos)
    resultado = datos.drop(columns=[c for c in ("y", "duration") if c in datos.columns]).copy()
    resultado.insert(0, "Probabilidad", pipeline.predict_proba(X)[:, 1])
    resultado.insert(1, "Decisión", np.where(resultado["Probabilidad"] > UMBRAL, "Llamar", "No priorizar"))
    resultado = resultado.sort_values("Probabilidad", ascending=False).reset_index(drop=True)

    a_llamar = int((resultado["Decisión"] == "Llamar").sum())
    st.markdown(f'<p class="resumen-lista">El modelo recomienda llamar a <strong>{a_llamar} de {len(resultado)}</strong> '
                'clientes.</p>', unsafe_allow_html=True)
    st.dataframe(
        resultado, hide_index=True, width="stretch",
        column_config={"Probabilidad": st.column_config.ProgressColumn(
            "Probabilidad", format="percent", min_value=0.0, max_value=1.0)},
    )
    st.download_button("Descargar lista priorizada", resultado.to_csv(index=False).encode("utf-8"),
                       "clientes_priorizados.csv", "text/csv", type="primary")


def main():
    st.set_page_config(page_title="A quién llamar | Depósitos a plazo", page_icon="🏦", layout="wide")
    st.markdown(ESTILOS, unsafe_allow_html=True)

    st.markdown('<h1 class="titulo">¿A quién llamar en la campaña de depósitos a plazo?</h1>', unsafe_allow_html=True)
    st.markdown('<p class="bajada">Ingresa los datos de un cliente para estimar la probabilidad de que contrate el '
                'depósito y saber si conviene llamarlo. También puedes priorizar una lista completa de clientes.</p>',
                unsafe_allow_html=True)

    pipeline = cargar_pipeline()
    if pipeline is None:
        st.error("No se encontró el archivo pipeline_modelo.pkl. Ejecuta el notebook "
                 "02_Modelado_Evaluacion_Bank_Marketing.ipynb y copia el archivo junto a app.py.")
        st.stop()

    tab_cliente, tab_lista = st.tabs(["Un cliente", "Lista de clientes"])

    with tab_cliente:
        col_form, col_resultado = st.columns([3, 2], gap="large")
        with col_form:
            cliente = formulario_cliente()
        with col_resultado:
            probabilidad = float(pipeline.predict_proba(preparar(cliente))[:, 1][0])
            st.markdown(comprobante(probabilidad), unsafe_allow_html=True)

    with tab_lista:
        lista_clientes(pipeline)

    st.markdown(f'<p class="pie">Modelo XGBoost optimizado con Bayes Search. En el conjunto de prueba obtuvo un ROC AUC '
                f'de {ROC_AUC_TEST:.3f} y una precision de {PRECISION_TEST:.2f}. Proyecto final de Minería de Datos: '
                'Mariana Arbeláez, María Lucía Cadavid y Juan Manuel Arias.</p>', unsafe_allow_html=True)


main()
