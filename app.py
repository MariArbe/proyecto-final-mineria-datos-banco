"""Despliegue del modelo Bank Marketing con Streamlit.

Carga el pipeline final (preprocesamiento + XGBoost optimizado) guardado en el notebook
02_Modelado_Evaluacion_Bank_Marketing.ipynb y predice si un cliente contratará el depósito a plazo.

Ejecutar con:  streamlit run app.py
"""
import base64
import math
import pickle
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

CARPETA = Path(__file__).parent

# Umbral de decisión y referencias del modelo (notebook 02)
UMBRAL = 0.5
TASA_PROMEDIO = 0.117        # proporción de clientes que contrataron en el conjunto de prueba
PRECISION_TEST = 0.62        # de cada 100 clientes recomendados, ~62 contratan
ROC_AUC_TEST = 0.804

COLUMNAS = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
            "contact", "day_of_week", "month", "campaign", "pdays", "previous", "poutcome"]
CATEGORICAS = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

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

# Nombre de cada variable para las gráficas
NOMBRES = {
    "age": "Edad", "job": "Ocupación", "marital": "Estado civil", "education": "Nivel educativo",
    "default": "Créditos en mora", "balance": "Saldo promedio", "housing": "Crédito hipotecario",
    "loan": "Crédito de consumo", "contact": "Medio de contacto", "day_of_week": "Día del mes",
    "month": "Mes de la llamada", "campaign": "Llamadas en la campaña", "pdays": "Días desde el último contacto",
    "previous": "Contactos anteriores", "poutcome": "Resultado de la campaña anterior",
}

# Paleta (validada con el script de la skill de visualización: contraste, croma y daltonismo)
PAPEL, TINTA, PIZARRA, LINEA = "#F3F6F4", "#16243F", "#56636B", "#D3DBD6"
VERDE, ORO, NIEBLA = "#0F7F5C", "#C8851A", "#A7B4BF"

# Desempeño de los modelos en validación cruzada (ROC AUC, notebook 02)
MODELOS_CV = [
    ("XGBoost optimizado (modelo final)", 0.795), ("Stacking", 0.791), ("XGBoost inicial", 0.790),
    ("Random Forest", 0.786), ("SVM lineal", 0.759), ("Árbol de Decisión", 0.717),
    ("KNN", 0.702), ("Red neuronal (MLP)", 0.671),
]


def patron_guilloche():
    """Líneas onduladas superpuestas, como el grabado de seguridad de un billete (SVG en base64)."""
    ancho, alto = 1400, 320
    lineas = []
    for k in range(22):
        amplitud, fase = 38 + k * 5, k * 0.33
        puntos = " ".join(
            f"{x},{alto / 2 + amplitud * math.sin(x / 95 + fase) * math.cos(x / 340 - fase / 2):.1f}"
            for x in range(0, ancho + 1, 8))
        lineas.append(f'<polyline points="{puntos}" fill="none" stroke="#F3F6F4" stroke-opacity="0.075" stroke-width="1"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" '
           f'viewBox="0 0 {ancho} {alto}" preserveAspectRatio="none">{"".join(lineas)}</svg>')
    return base64.b64encode(svg.encode()).decode()


ESTILOS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap');

:root {
  --papel: #F3F6F4; --tinta: #16243F; --pizarra: #56636B; --linea: #D3DBD6;
  --verde: #0F7F5C; --oro: #C8851A; --blanco: #FFFFFF;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
.stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stRadio, .stToggle, button, input, label {
  font-family: 'Archivo', system-ui, sans-serif;
}
[data-testid="stAppViewContainer"] { background: var(--papel); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

/* Encabezado: tinta de billete con grabado guilloché */
.encabezado {
  background: var(--tinta) url("data:image/svg+xml;base64,__GUILLOCHE__") center / cover no-repeat;
  color: var(--papel);
  border-radius: 14px;
  padding: 2.4rem 2.6rem 2.2rem;
  margin-bottom: 1.6rem;
}
.encabezado .contexto { margin: 0 0 0.6rem 0; font-size: 1rem; opacity: 0.78; }
.encabezado h1 {
  font-family: 'Archivo', sans-serif; font-stretch: 125%; font-weight: 800;
  font-size: clamp(2.4rem, 6vw, 4.2rem); line-height: 0.98; letter-spacing: -0.03em;
  color: var(--papel); margin: 0 0 0.9rem 0; padding: 0;
}
.encabezado .bajada { font-size: 1.08rem; line-height: 1.55; max-width: 60ch; margin: 0; opacity: 0.9; }

/* Pestañas */
[data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--linea); }
button[data-baseweb="tab"] p { font-size: 1.02rem; font-weight: 600; }
[data-baseweb="tab-highlight"] { background-color: var(--verde); height: 3px; }

/* Sin el ícono de enlace que Streamlit agrega a los títulos */
[data-testid="stHeaderActionElements"], h1 a, h3 a { display: none !important; }

/* Tarjetas (st.container con borde) */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) ,
div[data-testid="stVerticalBlock"][data-test-border="true"] {
  background: var(--blanco); border-color: var(--linea) !important; border-radius: 12px;
}

h3.grupo {
  font-family: 'Archivo', sans-serif; font-weight: 650; font-size: 1.05rem; color: var(--tinta);
  margin: 1.1rem 0 0.25rem 0; padding: 0 0 0.4rem 0; border-bottom: 1px solid var(--linea);
}
h3.grupo:first-child { margin-top: 0.2rem; }
h3.seccion {
  font-family: 'Archivo', sans-serif; font-stretch: 112%; font-weight: 700; font-size: 1.25rem;
  color: var(--tinta); margin: 0 0 0.2rem 0; padding: 0;
}
p.explicacion { color: var(--pizarra); font-size: 0.95rem; line-height: 1.5; margin: 0 0 0.6rem 0; max-width: 70ch; }

/* Comprobante de decisión: desprendible con borde perforado */
.comprobante {
  position: relative; background: var(--blanco); border: 1px solid var(--linea); border-top: none;
  border-radius: 0 0 12px 12px; padding: 1.9rem 1.7rem 1.5rem; color: var(--tinta); margin-top: 10px;
}
.comprobante::before {
  content: ""; position: absolute; left: -1px; right: -1px; top: -10px; height: 10px;
  background:
    radial-gradient(circle at 9px 10px, var(--papel) 5px, transparent 5.5px) 0 0 / 18px 10px repeat-x,
    var(--blanco);
  border-left: 1px solid var(--linea); border-right: 1px solid var(--linea);
}
.comprobante .etiqueta { color: var(--pizarra); font-size: 0.95rem; margin: 0; }
.comprobante .cifra {
  font-stretch: 125%; font-weight: 800; font-size: clamp(3.8rem, 7vw, 5.4rem); line-height: 1;
  letter-spacing: -0.035em; margin: 0.35rem 0 1.3rem 0; font-variant-numeric: tabular-nums;
}
.comprobante .cifra span { font-size: 0.42em; font-weight: 600; margin-left: 0.06em; }

.regla { position: relative; height: 10px; background: #E7EDE9; border-radius: 5px; margin-bottom: 2.6rem; }
.regla .relleno { position: absolute; inset: 0 auto 0 0; border-radius: 5px; background: var(--tinta);
                  transition: width 300ms ease-out; }
.regla.llamar .relleno { background: var(--verde); }
.regla .marca { position: absolute; top: -5px; width: 2px; height: 20px; background: var(--pizarra); }
.regla .marca small { position: absolute; top: 24px; transform: translateX(-50%); white-space: nowrap;
                      font-size: 0.78rem; color: var(--pizarra); line-height: 1.25; text-align: center; }

/* Sello de la recomendación */
.sello {
  display: inline-block; font-stretch: 112%; font-weight: 800; font-size: 1.35rem; letter-spacing: 0.01em;
  padding: 0.45rem 1rem; border: 3px solid currentColor; border-radius: 8px;
  box-shadow: inset 0 0 0 2px var(--blanco), inset 0 0 0 3.5px currentColor;
  transform: rotate(-3deg); margin: 0.2rem 0 1rem 0.2rem; animation: estampar 380ms cubic-bezier(.2,.9,.3,1.2);
}
.sello.llamar { color: var(--verde); }
.sello.esperar { color: var(--pizarra); }
@keyframes estampar { from { transform: rotate(-3deg) scale(1.35); opacity: 0; } to { transform: rotate(-3deg) scale(1); opacity: 1; } }

.comprobante .nota { color: var(--pizarra); font-size: 0.95rem; line-height: 1.55; margin: 0.35rem 0 0 0; max-width: 46ch; }
.comprobante .nota strong { color: var(--tinta); }

/* Cifras clave */
.cifras { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: 0.4rem 0 1.4rem 0; }
.cifra-item { background: var(--blanco); border: 1px solid var(--linea); border-radius: 12px; padding: 1.1rem 1.2rem; }
.cifra-item .valor { font-stretch: 118%; font-weight: 800; font-size: 2.1rem; line-height: 1.05; color: var(--tinta);
                     margin: 0 0 0.35rem 0; font-variant-numeric: tabular-nums; }
.cifra-item .valor.verde { color: var(--verde); }
.cifra-item .texto { color: var(--pizarra); font-size: 0.92rem; line-height: 1.45; margin: 0; }

.leyenda { display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; font-size: 0.88rem; color: var(--tinta); margin: 0 0 0.3rem 0; }
.leyenda span { display: inline-flex; align-items: center; gap: 0.4rem; }
.leyenda i { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }

.pie { color: var(--pizarra); font-size: 0.85rem; line-height: 1.5; border-top: 1px solid var(--linea);
       padding-top: 1rem; margin-top: 2.5rem; }

@media (max-width: 640px) {
  [data-testid="stHorizontalBlock"]:has(.comprobante) { flex-direction: column-reverse; }
  .encabezado { padding: 1.8rem 1.4rem; }
  .cifras { grid-template-columns: 1fr; }
  .sello { font-size: 1.1rem; white-space: nowrap; }
}
@media (prefers-reduced-motion: reduce) { .regla .relleno { transition: none; } .sello { animation: none; } }
</style>
""".replace("__GUILLOCHE__", patron_guilloche())


# ------------------------------------------------------------------ Modelo y datos

@st.cache_resource
def cargar_pipeline():
    """Carga el pipeline final guardado con pickle en el notebook de modelado."""
    for ruta in (CARPETA / "pipeline_modelo.pkl", CARPETA / "entregables_modelado_bank" / "pipeline_modelo.pkl"):
        if ruta.exists():
            with open(ruta, "rb") as archivo:
                return pickle.load(archivo)
    return None


def preparar(datos):
    """Deja los datos en el formato del dataset limpio: mismas columnas y valores no registrados como NaN."""
    datos = datos[COLUMNAS].copy()
    for columna in CATEGORICAS:
        valores = datos[columna].astype("object").replace({"unknown": np.nan, "": np.nan})
        datos[columna] = valores.where(valores.notna(), np.nan)
    return datos


@st.cache_resource
def historico(_pipeline):
    """Probabilidades del modelo para los clientes del dataset limpio (referencia para comparar)."""
    ruta = CARPETA / "bank_marketing_limpio_sin_balancear.csv"
    if not ruta.exists():
        return None, None
    datos = preparar(pd.read_csv(ruta))
    return datos, _pipeline.predict_proba(datos)[:, 1]


def contribuciones(pipeline, datos):
    """Aporte de cada variable a la predicción (TreeSHAP nativo de XGBoost, en escala log-odds).

    Las columnas one-hot de una misma variable se suman para hablar de la variable original."""
    preprocesador = pipeline.named_steps["preprocessor"]
    modelo = pipeline.named_steps["model"]
    aportes = modelo.get_booster().predict(xgb.DMatrix(preprocesador.transform(datos)), pred_contribs=True)[:, :-1]
    variables = []
    for nombre in preprocesador.get_feature_names_out():
        tipo, resto = nombre.split("__", 1)
        variables.append(resto if tipo == "num" else next(c for c in CATEGORICAS if resto.startswith(c + "_")))
    return pd.DataFrame(aportes, columns=variables, index=datos.index).T.groupby(level=0).sum().T


@st.cache_resource
def importancia_global(_pipeline, _datos):
    """Influencia promedio de cada variable sobre una muestra de clientes históricos."""
    muestra = _datos.sample(min(4000, len(_datos)), random_state=42)
    return contribuciones(_pipeline, muestra).abs().mean().sort_values(ascending=False)


def describir(variable, cliente):
    """Texto legible del valor que tiene el cliente en una variable."""
    valor = cliente[variable]
    if pd.isna(valor):
        return f"{NOMBRES[variable]}: no registrado"
    inversos = {"job": OCUPACIONES, "marital": ESTADO_CIVIL, "education": EDUCACION, "contact": MEDIO_CONTACTO,
                "month": MESES, "poutcome": RESULTADO_ANTERIOR}
    if variable in inversos:
        etiqueta = next(k for k, v in inversos[variable].items() if v == valor)
        return f"{NOMBRES[variable]}: {etiqueta.lower()}"
    if variable in ("default", "housing", "loan"):
        return f"{NOMBRES[variable]}: {'sí' if valor == 'yes' else 'no'}"
    if variable == "pdays":
        return f"{NOMBRES[variable]}: {'sin contacto previo' if valor == -1 else f'{int(valor)}'}"
    if variable == "balance":
        return f"{NOMBRES[variable]}: {int(valor):,} €".replace(",", ".")
    if variable == "age":
        return f"{NOMBRES[variable]}: {int(valor)} años"
    return f"{NOMBRES[variable]}: {int(valor)}"


# ------------------------------------------------------------------ Gráficas

def tema(grafica):
    return (grafica.configure(font="Archivo", background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=PIZARRA, titleColor=PIZARRA, labelFontSize=12, titleFontSize=12,
                            titleFontWeight=500, gridColor="#E3E9E6", domainColor=LINEA, tickColor=LINEA)
            .configure_legend(labelColor=TINTA, labelFontSize=12, symbolType="square", orient="top"))


def grafica_factores(aportes, cliente, maximo=6):
    """Barras divergentes: qué datos del cliente suben o bajan su probabilidad."""
    orden = aportes.abs().sort_values(ascending=False).index
    filas = sorted(({"Variable": describir(v, cliente), "Efecto": aportes[v]} for v in orden[:maximo]),
                   key=lambda fila: fila["Efecto"], reverse=True)
    resto = aportes[orden[maximo:]].sum()
    if abs(resto) >= 0.01:
        filas.append({"Variable": f"Otras {len(orden) - maximo} variables", "Efecto": resto})
    datos = pd.DataFrame(filas)
    datos["Dirección"] = np.where(datos["Efecto"] >= 0, "Sube la probabilidad", "Baja la probabilidad")

    barras = alt.Chart(datos).mark_bar(cornerRadius=3, height=18).encode(
        y=alt.Y("Variable:N", sort=None, title=None, axis=alt.Axis(labelLimit=280, ticks=False, domain=False)),
        x=alt.X("Efecto:Q", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False, domain=False)),
        color=alt.Color("Dirección:N", legend=None,
                        scale=alt.Scale(domain=["Sube la probabilidad", "Baja la probabilidad"], range=[VERDE, ORO])),
        tooltip=[alt.Tooltip("Variable:N", title="Dato del cliente"), alt.Tooltip("Dirección:N", title="Efecto"),
                 alt.Tooltip("Efecto:Q", title="Magnitud (log-odds)", format="+.2f")],
    )
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=TINTA, strokeWidth=1).encode(x="x:Q")
    return tema((barras + cero).properties(height=36 * len(datos)))


def grafica_distribucion(probabilidades, marcador=None, etiqueta_marcador=None):
    """Histograma de probabilidades con el umbral de decisión y, opcionalmente, la posición de un cliente."""
    conteos, bordes = np.histogram(probabilidades, bins=np.linspace(0, 1, 41))
    datos = pd.DataFrame({"inicio": bordes[:-1], "fin": bordes[1:], "clientes": conteos})
    datos["Rango"] = [f"{a:.1%} a {b:.1%}" for a, b in zip(datos["inicio"], datos["fin"])]
    datos["fin_barra"] = datos["fin"] - 0.004  # separación entre barras
    datos["base"] = 0

    barras = alt.Chart(datos).mark_bar(color=NIEBLA, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("inicio:Q", title="Probabilidad de contratar",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5, grid=False)),
        x2="fin_barra:Q",
        y=alt.Y("clientes:Q", title="Clientes (escala raíz cuadrada)", scale=alt.Scale(type="sqrt"),
                axis=alt.Axis(tickCount=4)),
        y2="base:Q",
        tooltip=[alt.Tooltip("Rango:N", title="Probabilidad"), alt.Tooltip("clientes:Q", title="Clientes", format=",")],
    )
    capas = [barras]
    referencias = [{"x": UMBRAL, "texto": "Umbral para llamar", "color": PIZARRA, "dy": 0, "guion": [4, 3]}]
    if marcador is not None:
        referencias.append({"x": marcador, "texto": etiqueta_marcador, "color": VERDE if marcador > UMBRAL else TINTA,
                            "dy": 16, "guion": [1, 0]})
    for ref in referencias:
        punto = pd.DataFrame({"x": [ref["x"]], "texto": [ref["texto"]]})
        alinear = "right" if ref["x"] > 0.8 else "left"
        capas.append(alt.Chart(punto).mark_rule(color=ref["color"], strokeWidth=2, strokeDash=ref["guion"]).encode(x="x:Q"))
        capas.append(alt.Chart(punto).mark_text(align=alinear, dx=6 if alinear == "left" else -6, dy=ref["dy"],
                                                baseline="top", color=ref["color"], fontSize=12, fontWeight=600)
                     .encode(x="x:Q", y=alt.value(0), text="texto:N"))
    return tema(alt.layer(*capas).properties(height=240))


def grafica_modelos():
    """Diagrama de puntos: ROC AUC en validación cruzada, con el modelo final resaltado."""
    datos = pd.DataFrame(MODELOS_CV, columns=["Modelo", "ROC AUC"])
    datos["Final"] = datos["Modelo"].str.contains("modelo final")
    base = alt.Chart(datos).encode(
        y=alt.Y("Modelo:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, ticks=False, domain=False)),
        x=alt.X("ROC AUC:Q", scale=alt.Scale(domain=[0.65, 0.81]), title="ROC AUC en validación cruzada",
                axis=alt.Axis(format=".2f", tickCount=5)),
    )
    puntos = base.mark_circle(size=150, opacity=1).encode(
        color=alt.condition("datum.Final", alt.value(VERDE), alt.value("#8996A2")),
        tooltip=["Modelo:N", alt.Tooltip("ROC AUC:Q", format=".3f")],
    )
    etiqueta = base.transform_filter("datum.Final").mark_text(align="left", dx=12, color=VERDE, fontWeight=700).encode(
        text=alt.Text("ROC AUC:Q", format=".3f"))
    return tema((puntos + etiqueta).properties(height=36 * len(datos)))


def grafica_importancia(importancia, maximo=10):
    """Barras horizontales: influencia promedio de cada variable."""
    datos = importancia.head(maximo).rename(index=NOMBRES).rename("Influencia").reset_index()
    datos.columns = ["Variable", "Influencia"]
    return tema(alt.Chart(datos).mark_bar(color=TINTA, cornerRadiusEnd=4, height=18).encode(
        y=alt.Y("Variable:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, ticks=False, domain=False)),
        x=alt.X("Influencia:Q", title="Influencia promedio en la predicción", axis=alt.Axis(labels=False, ticks=False)),
        tooltip=["Variable:N", alt.Tooltip("Influencia:Q", format=".3f", title="Influencia (log-odds)")],
    ).properties(height=34 * len(datos)))


# ------------------------------------------------------------------ Piezas de la interfaz

def comprobante(probabilidad, percentil=None):
    """HTML del comprobante de decisión para un cliente."""
    llamar = probabilidad > UMBRAL
    veces = probabilidad / TASA_PROMEDIO
    comparacion = (f"Es <strong>{veces:.1f} veces</strong> la tasa promedio de la campaña ({TASA_PROMEDIO:.1%})."
                   if veces >= 1 else f"Está por debajo de la tasa promedio de la campaña ({TASA_PROMEDIO:.1%}).")
    posicion = (f"<p class='nota'>Su probabilidad supera a la del <strong>{percentil:.0%}</strong> de los clientes "
                "históricos.</p>" if percentil is not None else "")
    if llamar:
        sello = '<div class="sello llamar">Llamar a este cliente</div>'
        nota = (f"En el conjunto de prueba, de cada 100 clientes con esta recomendación unos "
                f"{PRECISION_TEST * 100:.0f} contrataron el depósito.")
    else:
        sello = '<div class="sello esperar">No priorizar por ahora</div>'
        nota = "Puede quedar para una ronda posterior si hay capacidad de llamadas disponible."
    return f"""
<div class="comprobante" role="status" aria-live="polite">
  <p class="etiqueta">Probabilidad de que contrate el depósito</p>
  <p class="cifra">{probabilidad * 100:.0f}<span>%</span></p>
  <div class="regla {'llamar' if llamar else ''}" aria-hidden="true">
    <div class="relleno" style="width:{probabilidad * 100:.1f}%"></div>
    <div class="marca" style="left:{TASA_PROMEDIO * 100:.1f}%"><small>Promedio<br>{TASA_PROMEDIO:.1%}</small></div>
    <div class="marca" style="left:{UMBRAL * 100:.0f}%"><small>Umbral para llamar<br>{UMBRAL:.0%}</small></div>
  </div>
  {sello}
  <p class="nota">{comparacion}</p>
  {posicion}
  <p class="nota">{nota}</p>
</div>
"""


def cifras(items):
    """Fila de cifras clave: [(valor, texto, resaltar)]."""
    bloques = "".join(f'<div class="cifra-item"><p class="valor{" verde" if resaltar else ""}">{valor}</p>'
                      f'<p class="texto">{texto}</p></div>' for valor, texto, resaltar in items)
    st.markdown(f'<div class="cifras">{bloques}</div>', unsafe_allow_html=True)


def titulo_seccion(titulo, explicacion):
    st.markdown(f'<h3 class="seccion">{titulo}</h3><p class="explicacion">{explicacion}</p>', unsafe_allow_html=True)


def formulario_cliente():
    """Formulario de un cliente; devuelve un DataFrame de una fila con las variables del modelo."""
    st.markdown('<h3 class="grupo">Perfil del cliente</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    edad = c1.number_input("Edad", min_value=18, max_value=100, value=39, step=1)
    ocupacion = c2.selectbox("Ocupación", list(OCUPACIONES), index=1)
    estado_civil = c1.selectbox("Estado civil", list(ESTADO_CIVIL))
    educacion = c2.selectbox("Nivel educativo", list(EDUCACION), index=2)

    st.markdown('<h3 class="grupo">Situación financiera</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    saldo = c1.number_input("Saldo promedio anual (€)", min_value=-10000, max_value=150000, value=450, step=50,
                            help="Puede ser negativo si el cliente tiene sobregiros.")
    hipoteca = c2.toggle("Crédito hipotecario", value=True)
    consumo = c2.toggle("Crédito de consumo", value=False)
    mora = c2.toggle("Créditos en mora", value=False)

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


def pestana_cliente(pipeline, probabilidades_historicas):
    col_form, col_resultado = st.columns([1.15, 1], gap="large")
    with col_form:
        with st.container(border=True):
            cliente = formulario_cliente()
    datos = preparar(cliente)
    probabilidad = float(pipeline.predict_proba(datos)[:, 1][0])
    percentil = (None if probabilidades_historicas is None
                 else float((probabilidades_historicas < probabilidad).mean()))
    with col_resultado:
        st.markdown(comprobante(probabilidad, percentil), unsafe_allow_html=True)
        st.write("")
        titulo_seccion("Por qué esta recomendación",
                       "Los datos del cliente que más empujan su probabilidad hacia arriba o hacia abajo.")
        st.markdown('<div class="leyenda"><span><i style="background:var(--verde)"></i>Sube la probabilidad</span>'
                    '<span><i style="background:var(--oro)"></i>Baja la probabilidad</span></div>',
                    unsafe_allow_html=True)
        st.altair_chart(grafica_factores(contribuciones(pipeline, datos).iloc[0], datos.iloc[0]),
                        width="stretch")

    if probabilidades_historicas is not None:
        st.write("")
        titulo_seccion(f"Dónde queda este cliente frente a los {len(probabilidades_historicas):,} clientes históricos"
                       .replace(",", "."),
                       "La mayoría de clientes tiene una probabilidad baja de contratar; por eso el modelo recomienda "
                       "llamar solo a una fracción pequeña.")
        st.altair_chart(grafica_distribucion(probabilidades_historicas, probabilidad, "Este cliente"),
                        width="stretch")


def pestana_lista(pipeline):
    st.markdown('<p class="explicacion">Sube un archivo CSV con las mismas columnas del dataset limpio. Las columnas '
                '<code>y</code> y <code>duration</code> se ignoran si vienen en el archivo.</p>', unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 2])
    plantilla = pd.DataFrame(columns=COLUMNAS).to_csv(index=False).encode("utf-8")
    c1.download_button("Descargar plantilla CSV", plantilla, "plantilla_clientes.csv", "text/csv", width="stretch")
    ejemplo = CARPETA / "bank_marketing_datos_futuros.csv"
    if ejemplo.exists() and c2.button("Probar con clientes de ejemplo", width="stretch"):
        st.session_state["lista_ejemplo"] = True

    archivo = st.file_uploader("Archivo de clientes (CSV)", type="csv")
    if archivo is not None:
        st.session_state["lista_ejemplo"] = False
        try:
            datos = pd.read_csv(archivo)
        except Exception:
            st.error("No se pudo leer el archivo. Revisa que sea un CSV separado por comas.")
            return
    elif st.session_state.get("lista_ejemplo"):
        datos = pd.read_csv(ejemplo)
        st.caption("Mostrando los 12 clientes de ejemplo de bank_marketing_datos_futuros.csv.")
    else:
        st.caption("Cuando subas el archivo, aquí aparecerá la lista de clientes ordenada de mayor a menor probabilidad.")
        return

    faltantes = [c for c in COLUMNAS if c not in datos.columns]
    if faltantes:
        st.error(f"Al archivo le faltan estas columnas: {', '.join(faltantes)}. Usa la plantilla como referencia.")
        return
    if datos.empty:
        st.error("El archivo no tiene clientes. Agrega al menos una fila.")
        return

    resultado = datos.drop(columns=[c for c in ("y", "duration") if c in datos.columns]).copy()
    resultado.insert(0, "Probabilidad", pipeline.predict_proba(preparar(datos))[:, 1])
    resultado.insert(1, "Decisión", np.where(resultado["Probabilidad"] > UMBRAL, "Llamar", "No priorizar"))
    resultado = resultado.sort_values("Probabilidad", ascending=False).reset_index(drop=True)

    recomendados = resultado[resultado["Decisión"] == "Llamar"]
    cifras([
        (f"{len(resultado):,}".replace(",", "."), "clientes en la lista", False),
        (f"{len(recomendados):,}".replace(",", "."), "recomendados para llamar (probabilidad mayor que 50%)", True),
        (f"{recomendados['Probabilidad'].sum():.1f}", "depósitos esperados si se llama a los recomendados "
                                                       "(suma de sus probabilidades)", False),
    ])
    titulo_seccion("Distribución de la lista", "Cada barra agrupa a los clientes con una probabilidad parecida.")
    st.altair_chart(grafica_distribucion(resultado["Probabilidad"]), width="stretch")
    st.dataframe(
        resultado, hide_index=True, width="stretch",
        column_config={"Probabilidad": st.column_config.ProgressColumn(
            "Probabilidad", format="percent", min_value=0.0, max_value=1.0)},
    )
    st.download_button("Descargar lista priorizada", resultado.to_csv(index=False).encode("utf-8"),
                       "clientes_priorizados.csv", "text/csv", type="primary")


def pestana_modelo(pipeline, datos_historicos):
    cifras([
        (f"{PRECISION_TEST * 100:.0f} de 100", "clientes recomendados para llamar contrataron el depósito "
                                               "en el conjunto de prueba", True),
        (f"{PRECISION_TEST / TASA_PROMEDIO:.1f} veces", f"más efectivo que llamar sin priorizar, donde contrata el "
                                                        f"{TASA_PROMEDIO:.1%} de los clientes", False),
        (f"{ROC_AUC_TEST:.3f}", "ROC AUC en prueba: capacidad de ordenar a los clientes de más a menos probable", False),
    ])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        titulo_seccion("Modelos comparados",
                       "Se compararon siete modelos con validación cruzada de 10 particiones. Stacking, XGBoost y "
                       "Random Forest no tuvieron diferencias significativas (Tukey); se eligió XGBoost por su bajo "
                       "costo computacional y se optimizó con Bayes Search.")
        st.altair_chart(grafica_modelos(), width="stretch")
    with c2:
        if datos_historicos is not None:
            titulo_seccion("Variables que más influyen",
                           "Influencia promedio de cada variable en las predicciones, calculada sobre 4.000 clientes "
                           "históricos.")
            st.altair_chart(grafica_importancia(importancia_global(pipeline, datos_historicos)), width="stretch")


def main():
    st.set_page_config(page_title="A quién llamar | Depósitos a plazo", page_icon="🏦", layout="wide")
    st.markdown(ESTILOS, unsafe_allow_html=True)
    st.markdown("""
<div class="encabezado">
  <p class="contexto">Campaña de depósitos a plazo</p>
  <h1>¿A quién llamar?</h1>
  <p class="bajada">Estima la probabilidad de que un cliente contrate el depósito antes de llamarlo, entiende qué la
  explica y prioriza listas completas de clientes.</p>
</div>""", unsafe_allow_html=True)

    pipeline = cargar_pipeline()
    if pipeline is None:
        st.error("No se encontró el archivo pipeline_modelo.pkl. Ejecuta el notebook "
                 "02_Modelado_Evaluacion_Bank_Marketing.ipynb y copia el archivo junto a app.py.")
        st.stop()
    datos_historicos, probabilidades_historicas = historico(pipeline)

    tab_cliente, tab_lista, tab_modelo = st.tabs(["Un cliente", "Lista de clientes", "Cómo decide el modelo"])
    with tab_cliente:
        pestana_cliente(pipeline, probabilidades_historicas)
    with tab_lista:
        pestana_lista(pipeline)
    with tab_modelo:
        pestana_modelo(pipeline, datos_historicos)

    st.markdown(f'<p class="pie">Modelo XGBoost optimizado con Bayes Search. En el conjunto de prueba obtuvo un ROC AUC '
                f'de {ROC_AUC_TEST:.3f} y una precision de {PRECISION_TEST:.2f}. Proyecto final de Minería de Datos: '
                'Mariana Arbeláez, María Lucía Cadavid y Juan Manuel Arias.</p>', unsafe_allow_html=True)


main()
