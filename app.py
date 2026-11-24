from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sismo_model import CLASSES, DATA_PATH, classify, train_and_evaluate

st.set_page_config(
    page_title="SISMO AI | Laboratorio Huanuco",
    page_icon="〰",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#101B2D"
PAPER = "#F3F4EF"
MINT = "#B7F36B"
BLUE = "#5969EA"
MUTED = "#657185"
CLASS_COLORS = {"I-IV": "#70B7A2", "V-VI": "#E7B85B", "VII-XII": "#E97961"}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#101B2D; --paper:#F3F4EF; --mint:#B7F36B; --blue:#5969EA; }
    html, body, [class*="css"] { font-family:Manrope, sans-serif; }
    .stApp { background:var(--paper); color:var(--ink); }
    [data-testid="stSidebar"] { background:#101B2D; }
    [data-testid="stSidebar"] * { color:#F3F4EF; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#AEB9C9; }
    .block-container { max-width:1440px; padding-top:2rem; padding-bottom:3rem; }
    .hero { background:#101B2D; color:#F3F4EF; border-radius:20px; padding:30px 34px;
            position:relative; overflow:hidden; min-height:205px; }
    .hero:after { content:""; position:absolute; width:290px; height:290px; right:8%; top:-175px;
                  border:1px solid #B7F36B; border-radius:50%; box-shadow:0 0 0 24px #b7f36b0d,0 0 0 50px #b7f36b0d; }
    .eyebrow { font:500 11px 'DM Mono',monospace; letter-spacing:2px; color:#B7F36B; text-transform:uppercase; }
    .hero h1 { font-size:clamp(34px,4vw,55px); letter-spacing:-2px; margin:9px 0 3px; line-height:1.04; color:#F3F4EF; }
    .hero p { color:#B6C2D2; font-size:14px; margin:8px 0 0; }
    .metric { background:#fff; border:1px solid #E1E4DC; border-radius:15px; padding:17px 19px; min-height:104px; }
    .metric-label { font:500 10px 'DM Mono',monospace; color:#657185; letter-spacing:1px; text-transform:uppercase; }
    .metric-value { font-size:28px; font-weight:800; letter-spacing:-1px; color:#101B2D; margin-top:5px; }
    .metric-note { color:#657185; font-size:11px; }
    .section-title { font-size:20px; letter-spacing:-.6px; font-weight:800; margin:10px 0 2px; }
    .section-note { font-size:12px; color:#657185; margin:0 0 13px; }
    .panel { background:#fff; border:1px solid #E1E4DC; border-radius:15px; padding:18px 20px; }
    .warning { background:#FFF2D7; border-left:4px solid #E7B85B; padding:12px 15px; border-radius:6px; color:#614B21; font-size:12px; }
    .prediction { background:#101B2D; padding:19px; border-radius:14px; color:#F3F4EF; }
    .prediction strong { font-size:27px; color:#B7F36B; letter-spacing:-1px; }
    [data-testid="stMetric"] { background:white; border:1px solid #E1E4DC; padding:14px 17px; border-radius:14px; }
    [data-testid="stMetricLabel"] p { font:500 10px 'DM Mono',monospace; letter-spacing:1px; text-transform:uppercase; color:#657185; }
    .stButton button[kind="primary"] { background:#5969EA; border:0; border-radius:9px; font-weight:700; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_result():
    path = Path(__file__).parent / "data" / "sismos_sinteticos.csv"
    return train_and_evaluate(pd.read_csv(path))


result = load_result()
data = result["data"]

st.session_state.setdefault("active_nav", "El reto")

with st.sidebar:
    st.markdown("## ◉ SISMO AI")
    st.caption("LABORATORIO DE INTENSIDAD · HUÁNUCO")
    st.markdown("---")
    st.caption("PROYECTO EDUCATIVO · MACHINE LEARNING")
    st.caption("Laboratorio de intensidad sísmica")
    st.markdown("---")
    st.caption("Dataset sintético · No es una alerta oficial.")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Laboratorio de datos · Perú / 2026</div>
      <h1>Leer el movimiento.<br>Entender el impacto.</h1>
      <p>Un experimento transparente de clasificación KNN sobre escenarios sísmicos simulados.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def choose_page(label):
    st.session_state["active_nav"] = label


navigation_columns = st.columns([1, 1, 1, 1.15, 1, 1.3], gap="small")
for column, label in zip(
    navigation_columns,
    ["El reto", "Datos", "Prototipo", "Desarrollo", "Evaluación"],
):
    column.button(
        label,
        key=f"top_nav_{label}",
        type="primary" if st.session_state["active_nav"] == label else "secondary",
        use_container_width=True,
        on_click=choose_page,
        args=(label,),
    )
report_path = Path(__file__).parent / "informe" / "SISMO_AI_informe.pdf"
if report_path.exists():
    navigation_columns[5].download_button(
        "Imprimir guía",
        data=report_path.read_bytes(),
        file_name="SISMO_AI_informe.pdf",
        mime="application/pdf",
        key="download_report_top",
        use_container_width=True,
    )
else:
    navigation_columns[5].button(
        "Imprimir guía", disabled=True, use_container_width=True
    )
st.markdown(
    '<div style="height:1px;background:#D8DDD5;margin:8px 0 17px"></div>',
    unsafe_allow_html=True,
)
st.write("")

page = {
    "El reto": "Resumen",
    "Datos": "Datos y método",
    "Prototipo": "Simulador KNN",
    "Desarrollo": "Desarrollo",
    "Evaluación": "Evaluación",
}[st.session_state["active_nav"]]

if page == "Resumen":
    cols = st.columns(4)
    metrics = [
        (f"{len(data):,}", "REGISTROS", "36 eventos simulados · 18 localidades"),
        (f"{data['magnitud'].mean():.1f}", "MAGNITUD MEDIA", "Valor generado, no observado"),
        (f"{data['profundidad_km'].mean():.0f} km", "PROFUNDIDAD MEDIA", "Distribución artificial"),
        (f"K = {result['best_k']}", "MODELO ELEGIDO", "Validación agrupada por evento"),
    ]
    for col, (value, label, note) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    st.markdown(
        '<div class="warning"><b>Experimento educativo, no alerta sísmica.</b> '
        'Dataset 100 % sintético. No representa observaciones del IGP, CENSIS ni USGS; '
        'no predice terremotos ni debe usarse para decisiones de seguridad.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.markdown('<div class="section-title">Geografía de los escenarios</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Epicentros aleatorios simulados; cada punto representa un evento.</div>', unsafe_allow_html=True)
        events = data.drop_duplicates("evento_id")
        fig = px.scatter_geo(
            events, lat="latitud", lon="longitud", color="magnitud", size="magnitud",
            hover_name="evento_id", hover_data={"fecha": True, "magnitud": ":.2f", "latitud": False, "longitud": False},
            color_continuous_scale=["#B7F36B", "#E7B85B", "#E97961"],
            scope="south america", projection="natural earth",
        )
        fig.update_geos(
            showcountries=True, countrycolor="#CED4CC", showland=True, landcolor="#E9EBE4",
            showocean=True, oceancolor="#F3F4EF", lataxis_range=[-22, 2], lonaxis_range=[-84, -65],
            showcoastlines=True, coastlinecolor="#C5CBC4",
        )
        fig.update_layout(
            height=390, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="white",
            geo=dict(bgcolor="white"), coloraxis_colorbar=dict(title="Magnitud"),
            font=dict(family="Manrope", color=INK),
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Clases generadas</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Distribución de etiquetas didácticas sintéticas.</div>', unsafe_allow_html=True)
        counts = data["intensidad"].value_counts().reindex(CLASSES, fill_value=0).rename_axis("Clase").reset_index(name="Registros")
        donut = px.pie(counts, values="Registros", names="Clase", hole=0.68, color="Clase", color_discrete_map=CLASS_COLORS)
        donut.update_traces(textinfo="percent", textfont_size=12, marker=dict(line=dict(color="white", width=3)))
        donut.update_layout(height=250, margin=dict(l=0, r=0, t=5, b=0), paper_bgcolor="white", legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(donut, use_container_width=True)
        st.markdown('<div class="panel"><b>Pregunta de investigación</b><br><span style="color:#657185;font-size:13px">¿Cómo influyen magnitud, profundidad y distancia en la clase que asigna un KNN cuando los eventos se mantienen separados?</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Comparación de vecinos K</div>', unsafe_allow_html=True)
    cv_frame = pd.DataFrame({"K": [str(k) for k in result["cv_scores"]], "F1 macro medio": list(result["cv_scores"].values())})
    cv_fig = px.bar(cv_frame, x="K", y="F1 macro medio", color="K", color_discrete_sequence=[BLUE, "#70B7A2", "#E7B85B", "#E97961"])
    cv_fig.update_layout(height=250, showlegend=False, margin=dict(l=0, r=0, t=8, b=0), paper_bgcolor="white", plot_bgcolor="white", yaxis_range=[0, 1])
    st.plotly_chart(cv_fig, use_container_width=True)
    st.markdown('<div class="section-title">El reto tecnológico</div>', unsafe_allow_html=True)
    st.markdown(
        "Crear una herramienta educativa que permita explorar cómo tres variables de un escenario "
        "sísmico sintético se relacionan con una clase didáctica mediante KNN. No estima fechas, "
        "magnitudes futuras ni riesgo real."
    )
    objective, product, boundary = st.columns(3)
    objective.markdown('<div class="panel"><b>Objetivo</b><p style="color:#657185">Entrenar, validar y explicar un clasificador KNN con escalado y separación por evento.</p></div>', unsafe_allow_html=True)
    product.markdown('<div class="panel"><b>Producto</b><p style="color:#657185">Dashboard, mapa, explorador de datos, evaluación y simulador interactivo.</p></div>', unsafe_allow_html=True)
    boundary.markdown('<div class="panel"><b>Alcance</b><p style="color:#657185">Prototipo educativo con datos sintéticos; no es una herramienta oficial de seguridad.</p></div>', unsafe_allow_html=True)

elif page == "Simulador KNN":
    st.markdown('<div class="section-title">Un escenario, una votación</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Modifica las variables y observa los vecinos del conjunto de entrenamiento.</div>', unsafe_allow_html=True)
    a, b = st.columns([0.75, 1.25], gap="large")
    with a:
        with st.form("simulator"):
            magnitude = st.slider("Magnitud (Mw)", 0.0, 10.0, 5.4, 0.1)
            depth = st.slider("Profundidad (km)", 0, 700, 45, 1)
            distance = st.slider("Distancia epicentral (km)", 0, 1500, 60, 1)
            k = st.select_slider("Número de vecinos K", options=[3, 5, 7, 9], value=result["best_k"])
            submitted = st.form_submit_button("ESTIMAR CLASE DIDÁCTICA", type="primary", use_container_width=True)
        if submitted:
            st.session_state["estimate"] = classify(result, magnitude, depth, distance, k)
            st.session_state["input_values"] = (magnitude, depth, distance)
    with b:
        estimate = st.session_state.get("estimate")
        if estimate:
            st.markdown(
                f'<div class="prediction"><span class="eyebrow">CLASIFICACIÓN EXPERIMENTAL · K={estimate["k"]}</span>'
                f'<br><strong>{estimate["prediccion"]}</strong><br><span style="color:#B6C2D2">Clase didáctica asignada por el modelo entrenado</span></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            votes = pd.DataFrame({"Clase": list(estimate["votos"].keys()), "Votos": list(estimate["votos"].values())})
            vote_fig = px.bar(votes, x="Clase", y="Votos", color="Clase", color_discrete_map=CLASS_COLORS)
            vote_fig.update_layout(height=220, showlegend=False, margin=dict(l=0, r=0, t=4, b=0), paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(vote_fig, use_container_width=True)
            neighbors = estimate["vecinos"][["evento_id", "localidad", "magnitud", "profundidad_km", "distancia_km", "intensidad"]].copy()
            neighbors.insert(0, "Distancia normalizada", estimate["distancias"].round(3))
            st.dataframe(neighbors.reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.markdown(
                """
                <div class="panel">
                  <div style="font:500 10px 'DM Mono',monospace;letter-spacing:1.5px;color:#5969EA">MÉTODO · DISTANCIA EUCLIDIANA</div>
                  <svg viewBox="0 0 640 230" role="img" aria-label="Diagrama del proceso KNN" style="width:100%;height:auto;margin:20px 0">
                    <path d="M75 116H540" stroke="#D8DDD5" stroke-width="3" stroke-dasharray="7 8"/>
                    <circle cx="112" cy="116" r="44" fill="#101B2D"/><circle cx="112" cy="116" r="10" fill="#B7F36B"/>
                    <circle cx="280" cy="70" r="17" fill="#70B7A2"/><circle cx="296" cy="159" r="17" fill="#E7B85B"/>
                    <circle cx="343" cy="96" r="17" fill="#70B7A2"/><circle cx="384" cy="154" r="17" fill="#E97961"/>
                    <circle cx="426" cy="88" r="17" fill="#70B7A2"/><circle cx="503" cy="116" r="43" fill="#5969EA"/>
                    <text x="112" y="188" text-anchor="middle" fill="#657185" font-size="12">ESCENARIO</text>
                    <text x="352" y="213" text-anchor="middle" fill="#657185" font-size="12">VECINOS DEL DATASET</text>
                    <text x="503" y="188" text-anchor="middle" fill="#657185" font-size="12">VOTACIÓN</text>
                    <text x="503" y="121" text-anchor="middle" fill="white" font-size="12" font-weight="bold">KNN</text>
                  </svg>
                  <b>¿Qué hace el simulador?</b>
                  <p style="color:#657185">Escala las variables, encuentra los K registros de entrenamiento más próximos y muestra la clase estimada junto con el reparto de votos. Cambia una variable a la vez para explorar su efecto.</p>
                  <span style="font:11px 'DM Mono',monospace;color:#5969EA">STANDARD SCALER · KNN · VOTOS VISIBLES</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
    st.markdown('<div class="warning">La clase obtenida es experimental y procede de datos generados artificialmente. No describe una intensidad observada ni constituye pronóstico, alerta o consejo de seguridad.</div>', unsafe_allow_html=True)

elif page == "Datos y método":
    st.markdown('<div class="section-title">Datos abiertos, límites claros</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">El origen de cada fila está declarado; no se mezclan observaciones reales y datos de práctica.</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning"><b>Procedencia: 100 % sintética.</b> Se generaron 36 eventos × 18 localidades (648 filas). Las coordenadas de eventos y los atributos fueron muestreados; la etiqueta se construyó con una fórmula pedagógica ruidosa y no con testimonios ni mediciones Mercalli.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("#### Diccionario de variables")
    dictionary = pd.DataFrame(
        [
            ("evento_id", "Grupo de separación; SYN-###", "Sintético"),
            ("fecha", "Fecha aleatoria dentro de 2000-2025", "Sintético"),
            ("latitud / longitud", "Epicentro aproximado para visualizar", "Sintético"),
            ("localidad", "Nombre de localidad peruana", "Etiqueta geográfica"),
            ("magnitud", "Valor simulado (escala de referencia Mw)", "Sintético"),
            ("profundidad_km", "Profundidad simulada en km", "Sintético"),
            ("distancia_km", "Distancia simulada a la localidad", "Sintético"),
            ("intensidad", "I-IV, V-VI o VII-XII; etiqueta artificial", "Sintético"),
            ("procedencia", "SINTETICO_EDUCATIVO", "Metadato"),
        ],
        columns=["Campo", "Descripción", "Origen"],
    )
    st.dataframe(dictionary, use_container_width=True, hide_index=True)
    st.markdown("#### Muestra del CSV")
    st.dataframe(data.head(12), use_container_width=True, hide_index=True)
    st.download_button("Descargar CSV (UTF-8)", data=DATA_PATH.read_bytes(), file_name="sismos_sinteticos.csv", mime="text/csv")
    st.markdown("#### Flujo de trabajo")
    st.markdown("1. Dividir eventos completos: 80 % entrenamiento y 20 % prueba (`GroupShuffleSplit`).\n2. Comparar K = 3, 5, 7, 9 con F1 macro y validación `GroupKFold` solo en entrenamiento.\n3. Ajustar `StandardScaler` dentro de un `Pipeline` en cada pliegue.\n4. Entrenar K seleccionado y evaluar una sola vez en eventos no vistos.\n5. Reportar métricas y confusión, sin presentar el desempeño simulado como desempeño real.")
    st.markdown("Fuentes recomendadas para un estudio futuro: [IGP](https://www.gob.pe/igp) · [CENSIS](https://ultimosismo.igp.gob.pe/) · [USGS](https://earthquake.usgs.gov/earthquakes/search/). Verificar si cada fuente proporciona intensidad observada antes de usarla como etiqueta.")

elif page == "Desarrollo":
    st.markdown('<div class="section-title">Del dato al prototipo</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Recorrido reproducible de preparación, entrenamiento, validación e integración.</div>', unsafe_allow_html=True)
    stages = st.columns(4)
    for column, number, title, detail in zip(
        stages,
        ["01", "02", "03", "04"],
        ["Generar", "Separar", "Entrenar", "Explorar"],
        [
            "648 filas artificiales reproducibles; cada fila declara su procedencia.",
            "80/20 por evento. Los grupos de prueba nunca aparecen en entrenamiento.",
            "Escalado + KNN en pipeline; K se compara solo con GroupKFold.",
            "Dashboard, mapa, métricas y votos de los vecinos del simulador.",
        ],
    ):
        column.markdown(
            f'<div class="metric"><div class="metric-label">FASE {number}</div>'
            f'<div class="metric-value" style="font-size:20px">{title}</div>'
            f'<div class="metric-note">{detail}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.markdown("#### Estructura del proyecto")
    st.code(
        "SismoAI_Huanuco/\n"
        "├── app.py                     # Dashboard y navegación funcional\n"
        "├── sismo_model.py             # Pipeline KNN y evaluación agrupada\n"
        "├── data/sismos_sinteticos.csv # 648 registros con procedencia\n"
        "├── scripts/generar_datos.py   # Dataset reproducible\n"
        "├── scripts/generar_informe.py # Informe PDF y gráficas\n"
        "├── tests/test_model.py        # Pruebas automatizadas\n"
        "├── requirements.txt\n"
        "└── README.md",
        language="text",
    )
    st.markdown("#### Comandos para ejecutar en Windows")
    st.code(
        "py -m venv .venv\n"
        r".\.venv\Scripts\Activate.ps1" "\n"
        "python -m pip install -r requirements.txt\n"
        "python -m unittest discover -s tests -v\n"
        "streamlit run app.py",
        language="powershell",
    )
    st.markdown("#### Decisiones técnicas")
    st.markdown(
        "- **Escalado en pipeline:** evita que las unidades de profundidad dominen las distancias KNN.\n"
        "- **Separación por evento:** evita que filas del mismo sismo se filtren entre entrenamiento y prueba.\n"
        "- **K seleccionado en entrenamiento:** mantiene el conjunto de prueba reservado para evaluación.\n"
        "- **Etiqueta de intensidad:** solo simulada en esta entrega; un catálogo de magnitud no basta para atribuir intensidad observada."
    )
    st.markdown('<div class="warning">Para una versión de investigación, reemplaza el CSV únicamente con datos autorizados y etiquetados con intensidad observada. Registra fuente, escala, limpieza y posibles sesgos antes de recalcular el informe.</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="section-title">Medir sin ocultar el contexto</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Conjunto de prueba formado por eventos completos no usados durante el ajuste.</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    display = [
        ("Exactitud", "exactitud"),
        ("Precisión macro", "precision_macro"),
        ("Recall macro", "recall_macro"),
        ("F1 macro", "f1_macro"),
    ]
    for col, (label, key) in zip(metric_cols, display):
        col.metric(label, f"{result['metrics'][key]:.1%}")
    st.caption(f"Prueba: {result['test']['evento_id'].nunique()} eventos independientes · {len(result['test'])} filas. Métricas aplican solo a la simulación.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        matrix = go.Figure(data=go.Heatmap(
            z=result["confusion"], x=result["labels"], y=result["labels"],
            colorscale=[[0, "#F3F4EF"], [1, BLUE]], text=result["confusion"], texttemplate="%{text}",
            colorbar=dict(title="Filas"),
        ))
        matrix.update_layout(title="Matriz de confusión · prueba", xaxis_title="Predicción", yaxis_title="Etiqueta sintética", height=400, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(matrix, use_container_width=True)
    with right:
        score_data = pd.DataFrame({"K": [str(key) for key in result["cv_scores"]], "F1 macro CV": list(result["cv_scores"].values())})
        chart = px.line(score_data, x="K", y="F1 macro CV", markers=True)
        chart.update_traces(line_color=BLUE, marker_size=11)
        chart.update_layout(title="Selección de K · GroupKFold", height=400, yaxis_range=[0, 1], paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(chart, use_container_width=True)
    st.markdown("#### Tres escenarios hipotéticos ejecutados")
    scenarios = [
        ("Cercano y moderado", 4.5, 30, 25),
        ("Intermedio", 5.5, 45, 100),
        ("Mayor magnitud", 6.8, 20, 40),
    ]
    scenario_rows = []
    for label, magnitude, depth, distance in scenarios:
        prediction = classify(result, magnitude, depth, distance)
        winning_votes = max(prediction["votos"].values())
        scenario_rows.append(
            (
                label,
                magnitude,
                depth,
                distance,
                prediction["prediccion"],
                f"{winning_votes}/{prediction['k']} vecinos",
            )
        )
    st.dataframe(
        pd.DataFrame(
            scenario_rows,
            columns=[
                "Escenario", "Magnitud", "Profundidad (km)", "Distancia (km)",
                "Clase del modelo", "Voto ganador",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Lectura: compara cómo cambia la clase al modificar el escenario y observa cuántos vecinos respaldan el voto ganador. Estas salidas explican el modelo sintético; no son intensidades observadas.")
    st.markdown('<div class="warning">El puntaje no expresa certeza sobre el mundo real. La etiqueta objetivo es simulada; para una investigación aplicada hacen falta registros observados, procedencia verificable, revisión de sesgo y validación externa.</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("SISMO AI · LABORATORIO EDUCATIVO · PROTOTIPO NO OFICIAL · DATOS SINTÉTICOS")
