import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt

# Fondo personalizado y logo centrado
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #0e0d0c;
    }}
    .center-logo {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1.5rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo.jpeg", width=350)

# Lista de ETFs
lista_etfs = ["SPY", "DIA", "QQQ", "XLK", "IWV"]

# Inicializar valores en session_state si no existen
if 'etf' not in st.session_state:
    st.session_state.etf = lista_etfs[0]
if 'monto' not in st.session_state:
    st.session_state.monto = 1.0

st.set_page_config(
    page_title="ETF Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("ETF Investment Prediction Platform")
st.markdown("<p style='text-align: center; color: #94a3b8;'>Leverage advanced machine learning to forecast ETF performance and optimize your portfolio allocation.</p>", unsafe_allow_html=True)

# --- Selección de ETF ---
etf_ticker = st.selectbox(
    "Selecciona el ETF:", 
    lista_etfs,
    index=lista_etfs.index(st.session_state.etf) if st.session_state.etf in lista_etfs else 0,
    key='etf_selector'
)

# --- Descargar datos históricos completos ---
ticker = yf.Ticker(etf_ticker)
df = ticker.history(period="max")[['Close']]

if not df.empty:
    df = df.reset_index()
    activo_min = df['Date'].min().to_pydatetime()
    activo_max = df['Date'].max().to_pydatetime()
    fecha_inicio, fecha_fin = st.slider(
        'Seleccione las fechas de estudio',
        min_value=activo_min,
        max_value=activo_max,
        value=[activo_min, activo_max],
        format="DD/MM/YYYY"
    )
    # Filtrar el DataFrame según el rango seleccionado
    df_filt = df[(df['Date'] >= fecha_inicio) & (df['Date'] <= fecha_fin)]
    st.header(f'Cotización Bursátil de {etf_ticker} (USD)', divider='gray')
    st.line_chart(
        df_filt.set_index('Date'),
        y='Close'
    )
else:
    st.warning("No hay datos disponibles para el ETF seleccionado.")

# --- Campo para ingresar monto a invertir ---
st.markdown("---")
monto = st.number_input(
    "Monto a Invertir (USD):", 
    min_value=1.0, 
    value=st.session_state.monto,
    step=1.0, 
    format="%.2f",
    key='monto_input'
)

# --- Botón de predicción al final de la página ---
# Actualizar session_state con los valores actuales
st.session_state.etf = etf_ticker
st.session_state.monto = monto

predict = st.button("🚀 Generar Predicción", use_container_width=True)
etiqueta_temporal = dt.datetime.now().strftime("%d/%m/%Y")
if predict:
    st.session_state.fecha_prediccion = etiqueta_temporal
    # Guardar en query params como respaldo
    st.query_params['etf'] = etf_ticker
    st.query_params['monto'] = str(monto)
    st.query_params['fecha'] = etiqueta_temporal
    st.success(f"Redirigiendo a la página de predicción para {st.session_state.etf}...")
    st.switch_page('pages/Page_2.py')