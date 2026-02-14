import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import datetime as dt
from tensorflow.keras.models import load_model
import pickle
import os
import numpy as np

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

# Disclaimer
st.warning(
    "⚠️ **AVISO IMPORTANTE:** Las predicciones presentadas en esta plataforma son estimaciones generadas mediante modelos de machine learning "
    "y tienen fines exclusivamente educativos e informativos. No constituyen asesoramiento financiero ni recomendaciones de inversión. "
    "Los resultados pasados no garantizan rendimientos futuros. No nos hacemos responsables por las decisiones de inversión "
    "que los usuarios tomen basándose en esta información. Consulte siempre con un asesor financiero profesional antes de tomar decisiones de inversión."
)

# Recuperar valores de query params como respaldo si existen
query_etf = st.query_params.get('etf', None)
query_monto = st.query_params.get('monto', None)
query_fecha = st.query_params.get('fecha', None)

# Inicializar valores por defecto en session_state si no existen, usando query params como respaldo
if 'etf' not in st.session_state:
    st.session_state.etf = query_etf if query_etf else 'SPY'
if 'monto' not in st.session_state:
    st.session_state.monto = float(query_monto) if query_monto else 1000.0
if 'fecha_prediccion' not in st.session_state:
    st.session_state.fecha_prediccion = query_fecha if query_fecha else dt.datetime.now().strftime("%d/%m/%Y")

# Si hay query params, actualizar session_state con ellos (en caso de recarga)
if query_etf:
    st.session_state.etf = query_etf
if query_monto:
    st.session_state.monto = float(query_monto)
if query_fecha:
    st.session_state.fecha_prediccion = query_fecha

# Obtener valores del session_state
etf_ticker = st.session_state.etf
monto = st.session_state.monto
etiqueta_temporal = st.session_state.fecha_prediccion

# Asegurar que los query params permanezcan en la URL (para persistencia en recargas)
st.query_params['etf'] = etf_ticker
st.query_params['monto'] = str(monto)
st.query_params['fecha'] = etiqueta_temporal

# Normalizar y unificar formato de fechas en todos los DataFrames

def normalize_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index = df.index.strftime("%d/%m/%Y")
    return df

# --- DESCARGA DE VARIABLES EXÓGENAS Y DATOS ETF ---

# Fechas dinámicas
_df_temp = yf.Ticker(etf_ticker).history(period="max")[['Close']]
date_inicio = _df_temp.index.min().strftime("%Y-%m-%d")
date_hoy = dt.datetime.strptime(etiqueta_temporal, "%d/%m/%Y").strftime("%Y-%m-%d")

df_etf = yf.Ticker(etf_ticker).history(start=date_inicio, end=date_hoy)[['Close']]
df_etf = df_etf.rename(columns={'Close': etf_ticker})

# Normalizar fechas del ETF objetivo antes de cualquier join
df_etf = normalize_index(df_etf)

# Obtener datos de FRED usando alternativa (cargar desde CSV del dataset)
try:
    dataset_path = 'DataSet_General/DATASET_LIMPIO_E_IMPUTADO.csv'
    if os.path.exists(dataset_path):
        df_dataset = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
        exog_vars_fred = ['DGS10','DGS2','VIXCLS']
        data_exog_fred = df_dataset[exog_vars_fred].loc[date_inicio:date_hoy]
    else:
        # Si no existe el CSV, crear DataFrame vacío con las columnas necesarias
        data_exog_fred = pd.DataFrame(columns=['DGS10','DGS2','VIXCLS'])
except Exception as e:
    st.error(f"Error al cargar datos de FRED: {e}")
    data_exog_fred = pd.DataFrame(columns=['DGS10','DGS2','VIXCLS'])

activos_exogenos = ['GC=F','CL=F','DX-Y.NYB','^SPGSCI','^DJT','HG=F','^VXN']
data_exog_yf = pd.DataFrame()
for activo in activos_exogenos:
    precios_cierre = yf.Ticker(activo).history(start=date_inicio, end=date_hoy)['Close']
    data_exog_yf[activo] = precios_cierre

# Descargar todas las ETFs como variables exógenas, excepto la ETF objetivo
etfs_exogenas = [etf for etf in ['DIA', 'QQQ', 'XLK', 'IWV', 'SPY']]
data_etfs_exog = pd.DataFrame()
for etf in etfs_exogenas:
    precios_cierre = yf.Ticker(etf).history(start=date_inicio, end=date_hoy)['Close']
    data_etfs_exog[etf] = precios_cierre
# Evitar duplicar columna si el ETF objetivo también está en las exógenas
if etf_ticker in data_etfs_exog.columns:
    data_etfs_exog = data_etfs_exog.drop(columns=[etf_ticker])
# Normalizar fechas
data_etfs_exog = normalize_index(data_etfs_exog)
data_exog_fred = normalize_index(data_exog_fred)
data_exog_yf = normalize_index(data_exog_yf)

# Inicializar df_full con el ETF objetivo
df_full = df_etf.copy()
# Unir al DataFrame principal
df_full = df_full.join(data_etfs_exog, how='left')
df_full = df_full.join(data_exog_fred, how='left')
df_full = df_full.join(data_exog_yf, how='left')

# --- CARGA DE MODELOS Y SCALERS ---
carpeta_modelos = "modelos_directos_recientes"
modelos_directos_recientes = []

for dia in range(1, 6):
    modelo_path = os.path.join(carpeta_modelos, f"modelo_dia{dia}.h5")
    scaler_X_path = os.path.join(carpeta_modelos, f"scaler_X_dia{dia}.pkl")
    scaler_y_path = os.path.join(carpeta_modelos, f"scaler_y_dia{dia}.pkl")
    
    modelo = load_model(modelo_path)
    with open(scaler_X_path, "rb") as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)
    
    modelos_directos_recientes.append({
        "dia": dia,
        "modelo": modelo,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y
    })

# --- PREPARAR ENTRADA Y REALIZAR PREDICCIÓN ---
# Definir las columnas de features (deben coincidir con el entrenamiento)
feature_cols = ['SPY', 'DIA', 'QQQ', 'XLK', 'IWV', 'DGS10', 'DGS2', 'VIXCLS', 'GC=F', 'CL=F', 'DX-Y.NYB', '^SPGSCI', '^DJT', 'HG=F', '^VXN']

# Seleccionar la ventana más reciente para predecir (WINDOW=30)
WINDOW = 30

df_pred = df_full[feature_cols].dropna()
if len(df_pred) < WINDOW:
    st.error("No hay suficientes datos para realizar la predicción.")
    st.stop()

# Verificar que todas las columnas de features estén presentes en df_full
missing_cols = [col for col in feature_cols if col not in df_full.columns]
if missing_cols:
    st.error(f"Faltan las siguientes columnas en los datos descargados: {missing_cols}. Por favor, verifica que todos los activos estén disponibles en Yahoo Finance y FRED.")
    st.stop()

X_input = df_pred.values[-WINDOW:]
X_input = X_input.reshape(1, WINDOW, len(feature_cols))

# Ejecutar predicción para cada horizonte (día 1 a 5)
predicciones = []
for m in modelos_directos_recientes:
    modelo = m['modelo']
    scaler_X = m['scaler_X']
    scaler_y = m['scaler_y']
    # Escalar entrada
    X_scaled = scaler_X.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)
    # Predecir
    pred_scaled = modelo.predict(X_scaled, verbose=0)
    pred_real = scaler_y.inverse_transform(pred_scaled)[0]
    predicciones.append(pred_real)

# Función para calcular días hábiles (salta sábados y domingos)
def calcular_dias_habiles(fecha_inicio, num_dias):
    fecha_actual = dt.datetime.strptime(fecha_inicio, "%d/%m/%Y")
    fechas_habiles = []
    dias_agregados = 0
    
    while dias_agregados < num_dias:
        fecha_actual += dt.timedelta(days=1)
        # 0=Lunes, 6=Domingo
        if fecha_actual.weekday() < 5:  # Si no es sábado (5) ni domingo (6)
            fechas_habiles.append(fecha_actual.strftime("%d/%m/%Y"))
            dias_agregados += 1
    
    return fechas_habiles

# Mostrar resultados
st.subheader(f"Predicción para {etf_ticker} (monto: ${monto:,.2f})")

# Precio actual (último disponible) del ETF objetivo
precio_hoy = df_etf[etf_ticker].dropna().iloc[-1]
fecha_precio = df_etf[etf_ticker].dropna().index[-1]
st.metric(
    label=f"Precio de {etf_ticker} al {fecha_precio}",
    value=f"${precio_hoy:,.2f}"
)

# Calcular fechas de predicción (días hábiles)
fechas_prediccion = calcular_dias_habiles(etiqueta_temporal, 5)

resultados = []
for i, pred in enumerate(predicciones):
    valor_estimado = monto * (pred[feature_cols.index(etf_ticker)] / X_input[0, -1, feature_cols.index(etf_ticker)])
    resultados.append({
        'Fecha': fechas_prediccion[i],
        'Precio Estimado': pred[feature_cols.index(etf_ticker)],
        'Valor Estimado Inversión': valor_estimado
    })
df_resultados = pd.DataFrame(resultados)

# Analizar oportunidad de inversión
valor_minimo = df_resultados['Valor Estimado Inversión'].min()
idx_minimo = df_resultados['Valor Estimado Inversión'].idxmin()
fecha_minima = df_resultados.loc[idx_minimo, 'Fecha']

# Buscar si hay un valor mayor después del mínimo
valores_posteriores = df_resultados.loc[idx_minimo+1:, 'Valor Estimado Inversión']

if not valores_posteriores.empty and valores_posteriores.max() > valor_minimo:
    # Hay oportunidad de ganancia
    valor_maximo_posterior = valores_posteriores.max()
    idx_maximo_posterior = valores_posteriores.idxmax()
    fecha_maxima = df_resultados.loc[idx_maximo_posterior, 'Fecha']
    ganancia_estimada = valor_maximo_posterior - valor_minimo
    porcentaje_ganancia = ((valor_maximo_posterior - valor_minimo) / valor_minimo) * 100
    
    st.success(
        f"💰 **OPORTUNIDAD DE INVERSIÓN DETECTADA**\n\n"
        f"📈 **Recomendación:** Invertir en la fecha **{fecha_minima}** y vender en la fecha **{fecha_maxima}**\n\n"
        f"💵 **Ganancia estimada:** ${ganancia_estimada:,.2f} ({porcentaje_ganancia:.2f}%)\n\n"
        f"📊 Valor en compra: ${valor_minimo:,.2f} | Valor en venta: ${valor_maximo_posterior:,.2f}"
    )
else:
    # No hay oportunidad de ganancia
    st.error(
        f"⚠️ **RECOMENDACIÓN: ABSTENERSE DE INVERTIR**\n\n"
        f"📉 Según las predicciones, no se detecta una oportunidad favorable de inversión en este periodo.\n\n"
        f"El modelo no proyecta un incremento significativo del valor después del punto mínimo detectado."
    )

st.dataframe(df_resultados, use_container_width=True, hide_index=True)

# Visualización de resultados
st.subheader("Proyección de la Inversión (5 días)")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_resultados['Fecha'],
    y=df_resultados['Valor Estimado Inversión'],
    mode='lines+markers',
    name='Valor Estimado Inversión',
    line=dict(color='#3b82f6', width=4),
    marker=dict(size=10, color='#3b82f6', line=dict(color='white', width=2)),
    fill='tozeroy',
    fillcolor='rgba(59, 130, 246, 0.1)'
))
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(title='Fecha', gridcolor='#30363d'),
   yaxis=dict(title='Valor Estimado Inversión', gridcolor='#30363d'),
    height=400,
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Precio Estimado del ETF (5 días)")
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=df_resultados['Fecha'],
    y=df_resultados['Precio Estimado'],
    marker_color='#636efa',
    name='Precio Estimado'
))
fig2.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Fecha'),
    yaxis=dict(title='Precio Estimado'),
    height=350,
    showlegend=False
)
st.plotly_chart(fig2, use_container_width=True)

graphics = st.button("📊 Ver Gráficos", use_container_width=True)
if graphics:
    st.success(f"📊 Redirigiendo a la página de gráficos sobre Métricas de Evaluación...")
    st.switch_page('pages/Page_3.py')

