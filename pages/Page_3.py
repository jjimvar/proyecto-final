import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import datetime as dt
import pickle
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

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

# --- SECCIÓN 9.2: GRÁFICAS Y MÉTRICAS DE EVALUACIÓN ---

st.markdown("---")
st.title("📊 Métricas de Evaluación del Modelo")

# Cargar datos necesarios para las gráficas
# Se requieren los resultados del Walk Forward Testing del notebook Modelo_Predictivo.ipynb

try:
    # Intentar cargar los datos del walk forward testing
    with open("datos_wf_testing.pkl", "rb") as f:
        datos_wf = pickle.load(f)
    
    actuals_wf_ft = datos_wf.get('actuals_wf_ft')
    preds_wf_ft = datos_wf.get('preds_wf_ft')
    
    nombres_etfs = ['SPY', 'DIA', 'QQQ', 'XLK', 'IWV']
    
    # Colores para valores reales y predicciones
    color_real = "#0011ff"       # Cian para valores reales
    color_prediccion = "#c01515" # Rojo para predicciones
    
    # --- GRÁFICA 1: Comparación de valores reales vs predicciones ---
    st.subheader("Comparación: Valores Reales vs Predicciones")
    
    # Selectbox para elegir ETF
    etf_seleccionado = st.selectbox(
        "Selecciona un ETF:",
        nombres_etfs,
        index=0
    )
    
    fig = go.Figure()
    
    # Obtener índice del ETF seleccionado
    idx_etf = nombres_etfs.index(etf_seleccionado)
    
    # Valores reales (línea sólida)
    fig.add_trace(go.Scatter(
        y=actuals_wf_ft[:, idx_etf],
        mode='lines',
        name=f'{etf_seleccionado} (Real)',
        line=dict(color=color_real, width=3)
    ))
    
    # Predicciones (línea punteada)
    fig.add_trace(go.Scatter(
        y=preds_wf_ft[:, idx_etf],
        mode='lines',
        name=f'{etf_seleccionado} (Predicción)',
        line=dict(color=color_prediccion, width=2.5, dash='dot')
    ))
    
    fig.update_layout(
        title="Walk Forward Testing",
        xaxis_title="Paso",
        yaxis_title="Precio ($)",
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- GRÁFICA 2: Métricas de desempeño por ETF ---
    st.subheader("Métricas de Desempeño por ETF")
    
    metricas_por_etf = {}
    
    for i, etf in enumerate(nombres_etfs):
        y_true = actuals_wf_ft[:, i]
        y_pred = preds_wf_ft[:, i]
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        metricas_por_etf[etf] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE (%)': mape * 100
        }
    
    df_metricas = pd.DataFrame(metricas_por_etf).T
    df_metricas = df_metricas.round(4)
    
    st.dataframe(df_metricas, use_container_width=True)
    
    # Gráfica de barras con las métricas (excluyendo R²)
    fig_metricas = go.Figure()
    
    for metrica in df_metricas.columns:
        # Ocultar R² en la gráfica
        if metrica == 'R²':
            continue
        
        fig_metricas.add_trace(go.Bar(
            name=metrica,
            x=df_metricas.index,
            y=df_metricas[metrica],
            text=df_metricas[metrica].round(3),
            textposition='outside'
        ))
    
    fig_metricas.update_layout(
        title="Comparación de Métricas por ETF",
        xaxis_title="ETF",
        yaxis_title="Valor",
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_metricas, use_container_width=True)

    # --- GRÁFICA 3: Evolución de métricas por día de predicción ---
    st.subheader("Evolución de Métricas por Día de Predicción")

    metricas_por_dia = []
    dias_disponibles = []
    dias_faltantes = []

    for dia in range(1, 6):
        archivo_dia = f"walk_forward_dia{dia}.pkl"
        if not os.path.exists(archivo_dia):
            dias_faltantes.append(dia)
            continue

        with open(archivo_dia, "rb") as f:
            datos_dia = pickle.load(f)

        if isinstance(datos_dia, dict) and isinstance(datos_dia.get("metricas"), dict):
            mae = float(datos_dia["metricas"].get("MAE", np.nan))
            rmse = float(datos_dia["metricas"].get("RMSE", np.nan))
            mape = float(datos_dia["metricas"].get("MAPE", np.nan))
        else:
            y_true = np.array(datos_dia.get("actuals", []))
            y_pred = np.array(datos_dia.get("preds", []))
            if y_true.size == 0 or y_pred.size == 0:
                dias_faltantes.append(dia)
                continue
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = mean_absolute_percentage_error(y_true, y_pred)

        metricas_por_dia.append({
            "Día": dia,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape * 100
        })
        dias_disponibles.append(dia)

    if metricas_por_dia:
        df_metricas_dia = pd.DataFrame(metricas_por_dia).sort_values("Día")

        metrica_seleccionada = st.selectbox(
            "Selecciona una métrica:",
            ["MAE", "RMSE", "MAPE (%)"],
            index=0
        )

        fig_evolucion = go.Figure()
        fig_evolucion.add_trace(go.Scatter(
            x=df_metricas_dia["Día"],
            y=df_metricas_dia[metrica_seleccionada],
            mode="lines+markers",
            name=metrica_seleccionada,
            line=dict(color="#22c55e", width=3),
            marker=dict(size=8, color="#22c55e", line=dict(color="white", width=1))
        ))

        fig_evolucion.update_layout(
            title="Evolución por Horizonte (1-5 días)",
            xaxis_title="Día de Predicción",
            yaxis_title=metrica_seleccionada,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=400,
            hovermode="x unified"
        )

        st.plotly_chart(fig_evolucion, use_container_width=True)
        st.dataframe(df_metricas_dia.round(4), use_container_width=True, hide_index=True)

        if dias_faltantes:
            st.info(
                "No se encontraron archivos de walk forward para los días: "
                + ", ".join(str(d) for d in dias_faltantes)
                + ". Si deseas ver los 5 días, guarda los archivos 'walk_forward_diaN.pkl' al ejecutar el notebook."
            )
    else:
        st.warning(
            "No hay datos disponibles para mostrar la evolución por día. "
            "Ejecuta el notebook y guarda los archivos 'walk_forward_dia1.pkl' a 'walk_forward_dia5.pkl'."
        )
    
except FileNotFoundError:
    st.warning(
        "⚠️ Los datos del Walk Forward Testing no se encuentran disponibles. "
        "Por favor, ejecuta primero el notebook 'Modelo_Predictivo.ipynb' y asegúrate de guardar los datos con: "
        "`pickle.dump({'actuals_wf_ft': actuals_wf_ft, 'preds_wf_ft': preds_wf_ft}, open('datos_wf_testing.pkl', 'wb'))`"
    )
except Exception as e:
    st.error(f"Error al cargar los datos: {str(e)}")