# Predicción de Precios de ETFs Estadounidenses con LSTM
Proyecto final del Bootcamp de Data Science de 4Geeks Academy que utiliza redes neuronales LSTM para predecir los precios de los 5 ETFs más importantes de Estados Unidos con un horizonte de 5 días. Realizado por [Catherine Cazorla](https://github.com/cathycaz)], [Carlos Mairena](https://github.com/carlos060495) y [Jesús Jiménez](https://github.com/jjimvar)

## 📊 Descripción del Proyecto
Este proyecto implementa un sistema de predicción de precios basado en aprendizaje profundo que:
- **Predice** los precios de cierre de 5 ETFs principales: SPY, QQQ, IWV, DIA, XLK
- **Utiliza** una red LSTM independiente para cada día de predicción (5 modelos totales)
- **Analiza** una ventana temporal de 30 días históricos para hacer predicciones
- **Proporciona** tanto un notebook interactivo como una aplicación web con Streamlit

## 🎯 Objetivos
- Desarrollar modelos predictivos de series temporales usando LSTM
- Alcanzar predicciones fiables de precios en mercados financieros
- Crear una herramienta visual e interactiva para usuarios
- Demostrar capacidades en Deep Learning aplicado a finanzas

## 🏗️ Estructura del Proyecto
```
proyecto-final/
├── README.md                          # Este archivo
├── 0.INSTALACION.md                     # Guía de instalación detallada
├── requirements.txt                   # Dependencias del proyecto
|
├── Home.py                        # Página principal streamlit
|    ├── pages/
|    |    ├── Page_2.py                  # Página de análisis
|    |    └── Page_3.py                  # Página de predicciones
|    ├── datos_wf_testing.pkl              # Datos generales para el Walk Foward (Page_3.py)
|    ├── modelo_dia1.h5                    # Predicción para día 1 (Page_2.py) 
     ├── modelo_dia2.h5                    # Predicción para día 2
     ├── modelo_dia3.h5                    # Predicción para día 3
     ├── modelo_dia4.h5                    # Predicción para día 4
     └── modelo_dia5.h5                    # Predicción para día 5
│
├── Modelo_Predictivo.ipynb        # Notebook con análisis y entrenamiento
     ├── DataSet_General/
     │   └── DATASET_LIMPIO_E_IMPUTADO.csv  # Dataset procesado y listo para usar
     └── modelos_directos_recientes/    # Modelos preentrenados (.h5)
         ├── modelo_dia1.h5            # Predicción para día 1
         ├── modelo_dia2.h5            # Predicción para día 2
         ├── modelo_dia3.h5            # Predicción para día 3
         ├── modelo_dia4.h5            # Predicción para día 4
         └── modelo_dia5.h5            # Predicción para día 5
```

## 🔧 Tecnologías Utilizadas
- **Deep Learning**: TensorFlow/Keras (LSTM)
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Visualización**: Plotly, Matplotlib
- **Web Framework**: Streamlit
- **Datos Financieros**: yfinance
- **Otros**: Python 3.13+, h5py

## 📈 Arquitectura del Modelo
### Componentes Principales:
- **5 Modelos LSTM** - Uno para cada día de predicción (1 a 5 días adelante)
- **Ventana temporal**: 30 días históricos como entrada
- **Arquitectura LSTM**: Optimizada para capturar dependencias temporales en series financieras
- **ETFs objetivo**:
  - SPY: S&P 500
  - QQQ: Nasdaq-100
  - IWV: Russell 3000
  - DIA: Dow Jones Industrial Average
  - XLK: Technology Sector

## 🚀 Inicio Rápido

### Instalación

```powershell
# 1. Clonar el repositorio
git clone <url-repositorio>
cd proyecto-final

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Instalar pandas-datareader compatible con Python 3.13
pip install git+https://github.com/pydata/pandas-datareader.git
```

### Usar el Notebook

```powershell
# Activar el entorno virtual (si no está activo)
.\.venv\Scripts\Activate.ps1

# Abrir Jupyter
jupyter notebook Fuente/Modelo_Predictivo.ipynb
```

En el notebook encontrarás:
- Exploración y análisis del dataset
- Preparación de datos
- Entrenamiento de modelos LSTM
- Evaluación y validación
- Visualización de resultados

### Usar la Aplicación Web

```powershell
# Activar el entorno virtual (si no está activo)
.\.venv\Scripts\Activate.ps1

# Ejecutar Streamlit
streamlit run "Web app/Home.py"
```

La aplicación estará disponible en `http://localhost:8501`

## 📝 Dataset

El proyecto utiliza **DATASET_LIMPIO_E_IMPUTADO.csv** ubicado en `Fuente/DataSet_General/`

**Características del dataset**:
- Datos históricos de los 5 ETFs principales
- Limpieza y tratamiento de valores faltantes completado
- Listo para entrenar modelos
- Incluye variables como: Apertura, Cierre, Alto, Bajo, Volumen

## 🎓 Detalles del Entrenamiento

Para información detallada sobre:
- Metodología de preprocessing
- Arquitectura específica de cada modelo
- Métricas de evaluación
- Resultados alcanzados
- Análisis de predicciones

Consulta el notebook: **Fuente/Modelo_Predictivo.ipynb**

## 💾 Modelos Preentrenados

El proyecto incluye 5 modelos LSTM preentrenados listos para usar:
- `modelo_dia1.h5` - Predice precio para el día +1
- `modelo_dia2.h5` - Predice precio para el día +2
- `modelo_dia3.h5` - Predice precio para el día +3
- `modelo_dia4.h5` - Predice precio para el día +4
- `modelo_dia5.h5` - Predice precio para el día +5

Ubicación: `Fuente/modelos_directos_recientes/`

## 📊 Páginas de la Aplicación

### Home.py
Página principal con información general del proyecto y visualización de precios actuales.

### Page_2.py
Análisis técnico y visualización de tendencias históricas de los ETFs.

### Page_3.py
Predicciones futuras utilizando los modelos LSTM entrenados y comparación con valores reales.

## ⚙️ Requisitos del Sistema

- **Python**: 3.11+
- **Memoria**: Al menos 4GB RAM (recomendado 8GB)
- **Disco**: ~2GB para modelos y datos
- **Sistema Operativo**: Windows, macOS o Linux

## 🔍 Validación de Instalación

Después de instalar las dependencias, verifica que todo funciona:

```powershell
# Verificar instalación
python -c "import tensorflow; import streamlit; import pandas; print('Instalación OK')"
```

## 🚦 Troubleshooting

Si encuentras problemas:

1. **Error de pandas-datareader**: Usa la instalación especial de GitHub incluida en INSTALACION.md
2. **Error de TensorFlow**: Puede requerir versiones específicas de CUDA (opcional para CPU)
3. **Error de puertos Streamlit**: Intenta correr con `streamlit run "Web app/Home.py" --server.port 8502`

Consulta **INSTALACION.md** para más detalles.

## 📚 Recursos Útiles
- [LSTM y Series Temporales](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)
- [Documentación Streamlit](https://docs.streamlit.io/)
- [Análisis Técnico Financiero](https://es.wikipedia.org/wiki/An%C3%A1lisis_t%C3%A9cnico)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)

## 👨‍🎓 Sobre este Proyecto
Este proyecto fue desarrollado como trabajo final del **Bootcamp de Data Science en 4Geeks Academy**, demostrando competencias en:
- Machine Learning y Deep Learning
- Análisis de Series Temporales
- Desarrollo de Aplicaciones Web
- Visualización de Datos
- Buenas prácticas en Data Science

## 📄 Licencia
Este proyecto es de código abierto. Siéntete libre de usarlo, modificarlo y distribuirlo.

## 📧 Contacto
Para dudas, sugerencias o reportar problemas, contacta con el correo jjimenezvargas907@gmail.com.

---

**Última actualización**: Febrero 2026
