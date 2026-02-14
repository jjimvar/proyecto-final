# Guía de Instalación

## Primera vez - Crear entorno virtual e instalar dependencias

```powershell
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Para pandas-datareader compatible con Python 3.13
pip install git+https://github.com/pydata/pandas-datareader.git
```

## Próximas veces - Solo activar el entorno

```powershell
# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Ejecutar el notebook
jupyter notebook Modelo_Predictivo.ipynb

# O ejecutar Streamlit
streamlit run Home.py
```

## Si quieres que otros ejecuten tu proyecto

Solo necesitan:
1. Clonar el repositorio
2. Crear el entorno virtual
3. Instalar `requirements.txt`
4. Todo funcionará sin problemas de dependencias
