import streamlit as st
from backend.models.database import init_db
from pages.optimization import show_optimization_page
from pages.other_services import show_other_services_page

# Configuración inicial
st.set_page_config(page_title="Optimizador de Pozos", layout="wide")
st.title("🛢️ Optimización de Distribución de Gas")

# Inicializar la base de datos
init_db()

# --- Pestañas ---
tabs = st.sidebar.radio("Selecciona una opción", ["Optimización", "Otros servicios"])

if tabs == "Optimización":
    show_optimization_page()
elif tabs == "Otros servicios":
    show_other_services_page()