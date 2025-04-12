import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Configuración de rutas
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.services.optimization_pipeline import run_pipeline
from backend.models.database import init_db, get_session
from backend.models.optimizacion import Optimizacion

# Inicializar la base de datos
engine = init_db()

# Configuración de la página
st.set_page_config(page_title="Optimizador de Pozos", layout="wide")
st.title("🛢️ Optimización de Distribución de Gas")

# --- Sección de configuración ---
with st.expander("Configuración de Optimización", expanded=True):
    qgl_limit = st.number_input(
        "Límite total de QGL (Gas de Levantamiento)", 
        min_value=1000, 
        max_value=10000, 
        value=4600,
        step=100,
        key="qgl_limit"
    )
    user = st.text_input("Usuario", value="admin", key="current_user")

# --- Subida de archivo ---
uploaded_file = st.file_uploader("Sube tu archivo CSV con datos de producción", type="csv")

if uploaded_file:
    # Crear directorios necesarios
    upload_dir = project_root / "static" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar archivo temporalmente
    temp_path = upload_dir / "temp_data.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Ejecutar pipeline
    if st.button("Ejecutar Optimización"):
        with st.spinner("Procesando datos..."):
            try:
                # Ejecutar el pipeline con el límite de QGL configurado
                result_file = run_pipeline(
                    csv_file_path=str(temp_path),
                    output_file=str(project_root / "static" / "results" / "output.txt"),
                    plot_file=str(project_root / "static" / "plots" / "wells_plot.png"),
                    user=user,
                    qgl_limit=qgl_limit  # Pasamos el valor del frontend
                )
                
                # Mostrar resultados
                st.success("¡Optimización completada!")
                
                # --- Mostrar gráfica ---
                st.header("Curvas de Optimización")
                plot_path = project_root / "static" / "plots" / "wells_plot.png"
                if plot_path.exists():
                    st.image(str(plot_path), use_container_width=True)
                
                # --- Datos de la base de datos ---
                st.header("Historial de Optimización")
                session = get_session(engine)
                optimizacion = session.query(Optimizacion).order_by(Optimizacion.fecha_ejecucion.desc()).first()
                
                if optimizacion:
                    st.metric("Producción Total", f"{optimizacion.produccion_total:.2f} bbl")
                    st.metric("QGL Total Utilizado", f"{optimizacion.qgl_total:.2f} Mscf")
                    
                    st.subheader("Detalle por Pozo")
                    pozos_data = [{
                        "Pozo": pozo.numero_pozo,
                        "Producción (bbl)": pozo.produccion_optima,
                        "QGL (Mscf)": pozo.qgl_optimo
                    } for pozo in optimizacion.pozos]
                    
                    st.dataframe(
                        pd.DataFrame(pozos_data).set_index("Pozo"),
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ Error durante la optimización: {str(e)}")
                st.exception(e)  # Muestra el traceback completo para debugging
            finally:
                if 'session' in locals():
                    session.close()