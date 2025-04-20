import streamlit as st
from components.file_upload import show_file_upload
from components.optimization_charts import show_optimization_charts
from components.history import show_history

def show_optimization_page():
    uploaded_file, input_data_path = show_file_upload()

    # Mostrar pestañas de optimización
    tab1, tab2, tab3 = st.tabs(["Optimización Global", "Optimización con QGL dado", "Historial de Optimizaciones"])

    with tab1:
        if uploaded_file is not None:
            show_optimization_charts(input_data_path, global_opt=True)

    with tab2:
        if uploaded_file is not None:
            show_optimization_charts(input_data_path, global_opt=False)

    with tab3:
        show_history()
