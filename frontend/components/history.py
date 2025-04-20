import streamlit as st
import pandas as pd
from backend.models.database import get_session
from backend.models.optimizacion import Optimizacion
from components.results_table import show_well_results

def show_history():
    st.subheader("Historial de Optimizaciones")

    session = get_session()
    try:
        optimizaciones = session.query(Optimizacion).order_by(Optimizacion.fecha_ejecucion.desc()).all()

        historial_data = [{
            "ID": opt.id,
            "Fecha": opt.fecha_ejecucion.strftime("%Y-%m-%d %H:%M:%S"),
            "Planta": opt.nombre_planta,
            "Producción Total (bbl)": opt.produccion_total,
            "QGL Total (Mscf)": opt.qgl_total,
            "Límite QGL": opt.qgl_limit,
            "(USD/bbl)": opt.valor_barril,
            "(USD/Mscf)": opt.valor_gas
        } for opt in optimizaciones]

        df_historial = pd.DataFrame(historial_data)

        st.dataframe(
            df_historial.style.format({
                "Producción Total (bbl)": "{:.2f}",
                "QGL Total (Mscf)": "{:.2f}",
                "Límite QGL": "{:.2f}",
                "(USD/bbl)": "{:.2f}",
                "(USD/Mscf)": "{:.2f}"
            }),
            use_container_width=True,
            height=300
        )

        selected_id = st.selectbox(
            "Selecciona una optimización para ver detalles",
            options=[opt.id for opt in optimizaciones],
            format_func=lambda x: f"Optimización ID: {x} - {next((opt.nombre_planta for opt in optimizaciones if opt.id == x), '')}"
        )

        if selected_id:
            selected_optimizacion = next((opt for opt in optimizaciones if opt.id == selected_id), None)
            
            if selected_optimizacion:
                st.subheader(f"Resultados Detallados por Pozo de planta {selected_optimizacion.nombre_planta}")
                show_well_results(selected_optimizacion.pozos)
                
                st.warning("Las gráficas de comportamiento no están disponibles en el historial...")
                
                csv = pd.DataFrame([{
                    "Pozo": pozo.numero_pozo,
                    "Nombre": pozo.nombre_pozo,
                    "Producción (bbl)": pozo.produccion_optima,
                    "QGL (Mscf)": pozo.qgl_optimo
                } for pozo in selected_optimizacion.pozos]).to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Descargar resultados como CSV",
                    data=csv,
                    file_name=f"resultados_optimizacion_{selected_optimizacion.id}.csv",
                    mime='text/csv'
                )
    finally:
        session.close()