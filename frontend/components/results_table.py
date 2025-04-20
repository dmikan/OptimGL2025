import streamlit as st
import pandas as pd

def show_well_results(wells):
    pozos_data = [{
        "Pozo": pozo.numero_pozo,
        "Nombre": pozo.nombre_pozo,
        "Producción (bbl)": pozo.produccion_optima,
        "QGL (Mscf)": pozo.qgl_optimo
    } for pozo in wells]
    
    st.dataframe(
        pd.DataFrame(pozos_data).set_index("Pozo").style.format({
            "Producción (bbl)": "{:.2f}",
            "QGL (Mscf)": "{:.2f}"
        }),
        use_container_width=True
    )