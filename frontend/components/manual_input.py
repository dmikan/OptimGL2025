import streamlit as st
import pandas as pd

def show_manual_input():
    # Configuración de estilo
    st.markdown("""
    <style>
        .stDataFrame .row-headers {
            font-weight: bold;
            color: #1f77b4;
        }
        /* Ocultar valores None */
        .stDataFrame [data-testid='stDataFrame'] td:empty:after {
            content: '';
            display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Configuración de parámetros ---
    st.subheader("Configuración")
    col1, col2 = st.columns(2)
    
    with col1:
        num_wells = st.number_input("Número de pozos", min_value=1, max_value=20, value=5, step=1)
    
    with col2:
        num_filas = st.number_input("Número de filas de datos", min_value=1, max_value=100, value=5, step=1)

    # --- Información de campos (1 fila fija) ---
    st.subheader("Información de Campos y Pozos")
    
    well_columns = [f"well_{i+1}" for i in range(num_wells)]
    input_columns_info = ["field"] + well_columns
    
    input_df_info = pd.DataFrame(
        [["" for _ in input_columns_info]],
        columns=input_columns_info,
        index=[1]  # Índice inicial en 1
    )
    
    edited_input_df_info = st.data_editor(
        input_df_info,
        num_rows="fixed",
        hide_index=False,
        use_container_width=True,
        key="info_editor"
    )

    # --- Datos de producción (filas dinámicas) ---
    st.subheader("Datos de Producción")
    
    production_columns = []
    for i in range(1, num_wells + 1):
        production_columns.extend([f"q_inj_w{i}", f"fluid_w{i}"])
    
    # DataFrame con el número de filas especificado
    input_df = pd.DataFrame(
        [["" for _ in production_columns] for _ in range(num_filas)],
        columns=production_columns,
        index=range(1, num_filas + 1)  # Índices desde 1
    )
    
    edited_input_df = st.data_editor(
        input_df,
        num_rows="fixed",
        hide_index=False,
        use_container_width=True,
        key="prod_editor"
    )

    # Agregar columna index si no existe
    if "index" not in edited_input_df.columns:
        edited_input_df.insert(0, "index", range(1, len(edited_input_df) + 1))

    # --- Construir el CSV final ---
    # Número total de columnas esperadas (basado en edited_input_df)
    total_columns = len(edited_input_df.columns)  # 11 (index + 10 columns)

    # Crear un DataFrame vacío con el número correcto de columnas
    final_df = pd.DataFrame(columns=range(total_columns))

    # Fila 1: Descripción (primera celda + vacías)
    final_df.loc[0] = ["description"] + [""] * (total_columns - 1)

    # Fila 2: Nombres de campos/pozos (field, well_1, well_2, etc.)
    # Asegurar que tenga el mismo número de columnas (rellenar con "")
    wells_header = edited_input_df_info.columns.tolist() + [""] * (total_columns - len(edited_input_df_info.columns))
    final_df.loc[1] = wells_header

    # Fila 3: Valores de campos/pozos (field001, w001, etc.)
    if not edited_input_df_info.empty:
        wells_values = edited_input_df_info.iloc[0].tolist() + [""] * (total_columns - len(edited_input_df_info.columns))
        final_df.loc[2] = wells_values

    # Fila 4: Encabezados de datos (index, q_inj_w1, fluid_w1, etc.)
    final_df.loc[3] = edited_input_df.columns.tolist()

    # Filas siguientes: Datos de producción
    for i, row in edited_input_df.iterrows():
        final_df.loc[4 + i] = row.tolist()
    
    return final_df