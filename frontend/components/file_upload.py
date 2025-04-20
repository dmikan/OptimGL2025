import streamlit as st
from pathlib import Path
import pandas as pd
from io import StringIO
from components.manual_input import show_manual_input

def show_file_upload():
    uploaded_file = st.file_uploader("Sube tu archivo CSV con datos de producción", type="csv")
    on = st.toggle("Upload data manually")

    input_data_path = None

    if on:
        uploaded_file = show_manual_input()

    if uploaded_file is not None:
        # Crear directorios necesarios
        project_root = Path(__file__).parent.parent.parent
        upload_dir = project_root / "static" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Manejar ambos casos (archivo subido o datos manuales)
        if isinstance(uploaded_file, pd.DataFrame):
            # Caso de datos manuales
            df = uploaded_file
            nombre_planta = df.iloc[2, 0] if len(df) > 1 else "manual_data"
            input_data_path = upload_dir / f"data_{nombre_planta}.csv"

            if st.button("Upload data"): # Clicking on this button stores data entered manually into static/uploads
                try:
                    df.to_csv(input_data_path, index=False, header=False)
                    st.success(f"Datos guardados exitosamente en {input_data_path}")
                except Exception as e:
                    st.error(f"Error al guardar los datos: {e}")
        else:
            # Caso de archivo subido
            content = uploaded_file.getvalue()
            csv_str = content.decode("utf-8")
            df = pd.read_csv(StringIO(csv_str))
            nombre_planta = df.iloc[1, 0] if len(df) > 1 else "uploaded_data"

            input_data_path = upload_dir / f"data_{nombre_planta}.csv"
            with open(input_data_path, "wb") as f:
                f.write(content)

    return uploaded_file, input_data_path
