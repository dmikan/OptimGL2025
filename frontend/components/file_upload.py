import streamlit as st
from pathlib import Path
import pandas as pd
from io import StringIO 
from components.manual_input import show_manual_input

def show_file_upload():
    uploaded_file = st.file_uploader("Sube tu archivo CSV con datos de producción", type="csv")
    on = st.toggle("Upload data manually")

    temp_path = None
    
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
            temp_path = upload_dir / f"data_{nombre_planta}.csv"
            
            if st.button("Upload data"):
                try:
                    df.to_csv(temp_path, index=False, header=False)
                    st.success(f"Datos guardados exitosamente en {temp_path}")
                except Exception as e:
                    st.error(f"Error al guardar los datos: {e}")
        else:
            # Caso de archivo subido
            content = uploaded_file.getvalue()
            csv_str = content.decode("utf-8") 
            df = pd.read_csv(StringIO(csv_str))
            nombre_planta = df.iloc[1, 0] if len(df) > 1 else "uploaded_data"
            
            temp_path = upload_dir / f"data_{nombre_planta}.csv"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
    
    return uploaded_file, temp_path