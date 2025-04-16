import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from io import StringIO
import numpy as np

# Configuración de rutas
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.services.optimization_pipeline import run_pipeline, fitting_group
from backend.models.database import init_db, get_session
from backend.models.optimizacion import Optimizacion
from backend.services.optimization_pipeline_summary import run_pipeline_summary

# Inicializar la base de datos
engine = init_db()

# Configuración de la página
st.set_page_config(page_title="Optimizador de Pozos", layout="wide")
st.title("🛢️ Optimización de Distribución de Gas")

# --- Pestañas ---
tabs = st.sidebar.radio("Selecciona una opción", ["Optimización", "Otros servicios"])

if tabs == "Optimización":
    # --- Subida de archivo ---
    uploaded_file = st.file_uploader("Sube tu archivo CSV con datos de producción", type="csv")
    # Crear pestañas
    tab1, tab2, tab3 = st.tabs(["Optimización Global", "Optimización con QGL dado", "Historial de Optimizaciones"])

    if uploaded_file:
        # Crear directorios necesarios
        upload_dir = project_root / "static" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo permanentemente
        content = uploaded_file.getvalue()
        csv_str = content.decode("utf-8") 
        df = pd.read_csv(StringIO(csv_str))
        nombre_planta = df.iloc[1, 0]

        temp_path = upload_dir / f"temp_data_{nombre_planta}.csv"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

    with tab1:
        if st.button("Optimización Global"):    
             with st.spinner("Procesando datos..."):
                try:
                    fit = fitting_group(csv_file_path=str(temp_path))
                    # Obtener datos de la última optimización
                    session = get_session(engine)
                    optimizacion = session.query(Optimizacion).order_by(Optimizacion.fecha_ejecucion.desc()).first()
                    # --- Gráfica de Optimización Global ---
                    st.subheader(f"Curva de Optimización Global: {optimizacion.nombre_planta}")

                    def has_stabilized(values, window_size=3, tolerance=1e-6):
                        """Detecta si los valores se han estabilizado"""
                        if len(values) < window_size:
                            return False
                        recent = values[-window_size:]
                        return all(abs(x - recent[0]) < tolerance for x in recent)

                    qgl_history = []
                    num = 40
                    max_qgl = 20000 
                    optimization_results = {"qgl_limit": [], "total_production": [], "total_qgl": []}
                    #log_vals = np.linspace(0, 10000, 30, dtype=int)
                    #log_vals = np.logspace(start=0, stop=np.log10(20000), num=20, dtype=int)
                    log_vals = np.logspace(start=1, stop=np.log10(max_qgl), num=num)
                    log_vals = np.unique(log_vals)
                    progress_bar = st.progress(0, text="Calculando curva de optimización global...")
                    for i, qgl_limit in enumerate(log_vals):    
                        dic_optim_result = run_pipeline_summary(
                                q_gl_range=fit["qgl_range"],
                                y_pred_list=fit["y_pred_list"], 
                                qgl_limit = qgl_limit) 
                        # la siguiente línea vienes de la función run_pipeline que retorna un diccionario y tomamos su valor de "summary 
                        current_qgl = dic_optim_result["total_qgl"]
                        optimization_results["qgl_limit"].append(qgl_limit)
                        optimization_results["total_production"].append(dic_optim_result["total_production"])
                        optimization_results["total_qgl"].append(dic_optim_result["total_qgl"])
                        qgl_history.append(current_qgl)
                        # Actualizar barra de progreso con información de estabilidad
                        progress_text = f"Procesando QGL límite: {round(qgl_limit,2)}"
                        if has_stabilized(qgl_history):
                            progress_text += " (¡Estabilizado!)"
                        progress_bar.progress((i + 1) / num, text=progress_text)

                            # Terminar si está estabilizado
                        if has_stabilized(qgl_history, window_size=3):
                            st.info("El QGL total se ha estabilizado. Terminando optimización temprana.")
                            break

                    progress_bar.empty()

                    # Configuración de colores para tema oscuro de Streamlit
                    bg_color = "#0E1117"  # Color de fondo oscuro de Streamlit
                    grid_color = "#37474F"  # Color de grid que contrasta bien con el fondo
                    text_color = "#FFFFFF"  # Color de texto principal de Streamlit
                    line_color = "#00E676"  # Verde claro para las curvas
                    marker_color = "#C8E6C9"  # Verde claro para los puntos de datos
                    optimal_line_color = "#FF5252"  # Rojo brillante para las líneas de QGL óptimo
                    last_value_line_color = "#FF1744"

                    # Obtener el último valor de producción
                    last_production = optimization_results["total_production"][-1] if optimization_results["total_production"] else 0

                    # Crear figura
                    fig = go.Figure()

                    # Añadir curva de producción total vs qgl_limit
                    fig.add_trace(
                        go.Scatter(
                            x=optimization_results["total_qgl"],
                            y=optimization_results["total_production"],
                            mode='lines+markers',
                            name='Producción Total',
                            line=dict(width=3, color=line_color),
                            marker=dict(color=marker_color, size=7),
                            showlegend=True
                        )
                    )


                    # Añadir línea horizontal punteada roja para el último valor de producción
                    fig.add_hline(
                        y=last_production,
                        line=dict(
                            color=last_value_line_color,
                            width=2,
                            dash='dash'
                        ),
                        annotation_text=f'{last_production:.2f} bbl',
                        annotation_position="bottom left",
                        annotation_font=dict(color=last_value_line_color),
                        name='Último valor'
                    )

                    # Configurar ejes con colores del tema
                    fig.update_layout(
                        xaxis=dict(
                            title_text="Límite de Inyección de Gas (qgl_limit)",
                            gridcolor=grid_color,
                            linecolor=grid_color,
                            tickfont=dict(color=text_color),
                            title_font=dict(color=text_color)
                        ),
                        yaxis=dict(
                            title_text="Producción Total de Petróleo (bbl)",
                            gridcolor=grid_color,
                            linecolor=grid_color,
                            tickfont=dict(color=text_color),
                            title_font=dict(color=text_color)
                        ),
                        yaxis2=dict(
                            title_text="Consumo Total de Gas",
                            overlaying="y",
                            side="right",
                            gridcolor=grid_color,
                            linecolor=grid_color,
                            tickfont=dict(color=text_color),
                            title_font=dict(color=text_color)
                        )
                    )

                    # Configuración general del layout
                    fig.update_layout(
                        height=600,
                        width=900,
                        plot_bgcolor=bg_color,
                        paper_bgcolor=bg_color,
                        font=dict(color=text_color),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(color=text_color)
                        ),
                        margin=dict(l=50, r=50, b=80, t=80, pad=4),
                        hovermode="x unified"
                    )

                    # Configuración del grid
                    fig.update_xaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor=grid_color,
                        zerolinecolor=grid_color
                    )

                    fig.update_yaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor=grid_color,
                        zerolinecolor=grid_color
                    )

                    st.plotly_chart(fig, use_container_width=True) 
                except Exception as e:
                    st.error(f"❌ Error durante la optimización: {str(e)}")
                    st.exception(e)

    with tab2:
        # --- Sección de configuración ---
        with st.expander("Configuración de Optimización", expanded=True):
            qgl_limit = st.number_input(
                "Límite total de QGL (Gas de Levantamiento)", 
                min_value=0, 
                max_value=None, 
                value=1000,
                step=100,
                key="qgl_limit"
            )
        # Ejecutar pipeline
        if st.button("Ejecutar Optimización"):
            with st.spinner("Procesando datos..."):
                try:
                    fit = fitting_group(csv_file_path=str(temp_path))

                    # Ejecutar el pipeline
                    results = run_pipeline(
                        csv_file_path=str(temp_path),
                        q_gl_range=fit['qgl_range'],
                        y_pred_list=fit["y_pred_list"],
                        plot_data=fit["plot_data"],
                        output_file=str(project_root / "static" / "results" / "output_temp.txt"),
                        qgl_limit=qgl_limit
                    )
                    
                    st.success("¡Optimización completada!")
                    
                    # Obtener datos de la última optimización
                    session = get_session(engine)
                    optimizacion = session.query(Optimizacion).order_by(Optimizacion.fecha_ejecucion.desc()).first()
                    
                    # --- Resumen de optimización ---
                    st.subheader(f"Resumen de Optimización de Planta: {optimizacion.nombre_planta}")
                    
                    if optimizacion:
                        # Métricas globales
                        col1, col2, col3 = st.columns(3)

                        # Separador visual
                        st.markdown("---")
                        
                        with col1:
                            st.metric(
                                "Producción Total", 
                                f"{results['summary']['total_production']:.2f} bbl"
                            )
                        
                        with col2:
                            st.metric(
                                "QGL Total Utilizado", 
                                f"{results['summary']['total_qgl']:.2f} Mscf",
                                delta=f"{(results['summary']['total_qgl']/results['summary']['qgl_limit']*100):.1f}% del límite"
                            )
                        
                        with col3:
                            st.metric(
                                "Límite QGL Configurado", 
                                f"{results['summary']['qgl_limit']:.2f} Mscf"
                            )
                        
                        # --- Gráficas de optimización ---
                        st.subheader("Curvas de Comportamiento por Pozo")
                        
                        # Configuración de colores para tema oscuro de Streamlit
                        bg_color = "#0E1117"  # Color de fondo oscuro de Streamlit
                        grid_color = "#37474F"  # Color de grid que contrasta bien con el fondo
                        text_color = "#FFFFFF"  # Color de texto principal de Streamlit
                        line_color = "#00E676"  # Verde claro para las curvas
                        marker_color = "#C8E6C9"  # Verde claro para los puntos de datos
                        optimal_line_color = "#FF5252"  # Rojo brillante para las líneas de QGL óptimo
                        
                        # Crear figura con subplots
                        fig = make_subplots(
                            rows=2, 
                            cols=3, 
                            subplot_titles=[f"Well {pozo.nombre_pozo}" 
                                           for pozo in optimizacion.pozos],
                            horizontal_spacing=0.1,
                            vertical_spacing=0.15
                        )
                        
                        for idx, (well_data, pozo) in enumerate(zip(results['plot_data'], optimizacion.pozos)):
                            row = (idx // 3) + 1
                            col = (idx % 3) + 1
                            
                            # Añadir curva ajustada
                            fig.add_trace(
                                go.Scatter(
                                    x=well_data["q_gl_range"],
                                    y=well_data["q_oil_predicted"],
                                    mode='lines',
                                    name='Curva ajustada',
                                    line=dict(width=3, color=line_color),
                                    showlegend=True if idx == 0 else False,
                                    legendgroup='group1'
                                ),
                                row=row, col=col
                            )
                            
                            # Añadir puntos reales
                            fig.add_trace(
                                go.Scatter(
                                    x=well_data["q_gl_actual"],
                                    y=well_data["q_oil_actual"],
                                    mode='markers',
                                    name='Datos reales',
                                    marker=dict(color=marker_color, size=7, line=dict(width=1, color='DarkSlateGrey')),
                                    showlegend=True if idx == 0 else False,
                                    legendgroup='group2'
                                ),
                                row=row, col=col
                            )
                            
                            # Añadir línea vertical para QGL óptimo
                            optimal_qgl = pozo.qgl_optimo
                            optimal_prod = pozo.produccion_optima
                            
                            # Crear segmento desde (optimal_qgl, 0) hasta (optimal_qgl, optimal_prod)
                            fig.add_trace(
                                go.Scatter(
                                    x=[optimal_qgl, optimal_qgl],
                                    y=[0, optimal_prod],
                                    mode='lines',
                                    name='QGL óptimo',
                                    line=dict(color=optimal_line_color, width=2, dash='dash'),
                                    showlegend=True if idx == 0 else False,
                                    legendgroup='group3'
                                ),
                                row=row, col=col
                            )
                            
                            # Añadir marcador en el punto óptimo
                            fig.add_trace(
                                go.Scatter(
                                    x=[optimal_qgl],
                                    y=[optimal_prod],
                                    mode='markers',
                                    name='Punto óptimo',
                                    marker=dict(color=optimal_line_color, size=10, symbol='x'),
                                    showlegend=True if idx == 0 else False,
                                    legendgroup='group4'
                                ),
                                row=row, col=col
                            )
                            
                            # Configurar ejes con colores del tema
                            fig.update_xaxes(
                                title_text="Inyección de gas (q_gl)", 
                                row=row, 
                                col=col,
                                gridcolor=grid_color,
                                linecolor=grid_color,
                                tickfont=dict(color=text_color),
                                title_font=dict(color=text_color)
                            )
                            
                            fig.update_yaxes(
                                title_text="Producción de petróleo (q_oil)", 
                                row=row, 
                                col=col,
                                gridcolor=grid_color,
                                linecolor=grid_color,
                                tickfont=dict(color=text_color),
                                title_font=dict(color=text_color)
                            )
                        
                        # Configuración general del layout
                        fig.update_layout(
                            height=800,
                            width=1200,
                            plot_bgcolor=bg_color,
                            paper_bgcolor=bg_color,
                            font=dict(color=text_color),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                                font=dict(color=text_color)
                            ),
                            margin=dict(l=50, r=50, b=80, t=80, pad=4),
                        )
                        
                        # Configuración del grid
                        fig.update_xaxes(
                            showgrid=True, 
                            gridwidth=1, 
                            gridcolor=grid_color,
                            zerolinecolor=grid_color
                        )
                        
                        fig.update_yaxes(
                            showgrid=True, 
                            gridwidth=1, 
                            gridcolor=grid_color,
                            zerolinecolor=grid_color
                        )
                        
                        # Ajustar títulos de subplots
                        for annotation in fig['layout']['annotations']:
                            annotation['font'] = dict(color=text_color, size=12)
                        
                        st.plotly_chart(fig, use_container_width=True)

                        # Separador visual
                        st.markdown("---")
                        # --- Tabla de resultados ---
                        st.subheader("Resultados Detallados por Pozo")
                        
                        pozos_data = [{
                            "Pozo": pozo.numero_pozo,
                            "Nombre": pozo.nombre_pozo,
                            "Producción (bbl)": pozo.produccion_optima,
                            "QGL (Mscf)": pozo.qgl_optimo
                        } for pozo in optimizacion.pozos]
                        
                        st.dataframe(
                            pd.DataFrame(pozos_data).set_index("Pozo").style.format({
                                "Producción (bbl)": "{:.2f}",
                                "QGL (Mscf)": "{:.2f}"
                            }),
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error durante la optimización: {str(e)}")
                    st.exception(e)
                finally:
                    if 'session' in locals():
                        session.close()
    with tab3:
        # --- Historial de Optimizaciones ---
        st.subheader("Historial de Optimizaciones")

        session = get_session(engine)
        try:
            optimizaciones = session.query(Optimizacion).order_by(Optimizacion.fecha_ejecucion.desc()).all()

            historial_data = [{
                "ID": opt.id,
                "Fecha": opt.fecha_ejecucion.strftime("%Y-%m-%d %H:%M:%S"),
                "Planta": opt.nombre_planta,
                "Producción Total (bbl)": opt.produccion_total,
                "QGL Total (Mscf)": opt.qgl_total,
                "Límite QGL": opt.qgl_limit,
                "Archivo": Path(opt.archivo_origen).name
            } for opt in optimizaciones]

            df_historial = pd.DataFrame(historial_data)

            # Mostrar tabla de historial
            st.dataframe(
                df_historial.style.format({
                    "Producción Total (bbl)": "{:.2f}",
                    "QGL Total (Mscf)": "{:.2f}",
                    "Límite QGL": "{:.2f}"
                }),
                use_container_width=True,
                height=300
            )

            # Selección de optimización para ver detalles
            selected_id = st.selectbox(
                "Selecciona una optimización para ver detalles",
                options=[opt.id for opt in optimizaciones],
                format_func=lambda x: f"Optimización ID: {x} - {next((opt.nombre_planta for opt in optimizaciones if opt.id == x), '')}"
            )

            if selected_id:
                selected_optimizacion = next((opt for opt in optimizaciones if opt.id == selected_id), None)
                
                if selected_optimizacion:
                    # --- Tabla de resultados ---
                    st.subheader(f"Resultados Detallados por Pozo de planta {selected_optimizacion.nombre_planta}")
                    
                    pozos_data = [{
                        "Pozo": pozo.numero_pozo,
                        "Nombre": pozo.nombre_pozo,
                        "Producción (bbl)": pozo.produccion_optima,
                        "QGL (Mscf)": pozo.qgl_optimo
                    } for pozo in selected_optimizacion.pozos]
                    
                    st.dataframe(
                        pd.DataFrame(pozos_data).set_index("Pozo").style.format({
                            "Producción (bbl)": "{:.2f}",
                            "QGL (Mscf)": "{:.2f}"
                        }),
                        use_container_width=True
                    )
                    
                    # Nota sobre gráficas (hasta que tengamos los datos necesarios)
                    st.warning("Las gráficas de comportamiento no están disponibles en el historial porque no se almacenaron los parámetros del modelo. Para habilitar esta función, necesitamos modificar el modelo de datos.")
                    
                    # Opción para descargar resultados
                    csv = pd.DataFrame(pozos_data).to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar resultados como CSV",
                        data=csv,
                        file_name=f"resultados_optimizacion_{selected_optimizacion.id}.csv",
                        mime='text/csv'
                    )
        finally:
            session.close()

elif tabs == "Otros servicios":
    st.subheader("Otros servicios")   