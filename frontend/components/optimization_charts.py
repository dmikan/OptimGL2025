import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backend.services.optimization_pipeline import run_pipeline, fitting_group
from backend.models.optimizacion import Optimizacion
from backend.services.optimization_pipeline_summary import run_pipeline_summary
from utils.config import get_project_root
from backend.models.database import get_session

def show_optimization_charts(temp_path, global_opt=False):
    if global_opt:
        # Configuración para optimización global
        with st.expander("Configuración de Optimización", expanded=True):
            col1, col2 = st.columns(2)          
            with col1:
                p_qoil_global = st.number_input(
                    "Precio del petróleo (USD/bbl)", 
                    min_value=0.1, 
                    max_value=None, 
                    value=70.0,
                    step=1.0,
                    key="p_qoil_global"
                )
            
            with col2:
                p_qgl_global = st.number_input(
                    "Costo del gas (USD/Mscf)", 
                    min_value=0.1, 
                    max_value=None, 
                    value=5.0,
                    step=0.5,
                    key="p_qgl_global"
                )
        
        if st.button("Optimización Global"):    
            with st.spinner("Procesando datos..."):
                try:
                    fit = fitting_group(csv_file_path=str(temp_path))
                    session = get_session()
                    optimizacion = session.query(Optimizacion).order_by(Optimizacion.fecha_ejecucion.desc()).first()
                    
                    st.subheader(f"Curva de Optimización Global: {optimizacion.nombre_planta}")
                    _plot_global_optimization(fit, p_qoil_global, p_qgl_global)
                    
                except Exception as e:
                    st.error(f"❌ Error durante la optimización: {str(e)}")
                    st.exception(e)
    else:
        # Configuración para optimización con QGL dado
        with st.expander("Configuración de Optimización", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                qgl_limit = st.number_input(
                    "Límite total de QGL (Mscf)", 
                    min_value=0, 
                    max_value=None, 
                    value=1000,
                    step=100,
                    key="qgl_limit"
                )
            
            with col2:
                p_qoil = st.number_input(
                    "Precio del petróleo (USD/bbl)", 
                    min_value=0.1, 
                    max_value=None, 
                    value=70.0,
                    step=1.0,
                    key="p_qoil"
                )
            
            with col3:
                p_qgl = st.number_input(
                    "Costo del gas (USD/Mscf)", 
                    min_value=0.1, 
                    max_value=None, 
                    value=5.0,
                    step=0.5,
                    key="p_qgl"
                )
        
        if st.button("Ejecutar Optimización"):
            with st.spinner("Procesando datos..."):
                try:
                    fit = fitting_group(csv_file_path=str(temp_path))
                    results = run_pipeline(
                        csv_file_path=str(temp_path),
                        q_gl_range=fit['qgl_range'],
                        y_pred_list=fit["y_pred_list"],
                        plot_data=fit["plot_data"],
                        output_file=str(get_project_root() / "static" / "results" / "output_temp.txt"),
                        qgl_limit=qgl_limit,
                        p_qoil=p_qoil,
                        p_qgl=p_qgl
                    )
                    
                    st.success("¡Optimización completada!")
                    session = get_session()
                    optimizacion = session.query(Optimizacion).order_by(Optimizacion.fecha_ejecucion.desc()).first()
                    
                    # Mostrar resultados
                    _show_optimization_results(results, optimizacion)
                    
                except Exception as e:
                    st.error(f"❌ Error durante la optimización: {str(e)}")
                    st.exception(e)
                finally:
                    if 'session' in locals():
                        session.close()

def _plot_global_optimization(fit, p_qoil, p_qgl):
    """Función auxiliar para graficar optimización global"""
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
    log_vals = np.logspace(start=1, stop=np.log10(max_qgl), num=num)
    log_vals = np.unique(log_vals)
    
    progress_bar = st.progress(0, text="Calculando curva de optimización global...")
    for i, qgl_limit in enumerate(log_vals):    
        dic_optim_result = run_pipeline_summary(
                q_gl_range=fit["qgl_range"],
                y_pred_list=fit["y_pred_list"], 
                qgl_limit = qgl_limit,
                p_qoil=p_qoil,
                p_qgl=p_qgl) 
        
        current_qgl = dic_optim_result["total_qgl"]
        optimization_results["qgl_limit"].append(qgl_limit)
        optimization_results["total_production"].append(dic_optim_result["total_production"])
        optimization_results["total_qgl"].append(dic_optim_result["total_qgl"])
        qgl_history.append(current_qgl)
        
        progress_text = f"Procesando QGL límite: {round(qgl_limit,2)}"
        if has_stabilized(qgl_history):
            progress_text += " (¡Estabilizado!)"
        progress_bar.progress((i + 1) / num, text=progress_text)

        if has_stabilized(qgl_history, window_size=3):
            st.info("El QGL total se ha estabilizado. Terminando optimización temprana.")
            break

    progress_bar.empty()

    # Configuración de colores para tema oscuro de Streamlit
    bg_color = "#0E1117"
    grid_color = "#37474F"
    text_color = "#FFFFFF"
    line_color = "#00E676"
    marker_color = "#C8E6C9"
    optimal_line_color = "#FF5252"
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
            title_font=dict(color=text_color),
            # Añadir estas líneas para el grid vertical
            showgrid=True,
            gridwidth=1,
            dtick=1000,  # Esto establecerá marcas cada 1000 unidades
            minor_griddash="dot",  # Estilo de línea punteada para el grid
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

    st.plotly_chart(fig, use_container_width=True)

def _show_optimization_results(results, optimizacion):
    """Muestra los resultados de la optimización"""
    # Métricas globales
    col1, col2, col3 = st.columns(3)

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
    
    # Gráficas de optimización
    st.subheader("Curvas de Comportamiento por Pozo")
    _plot_well_curves(results, optimizacion)
    
    # Tabla de resultados
    st.subheader("Resultados Detallados por Pozo")
    from components.results_table import show_well_results
    show_well_results(optimizacion.pozos)

def _plot_well_curves(results, optimizacion):
    # Verificación básica de datos
    if not results.get('plot_data') or not optimizacion.pozos:
        st.warning("No hay datos suficientes para graficar")
        return
    
    # Configuración de colores para tema oscuro de Streamlit (IDÉNTICO AL ORIGINAL)
    bg_color = "#0E1117"  # Color de fondo oscuro de Streamlit
    grid_color = "#37474F"  # Color de grid que contrasta bien con el fondo
    text_color = "#FFFFFF"  # Color de texto principal de Streamlit
    line_color = "#00E676"  # Verde claro para las curvas
    marker_color = "#C8E6C9"  # Verde claro para los puntos de datos
    optimal_line_color = "#FF5252"  # Rojo brillante para las líneas de QGL óptimo
    
    # FIGURA 1: CURVA DE PRODUCCIÓN (q_oil vs q_gl) - MANTENIENDO TODOS LOS DETALLES
    fig_prod = make_subplots(
        rows=2, 
        cols=3, 
        subplot_titles=[f"Well {pozo.nombre_pozo} - Producción" for pozo in optimizacion.pozos],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    for idx, (well_data, pozo) in enumerate(zip(results['plot_data'], optimizacion.pozos)):
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        
        # 1. Curva ajustada (EXACTAMENTE IGUAL)
        fig_prod.add_trace(
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
        
        # 2. Puntos reales (EXACTAMENTE IGUAL)
        fig_prod.add_trace(
            go.Scatter(
                x=well_data["q_gl_actual"],
                y=well_data["q_oil_actual"],
                mode='markers',
                name='Datos reales',
                marker=dict(
                    color=marker_color, 
                    size=7, 
                    line=dict(width=1, color='DarkSlateGrey')  # Detalle exacto del borde
                ),
                showlegend=True if idx == 0 else False,
                legendgroup='group2'
            ),
            row=row, col=col
        )
        
        # 3. Óptimo según MRP (MANTENIENDO LOS results[] ORIGINALES)
        fig_prod.add_trace(
            go.Scatter(
                x=[results["p_qgl_optim_list"][idx], results["p_qgl_optim_list"][idx]],
                y=[0, results["p_qoil_optim_list"][idx]],
                mode='lines',
                name='QGL máximo óptimo',
                line=dict(color=marker_color, width=2, dash='dash'),
                showlegend=True if idx == 0 else False,
                legendgroup='group5'
            ),
            row=row, col=col
        )

        # 4. Línea y marcador óptimo (IDÉNTICO AL ORIGINAL)
        optimal_qgl = pozo.qgl_optimo
        optimal_prod = pozo.produccion_optima
        
        fig_prod.add_trace(
            go.Scatter(
                x=[optimal_qgl, optimal_qgl],
                y=[0, optimal_prod],
                mode='lines',
                name='QGL óptimo',
                line=dict(color=optimal_line_color, width=2, dash='dash'),
                showlegend=True if idx == 0 else False,
                legendgroup='group'
            ),
            row=row, col=col
        )
        
        fig_prod.add_trace(
            go.Scatter(
                x=[optimal_qgl],
                y=[optimal_prod],
                mode='markers',
                name='Punto óptimo',
                marker=dict(
                    color=optimal_line_color, 
                    size=10, 
                    symbol='x'  # Mismo símbolo original
                ),
                showlegend=True if idx == 0 else False,
                legendgroup='group4'
            ),
            row=row, col=col
        )
        
        # Configuración de ejes (COPIADO DIRECTAMENTE)
        fig_prod.update_xaxes(
            title_text="Inyección de gas (q_gl)", 
            row=row, col=col,
            gridcolor=grid_color,
            linecolor=grid_color,
            tickfont=dict(color=text_color),
            title_font=dict(color=text_color)
        )
        
        fig_prod.update_yaxes(
            title_text="Producción (bbl/d)", 
            row=row, col=col,
            gridcolor=grid_color,
            linecolor=grid_color,
            tickfont=dict(color=text_color),
            title_font=dict(color=text_color)
        )

    # Layout figura 1 (PRESERVANDO TODOS LOS VALORES ORIGINALES)
    fig_prod.update_layout(
        height=800,  # Misma altura
        width=1200,  # Mismo ancho
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,  # Misma posición
            xanchor="right",
            x=1,     # Misma posición
            font=dict(color=text_color)
        ),
        margin=dict(
            l=50,  # Izquierda
            r=50,  # Derecha
            b=80,  # Inferior
            t=100, # Superior
            pad=4  # Padding
        )
    )

    st.plotly_chart(fig_prod, use_container_width=True)