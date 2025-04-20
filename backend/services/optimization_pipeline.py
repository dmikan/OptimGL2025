# services/optimization_pipeline.py
from backend.services.data_loader import DataLoader
from backend.services.optimization_model import OptimizationModel
from backend.services.fitting import Fitting
from backend.services.results_service import save_optimization_results
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def fitting_group(csv_file_path):
    # Cargar datos
    loader = DataLoader(csv_file_path)
    q_gl_list, q_oil_list, _ = loader.load_data()

    q_gl_max = max([np.max(j) for j in q_gl_list])
    #q_gl_range = np.linspace(0, q_gl_max, 1000)
    q_gl_range = np.logspace(0.1, np.log10(q_gl_max), 1000) # granularidad del fitting
    # Preparar datos para las gráficas
    plot_data = []
    y_pred_list = []

    for well in range(len(q_oil_list)):
        q_gl = q_gl_list[well]
        q_oil = q_oil_list[well]

        fitter = Fitting(q_gl, q_oil)
        y_pred = fitter.fit(fitter.model_namdar, q_gl_range)
        y_pred_list.append(y_pred)

        # Guardar datos para gráficas interactivas
        well_data = {
            "well_num": well + 1,
            "q_gl_actual": q_gl,
            "q_oil_actual": q_oil,
            "q_gl_range": q_gl_range,
            "q_oil_predicted": y_pred
        }
        plot_data.append(well_data)
    return {
        "qgl_range": q_gl_range,
        "y_pred_list": y_pred_list,
        "plot_data": plot_data,
    }


def run_pipeline(csv_file_path: str,
                 q_gl_range,
                 y_pred_list,
                 plot_data,
                 output_file: str = "static/results/output.txt",
                 qgl_limit: float = 4600,
                 p_qoil: float = 0.0,
                 p_qgl: float = 0.0) -> dict:  # Cambiamos el return type
    """
    Ejecuta el pipeline completo de optimización y guarda resultados en DB

    Args:
        csv_file_path: Ruta al archivo CSV con datos de entrada
        output_file: Ruta para guardar los resultados en texto
        user: Nombre del usuario que ejecuta la optimización
        qgl_limit: Límite total de gas de levantamiento disponible

    Returns:
        Diccionario con resultados y datos para gráficas
    """
    # Calcular MRP
    delta_q_gl = np.diff(q_gl_range)
    p_qgl_optim_list = []
    p_qoil_optim_list = []
    for well in range(len(y_pred_list)):
        delta_q_oil = np.diff(y_pred_list[well])
        mp = delta_q_oil / delta_q_gl
        mrp = p_qoil * mp  # Marginal Revenue Product
        qgl_values = q_gl_range[:-1]  # Valores de qgl para el MRP
        # Buscar el punto donde MRP >= P_qgl por última vez
        optimal_idx = np.where(mrp >= p_qgl)[0][-1] if any(mrp >= p_qgl) else len(mrp)-1
        qgl_optimo = qgl_values[optimal_idx]
        y_pred_optimo = y_pred_list[well][optimal_idx]
        p_qgl_optim_list.append(qgl_optimo)
        p_qoil_optim_list.append(y_pred_optimo)

    # Cargar datos para list_info
    loader = DataLoader(csv_file_path)
    _, _, list_info = loader.load_data()
    # Optimización
    model = OptimizationModel(
        q_gl=q_gl_range,
        q_fluid_wells=y_pred_list,
        available_qgl_total=qgl_limit,
        p_qgl_list=p_qgl_optim_list
    )
    model.define_optimisation_problem()
    model.define_variables()
    model.build_objective_function()
    model.add_constraints()
    model.solve_prob()

    result_prod_rates = model.get_maximised_prod_rates()
    result_optimal_qgl = model.get_optimal_injection_rates()
    results = list(zip(result_prod_rates, result_optimal_qgl))

    # Guardar resultados en archivo
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as file:
        file.write("=== Resultados de Optimización ===\n\n")
        file.write(f"Límite QGL configurado: {qgl_limit}\n")
        file.write(f"Producción total: {sum(result_prod_rates):.2f}\n")
        file.write(f"QGL total utilizado: {sum(result_optimal_qgl):.2f} ({sum(result_optimal_qgl)/qgl_limit:.1%})\n\n")
        file.write("Resultados por pozo:\n")
        for i, (prod, qgl) in enumerate(results):
            file.write(f"Pozo {i+1}: Producción = {prod:.2f}, QGL = {qgl:.2f}, Eficiencia = {prod/qgl:.4f}\n")

    # Guardar en base de datos
    wells_data = [
        {'numero': i+1, 'produccion': prod, 'qgl': qgl}
        for i, (prod, qgl) in enumerate(results)
    ]

    save_optimization_results(
        total_prod=sum(result_prod_rates),
        total_qgl=sum(result_optimal_qgl),
        info=list_info,
        wells_data=wells_data,
        filename=csv_file_path,
        qgl_limit=qgl_limit,
        p_qoil=p_qoil,
        p_qgl=p_qgl
    )

    return {
        "results": results,
        "plot_data": plot_data,
        "summary": {
            "total_production": sum(result_prod_rates),
            "total_qgl": sum(result_optimal_qgl),
            "qgl_limit": qgl_limit
            },
        "qgl_range": q_gl_range,
        "y_pred_list": y_pred_list,
        "p_qgl_optim_list": p_qgl_optim_list,
        "p_qoil_optim_list": p_qoil_optim_list
    }
