# services/optimization_pipeline.py
from backend.services.data_loader import DataLoader
from backend.services.optimization_model import OptimizationModel
from backend.services.fitting import Fitting
from backend.services.results_service import save_optimization_results
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def run_pipeline(csv_file_path: str, 
                 output_file: str = "static/results/output.txt",
                 plot_file: str = "static/plots/wells_plot.png",
                 user: str = None,
                 qgl_limit: float = 4600) -> str:
    """
    Ejecuta el pipeline completo de optimización y guarda resultados en DB
    
    Args:
        csv_file_path: Ruta al archivo CSV con datos de entrada
        output_file: Ruta para guardar los resultados en texto
        plot_file: Ruta para guardar la gráfica de resultados
        user: Nombre del usuario que ejecuta la optimización
        qgl_limit: Límite total de gas de levantamiento disponible (default: 4600)
    
    Returns:
        Ruta al archivo de resultados generado
    """
    
    # Cargar datos
    loader = DataLoader(csv_file_path)
    q_gl_list, q_oil_list, list_info = loader.load_data()

    # Configurar figura para las gráficas
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    q_gl_max = max([np.max(j) for j in q_gl_list])
    q_gl_range = np.linspace(0, q_gl_max, 1000)

    y_pred_list = []
    for well in range(len(q_oil_list)):
        q_gl = q_gl_list[well]
        q_oil = q_oil_list[well]

        fitter = Fitting(q_gl, q_oil)
        y_pred = fitter.fit(fitter.model_namdar, q_gl_range)
        y_pred_list.append(y_pred)
        
        # Graficar
        ax = axes[well]
        ax.plot(q_gl_range, y_pred, label="Curva ajustada", linewidth=2)
        ax.scatter(q_gl, q_oil, color='red', label='Datos reales')
        ax.set_xlabel('Inyección de Gas (q_gl)')
        ax.set_ylabel('Producción de Petróleo (q_oil)')
        ax.set_title(f'Pozo {well+1}')
        ax.legend()
        ax.grid()

    # Ajustar layout y guardar gráfica
    plt.tight_layout()
    
    # Asegurar que exista el directorio para las gráficas
    plot_dir = Path(plot_file).parent
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Optimización con el límite configurado
    model = OptimizationModel(
        q_gl=q_gl_range,
        q_fluid_wells=y_pred_list,
        available_qgl_total=qgl_limit  # Usamos el parámetro aquí
    )
    model.define_optimisation_problem()
    model.define_variables()
    model.build_objective_function()
    model.add_constraints()
    model.solve_prob()

    result_prod_rates = model.get_maximised_prod_rates()
    result_optimal_qgl = model.get_optimal_injection_rates()
    results = list(zip(result_prod_rates, result_optimal_qgl))

    # Asegurar que exista el directorio para resultados
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar resultados en archivo
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
        info = list_info,
        wells_data=wells_data,
        filename=csv_file_path,
        user=user,
        qgl_limit=qgl_limit,
    )

    return output_file