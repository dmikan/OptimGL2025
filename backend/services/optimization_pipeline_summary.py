# services/optimization_pipeline.py
from backend.services.optimization_model import OptimizationModel
import numpy as np



def run_pipeline_summary(q_gl_range,
                         y_pred_list, 
                         qgl_limit: int = 0,
                         p_qoil: float = 0.0,
                         p_qgl: float = 0.0) -> dict:

    # Calcular MRP
    delta_q_gl = np.diff(q_gl_range)
    p_qgl_optim_list = []
    for well in range(len(y_pred_list)):
        delta_q_oil = np.diff(y_pred_list[well])
        mp = delta_q_oil / delta_q_gl
        mrp = p_qoil * mp  # Marginal Revenue Product
        qgl_values = q_gl_range[:-1]  # Valores de qgl para el MRP
        # Buscar el punto donde MRP >= P_qgl por última vez
        optimal_idx = np.where(mrp >= p_qgl)[0][-1] if any(mrp >= p_qgl) else len(mrp)-1
        qgl_optimo = qgl_values[optimal_idx]
        p_qgl_optim_list.append(qgl_optimo)  

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


    return {
            "total_production": sum(result_prod_rates),
            "total_qgl": sum(result_optimal_qgl),
            "qgl_limit": qgl_limit
    }