# services/optimization_pipeline.py
from backend.services.optimization_model import OptimizationModel



def run_pipeline_summary(q_gl_range,
                         y_pred_list, 
                         qgl_limit: int = 0) -> dict:  

    model = OptimizationModel(
        q_gl=q_gl_range,
        q_fluid_wells=y_pred_list,
        available_qgl_total=qgl_limit
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