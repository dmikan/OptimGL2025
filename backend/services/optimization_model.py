import pulp
from backend.services.data_loader import DataLoader


class OptimizationModel:
    """Class representing the optimisation model"""

    def __init__(self, q_gl, q_fluid_wells, available_qgl_total):
        self.q_gl = q_gl
        self.q_fluid_wells = q_fluid_wells
        self.available_qgl_total = available_qgl_total
        #self.prob = pulp.LpProblem("Maximizar_Suma_Wells", pulp.LpMaximize)
        self.variables = self.define_variables()
        #self.build_objective_function()
        #self.agregar_restricciones()

    def define_optimisation_problem(self):
        self.prob = pulp.LpProblem("Maximise the sum of wells' production", pulp.LpMaximize)


    def define_variables(self):
        """Define the binary variables for each q_gl and q_fluid"""
        binary_variables = [
            [pulp.LpVariable(f'y{well_index}_{i}', cat='Binary') for i in range(len(self.q_gl))]
            for well_index in range(len(self.q_fluid_wells))
        ]
        return binary_variables

    def build_objective_function(self):
        """Defines the objective function to be maximised"""
        self.prob += pulp.lpSum(
            self.variables[i][j] * self.q_fluid_wells[i][j]
            for i in range(len(self.q_fluid_wells))
            for j in range(len(self.q_gl))
        ), "Objective function"
        #return self.prob

    def add_constraints(self):
        """Make sure that each well selects only one value of q_gl"""
        for index, col in enumerate(self.variables):
            self.prob += pulp.lpSum(col) == 1, f"Restriccion_Seleccion_Unica_{index}"
        self.prob += pulp.lpSum(
            self.variables[i][j] * self.q_gl[j]
            for j in range(len(self.q_gl))
            for i in range(len(self.q_fluid_wells))
        ) <= self.available_qgl_total, "constraint q_gl available"
        #return self.prob


    def solve_prob(self):
        """Solve the optimisation problem"""
        self.prob.solve()
        #return self.prob


    def get_maximised_prod_rates(self):
        """Get production value for each well"""
        var_results = [
        pulp.value(pulp.lpSum(self.variables[i][j] * self.q_fluid_wells[i][j]
            for j in range(len(self.q_gl))))
            for i in range(len(self.q_fluid_wells))
        ]
        return var_results

    def get_optimal_injection_rates(self):
        """Get production value for each well"""
        var_results = [
        pulp.value(pulp.lpSum(self.variables[i][j] * self.q_gl[j]
            for j in range(len(self.q_gl))))
            for i in range(len(self.q_fluid_wells))
        ]
        return var_results


if __name__ == "__main__":
    path_data = './data/fitted_curves.csv'
    data = DataLoader(path_data)
    q_gl,q_oil = data.load_data()
    model = OptimizationModel(q_gl, q_oil, 10)
    model.define_optimisation_problem()
    model.define_variables()
    model.build_objective_function()
    model.add_constraints()
    model.solve_prob()
    result_prod_rates = model.get_maximised_prod_rates()
    result_optimal_qgl = model.get_optimal_injection_rates()

    print("Los valores de producción óptimos son: ", result_prod_rates)
    print("La producción total es: ", sum(result_prod_rates))
    print("Los valores de qgl óptimos son: ", result_optimal_qgl)
    print("El valor óptimo de qgl es: ", sum(result_optimal_qgl))
