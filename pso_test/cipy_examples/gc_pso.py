# Copyright 2016 Andrich van Wyk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Example of gbest PSO algorithm
"""
from cipy.algorithms.core import Domain
from cipy.algorithms.core import max_iterations
from cipy.algorithms.core import minimize
from cipy.algorithms.pso import optimize
from cipy.algorithms.pso.functions import fitness_measurement
from cipy.algorithms.pso.functions import gc_velocity_update
from cipy.algorithms.pso.functions import update_rho
from cipy.benchmarks import functions


def main(dimension, iterations):
    """ Main function to execute gbest GC PSO algorithm.
    """
    objective_function = minimize(functions.sphere)
    stopping_condition = max_iterations(iterations)
    (solution, metrics) = optimize(objective_function=objective_function,
                                   domain=Domain(-5.12, 5.12, dimension),
                                   stopping_condition=stopping_condition,
                                   parameters={'seed': 3758117674,
                                               'rho': 1.0, 'e_s': 15, 'e_f': 5},
                                   velocity_update=gc_velocity_update,
                                   parameter_update=update_rho,
                                   measurements=[fitness_measurement])
    return solution


if __name__ == "__main__":
    solution = main(30, 1000)
    print("Result fitness: %s" % solution.best_fitness)
    print("Result: %s" % solution.position)