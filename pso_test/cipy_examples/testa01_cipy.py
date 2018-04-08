from cipy.optimizers.pso_optimizer import PSOOptimizer
from cipy.algorithms.core import max_iterations

from cipy.benchmarks.functions import rastrigin

def banana(x):
    x1 = x[0]
    x2 = x[1]
    return x1 + x2 

def test1():
    optimizer = PSOOptimizer()
    solution = optimizer.minimize(banana, -5.12, 5.12, 30,
                                max_iterations(1000))

    print("Solution error: " + str(optimizer.accuracy())) # Solution: optimizer.solution()
    print("Solution: ",  solution)

from cipy.algorithms.pso.functions import gbest_topology

parameters = {'swarm_size': 25, 'inertia': 0.729844,
            'c_1': 1.496180, 'c_2': 1.496180, 'v_max': 0.1,
            'topology': gbest_topology, 'seed': 208341084}

optimizer = PSOOptimizer(params=parameters)
solution = optimizer.minimize(banana, -5.12, 5.12, 2,
                              max_iterations(1000))

print("Solution error: " + str(optimizer.accuracy())) # Solution: optimizer.solution()
print("Solution: ",  solution)
                                