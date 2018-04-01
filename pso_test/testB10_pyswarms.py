import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)

# Perform optimization
best_cost, best_pos = optimizer.optimize(fx.sphere_func, iters=100, verbose=3, print_step=25)
print('best_cost', best_cost)
print('best_pos', best_pos)