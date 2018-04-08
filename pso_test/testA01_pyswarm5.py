from pyswarm import pso

def banana(x):
    x1 = x[0]
    x2 = x[1]
    return x1 + x2


lb = [-3, -1]
ub = [2, 6]

xopt, fopt = pso(banana, lb, ub)

print('xopt', xopt)
print('fopt', fopt)

# Optimum should be around x=[0.5, 0.76] with banana(x)=4.5 and con(x)=0