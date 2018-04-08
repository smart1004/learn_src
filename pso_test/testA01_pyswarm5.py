from pyswarm import pso

def func(x): # The function to be minimized
    x0 = x[0]
    x1 = x[1]
    # x2 = x[2]
    # return x0 + x1 + x2
    return x0 + x1

''' f_ieqcons
Returns a 1-D array in which each element must be greater or equal 
to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
ieqcons is ignored (Default: None)'''
def constraints(x):
    x0 = x[0]
    x1 = x[1]
    # x2 = x[2]    
    # return [x0 - x1]  #x0가 더 크야 한다  x0 >= x1 
    return [x0 - (x1+1)]  #x0가 더 크야 한다  x0 >= x1 

lb = [1, 1] #The lower bounds of the design variable(s)
ub = [5, 5] #The upper bounds of the design variable(s)
# lb = [1, 1, 1] #The lower bounds of the design variable(s)
# ub = [5, 5, 3] #The upper bounds of the design variable(s)
xopt, fopt = pso(func, lb, ub, f_ieqcons=constraints)

print('xopt', xopt)
print('fopt', fopt)

# xopt [-3. -1.  1.]
# fopt -3.0
