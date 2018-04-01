"""

Copyright (c) 2017, Yarpiz (www.yarpiz.com)
All rights reserved. Please read the "license.txt" for usage terms.
__________________________________________________________________________

Project Code: YPEA127
Project Title: Implementation of Particle Swarm Optimization in Python
Publisher: Yarpiz (www.yarpiz.com)

Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)

Contact Info: sm.kalami@gmail.com, info@yarpiz.com

"""

import yarpiz as yp;

# A Sample Cost Function
def Sphere(x):
    return sum(x**2);

# Define Optimization Problem
problem = {
        'CostFunction': Sphere,
        'nVar': 10,
        'VarMin': -5,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VarMax': 5,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
    };

# Running PSO
yp.tic();
print('Running PSO ...');
gbest, pop = yp.PSO(problem, MaxIter = 200, PopSize = 50, c1 = 1.5, c2 = 2, w = 1, wdamp = 0.995);
print();
yp.toc();
print();

# Final Result
print('Global Best:');
print(gbest);
print();
