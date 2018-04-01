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

import numpy as np;

# Particle Swarm Optimization
def PSO(problem, MaxIter = 100, PopSize = 100, c1 = 1.4962, c2 = 1.4962, w = 0.7298, wdamp = 1.0):

    # Empty Particle Template
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'best_position': None,
        'best_cost': None,
    };

    # Extract Problem Info
    CostFunction = problem['CostFunction'];
    VarMin = problem['VarMin'];
    VarMax = problem['VarMax'];
    nVar = problem['nVar'];

    # Initialize Global Best
    gbest = {'position': None, 'cost': np.inf};

    # Create Initial Population
    pop = [];
    for i in range(0, PopSize):
        pop.append(empty_particle.copy());
        pop[i]['position'] = np.random.uniform(VarMin, VarMax, nVar);
        pop[i]['velocity'] = np.zeros(nVar);
        pop[i]['cost'] = CostFunction(pop[i]['position']);
        pop[i]['best_position'] = pop[i]['position'].copy();
        pop[i]['best_cost'] = pop[i]['cost'];
        
        if pop[i]['best_cost'] < gbest['cost']:
            gbest['position'] = pop[i]['best_position'].copy();
            gbest['cost'] = pop[i]['best_cost'];
    
    # PSO Loop
    for it in range(0, MaxIter):
        for i in range(0, PopSize):
            
            pop[i]['velocity'] = w*pop[i]['velocity'] \
                + c1*np.random.rand(nVar)*(pop[i]['best_position'] - pop[i]['position']) \
                + c2*np.random.rand(nVar)*(gbest['position'] - pop[i]['position']);

            pop[i]['position'] += pop[i]['velocity'];
            pop[i]['position'] = np.maximum(pop[i]['position'], VarMin);
            pop[i]['position'] = np.minimum(pop[i]['position'], VarMax);

            pop[i]['cost'] = CostFunction(pop[i]['position']);
            
            if pop[i]['cost'] < pop[i]['best_cost']:
                pop[i]['best_position'] = pop[i]['position'].copy();
                pop[i]['best_cost'] = pop[i]['cost'];

                if pop[i]['best_cost'] < gbest['cost']:
                    gbest['position'] = pop[i]['best_position'].copy();
                    gbest['cost'] = pop[i]['best_cost'];

        w *= wdamp;
        print('Iteration {}: Best Cost = {}'.format(it, gbest['cost']));

    return gbest, pop;
