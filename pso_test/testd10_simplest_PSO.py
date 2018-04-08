#  http://adowney2.public.iastate.edu/projects/The_simplest_Particle_Swarm/The_simplest_PSO_2D.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The simplest particle swarm optimization code (2d Python 3.x)

This code is meant to be a very simple particle swarm optimization (PSO) code 
implementation using only the basic computing packages in packages (mainly NumPy). 
This code favors readability and ease of understanding over speed and robustness 
and is meant to be used as either a research tool or a study guide for anyone 
intrested in learning about particle swarm optimization.  

This code is set up for a 2D optimization, however, this code can be simply 
updated to allow for any number of dimensions with real-number solutions. 

@author: austindowney@gmail.com
copyright: Austin R.J. Downey (2017)
"""

#%% import modules
import IPython as IP
# IP.get_ipython().magic('reset -sf')
import numpy as np
import time as time
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.close('all')
# %% Define the inputs to the PSO

# Define the function that will be optimized. Here, the optimal value is x=10, y=20
def Function(x,y):
    z = (x-10)**2 + (y-20)**2
    return(z)

# Set the parameters for the particle swarm optimization. The parameters set here 
# were selected as they provide for a simple-to-read final plot. Experimentation 
#with the parameters is encouraged. 
swarm_size = 5                       # number of the swarm particles
iterations = 26                      # maximum number of iterations
inertia = 0.5                        # inertia of a particle
dimensions = 2                       # number of values the PSO will optimize
local_weight = 2                     # weighted factor for the particles historical best
global_weight =  2                   # weighted factor for the the global best
max_velocity = 1                     # the highest velocity allowed for a particle
step_size = 1                        # step size for updating each particle, or how far a particle 
                                     # travels before its velocity is readjusted  
#%% Setup the PSO
                                     
# set the x,y location of the particle's (initial guess) scattered around the x=0,y=0
particle_location = np.random.rand(swarm_size,dimensions)-0.5 

# set the initial velocity of the particles in each direction
particle_velocity = np.random.rand(swarm_size,dimensions)

# solve the function for the particle's locations and save as their local best
particle_best_value = Function(particle_location[:,0],particle_location[:,1])
particle_best_location = np.copy(particle_location)

# find the global best location of the initial guess and update the global best location
global_best_value = np.min(particle_best_value)
global_best_location = particle_location[np.argmin(particle_best_value)].copy()

# create empty lists that are updated during the processes and used only for 
# plotting the final results

best_value = []                  # for the best fitting value
best_locaion = []                # for the location of the best fitting value 
iteration_value_best = []        # for the best fitting value of the iteration
iteration_locaion_best = []      # for the location of the best fitting value of the iteration
iteration_value = []             # for the values of the iteration
iteration_locaion = []           # for the locations of the iteration

#%% Run the PSO code
for iteration_i in range(iterations): # for each iteration
    for particle_i in range(swarm_size): # for each particle
        for dimension_i in range(dimensions): # for each dimension 
        
            # generate 2 random numbers between 0 and 1
            u = np.random.rand(dimensions) 

            # calculate the error between the particle's best location and its current location
            error_particle_best = particle_best_location[particle_i,dimension_i] - \
                particle_location[particle_i,dimension_i]
            # calculate the error between the global best location and the particle's current location
            error_global_best = global_best_location[dimension_i] - \
                particle_location[particle_i,dimension_i]
            
            # update the velocity vector in a given dimension            
            v_new = inertia*particle_velocity[particle_i,dimension_i] + \
                local_weight*u[0]*error_particle_best + \
                global_weight*u[1]*error_global_best

            # bound a particle's velocity to the maximum value set above       
            if v_new < -max_velocity:
                v_new = -max_velocity
            elif v_new > max_velocity:
                v_new = max_velocity
            
            # update the particle location
            particle_location[particle_i,dimension_i] = particle_location[particle_i,dimension_i] + \
                v_new*step_size
            
            # update the particle velocity
            particle_velocity[particle_i,dimension_i] = v_new
        
        # for the new location, check if this is a new local or global best
        v = Function(particle_location[particle_i,0],particle_location[particle_i,1])
        # update if its a new local best
        if v < particle_best_value[particle_i]: 
            particle_best_value[particle_i]=v
            particle_best_location[particle_i,:] = particle_location[particle_i,:].copy()
        # update if its a new global best
        if v < global_best_value:
            global_best_value=v
            global_best_location = particle_location[particle_i,:].copy()

    # print the current best location to the console
    print('solution at x='+'%.2f' % global_best_location[0]+', y='+'%.2f' % global_best_location[1])

    # update the lists
    best_value.append(global_best_value.copy())
    best_locaion.append(global_best_location.copy())
    iteration_value.append(v)
    iteration_locaion.append(particle_location.copy())
    v = Function(particle_location[:,0].copy(),particle_location[:,1].copy())
    iteration_value_best.append(np.min(v))
    iteration_locaion_best.append(particle_location[np.argmin(v),:])

#%% Plot the final results
plt.figure(figsize=(5,5))
plt.grid('on')
plt.rc('axes', axisbelow=True)
plt.scatter(10,20,100,marker='*',facecolors='k', edgecolors='k')
#plt.text(10,21,'optimal solution',horizontalalignment='center')
for i in range(len(iteration_locaion)):
    plt.scatter(iteration_locaion[i][:,0],iteration_locaion[i][:,1],10,marker='x')
    plt.scatter(best_locaion[i][0],best_locaion[i][1],50,marker='o',facecolors='none',edgecolors='k',linewidths=0.4)
    plt.text(best_locaion[i][0]+0.1,best_locaion[i][1]+0.1,str(i),fontsize=8)
plt.xlim(-1,15)
plt.ylim(-1,23)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('2-dimensional particle swarm optimization')
plt.savefig('results',dpi=300)

