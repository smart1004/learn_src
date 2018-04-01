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

from yarpiz.pso import *;

def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time, math;
    if 'startTime_for_tictoc' in globals():
        dt = math.floor(100*(time.time() - startTime_for_tictoc))/100.;
        print('Elapsed time is {} second(s).'.format(dt));
    else:
        print('Start time not set. You should call tic before toc.');
