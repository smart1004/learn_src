# Travelling Salesperson Problem Example 1
# http://mnemstudio.org/particle-swarm-tsp-example-1.htm
import win_unicode_console; win_unicode_console.enable()
import random
import math
import sys

PARTICLE_COUNT = 10
V_MAX = 4 # Maximum velocity change allowed.  Range: 0 >= V_MAX < CITY_COUNT

MAX_EPOCHS = 10000

particles = []

map = []
CITY_COUNT = 8
TARGET = 86.63 # Number for algorithm to find.
XLocs = [30, 40, 40, 29, 19, 9, 9, 20]
YLocs = [5, 10, 20, 25, 25, 19, 9, 5]

class Particle:
    def __init__(self):
        self.mData = [0] * CITY_COUNT
        self.mpBest = 0
        self.mVelocity = 0.0

    def get_data(self, index):
        return self.mData[index]

    def set_data(self, index, value):
        self.mData[index] = value

    def get_pBest(self):
        return self.mpBest

    def set_pBest(self, value):
        self.mpBest = value

    def get_velocity(self):
        return self.mVelocity

    def set_velocity(self, velocityScore):
        self.mVelocity = velocityScore

class City:
    def __init__(self):
        self.mX = 0
        self.mY = 0
    
    def get_x(self):
        return self.mX
    
    def set_x(self, xCoordinate):
        self.mX = xCoordinate
    
    def get_y(self):
        return self.mY
    
    def set_y(self, yCoordinate):
        self.mY = yCoordinate

def get_distance(firstCity, secondCity):
    cityA = map[firstCity]
    cityB = map[secondCity]
    a2 = math.pow(math.fabs(cityA.get_x() - cityB.get_x()), 2)
    b2 = math.pow(math.fabs(cityA.get_y() - cityB.get_y()), 2)
    return math.sqrt(a2 + b2)

def get_total_distance(index):
    particles[index].set_pBest(0.0)
    
    for i in range(CITY_COUNT):
        if i == CITY_COUNT - 1:
            particles[index].set_pBest(particles[index].get_pBest() + get_distance(particles[index].get_data(CITY_COUNT - 1), particles[index].get_data(0))) # Complete trip.
        else:
            particles[index].set_pBest(particles[index].get_pBest() + get_distance(particles[index].get_data(i), particles[index].get_data(i + 1)))
    
    return

def initialize_map():
    for i in range(CITY_COUNT):
        newCity = City()
        newCity.set_x(XLocs[i])
        newCity.set_y(YLocs[i])
        map.append(newCity)
    
    return

def randomly_arrange(index = 0):
    cityA = random.randrange(0, CITY_COUNT)
    cityB = 0
    done = False
    
    while not done:
        cityB = random.randrange(0, CITY_COUNT)
        if cityB != cityA:
            done = 	True
    
    # swap cityA and cityB.
    temp = particles[index].get_data(cityA)
    particles[index].set_data(cityA, particles[index].get_data(cityB))
    particles[index].set_data(cityB, temp)
    return

def initialize_particles():
    for i in range(PARTICLE_COUNT):
        newParticle = Particle()
        
        for j in range(CITY_COUNT):
            newParticle.set_data(j, j)
        
        particles.append(newParticle)
        
        for j in range(10): # just any number of times to randomize them.
            randomly_arrange(len(particles) - 1)
        
        get_total_distance(len(particles) - 1)
    
    return

def quicksort(array, left, right):
    pivot = quicksort_partition(array, left, right)
    
    if left < pivot:
        quicksort(array, left, pivot - 1)
    
    if right > pivot:
        quicksort(array, pivot + 1, right)
    
    return array

def quicksort_partition(numbers, left, right):
    # The comparison is on each particle's pBest value.
    I_hold = left
    r_hold = right
    pivot = numbers[left]
    
    while left < right:
        while (numbers[right].get_pBest() >= pivot.get_pBest()) and (left < right):
            right -= 1
        
        if left != right:
            numbers[left] = numbers[right]
            left += 1
        
        while (numbers[left].get_pBest() <= pivot.get_pBest()) and (left < right):
            left += 1
        
        if left != right:
            numbers[right] = numbers[left]
            right -= 1
    
    numbers[left] = pivot
    pivot = left
    left = I_hold
    right = r_hold
    
    return pivot

def get_velocity():
    worstResults = 0.0
    vValue = 0.0
    
    # After sorting, worst will be last in list.
    worstResults = particles[PARTICLE_COUNT - 1].get_pBest()
    
    for i in range(PARTICLE_COUNT):
        vValue = (V_MAX * particles[i].get_pBest()) / worstResults
        
        if vValue > V_MAX:
            particles[i].set_velocity(V_MAX)
        elif vValue < 0.0:
            particles[i].set_velocity(0.0)
        else:
            particles[i].set_velocity(vValue)
    
    return

def copy_from_particle(source, destination):
    # push destination's data points closer to source's data points.
    targetA = random.randrange(0, CITY_COUNT) # source's city to target.
    targetB = 0
    indexA = 0
    indexB = 0
    tempIndex = 0
    
    # targetB will be source's neighbor immediately succeeding targetA (circular).
    for i in range(CITY_COUNT):
        if particles[source].get_data(i) == targetA:
            if i == CITY_COUNT - 1:
                targetB = particles[source].get_data(0) # if end of array, take from beginning.
            else:
                targetB = particles[source].get_data(i + 1)
            
            break
    
    # Move targetB next to targetA by switching values.
    for j in range(CITY_COUNT):
        if particles[destination].get_data(j) == targetA:
            indexA = j
        
        if particles[destination].get_data(j) == targetB:
            indexB = j
    
    # get temp index succeeding indexA.
    if indexA == CITY_COUNT - 1:
        tempIndex = 0
    else:
        tempIndex = indexA + 1
    
    # Switch indexB value with tempIndex value.
    temp = particles[destination].get_data(tempIndex)
    particles[destination].set_data(tempIndex, particles[destination].get_data(indexB))
    particles[destination].set_data(indexB, temp)
    
    return

def update_particles():
    # Best was previously sorted to index 0, so start from the second best.
    for i in range(PARTICLE_COUNT):
        if i > 0:
            # The higher the velocity score, the more changes it will need.
            changes = math.floor(math.fabs(particles[i].get_velocity()))
            sys.stdout.write("Changes for particle " + str(i) + ": " + str(changes) + "\n")
            for j in range(changes):
                # 50/50 chance.
                if random.random() > 0.5:
                    randomly_arrange(i)
                
                # Push it closer to it's best neighbor.
                copy_from_particle(i - 1, i)
            
            # Update pBest value.
            get_total_distance(i)
    
    return

def PSO_algorithm():
    epoch = 0
    done = False
    
    initialize_particles()
    
    while not done:
        # Two conditions can end this loop:
        # if the maximum number of epochs allowed has been reached, or,
        # if the Target value has been found.
        if epoch < MAX_EPOCHS:
            for i in range(PARTICLE_COUNT):
                sys.stdout.write("Route: ")
                
                for j in range(CITY_COUNT):
                    sys.stdout.write(str(particles[i].get_data(j)) + ", ")
                
                get_total_distance(i)
                sys.stdout.write("Distance: " + str(particles[i].get_pBest()) + "\n")
                
                if particles[i].get_pBest() <= TARGET:
                    done = True
            
            quicksort(particles, 0, len(particles) - 1)
            # list has to sorted in order for get_velocity() to work.
            get_velocity()
            
            update_particles()
            
            sys.stdout.write("epoch number: " + str(epoch) + "\n")
            
            epoch += 1
        
        else:
            done = True
    
    return

def print_best_solution():
    if particles[0].get_pBest() <= TARGET:
        sys.stdout.write("Target reached.\n")
    else:
        sys.stdout.write("Target not reached.\n")
    
    sys.stdout.write("Shortest Route: ")
    for j in range(CITY_COUNT):
        sys.stdout.write(str(particles[0].get_data(j)) + ", ")
    
    sys.stdout.write("Distance: " + str(particles[0].get_pBest()) + "\n")
    return

if __name__ == '__main__':
    initialize_map()
    PSO_algorithm()
    print_best_solution()
    
