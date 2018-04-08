# http://mnemstudio.org/particle-swarm-example-3.htm
# Pattern Search Example
import win_unicode_console; win_unicode_console.enable()

import math
import random
import sys

MAX_LENGTH = 6
MIN_LENGTH = 2
TARGET = ['b', 'i', 'n', 'g', 'o']
CHAR_SET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
    'u', 'v', 'w', 'x', 'y', 'z']

PARTICLE_COUNT = 10
V_MAX = 4.0 # Maximum velocity change allowed.
V_MIN = 1.0
MAX_EPOCHS = 20000

particles = []

class Particle:
    def __init__(self):
        self.mData = []
        self.mpBest = 0
        self.mVelocity = 0.0

    def get_data(self, index):
        return self.mData[index]

    def get_data_array(self):
        return self.mData

    def get_data_length(self):
        return len(self.mData)

    def append_data(self, value):
        self.mData.append(value)

    def insert_data(self, index, value):
        self.mData[index] = value

    def remove_data_at(self, index):
        self.mData.pop(index)

    def get_pBest(self):
        return self.mpBest

    def set_pBest(self, value):
        self.mpBest = value

    def get_velocity(self):
        return self.mVelocity

    def set_velocity(self, velocityScore):
        self.mVelocity = velocityScore

    def to_string(self):
        dataLength = len(self.mData)
        temp = ""
        for i in range(dataLength):
            temp += self.mData[i]

        return temp

def initialize_particles():
    for i in range(PARTICLE_COUNT):
        newParticle = Particle()
        newLength = random.randrange(0, MAX_LENGTH)
        for j in range(newLength):
            newParticle.append_data(CHAR_SET[random.randrange(0, len(CHAR_SET))])

        newParticle.set_pBest(MAX_LENGTH) # Any large number should do.  Algorithm will push this down to 0.
        particles.append(newParticle)

    return

def test_problem(index):
    return levenshtein_distance(TARGET, particles[index].get_data_array())

def get_minimum_for_LD(a, b, c):
    # Get minimum of three values.
    mi = a
    if b < mi:
        mi = b

    if c < mi:
        mi = c

    return mi

def levenshtein_distance(s, t):
    n = len(s)
    m = len(t)

    d = []
    for i in range(m + 1):
        d.append([0] * (n + 1))

    # Initialize the first row to 0,...,m
    for i in range(m + 1):
        d[i][0] = i

    # Initialize the first column to 0,...,n
    for j in range(n + 1):
        d[0][j] = j

    # Examine each character of s (j from 1 to n).
    for j in range(1, n + 1):
        #Examine each character of t (i from 1 to m).
        for i in range(1, m + 1):
            if s[j - 1] == t[i - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                # Set cell d[i][j] of the matrix equal to the minimum of
                # The cell immediately above: d[i-1][j] + 1.
                # The cell immediately to left: d[i][j-1] + 1.
                # The cell diagonally above and left: d[i-1][j-1] + cost.
                d[i][j] = get_minimum_for_LD((d[i - 1][j]) + 1, (d[i][j - 1]) + 1, (d[i - 1][j - 1]) + 1)

    # After the iteration steps (3, 4, 5, 6) are complete, the distance is found in cell d[n,m]
    return d[m][n]

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
        elif vValue < V_MIN:
            particles[i].set_velocity(V_MIN)
        else:
            particles[i].set_velocity(vValue)

    return

def randomly_arrange(index = 0):
    dataLength = particles[index].get_data_length()

    # 50/50 chance of removing a character.
    if dataLength > 0 and random.random() > 0.5:
        target = random.randrange(0, dataLength)
        particles[index].remove_data_at(target)
    elif dataLength > MIN_LENGTH:
        # 50/50 chance of appending a new character.
        if dataLength < MAX_LENGTH and random.random() > 0.5:
            particles[index].append_data(CHAR_SET[random.randrange(len(CHAR_SET) - 1) + 1])
        else:
            target = random.randrange(0, dataLength)
            particles[index].insert_data(target, CHAR_SET[random.randrange(len(CHAR_SET) - 1) + 1])

    return

def copy_from_particle(source, destination):
    # Push destination's data points closer to source's data points.
    srcLength = particles[source].get_data_length()
    if srcLength > 0:
        destLength = particles[destination].get_data_length()
        target = random.randrange(0, srcLength) # source's character to target.

        if destLength >= srcLength:
            particles[destination].insert_data(target, particles[source].get_data(target))
        else:
            i = 0
            if destLength > 0:
                i = destLength - 1

            for j in range(i, srcLength):
                particles[destination].append_data(particles[source].get_data(target))

    return

def update_particles():
    # Best is at index 0, so start from the second best.
    for i in range(1, PARTICLE_COUNT):
        #The higher the velocity score, the more changes it will need.
        changes = math.floor(math.fabs(particles[i].get_velocity()))
        sys.stdout.write("Changes for particle " + str(i) + ": " + str(changes) + "\n")
        for j in range(changes):
            # 50/50 chance.
            if random.random() > 0.5:
                randomly_arrange(i)

            if random.random() > 0.5:
                copy_from_particle(i - 1, i) # Push it closer to it's best neighbor.
            else:
                copy_from_particle(0, i) # Push it closer to the best particle.
        
        # Update pBest value.
        particles[i].set_pBest(levenshtein_distance(TARGET, particles[i].get_data_array()))
    
    return

def PSO_algorithm():
    epoch = 0
    done = False

    while not done:
        # Two conditions can end this loop:
        # if the maximum number of epochs allowed has been reached, or,
        # if the Target value has been found.
        if epoch < MAX_EPOCHS:
            for i in range(PARTICLE_COUNT):
                sys.stdout.write(particles[i].to_string() + " = " + str(test_problem(i)) + "\n")
                if test_problem(i) <= 0:
                    done = True

            quicksort(particles, 0, PARTICLE_COUNT - 1) # sort particles by their pBest scores, best to worst.

            get_velocity()            
            update_particles()            
            sys.stdout.write("epoch number: " + str(epoch) + "\n")
            epoch += 1
        else:
            done = True
    
    return

if __name__ == '__main__':
    initialize_particles()
    PSO_algorithm()
    