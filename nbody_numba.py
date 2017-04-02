"""
N-body simulation with Numba

Aditi Nair (asn264)
March 29 2016

In this script, I made the following modifications:
- Add jit decorators to all functions
- Add function signatures to all functions

Also, since numba cannot specifically handle dictionaries and tuples of mixed types I tried two experiments. 
In one, I simply declared dictionaries (BODIES) and tuples of mixed types (BODIES[reference]) with 'void' in function signatures.
In another, I changed the logic of my code so that I did not need to include either dictionaries or tuples of mixed types in the 
function signatures.

Since the second method was faster, I used that.

Next, I created a vectorized function called vec_deltas and used it to compute (dx,dy,dz) in the report_energy and advance functions.

"""

import numpy as np
from numba import jit, vectorize, int32, float64

BODIES = {
    'sun': (np.array([0.0, 0.0, 0.0],dtype=np.float64), [0.0, 0.0, 0.0], 39.47841760435743),

    'jupiter': (np.array([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01],dtype=np.float64),
                [0.606326392995832,
                 2.81198684491626,
                 -0.02521836165988763],
                0.03769367487038949),

    'saturn': (np.array([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01],dtype=np.float64),
               [-1.0107743461787924,
                1.8256623712304119,
                0.008415761376584154],
               0.011286326131968767),

    'uranus': (np.array([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01],dtype=np.float64),
               [1.0827910064415354,
                0.8687130181696082,
                -0.010832637401363636],
               0.0017237240570597112),

    'neptune': (np.array([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01],dtype=np.float64),
                [0.979090732243898,
                 0.5946989986476762,
                 -0.034755955504078104],
                0.0020336868699246304)}

BODIES_KEYS = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']

@vectorize([float64(float64, float64)])
def vec_deltas(x, y):
    return x - y

@jit('void(char[:],float64,int32)')
def advance(BODIES_KEYS, dt, iterations):
    '''
        advance the system one timestep

        Initially: modified extra function calls here. Later: did not call, and put directly into nbody fn
    '''

    for _ in range(iterations):
        for idx, body1 in enumerate(BODIES_KEYS):

            (a1,v1,m1) = BODIES[body1]

            for body2 in BODIES_KEYS[idx+1:]:
                
                (a2,v2,m2) = BODIES[body2]

                (dx, dy, dz) = vec_deltas(a1,a2)

                val = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
                m2_val = m2*val
                m1_val = m1*val
                v1[0] -= dx * m2_val
                v1[1] -= dy * m2_val
                v1[2] -= dz * m2_val
                v2[0] += dx * m1_val
                v2[1] += dy * m1_val
                v2[2] += dz * m1_val

            
        for body in BODIES_KEYS:
            (r, [vx, vy, vz], m) = BODIES[body]
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz

@jit('float64(char[:],float64)')
def report_energy(BODIES_KEYS, e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''

    seenit = set()
    for idx, body1 in enumerate(BODIES_KEYS):

        (a1,v1,m1) = BODIES[body1]

        for body2 in BODIES_KEYS[idx+1:]:

            (a2,v2,m2) = BODIES[body2]

            (dx, dy, dz) = vec_deltas(a1,a2)

            e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)


    for body in BODIES_KEYS:
        (r, [vx, vy, vz], m) = BODIES[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e

@jit('void(char[:],char,float64,float64,float64)')
def offset_momentum(BODIES_KEYS, ref, px=0.0, py=0.0, pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''

    for body in BODIES_KEYS:
        (r, [vx, vy, vz], m) = BODIES[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m
        
    (r, v, m) = BODIES[ref]
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m

@jit('void(int32,char,int32)')
def nbody(loops, reference, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''

    offset_momentum(BODIES_KEYS,reference)

    for _ in range(loops):

        advance(BODIES_KEYS, dt=0.01, iterations=iterations)
        print(report_energy(BODIES_KEYS))

if __name__ == '__main__':
    
    import timeit
    print timeit.timeit("nbody(100, 'sun', 20000)", setup="from __main__ import nbody", number=1)
