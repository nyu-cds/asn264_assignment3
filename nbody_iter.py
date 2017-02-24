"""
Aditi Nair (asn264)
Feb 23 2017

Modifications using itertools include:
1. nbody function uses islice instead of range function
2. Created a variable called combos which is all possible combinations of planets. 
This is used instead of nested for loops in report_energy and advance function. 

I did not change the way the BODIES_KEYS list was iterated over since it seemed like a minor and trivial change - especially
given the size of the list. 

"""
from itertools import *

BODIES = {
    'sun': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 39.47841760435743),

    'jupiter': ([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01],
                [0.606326392995832,
                 2.81198684491626,
                 -0.02521836165988763],
                0.03769367487038949),

    'saturn': ([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01],
               [-1.0107743461787924,
                1.8256623712304119,
                0.008415761376584154],
               0.011286326131968767),

    'uranus': ([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01],
               [1.0827910064415354,
                0.8687130181696082,
                -0.010832637401363636],
               0.0017237240570597112),

    'neptune': ([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01],
                [0.979090732243898,
                 0.5946989986476762,
                 -0.034755955504078104],
                0.0020336868699246304)}

BODIES_KEYS = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']

combos=list(combinations(BODIES_KEYS,2))


def advance(BODIES, BODIES_KEYS, combos, dt, iterations):
    '''
        advance the system one timestep

        Initially: modified extra function calls here. Later: did not call, and put directly into nbody fn
    '''

    for _ in range(iterations):
        for combo in combos:
            ([x1, y1, z1], v1, m1) = BODIES[combo[0]]
            ([x2, y2, z2], v2, m2) = BODIES[combo[1]]
                    
            (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
            
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

    
def report_energy(BODIES, BODIES_KEYS, combos, e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''

    seenit = set()
    for combo in combos:
        ([x1, y1, z1], v1, m1) = BODIES[combo[0]]
        ([x2, y2, z2], v2, m2) = BODIES[combo[1]]
            
        (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)

        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)


    for body in BODIES_KEYS:
        (r, [vx, vy, vz], m) = BODIES[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e

def offset_momentum(BODIES, BODIES_KEYS, ref, px=0.0, py=0.0, pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''

    for body in BODIES_KEYS:
        (r, [vx, vy, vz], m) = BODIES[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m
        
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m


def nbody(loops, reference, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''

    offset_momentum(BODIES, BODIES_KEYS, BODIES[reference])

    for i in islice(count(),0,loops,1):
        advance(BODIES, BODIES_KEYS, combos, dt=0.01, iterations=iterations)
        print(report_energy(BODIES, BODIES_KEYS, combos))

if __name__ == '__main__':
    
    import timeit
    print timeit.timeit("nbody(100, 'sun', 20000)", setup="from __main__ import nbody", number=1)
