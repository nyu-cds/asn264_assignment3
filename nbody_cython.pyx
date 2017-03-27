
'''
Aditi Nair
March 26 2017

For this assignment, I did the following:
- Add cdef declarations for all variables
- Use C types in function parameter declarations
- Modified functions to cpdef - so they can be used in imported notebooks as well as directly in the module.

I did not use numpy, so I did not do any ndarray declarations or efficient indexing.

Runtime: 8.57 sec
Runtime before cython: 38.42 sec

RELATIVE SPEED-UP: ~4.48

'''


cdef dict BODIES = {
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

cdef list BODIES_KEYS = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']


cpdef void advance(dict BODIES, list BODIES_KEYS, float dt, int iterations):
    '''
        advance the system one timestep

        Initially: modified extra function calls here. Later: did not call, and put directly into nbody fn
    '''

    cdef int idx
    cdef str body1, body2, body
    cdef float x1, y1, z1, m1, x2, y2, z2, m2, dx, dy, dz, val, m1_val, m2_val, vx, vy, vz, m
    cdef list v1, v2, r


    for _ in range(iterations):
        for idx, body1 in enumerate(BODIES_KEYS):
            

            ([x1, y1, z1], v1, m1) = BODIES[body1]
            for body2 in BODIES_KEYS[idx+1:]:
                ([x2, y2, z2], v2, m2) = BODIES[body2]
                
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

    
cpdef float report_energy(dict BODIES, list BODIES_KEYS, float e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''

    cdef int idx
    cdef str body1, body2, body
    cdef float x1, y1, z1, x2, y2, z2, m1, m2, dx, dy, dz, vx, vy, vz, m
    cdef list v1, v2, r

    seenit = set()
    for idx, body1 in enumerate(BODIES_KEYS):
        ((x1, y1, z1), v1, m1) = BODIES[body1]
        for body2 in BODIES_KEYS[idx+1:]:
            ((x2, y2, z2), v2, m2) = BODIES[body2]
            
            (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)

            e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)


    for body in BODIES_KEYS:
        (r, [vx, vy, vz], m) = BODIES[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e

cpdef void offset_momentum(dict BODIES, list BODIES_KEYS, tuple ref, float px=0.0, float py=0.0, float pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''


    cdef float m, vx, vy, vz
    cdef list r, v
    cdef str body

    for body in BODIES_KEYS:
        (r, [vx, vy, vz], m) = BODIES[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m
    
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m


cpdef void nbody(int loops, str reference, int iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''

    offset_momentum(BODIES, BODIES_KEYS, BODIES[reference])

    cdef int i
    for i in range(loops):

        advance(BODIES, BODIES_KEYS, dt=0.01, iterations=iterations)
        print(report_energy(BODIES, BODIES_KEYS))


if __name__ == '__main__':
    
    import timeit
    print timeit.timeit("nbody(100, 'sun', 20000)", setup="from __main__ import nbody", number=1)
