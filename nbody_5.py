"""
Precompute all of the values associated with the bodies dictionary - including the list of keys.

    N-body simulation.

TIME: 127.394190788 SECONDS

THIRD MOST IMPROVEMENT
"""

#PI = 3.14159265358979323
#SOLAR_MASS = 4 * PI * PI #39.47841760435743
#DAYS_PER_YEAR = 365.24


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

def compute_deltas(x1, x2, y1, y2, z1, z2):
    return (x1-x2, y1-y2, z1-z2)
    
def compute_b(m, dt, dx, dy, dz):
    mag = compute_mag(dt, dx, dy, dz)
    return m * mag

def compute_mag(dt, dx, dy, dz):
    return dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))

def update_vs(v1, v2, dt, dx, dy, dz, m1, m2):
    v1[0] -= dx * compute_b(m2, dt, dx, dy, dz)
    v1[1] -= dy * compute_b(m2, dt, dx, dy, dz)
    v1[2] -= dz * compute_b(m2, dt, dx, dy, dz)
    v2[0] += dx * compute_b(m1, dt, dx, dy, dz)
    v2[1] += dy * compute_b(m1, dt, dx, dy, dz)
    v2[2] += dz * compute_b(m1, dt, dx, dy, dz)

def update_rs(r, dt, vx, vy, vz):
    r[0] += dt * vx
    r[1] += dt * vy
    r[2] += dt * vz

def advance(dt):
    '''
        advance the system one timestep
    '''
    seenit = []
    for body1 in BODIES_KEYS:
        for body2 in BODIES_KEYS:
            if (body1 != body2) and not (body2 in seenit):
                ([x1, y1, z1], v1, m1) = BODIES[body1]
                ([x2, y2, z2], v2, m2) = BODIES[body2]
                (dx, dy, dz) = compute_deltas(x1, x2, y1, y2, z1, z2)
                update_vs(v1, v2, dt, dx, dy, dz, m1, m2)
                seenit.append(body1)
        
    for body in BODIES_KEYS:
        (r, [vx, vy, vz], m) = BODIES[body]
        update_rs(r, dt, vx, vy, vz)

def compute_energy(m1, m2, dx, dy, dz):
    return (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
    
def report_energy(e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''
    seenit = []
    for body1 in BODIES_KEYS:
        for body2 in BODIES_KEYS:
            if (body1 != body2) and not (body2 in seenit):
                ((x1, y1, z1), v1, m1) = BODIES[body1]
                ((x2, y2, z2), v2, m2) = BODIES[body2]
                (dx, dy, dz) = compute_deltas(x1, x2, y1, y2, z1, z2)
                e -= compute_energy(m1, m2, dx, dy, dz)
                seenit.append(body1)
        
    for body in BODIES_KEYS:
        (r, [vx, vy, vz], m) = BODIES[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e

def offset_momentum(ref, px=0.0, py=0.0, pz=0.0):
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
    # Set up global state
    offset_momentum(BODIES[reference])

    for _ in range(loops):
        report_energy()
        for _ in range(iterations):
            advance(0.01)
        print(report_energy())

if __name__ == '__main__':
    #nbody(100, 'sun', 20000)
    
    import timeit
    print timeit.timeit("nbody(100, 'sun', 20000)", setup="from __main__ import nbody", number=1)
