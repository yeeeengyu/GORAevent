import random

NUM_PARTICLES = 60
BASE_VY = (0.01, 0.03)
WIND_VX = (-0.004, 0.004)
GRAVITY = 0.0006
RADIUS_PX = (2, 4)
ALPHA = 0.55
PRUNE_AFTER = 15

color_choices = [(0,255,255), (255,200,0), (255,105,180), (180,105,255), (200,255,200)]
def init_particle(n=NUM_PARTICLES):
    ps = []
    for _ in range(n):
        u = random.random()
        v = random.uniform(-0.1, 0.0)
        vu = random.uniform(*WIND_VX)
        vv = random.uniform(*BASE_VY)
        r = random.randint(*RADIUS_PX)
        ps.append([u, v, vu, vv, r])
    return ps

def update_particle(particles):
    for p in particles:
        p[3] += GRAVITY   # vv += g
        p[0] += p[2]      # u += vu
        p[1] += p[3]      # v += vv
        if p[0] < 0:
            p[0] = 0; p[2] *= -0.6
        elif p[0] > 1:
            p[0] = 1; p[2] *= -0.6
        if p[1] > 1.05:
            p[0] = random.random()
            p[1] = random.uniform(-0.1, 0.0)
            p[2] = random.uniform(*WIND_VX)
            p[3] = random.uniform(*BASE_VY)