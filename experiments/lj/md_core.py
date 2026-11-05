"""2D force-shifted Lennard-Jones MD; reduced units m=sigma=epsilon=kB=1.

Periodic rectangular cells, triangular reference lattice, BAOAB Langevin NVT.
No atoms are pinned and neither velocities nor thermostat noise are projected.
Straight-bond fractions support the spatial Irving--Kirkwood estimator.
"""
import numpy as np
from numba import njit

RC = 2.5
SKIN = 0.35
UC = 4.0 * (RC**-12 - RC**-6)
DUC = 24.0 * (RC**-7 - 2.0 * RC**-13)


def lattice(n, a):
    assert n % 2 == 0
    q = np.array([((i + 0.5 * (j % 2)) * a, j * np.sqrt(3)/2*a)
                  for j in range(n) for i in range(n)])
    center_id = (n//2)*n + n//2
    q -= q[center_id].copy()
    return q, np.array([n*a, n*np.sqrt(3)/2*a])


@njit(cache=True)
def neighbors(q, box):
    n = len(q)
    nc = np.maximum(np.floor(box / (RC + SKIN)).astype(np.int64), 1)
    head = np.full(nc[0]*nc[1], -1, dtype=np.int64)
    nxt = np.empty(n, dtype=np.int64)
    cell = np.empty((n, 2), dtype=np.int64)
    for i in range(n):
        for d in range(2):
            x = q[i, d] % box[d]
            cell[i, d] = min(int(x / box[d] * nc[d]), nc[d]-1)
        k = cell[i, 0] + nc[0]*cell[i, 1]
        nxt[i] = head[k]
        head[k] = i
    pairs = np.empty((n*80, 2), dtype=np.int64)
    count = 0
    for i in range(n):
        visited = np.full(9, -1, dtype=np.int64)
        nv = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                c = (cell[i, 0]+dx) % nc[0] + nc[0]*((cell[i, 1]+dy) % nc[1])
                seen = False
                for z in range(nv):
                    if visited[z] == c:
                        seen = True
                if seen:
                    continue
                visited[nv] = c
                nv += 1
                j = head[c]
                while j >= 0:
                    if j > i:
                        rx = q[i, 0]-q[j, 0]
                        ry = q[i, 1]-q[j, 1]
                        rx -= box[0]*np.rint(rx/box[0])
                        ry -= box[1]*np.rint(ry/box[1])
                        if rx*rx+ry*ry < (RC+SKIN)**2:
                            if count >= len(pairs):
                                raise ValueError('Neighbor capacity exceeded')
                            pairs[count, 0], pairs[count, 1] = i, j
                            count += 1
                    j = nxt[j]
    return pairs[:count].copy()


@njit(cache=True)
def forces(q, box, pairs):
    f = np.zeros_like(q)
    energy, virial = 0.0, 0.0
    for k in range(len(pairs)):
        i, j = pairs[k, 0], pairs[k, 1]
        rx, ry = q[i, 0]-q[j, 0], q[i, 1]-q[j, 1]
        rx -= box[0]*np.rint(rx/box[0])
        ry -= box[1]*np.rint(ry/box[1])
        r2 = rx*rx+ry*ry
        if r2 < RC*RC:
            r = np.sqrt(r2)
            r6 = r2**-3
            c = 24*(2*r6*r6-r6)/r2 + DUC/r
            fx, fy = c*rx, c*ry
            f[i, 0] += fx
            f[i, 1] += fy
            f[j, 0] -= fx
            f[j, 1] -= fy
            energy += 4*(r6*r6-r6)-UC-(r-RC)*DUC
            virial += c*r2
    return f, energy, virial


@njit(cache=True)
def bond_fraction(mx, my, rx, ry, radius):
    """Fraction of segment midpoint m, vector r inside disk at the origin."""
    r2 = rx*rx+ry*ry
    dot = mx*rx+my*ry
    disc = dot*dot-r2*(mx*mx+my*my-radius*radius)
    if disc <= 0:
        return 0.0
    left = max(-0.5, (-dot-np.sqrt(disc))/r2)
    right = min(0.5, (-dot+np.sqrt(disc))/r2)
    return max(0.0, right-left)

